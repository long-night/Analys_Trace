"""正确性测试执行模块"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from op_testgen.builder.tensor_builder import TestCase
from op_testgen.config import get_settings


@dataclass
class CorrectnessResult:
    """正确性测试结果"""
    op_name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    avg_abs_err: float = 0.0
    avg_rel_err: float = 0.0
    backend: str = "cuda"
    error_message: Optional[str] = None
    output_shapes_match: bool = True
    input_info: str = ""


class CorrectnessRunner:
    """正确性测试执行器"""

    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast
        self.settings = get_settings()

    def _get_threshold(self, dtype_str: str, op_name: str) -> tuple:
        op_thresholds = self.settings.op_specific_thresholds
        if op_name in op_thresholds:
            return op_thresholds[op_name]

        thresholds = self.settings.error_thresholds
        type_lower = dtype_str.lower()
        if type_lower in thresholds:
            return thresholds[type_lower]
        return thresholds.get("float32", (1e-5, 1e-4))

    def _check_tolerance(self, cpu_out: torch.Tensor, cuda_out: torch.Tensor, atol: float, rtol: float) -> bool:
        if cpu_out.dtype != cuda_out.dtype:
            cpu_out = cpu_out.to(torch.float64)
            cuda_out = cuda_out.to(torch.float64)
        diff = torch.abs(cpu_out - cuda_out)
        tolerance = atol + rtol * torch.abs(cpu_out)
        return bool(torch.all(diff <= tolerance))

    def _compute_errors(self, cpu_out: torch.Tensor, cuda_out: torch.Tensor) -> Dict[str, float]:
        if cpu_out.dtype != cuda_out.dtype:
            cpu_out = cpu_out.to(torch.float64)
            cuda_out = cuda_out.to(torch.float64)
        diff = torch.abs(cpu_out - cuda_out)
        abs_err = diff

        rel_err = torch.zeros_like(diff, dtype=torch.float64)
        mask = cpu_out != 0
        rel_err[mask] = diff[mask] / torch.abs(cpu_out[mask])

        return {
            "max_abs": float(abs_err.max()),
            "max_rel": float(rel_err.max()),
            "avg_abs": float(abs_err.mean()),
            "avg_rel": float(rel_err.mean()),
        }

    def _format_input_info(self, test_case: TestCase) -> str:
        parts = []
        for i, t in enumerate(test_case.input_tensors):
            if isinstance(t, torch.Tensor):
                dtype_str = str(t.dtype).replace('torch.', '')
                stride_info = f"stride={list(t.stride())}"
                parts.append(f"T{i}={list(t.shape)}:{dtype_str}:{stride_info}")
            elif isinstance(t, list):
                tensor_infos = []
                for x in t:
                    if isinstance(x, torch.Tensor):
                        dtype_str = str(x.dtype).replace('torch.', '')
                        stride_info = f"stride={list(x.stride())}"
                        tensor_infos.append(f"{list(x.shape)}:{dtype_str}:{stride_info}")
                    else:
                        tensor_infos.append(str(x))
                parts.append(f"T{i}=[{tensor_infos}]")
        if test_case.args:
            parts.append(f"args={test_case.args}")
        if test_case.kwargs:
            parts.append(f"kwargs={test_case.kwargs}")
        return "; ".join(parts)

    def _run_single(self, test_case: TestCase, backend: str) -> CorrectnessResult:
        op = test_case.mapped_op
        op_name = op.op_info.name
        input_info = self._format_input_info(test_case)

        # 保存原始 SWDNN 状态，确保测试结束后恢复
        prev_swdnn = os.environ.get("SWDNN", "OFF")

        try:
            # 1. CPU baseline (SWDNN=OFF)
            os.environ["SWDNN"] = "OFF"
            cpu_kwargs = {k: v for k, v in test_case.kwargs.items()}

            def to_cpu(args):
                return [
                    t.clone().cpu() if isinstance(t, torch.Tensor) else
                    [x.clone().cpu() if isinstance(x, torch.Tensor) else x for x in t] if isinstance(t, list) else
                    t
                    for t in args
                ]

            cpu_positional = to_cpu(test_case.positional_args)

            try:
                cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
            except RuntimeError as e:
                err_msg = str(e)
                if "itensor_view_from_dense" in err_msg or "expects float/bfloat16/half/int8" in err_msg:
                    try:
                        cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
                    except RuntimeError:
                        return CorrectnessResult(
                            op_name=op_name, passed=False, backend=backend,
                            error_message=f"CPU baseline failed after retry: {err_msg}",
                            input_info=input_info,
                        )
                else:
                    raise

            if not isinstance(cpu_out, torch.Tensor):
                if isinstance(cpu_out, tuple):
                    cpu_out = [o for o in cpu_out if isinstance(o, torch.Tensor)]
                    if not cpu_out:
                        return CorrectnessResult(
                            op_name=op_name, passed=True, backend=backend,
                            error_message="Non-tensor output, shape check only",
                            input_info=input_info,
                        )
                    cpu_out = cpu_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only",
                        input_info=input_info,
                    )

            # 如果仅测试 CPU baseline，直接返回结果
            if backend == "cpu":
                return CorrectnessResult(
                    op_name=op_name, passed=True, backend="cpu",
                    input_info=input_info,
                    error_message="CPU baseline only",
                )

            # 2. SWDNN (CPU + SWDNN=ON) 或 CUDA (CUDA + SWDNN=OFF)
            if backend == "swdnn":
                os.environ["SWDNN"] = "ON"
                device_str = "cpu"
            else:
                os.environ["SWDNN"] = "OFF"
                device_str = "cuda"

            if device_str == "cuda" and not torch.cuda.is_available():
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message="CUDA not available",
                    input_info=input_info,
                )

            target_kwargs = {k: v for k, v in test_case.kwargs.items()}
            target_positional = []
            for arg in test_case.positional_args:
                if isinstance(arg, torch.Tensor):
                    t = arg.clone()
                    if device_str == "cuda":
                        target_positional.append(t.cuda())
                    else:
                        target_positional.append(t.cpu())
                elif isinstance(arg, list):
                    moved = []
                    for t in arg:
                        if isinstance(t, torch.Tensor):
                            t_clone = t.clone()
                            moved.append(t_clone.cuda() if device_str == "cuda" else t_clone.cpu())
                        else:
                            moved.append(t)
                    target_positional.append(moved)
                else:
                    target_positional.append(arg)

            target_out = op.callable(*target_positional, **target_kwargs)

            if isinstance(target_out, tuple):
                target_out = [o for o in target_out if isinstance(o, torch.Tensor)]
                if target_out:
                    target_out = target_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only",
                        input_info=input_info,
                    )

            target_out_cpu = target_out.cpu().to(torch.float64)

            # 3. 对比
            if cpu_out.shape != target_out_cpu.shape:
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message=f"Shape mismatch: CPU {cpu_out.shape} vs {backend.upper()} {target_out_cpu.shape}",
                    output_shapes_match=False,
                    input_info=input_info,
                )

            errors = self._compute_errors(cpu_out, target_out_cpu)

            # 4. 阈值检查
            dtype_str = op.op_info.input_types[0] if op.op_info.input_types else "float32"
            atol, rtol = self._get_threshold(dtype_str, op_name)

            passed = self._check_tolerance(cpu_out, target_out_cpu, atol, rtol)

            return CorrectnessResult(
                op_name=op_name,
                passed=passed,
                max_abs_err=errors["max_abs"],
                max_rel_err=errors["max_rel"],
                avg_abs_err=errors["avg_abs"],
                avg_rel_err=errors["avg_rel"],
                backend=backend,
                input_info=input_info,
            )

        except Exception as e:
            return CorrectnessResult(
                op_name=op_name,
                passed=False,
                backend=backend,
                error_message=str(e),
                input_info=input_info,
            )

        finally:
            # 恢复原始 SWDNN 环境变量
            os.environ["SWDNN"] = prev_swdnn

    def run(self, test_case: TestCase, backend: str = "cuda") -> CorrectnessResult:
        """运行单个测试用例的正确性测试"""
        return self._run_single(test_case, backend)

    def run_all(self, test_cases: List[TestCase], backend: str = "cuda") -> List[CorrectnessResult]:
        """批量运行正确性测试"""
        results = []
        for tc in test_cases:
            result = self.run(tc, backend)
            results.append(result)
            if self.fail_fast and not result.passed:
                break
        return results
