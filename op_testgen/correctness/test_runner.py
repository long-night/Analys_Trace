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
        if cpu_out.dtype == torch.bool:
            cpu_out = cpu_out.to(torch.int32)
            cuda_out = cuda_out.to(torch.int32)
        diff = torch.abs(cpu_out - cuda_out)
        tolerance = atol + rtol * torch.abs(cpu_out)
        return bool(torch.all(diff <= tolerance))

    def _compute_errors(self, cpu_out: torch.Tensor, cuda_out: torch.Tensor) -> Dict[str, float]:
        """计算绝对误差和相对误差"""
        if cpu_out.dtype == torch.bool:
            cpu_out = cpu_out.to(torch.int32)
            cuda_out = cuda_out.to(torch.int32)
        diff = torch.abs(cpu_out - cuda_out)
        abs_err = diff

        rel_err = torch.zeros_like(diff)
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
                parts.append(f"T{i}={list(t.shape)}:{str(t.dtype).replace('torch.', '')}")
            elif isinstance(t, list):
                shapes = [list(x.shape) if isinstance(x, torch.Tensor) else str(x) for x in t]
                parts.append(f"T{i}=[{shapes}]")
        if test_case.args:
            parts.append(f"args={test_case.args}")
        if test_case.kwargs:
            parts.append(f"kwargs={test_case.kwargs}")
        return "; ".join(parts)

    def _run_single(self, test_case: TestCase, backend: str) -> CorrectnessResult:
        op = test_case.mapped_op
        op_name = op.op_info.name
        input_info = self._format_input_info(test_case)

        try:
            cpu_kwargs = {k: v for k, v in test_case.kwargs.items()}

            def to_cpu(args):
                return [
                    t.clone().cpu() if isinstance(t, torch.Tensor) else
                    [x.clone().cpu() if isinstance(x, torch.Tensor) else x for x in t] if isinstance(t, list) else
                    t
                    for t in args
                ]

            cpu_positional = []
            if test_case.positional_args:
                cpu_positional = to_cpu(test_case.positional_args)
            def _clone_to_cpu(t):
                if isinstance(t, torch.Tensor):
                    return t.clone().cpu()
                elif isinstance(t, list):
                    return [_clone_to_cpu(x) for x in t]
                return t
            cpu_tensors = [_clone_to_cpu(t) for t in test_case.input_tensors]

            try:
                if test_case.positional_args and not cpu_kwargs:
                    cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
                else:
                    cpu_out = op.callable(*cpu_tensors, *test_case.args, **cpu_kwargs)
            except RuntimeError as e:
                err_msg = str(e)
                if "itensor_view_from_dense" in err_msg or "expects float/bfloat16/half/int8" in err_msg:
                    if test_case.positional_args:
                        cpu_positional = to_cpu(test_case.positional_args)
                        cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
                    else:
                        cpu_tensors = [t.clone().cpu() for t in test_case.input_tensors]
                        cpu_out = op.callable(*cpu_tensors, *test_case.args, **cpu_kwargs)
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

            if backend == "cpu":
                return CorrectnessResult(op_name=op_name, passed=True, backend="cpu", input_info=input_info)

            # 2. CUDA / SWDNN
            if backend == "swdnn":
                os.environ["SWDNN"] = "ON"
            else:
                os.environ["SWDNN"] = "OFF"

            if not torch.cuda.is_available():
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message="CUDA not available",
                    input_info=input_info,
                )

            cuda_kwargs = {k: v for k, v in test_case.kwargs.items()}
            if test_case.positional_args and not cuda_kwargs:
                cuda_positional = []
                for arg in test_case.positional_args:
                    if isinstance(arg, torch.Tensor):
                        cuda_positional.append(arg.cuda())
                    elif isinstance(arg, list):
                        cuda_positional.append([t.cuda() if isinstance(t, torch.Tensor) else t for t in arg])
                    else:
                        cuda_positional.append(arg)
                cuda_out = op.callable(*cuda_positional, **cuda_kwargs)
            else:
                cuda_tensors = [t.clone().cuda() for t in test_case.input_tensors]
                cuda_out = op.callable(*cuda_tensors, *test_case.args, **cuda_kwargs)

            if isinstance(cuda_out, tuple):
                cuda_out = [o for o in cuda_out if isinstance(o, torch.Tensor)]
                if cuda_out:
                    cuda_out = cuda_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only"
                    )

            cuda_out_cpu = cuda_out.cpu().to(torch.float64)

            # 3. 对比
            if cpu_out.shape != cuda_out_cpu.shape:
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message=f"Shape mismatch: CPU {cpu_out.shape} vs CUDA {cuda_out_cpu.shape}",
                    output_shapes_match=False,
                )

            errors = self._compute_errors(cpu_out, cuda_out_cpu)

            # 4. 阈值检查
            dtype_str = op.op_info.input_types[0] if op.op_info.input_types else "float32"
            atol, rtol = self._get_threshold(dtype_str, op_name)

            passed = self._check_tolerance(cpu_out, cuda_out_cpu, atol, rtol)

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
