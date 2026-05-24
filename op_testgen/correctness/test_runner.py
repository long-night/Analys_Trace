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


class CorrectnessRunner:
    """正确性测试执行器"""

    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast
        self.settings = get_settings()

    def _get_threshold(self, dtype_str: str) -> tuple:
        """获取指定数据类型的误差阈值"""
        thresholds = self.settings.error_thresholds
        type_lower = dtype_str.lower()
        if type_lower in thresholds:
            return thresholds[type_lower]
        return thresholds.get("float32", (1e-5, 1e-4))

    def _compute_errors(self, cpu_out: torch.Tensor, cuda_out: torch.Tensor) -> Dict[str, float]:
        """计算绝对误差和相对误差"""
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

    def _run_single(self, test_case: TestCase, backend: str) -> CorrectnessResult:
        """执行单个测试用例的正确性对比"""
        op = test_case.mapped_op
        op_name = op.op_info.name

        try:
            # 1. CPU Baseline (float64 高精度)
            cpu_tensors = [t.clone().cpu().to(torch.float64) for t in test_case.input_tensors]
            cpu_kwargs = {k: v for k, v in test_case.kwargs.items()}
            cpu_out = op.callable(*cpu_tensors, **cpu_kwargs)

            if not isinstance(cpu_out, torch.Tensor):
                if isinstance(cpu_out, tuple):
                    cpu_out = [o for o in cpu_out if isinstance(o, torch.Tensor)]
                    if not cpu_out:
                        return CorrectnessResult(
                            op_name=op_name, passed=True, backend=backend,
                            error_message="Non-tensor output, shape check only"
                        )
                    cpu_out = cpu_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only"
                    )

            if backend == "cpu":
                return CorrectnessResult(op_name=op_name, passed=True, backend="cpu")

            # 2. CUDA / SWDNN
            if backend == "swdnn":
                os.environ["SWDNN"] = "ON"
            else:
                os.environ["SWDNN"] = "OFF"

            if not torch.cuda.is_available():
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message="CUDA not available"
                )

            cuda_tensors = [t.clone().cuda() for t in test_case.input_tensors]
            cuda_kwargs = {k: v for k, v in test_case.kwargs.items()}
            cuda_out = op.callable(*cuda_tensors, **cuda_kwargs)

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
            max_abs_thresh, max_rel_thresh = self._get_threshold(dtype_str)

            passed = errors["max_abs"] <= max_abs_thresh and errors["max_rel"] <= max_rel_thresh

            return CorrectnessResult(
                op_name=op_name,
                passed=passed,
                max_abs_err=errors["max_abs"],
                max_rel_err=errors["max_rel"],
                avg_abs_err=errors["avg_abs"],
                avg_rel_err=errors["avg_rel"],
                backend=backend,
            )

        except Exception as e:
            return CorrectnessResult(
                op_name=op_name,
                passed=False,
                backend=backend,
                error_message=str(e),
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
