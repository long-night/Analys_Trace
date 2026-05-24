"""性能测试执行模块"""
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from op_testgen.builder.tensor_builder import TestCase, TensorBuilder
from op_testgen.config import get_settings
from op_testgen.perf.classifier import OpClassifier
from op_testgen.perf.metrics import MetricsCalculator


@dataclass
class PerfResult:
    """性能测试结果"""
    op_name: str
    category: str
    backend: str
    avg_time_ms: float = 0.0
    flops: float = 0.0
    bandwidth_gbps: float = 0.0
    speedup: float = 1.0
    error_message: Optional[str] = None


class PerfBenchmark:
    """性能测试执行器"""

    def __init__(self, warmup_iters: int = 3, benchmark_iters: int = 10):
        self.settings = get_settings()
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.classifier = OpClassifier()
        self.metrics = MetricsCalculator()

    def _measure_time(self, test_case: TestCase, device: str) -> float:
        """测量算子平均执行时间（秒）"""
        op = test_case.mapped_op
        tensors = [t.clone().to(device) for t in test_case.input_tensors]
        kwargs = {k: v for k, v in test_case.kwargs.items()}

        for _ in range(self.warmup_iters):
            op.callable(*tensors, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            op.callable(*tensors, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / self.benchmark_iters

    def run(self, test_case: TestCase, backend: str = "cuda") -> PerfResult:
        """执行单个算子的性能测试"""
        op = test_case.mapped_op
        op_name = op.op_info.name
        category = self.classifier.classify(op_name)

        try:
            cpu_time = self._measure_time(test_case, "cpu")

            cuda_time = cpu_time
            if backend != "cpu" and torch.cuda.is_available():
                if backend == "swdnn":
                    import os
                    os.environ["SWDNN"] = "ON"
                else:
                    import os
                    os.environ["SWDNN"] = "OFF"
                cuda_time = self._measure_time(test_case, "cuda")

            speedup = cpu_time / cuda_time if cuda_time > 0 else 1.0

            flops = 0.0
            bandwidth = 0.0

            dtype = torch.float32
            if op.op_info.input_types:
                tb = TensorBuilder()
                dtype = tb._map_dtype(op.op_info.input_types[0])

            if category == "compute":
                formula = self.classifier.get_flops_formula(op_name)
                total_flops = self.metrics.compute_flops(op_name, op.op_info.input_dims, formula)
                flops = (total_flops / (cuda_time * 1e9)) if cuda_time > 0 else 0.0

            elif category in ("memory", "mixed"):
                bytes_total = self.metrics.compute_bytes(op.op_info.input_dims, dtype)
                bandwidth = (bytes_total / (cuda_time * 1e9)) if cuda_time > 0 else 0.0

            elif category == "communication":
                bytes_total = self.metrics.compute_communication_bytes(op.op_info.input_dims, dtype)
                bandwidth = (bytes_total / (cuda_time * 1e9)) if cuda_time > 0 else 0.0

            return PerfResult(
                op_name=op_name,
                category=category,
                backend=backend,
                avg_time_ms=cuda_time * 1000,
                flops=flops,
                bandwidth_gbps=bandwidth,
                speedup=speedup,
            )

        except Exception as e:
            return PerfResult(
                op_name=op_name,
                category=category,
                backend=backend,
                error_message=str(e),
            )

    def run_all(self, test_cases: List[TestCase], backend: str = "cuda") -> List[PerfResult]:
        """批量性能测试"""
        return [self.run(tc, backend) for tc in test_cases]
