"""报告基类"""
from abc import ABC, abstractmethod
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult


class BaseReporter(ABC):
    """报告生成器基类"""

    @abstractmethod
    def generate(self, correctness_results: List[CorrectnessResult],
                 perf_results: List[PerfResult], output_path: str) -> None:
        """生成报告文件"""
        pass
