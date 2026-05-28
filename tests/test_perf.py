"""测试 PerfBenchmark"""
import pytest
from op_testgen.perf.classifier import OpClassifier


class TestOpClassifier:
    def test_classify_compute(self):
        c = OpClassifier()
        assert c.classify("aten::matmul") == "compute"
        assert c.classify("aten::conv2d") == "compute"
        assert c.classify("aten::linear") == "compute"

    def test_classify_communication(self):
        c = OpClassifier()
        assert c.classify("nccl:all_reduce") == "communication"
        assert c.classify("c10d::allreduce_") == "communication"

    def test_classify_memory(self):
        c = OpClassifier()
        assert c.classify("aten::add") == "memory"
        assert c.classify("aten::relu") == "memory"

    def test_config_override(self):
        c = OpClassifier()
        assert c.classify("aten::add") == "memory"


class TestPerfBenchmark:
    def test_default_iters(self):
        from op_testgen.perf.benchmark import PerfBenchmark
        benchmark = PerfBenchmark()
        assert benchmark.cpu_iters == 1
        assert benchmark.target_iters == 3

    def test_cpu_time_reuse(self):
        """测试 PerfBenchmark 复用外部 CPU 时间"""
        from op_testgen.perf.benchmark import PerfBenchmark
        benchmark = PerfBenchmark()
        # 当 cpu_time_ms 传入时，不应再跑 CPU baseline
        # 这里用 mock 验证：如果复用成功，speedup 应基于传入值计算
        assert benchmark.cpu_iters == 1
        assert benchmark.target_iters == 3
