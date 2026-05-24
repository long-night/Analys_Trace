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
