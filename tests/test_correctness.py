"""测试 CorrectnessRunner"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.correctness.test_runner import CorrectnessRunner, CorrectnessResult


class TestCorrectnessRunner:
    @pytest.fixture
    def runner(self):
        return CorrectnessRunner()

    def test_cpu_only_baseline(self, runner):
        """测试仅 CPU 执行（无 CUDA 时）"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        assert mapped is not None
        builder = TensorBuilder(seed=42)
        test_case = builder.build(mapped)

        result = runner.run(test_case, backend="cpu")
        assert result.passed is True
        assert result.max_abs_err == 0.0

    def test_error_thresholds(self):
        """测试不同 dtype 的误差阈值"""
        runner = CorrectnessRunner()
        assert runner._get_threshold("float32") == (1e-5, 1e-4)
        assert runner._get_threshold("float16") == (1e-3, 1e-2)
        assert runner._get_threshold("bfloat16") == (5e-3, 5e-2)
        assert runner._get_threshold("float64") == (1e-10, 1e-9)

    def test_fail_fast(self, runner):
        """测试 fail-fast 模式"""
        runner_failfast = CorrectnessRunner(fail_fast=True)
        assert runner_failfast.fail_fast is True
