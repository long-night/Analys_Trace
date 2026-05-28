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
        runner = CorrectnessRunner()
        assert runner._get_threshold("float32", "aten::add") == (1e-5, 1e-4)
        assert runner._get_threshold("float16", "aten::add") == (1e-3, 1e-2)
        assert runner._get_threshold("bfloat16", "aten::add") == (5e-3, 5e-2)
        assert runner._get_threshold("float64", "aten::add") == (1e-10, 1e-9)
        assert runner._get_threshold("float32", "aten::conv2d") == (1e-1, 1e-1)

    def test_fail_fast(self, runner):
        """测试 fail-fast 模式"""
        runner_failfast = CorrectnessRunner(fail_fast=True)
        assert runner_failfast.fail_fast is True

    def test_cpu_baseline_timing(self, runner):
        """测试 CPU baseline 执行时间被记录"""
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
        assert result.cpu_time_ms > 0
        assert result.cpu_out is not None

    def test_cpu_baseline_cache_reuse(self, runner):
        """测试跨 backend 复用 CPU baseline 缓存"""
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

        # 第一次：完整执行（含 CPU baseline）
        result1 = runner.run(test_case, backend="cpu")
        assert result1.passed is True
        assert result1.cpu_time_ms > 0
        cache = (result1.cpu_out, result1.cpu_time_ms)

        # 第二次：复用缓存（跳过 CPU baseline）
        result2 = runner.run(test_case, backend="cpu", cpu_baseline_cache=cache)
        assert result2.passed is True
        assert result2.cpu_time_ms == result1.cpu_time_ms
