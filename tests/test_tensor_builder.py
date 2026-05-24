"""测试 TensorBuilder"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.builder.tensor_builder import TensorBuilder, TestCase


class TestTensorBuilder:
    @pytest.fixture
    def builder(self):
        return TensorBuilder(seed=42)

    def test_build_float_tensors(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        assert mapped is not None

        test_case = builder.build(mapped)
        assert len(test_case.input_tensors) == 2
        assert all(t.dtype == torch.float32 for t in test_case.input_tensors)
        assert all(t.shape == torch.Size([2, 3]) for t in test_case.input_tensors)

    def test_build_different_types(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "long"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert test_case.input_tensors[0].dtype == torch.float32
        assert test_case.input_tensors[1].dtype == torch.int64

    def test_build_with_strides(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert test_case.input_tensors[0].stride() == (3, 1)

    def test_dtype_mapping(self, builder):
        assert builder._map_dtype("float") == torch.float32
        assert builder._map_dtype("float64") == torch.float64
        assert builder._map_dtype("half") == torch.float16
        assert builder._map_dtype("bfloat16") == torch.bfloat16
        assert builder._map_dtype("long") == torch.int64
        assert builder._map_dtype("int") == torch.int32
        assert builder._map_dtype("bool") == torch.bool

    def test_scalar_not_tensor(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_types=["float", "Scalar"],
            concrete_inputs=[1.5],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert len(test_case.input_tensors) == 1
        assert any(arg == 1.5 for arg in test_case.positional_args if not isinstance(arg, torch.Tensor))
