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

    def test_large_stride(self, builder):
        """stride 超过 numel 时，应成功创建非连续张量"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_strides=[[100, 1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 3])
        assert t.stride() == (100, 1)
        assert not t.is_contiguous()

    def test_broadcast_stride_zero(self, builder):
        """stride=0（广播张量）应内存最优，底层存储=1"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[1000]],
            input_strides=[[0]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([1000])
        assert t.stride() == (0,)
        # 广播张量底层存储应为 1
        assert t.storage().size() == 1

    def test_negative_stride(self, builder):
        """负 stride 应通过 flip 模拟创建"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[3]],
            input_strides=[[-1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_case = builder.build(mapped)
            # 负 stride 模拟路径不应发出 RuntimeWarning
            assert not any(issubclass(warning.category, RuntimeWarning) for warning in w)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([3])
        assert t.numel() == 3

    def test_mixed_positive_negative_stride(self, builder):
        """混合正负 stride 应正确处理"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_strides=[[3, -1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 3])
        assert t.numel() == 6

    def test_overlapping_memory(self, builder):
        """重叠内存（非广播）应正确处理"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 2]],
            input_strides=[[1, 1]],
            input_types=["float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)

        t = test_case.input_tensors[0]
        assert t.shape == torch.Size([2, 2])
        assert t.stride() == (1, 1)
        assert t.storage().size() == 3  # 重叠内存，底层存储小于 numel

    def test_no_stride_fallback(self, builder):
        """无 stride 信息时不应发出警告"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_case = builder.build(mapped)
            # 不应有 RuntimeWarning
            assert not any(issubclass(warning.category, RuntimeWarning) for warning in w)

        assert len(test_case.input_tensors) == 2
        assert all(t.is_contiguous() for t in test_case.input_tensors)
