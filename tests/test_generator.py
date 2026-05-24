"""测试 generator 模块"""
import pytest
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper, MappedOp
from op_testgen.generator.test_case_generator import TestCaseGenerator


class TestSerializeTestCase:
    def test_serialize_basic_op(self):
        """测试基本算子的序列化"""
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[3, 4], [3, 4]],
            input_strides=[[4, 1], [4, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        data = gen._serialize_test_case(mapped)

        assert data["op_name"] == "aten::add"
        assert data["callable_path"] == "torch.add"
        assert data["input_dims"] == [[3, 4], [3, 4]]
        assert data["input_strides"] == [[4, 1], [4, 1]]
        assert data["input_types"] == ["float", "float"]
        assert data["concrete_inputs"] == []

    def test_serialize_op_with_concrete_inputs(self):
        """测试带 concrete_inputs 的算子序列化"""
        op_info = OpInfo(
            name="aten::sum",
            input_dims=[[2, 3, 4]],
            input_strides=[[12, 4, 1]],
            input_types=["float"],
            concrete_inputs=[0, True],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        data = gen._serialize_test_case(mapped)

        assert data["op_name"] == "aten::sum"
        assert data["concrete_inputs"] == [0, True]
