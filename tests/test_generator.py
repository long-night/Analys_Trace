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


class TestTemplateRendering:
    def test_rendered_file_contains_header(self):
        """测试生成的文件包含头部注释"""
        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        assert "Auto-generated test cases" in content
        assert "test.json" in content

    def test_rendered_file_contains_test_data(self):
        """测试生成的文件包含 TEST_CASES_DATA"""
        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        assert "TEST_CASES_DATA" in content
        assert "aten::add" in content
        assert "torch.add" in content

    def test_rendered_file_is_valid_python(self):
        """测试生成的文件是合法 Python 语法"""
        import ast

        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        ast.parse(content)
