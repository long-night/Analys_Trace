"""测试 generator 模块"""
import ast
import os
import tempfile

import pytest

from op_testgen.generator.test_case_generator import TestCaseGenerator
from op_testgen.generator.test_case_runner import InProcessRunner
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.parser.trace_parser import OpInfo


class TestSerializeTestCase:
    def test_serialize_basic_op(self):
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


class TestGenerateMethod:
    def test_generate_creates_file(self):
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped], output_path, source_trace="test.json", backend="cpu", seed=42)
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "TEST_CASES_DATA" in content
            assert "aten::add" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_generate_with_multiple_ops(self):
        op1 = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        op2 = OpInfo(
            name="aten::mul",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped1 = mapper.map_operator(op1)
        mapped2 = mapper.map_operator(op2)
        assert mapped1 is not None
        assert mapped2 is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped1, mapped2], output_path, source_trace="test.json")
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content.count("aten::add") >= 1
            assert content.count("aten::mul") >= 1
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestDecoupling:
    def test_generated_file_no_parser_mapper_imports(self):
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped], output_path, source_trace="test.json", backend="cpu", seed=42)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "op_testgen.parser" not in content
            assert "op_testgen.mapper" not in content
            assert "from op_testgen" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestInProcessRunner:
    def test_run_generated_file(self):
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped], output_path, source_trace="test.json", backend="cpu", seed=42)

            runner = InProcessRunner()
            returncode = runner.run(output_path, backend="cpu", only_correctness=True)

            assert returncode == 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_run_with_fail_fast(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name
            f.write('''
import sys

def main(argv):
    return 1
''')

        try:
            runner = InProcessRunner()
            returncode = runner.run(output_path)
            assert returncode == 1
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCLIE2E:
    def test_generate_subcommand(self):
        import subprocess

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    "python", "run.py", "generate",
                    "profiler_trace.json",
                    "-o", output_path,
                    "--max-ops", "5",
                    "--backend", "cpu",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "TEST_CASES_DATA" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_run_subcommand(self):
        import subprocess

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    "python", "run.py", "generate",
                    "profiler_trace.json",
                    "-o", output_path,
                    "--max-ops", "5",
                    "--backend", "cpu",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

            result = subprocess.run(
                ["python", "run.py", "run", output_path, "--only-correctness"],
                capture_output=True,
                text=True,
            )
            assert "测试完成" in result.stdout
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
