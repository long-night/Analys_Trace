"""测试用例生成器"""
from datetime import datetime
from typing import Any, Dict, List

from op_testgen.mapper.op_mapper import MappedOp
from op_testgen.generator.template import TEST_FILE_TEMPLATE


class TestCaseGenerator:
    """将 MappedOp 序列化为可执行的 Python 测试文件"""

    def __init__(self):
        pass

    def _serialize_test_case(self, mapped_op: MappedOp) -> Dict[str, Any]:
        """将单个 MappedOp 序列化为字典"""
        op_info = mapped_op.op_info
        return {
            "op_name": op_info.name,
            "callable_path": mapped_op.callable_path,
            "input_dims": op_info.input_dims,
            "input_strides": op_info.input_strides,
            "input_types": op_info.input_types,
            "concrete_inputs": op_info.concrete_inputs,
        }

    def _render_template(self, data: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """渲染模板生成 Python 源码"""
        from string import Template

        template = Template(TEST_FILE_TEMPLATE)
        return template.substitute(
            source_trace=options.get("source_trace", "unknown"),
            generated_at=datetime.now().isoformat(),
            backend=options.get("backend", "cuda"),
            seed=options.get("seed", 42),
            iters=options.get("iters", 10),
            test_cases_data_repr=repr(data),
        )
