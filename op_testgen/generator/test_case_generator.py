"""测试用例生成器"""
from typing import Any, Dict, List

from op_testgen.mapper.op_mapper import MappedOp


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
