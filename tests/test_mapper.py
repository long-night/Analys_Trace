"""测试 OpMapper"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper, MappedOp


class TestOpMapper:
    @pytest.fixture
    def mapper(self):
        return OpMapper()

    def test_whitelist_mapping(self, mapper):
        op = OpInfo(name="aten::add", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.add
        assert mapped.callable_path == "torch.add"
        assert mapped.namespace == "torch"
        assert mapped.support_status == "whitelisted"

    def test_dynamic_mapping(self, mapper):
        # aten::abs 不在初始白名单中，但可通过动态反射映射
        op = OpInfo(name="aten::abs", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.abs
        assert mapped.support_status == "auto_discovered"

    def test_tensor_suffix_stripping(self, mapper):
        op = OpInfo(name="aten::add.Tensor", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.add

    def test_filter_unsupported(self, mapper):
        # aten::empty 在黑名单中
        op = OpInfo(name="aten::empty", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        assert mapped is None

    def test_filter_namespace(self, mapper):
        # aten::_foobar 无法映射到任何命名空间
        op = OpInfo(name="aten::_foobar", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        # 如果动态反射也找不到，返回 None
        assert mapped is None or mapped.support_status == "unsupported"

    def test_map_all(self, mapper):
        ops = [
            OpInfo(name="aten::add", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"]),
            OpInfo(name="aten::empty", input_dims=[[2, 3]], input_types=["float"]),
            OpInfo(name="aten::conv2d", input_dims=[[1, 3, 32, 32], [16, 3, 3, 3]], input_types=["float", "float"]),
        ]
        mapped = mapper.map_all(ops)
        assert len(mapped) == 2  # add 和 conv2d，empty 被过滤
        names = [m.op_info.name for m in mapped]
        assert "aten::add" in names
        assert "aten::conv2d" in names
