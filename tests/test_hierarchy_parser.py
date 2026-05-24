"""层级构建算法测试"""
import pytest

from op_testgen.parser.trace_parser import OpInfo, HierarchicalOpInfo, TraceParser


class TestBuildHierarchy:
    """测试 _build_hierarchy 算法"""

    def test_simple_nested(self):
        """简单嵌套：parent -> child"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="parent", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="child", start_ts=10, duration_us=50, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert len(roots) == 1
        assert roots[0].name == "parent"
        assert roots[0].is_root is True
        assert len(roots[0].children) == 1
        assert roots[0].children[0].name == "child"
        assert roots[0].children[0].parent == roots[0]
        assert roots[0].children[0].depth == 1

    def test_deep_nesting(self):
        """深层嵌套：A -> B -> C"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=10, duration_us=80, tid=1),
            OpInfo(name="C", start_ts=20, duration_us=60, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert roots[0].name == "A"
        assert roots[0].depth == 0
        assert roots[0].children[0].name == "B"
        assert roots[0].children[0].depth == 1
        assert roots[0].children[0].children[0].name == "C"
        assert roots[0].children[0].children[0].depth == 2

    def test_sibling_ops(self):
        """同级算子：A -> [B, C]（顺序执行）"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=10, duration_us=20, tid=1),
            OpInfo(name="C", start_ts=40, duration_us=20, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert len(roots) == 1
        assert roots[0].name == "A"
        assert len(roots[0].children) == 2
        assert roots[0].children[0].name == "B"
        assert roots[0].children[1].name == "C"
        assert roots[0].children[0].parent == roots[0]
        assert roots[0].children[1].parent == roots[0]

    def test_multi_tid(self):
        """多线程独立处理"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=0, duration_us=50, tid=2),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert len(roots) == 2
        assert all(r.is_root for r in roots)

    def test_unmatched_events(self):
        """未匹配事件（孤儿节点，时间不与任何节点重叠）"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="parent", start_ts=0, duration_us=50, tid=1),
            OpInfo(name="orphan", start_ts=100, duration_us=10, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert len(roots) == 2
        root_names = {r.name for r in roots}
        assert root_names == {"orphan", "parent"}

    def test_self_nested(self):
        """同名算子嵌套：A -> A"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="A", start_ts=10, duration_us=50, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)

        assert roots[0].name == "A"
        assert roots[0].is_root is True
        assert roots[0].children[0].name == "A"
        assert roots[0].children[0].depth == 1
