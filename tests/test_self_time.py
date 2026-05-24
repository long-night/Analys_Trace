"""自耗时计算边界条件测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo


class TestSelfTime:
    def test_single_node_no_children(self):
        node = HierarchicalOpInfo(name="A", duration_us=100)
        assert node.self_duration_us == 100.0

    def test_parent_with_children(self):
        parent = HierarchicalOpInfo(name="parent", duration_us=100)
        child1 = HierarchicalOpInfo(name="child1", duration_us=30)
        child2 = HierarchicalOpInfo(name="child2", duration_us=40)
        parent.children = [child1, child2]

        assert parent.self_duration_us == 30.0

    def test_zero_duration_children(self):
        parent = HierarchicalOpInfo(name="parent", duration_us=100)
        child = HierarchicalOpInfo(name="child", duration_us=0)
        parent.children = [child]

        assert parent.self_duration_us == 100.0

    def test_negative_self_time_clamped(self):
        parent = HierarchicalOpInfo(name="parent", duration_us=50)
        child = HierarchicalOpInfo(name="child", duration_us=60)
        parent.children = [child]

        assert parent.self_duration_us == 0.0

    def test_deep_tree_cumulative(self):
        root = HierarchicalOpInfo(name="root", duration_us=100)
        child = HierarchicalOpInfo(name="child", duration_us=80)
        grandchild = HierarchicalOpInfo(name="grandchild", duration_us=50)

        root.children = [child]
        child.children = [grandchild]

        assert root.self_duration_us == 20.0
        assert child.self_duration_us == 30.0
        assert grandchild.self_duration_us == 50.0
