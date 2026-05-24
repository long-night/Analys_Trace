"""递归/循环调用检测测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer


class TestRecursionDetection:
    def test_no_recursion(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)

        root.children = [b]
        b.children = [c]
        b.parent = root
        c.parent = b

        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()

        assert len(patterns) == 0

    def test_self_recursion(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        a2 = HierarchicalOpInfo(name="A", start_ts=10, duration_us=80)

        root.children = [a2]
        a2.parent = root

        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()

        assert len(patterns) >= 1
        cycles = [p.cycle for p in patterns]
        assert ("A", "A") in cycles

    def test_mutual_recursion(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        a2 = HierarchicalOpInfo(name="A", start_ts=20, duration_us=60)

        root.children = [b]
        b.children = [a2]
        b.parent = root
        a2.parent = b

        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()

        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "A") in cycles

    def test_longer_cycle(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        a2 = HierarchicalOpInfo(name="A", start_ts=30, duration_us=40)

        root.children = [b]
        b.children = [c]
        c.children = [a2]
        b.parent = root
        c.parent = b
        a2.parent = c

        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()

        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "C", "A") in cycles

    def test_max_cycle_length_filter(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        d = HierarchicalOpInfo(name="D", start_ts=30, duration_us=40)
        a2 = HierarchicalOpInfo(name="A", start_ts=40, duration_us=30)

        root.children = [b]
        b.children = [c]
        c.children = [d]
        d.children = [a2]
        b.parent = root
        c.parent = b
        d.parent = c
        a2.parent = d

        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns(max_cycle_length=3)

        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "C", "D", "A") not in cycles
