"""调用链提取测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer


class TestCallChains:
    def test_simple_chain(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)

        root.children = [b]
        b.parent = root
        b.children = [c]
        c.parent = b

        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1)

        chain_tuples = [c.chain for c in chains]
        assert ("A", "B", "C") in chain_tuples
        assert ("A", "B") in chain_tuples
        assert ("A",) in chain_tuples

    def test_branching_tree(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=40)
        c = HierarchicalOpInfo(name="C", start_ts=60, duration_us=30)

        root.children = [b, c]
        b.parent = root
        c.parent = root

        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1)

        chain_tuples = [c.chain for c in chains]
        assert ("A", "B") in chain_tuples
        assert ("A", "C") in chain_tuples

    def test_min_occurrence_filter(self):
        root1 = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b1 = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        root1.children = [b1]
        b1.parent = root1

        root2 = HierarchicalOpInfo(name="A", start_ts=200, duration_us=100, is_root=True)
        b2 = HierarchicalOpInfo(name="B", start_ts=210, duration_us=80)
        root2.children = [b2]
        b2.parent = root2

        analyzer = HierarchyAnalyzer([root1, root2])

        chains = analyzer.get_call_chains(min_occurrence=2)
        assert len(chains) >= 1
        chain = next(c for c in chains if c.chain == ("A", "B"))
        assert chain.occurrence_count == 2

        chains = analyzer.get_call_chains(min_occurrence=3)
        assert not any(c.chain == ("A", "B") for c in chains)

    def test_max_depth_filter(self):
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        d = HierarchicalOpInfo(name="D", start_ts=30, duration_us=40)

        root.children = [b]
        b.children = [c]
        c.children = [d]
        b.parent = root
        c.parent = b
        d.parent = c

        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1, max_depth=2)

        chain_tuples = [c.chain for c in chains]
        assert ("A", "B", "C", "D") not in chain_tuples
        assert ("A", "B") in chain_tuples
