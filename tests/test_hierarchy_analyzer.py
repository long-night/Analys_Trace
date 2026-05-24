"""层级分析器测试"""
import pytest
import tempfile
import json

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer, CallChain, RecursivePattern


@pytest.fixture
def sample_tree():
    """构建样本调用树"""
    root1 = HierarchicalOpInfo(name="aten::conv2d", start_ts=0, duration_us=100, tid=1, is_root=True)
    child1 = HierarchicalOpInfo(name="aten::convolution", start_ts=10, duration_us=60, tid=1)
    grandchild1 = HierarchicalOpInfo(name="aten::_convolution", start_ts=20, duration_us=40, tid=1)
    child2 = HierarchicalOpInfo(name="aten::add", start_ts=80, duration_us=10, tid=1)
    root2 = HierarchicalOpInfo(name="aten::matmul", start_ts=0, duration_us=50, tid=1, is_root=True)

    root1.children = [child1, child2]
    child1.parent = root1
    child2.parent = root1
    child1.children = [grandchild1]
    grandchild1.parent = child1

    root1.depth = 0
    child1.depth = 1
    grandchild1.depth = 2
    child2.depth = 1
    root2.depth = 0

    return [root1, root2]


class TestHierarchyAnalyzer:
    def test_all_nodes(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        nodes = analyzer.all_nodes
        assert len(nodes) == 5
        names = {n.name for n in nodes}
        assert names == {"aten::conv2d", "aten::convolution", "aten::_convolution", "aten::add", "aten::matmul"}

    def test_self_time_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_self_time_stats()

        assert stats["aten::conv2d"].total_duration_us == 30.0
        assert stats["aten::convolution"].total_duration_us == 20.0
        assert stats["aten::_convolution"].total_duration_us == 40.0
        assert stats["aten::matmul"].total_duration_us == 50.0

    def test_total_time_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_total_time_stats()

        assert stats["aten::conv2d"].total_duration_us == 100.0
        assert stats["aten::_convolution"].total_duration_us == 40.0

    def test_get_call_chains(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        chains = analyzer.get_call_chains(min_occurrence=1)

        assert len(chains) == 5

        conv_chain = next((c for c in chains if c.chain == ("aten::conv2d", "aten::convolution", "aten::_convolution")), None)
        assert conv_chain is not None
        assert conv_chain.occurrence_count == 1

    def test_get_root_ops_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_root_ops_stats()

        assert len(stats) == 2
        assert "aten::conv2d" in stats
        assert "aten::matmul" in stats

    def test_get_depth_distribution(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        dist = analyzer.get_depth_distribution()

        assert dist[0] == 2
        assert dist[1] == 2
        assert dist[2] == 1

    def test_export_call_tree_text(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        analyzer.export_call_tree_text(temp_path)

        with open(temp_path, "r") as f:
            content = f.read()

        assert "aten::conv2d" in content
        assert "aten::convolution" in content
        assert "aten::_convolution" in content
        assert "self=" in content

    def test_export_call_tree_json(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        analyzer.export_call_tree_json(temp_path)

        with open(temp_path, "r") as f:
            tree = json.load(f)

        assert len(tree) == 2
        assert tree[0]["name"] == "aten::conv2d"
        assert len(tree[0]["children"]) == 2
