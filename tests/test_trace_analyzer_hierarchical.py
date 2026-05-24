"""测试 TraceAnalyzer 层级化聚合"""
from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.trace_analyzer import TraceAnalyzer


def test_analyzer_hierarchical_aggregation():
    root = HierarchicalOpInfo(name="aten::convolution_backward", duration_us=1000.0)
    root.is_root = True
    root.depth = 0

    child = HierarchicalOpInfo(name="aten::contiguous", duration_us=200.0)
    child.parent = root
    child.depth = 1
    root.children.append(child)

    analyzer = TraceAnalyzer([root, child], use_hierarchical=True)

    assert len(analyzer.operators) == 2
    assert "aten::convolution_backward" in analyzer.operators
    assert "aten::convolution_backward/aten::contiguous" in analyzer.operators

    root_stats = analyzer.operators["aten::convolution_backward"]
    assert root_stats.depth == 0
    assert root_stats.root_name == "aten::convolution_backward"

    child_stats = analyzer.operators["aten::convolution_backward/aten::contiguous"]
    assert child_stats.depth == 1
    assert child_stats.root_name == "aten::convolution_backward"


def test_analyzer_flat_mode_backward_compatible():
    root = HierarchicalOpInfo(name="aten::add", duration_us=100.0)
    child = HierarchicalOpInfo(name="aten::add", duration_us=50.0)
    child.parent = root
    child.depth = 1

    analyzer = TraceAnalyzer([root, child], use_hierarchical=False)
    assert len(analyzer.operators) == 1
    assert analyzer.operators["aten::add"].call_count == 2


def test_analyzer_hierarchical_csv_columns():
    root = HierarchicalOpInfo(name="aten::convolution_backward", duration_us=1000.0)
    root.is_root = True
    root.depth = 0

    child = HierarchicalOpInfo(name="aten::contiguous", duration_us=200.0)
    child.parent = root
    child.depth = 1
    root.children.append(child)

    analyzer = TraceAnalyzer([root, child], use_hierarchical=True)

    import tempfile
    import os
    import csv

    with tempfile.TemporaryDirectory() as tmpdir:
        ops_file = os.path.join(tmpdir, "ops.csv")
        shapes_file = os.path.join(tmpdir, "shapes.csv")
        analyzer.export_to_csv(ops_file, shapes_file)

        with open(ops_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
            assert "深度" in fieldnames
            assert "根算子" in fieldnames
            assert rows[0]["深度"] == "0"
            assert rows[1]["深度"] == "1"

        with open(shapes_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            assert "深度" in fieldnames
            assert "根算子" in fieldnames
