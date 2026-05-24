"""测试 HierarchicalOpInfo.hierarchical_name"""
from op_testgen.parser.trace_parser import HierarchicalOpInfo


def test_hierarchical_name_single_node():
    root = HierarchicalOpInfo(name="aten::add")
    root.is_root = True
    root.depth = 0
    assert root.hierarchical_name == "aten::add"


def test_hierarchical_name_with_children():
    root = HierarchicalOpInfo(name="aten::convolution_backward")
    root.is_root = True
    root.depth = 0

    child1 = HierarchicalOpInfo(name="aten::contiguous")
    child1.parent = root
    child1.depth = 1
    root.children.append(child1)

    child2 = HierarchicalOpInfo(name="aten::clone")
    child2.parent = child1
    child2.depth = 2
    child1.children.append(child2)

    assert root.hierarchical_name == "aten::convolution_backward"
    assert child1.hierarchical_name == "aten::convolution_backward/aten::contiguous"
    assert child2.hierarchical_name == "aten::convolution_backward/aten::contiguous/aten::clone"
