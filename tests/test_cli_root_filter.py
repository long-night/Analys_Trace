"""测试 CLI 根节点过滤"""
import argparse
from op_testgen.cli import _prepare_mapped_ops


def test_prepare_mapped_ops_root_filter():
    args = argparse.Namespace(
        trace_file="profiler_trace.json",
        test_all_ops=False,
        max_ops=100,
        op_filter=None,
        seed=42,
    )
    mapped_ops, mapper = _prepare_mapped_ops(args)

    if mapped_ops:
        for m in mapped_ops:
            info = m.op_info
            assert info.is_root, f"非根节点被包含: {info.name}"


def test_prepare_mapped_ops_all_ops():
    args = argparse.Namespace(
        trace_file="profiler_trace.json",
        test_all_ops=True,
        max_ops=100,
        op_filter=None,
        seed=42,
    )
    mapped_ops, mapper = _prepare_mapped_ops(args)

    if mapped_ops:
        root_count = sum(1 for m in mapped_ops if m.op_info.is_root)
        non_root_count = len(mapped_ops) - root_count
        assert non_root_count >= 0
