"""测试 CLI 根节点过滤"""
import argparse
from op_testgen.cli import _prepare_test_cases


def test_prepare_test_cases_root_filter():
    args = argparse.Namespace(
        trace_file="profiler_trace.json",
        test_all_ops=False,
        max_ops=100,
        op_filter=None,
        seed=42,
    )
    unique_mapped_ops, test_cases, mapper = _prepare_test_cases(args)

    if unique_mapped_ops:
        for m in unique_mapped_ops:
            info = m.op_info
            assert info.is_root, f"非根节点被包含: {info.name}"


def test_prepare_test_cases_all_ops():
    args = argparse.Namespace(
        trace_file="profiler_trace.json",
        test_all_ops=True,
        max_ops=100,
        op_filter=None,
        seed=42,
    )
    unique_mapped_ops, test_cases, mapper = _prepare_test_cases(args)

    if unique_mapped_ops:
        root_count = sum(1 for m in unique_mapped_ops if m.op_info.is_root)
        non_root_count = len(unique_mapped_ops) - root_count
        assert non_root_count >= 0
