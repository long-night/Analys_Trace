"""测试文件模板

生成的 .py 文件仅依赖 op_testgen 核心测试执行模块，不依赖 parser/mapper 等模块。
"""

TEST_FILE_TEMPLATE = '''#!/usr/bin/env python3
# Auto-generated test cases from trace: ${source_trace}
# Generated at: ${generated_at}
# Default options: backend=${backend}, seed=${seed}, iters=${iters}

import argparse
import fnmatch
import importlib
import sys
from typing import Any, Dict, List, Optional

import torch

# 仅依赖 op_testgen 的核心测试执行模块
from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.perf.benchmark import PerfBenchmark
from op_testgen.reporter.markdown_reporter import MarkdownReporter
try:
    from op_testgen.reporter.html_reporter import HTMLReporter
except ImportError:
    HTMLReporter = None
try:
    from op_testgen.reporter.excel_reporter import ExcelReporter
except ImportError:
    ExcelReporter = None

TEST_CASES_DATA = ${test_cases_data_repr}


# 简化数据类，避免导入 parser/mapper 模块
class _OpInfo:
    __slots__ = ("name", "input_dims", "input_strides", "input_types", "concrete_inputs")

    def __init__(self, name, input_dims, input_strides, input_types, concrete_inputs):
        self.name = name
        self.input_dims = input_dims
        self.input_strides = input_strides
        self.input_types = input_types
        self.concrete_inputs = concrete_inputs


class _MappedOp:
    __slots__ = ("op_info", "callable", "callable_path", "namespace", "support_status")

    def __init__(self, op_info, callable_obj, callable_path, namespace):
        self.op_info = op_info
        self.callable = callable_obj
        self.callable_path = callable_path
        self.namespace = namespace
        self.support_status = "generated"


def _build_mapped_op(data: Dict[str, Any]) -> Optional[_MappedOp]:
    """从序列化数据构造 _MappedOp（不构建张量）"""
    op_info = _OpInfo(
        name=data["op_name"],
        input_dims=data["input_dims"],
        input_strides=data["input_strides"],
        input_types=data["input_types"],
        concrete_inputs=data["concrete_inputs"],
    )

    parts = data["callable_path"].split(".")
    try:
        module = importlib.import_module(".".join(parts[:-1]))
        callable_obj = getattr(module, parts[-1])
    except (ImportError, AttributeError) as e:
        print(f"  警告: 无法导入 {data['callable_path']}: {e}")
        return None

    return _MappedOp(
        op_info=op_info,
        callable_obj=callable_obj,
        callable_path=data["callable_path"],
        namespace=parts[0],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auto-generated operator tests")
    parser.add_argument("--backend", default="${backend}", choices=["cuda", "swdnn", "cpu"])
    parser.add_argument("--only-correctness", action="store_true")
    parser.add_argument("--only-performance", action="store_true")
    parser.add_argument("--iters", type=int, default=${iters})
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=${seed})
    parser.add_argument("--op-filter", type=str, default=None)
    parser.add_argument("--format", choices=["markdown", "html", "json", "excel", "all"],
                        default="${format}")
    parser.add_argument("-o", "--output", default="${output}")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Auto-generated Operator Tests")
    print("=" * 60)

    # 过滤数据（不构建张量）
    test_data = TEST_CASES_DATA
    if args.op_filter:
        test_data = [
            d for d in TEST_CASES_DATA
            if fnmatch.fnmatch(d["op_name"], args.op_filter)
        ]
        print(f"\\n应用过滤 '{args.op_filter}' 后: {len(test_data)} 个测试用例")
    else:
        print(f"\\n共 {len(test_data)} 个测试用例待执行")

    if not test_data:
        print("错误: 没有可测试的算子")
        return 1

    builder = TensorBuilder(seed=args.seed)
    all_correctness = []
    all_perf = []

    if not args.only_performance:
        print("\\n执行正确性测试...")
        runner = CorrectnessRunner(fail_fast=args.fail_fast)

        for data in test_data:
            mapped_op = _build_mapped_op(data)
            if mapped_op is None:
                continue
            test_case = builder.build(mapped_op)
            try:
                result = runner.run(test_case, backend=args.backend)
                all_correctness.append(result)
                status = "通过" if result.passed else "失败"
                if result.error_message:
                    print(f"    [{status}] {result.op_name}: {result.error_message}")
                else:
                    print(f"    [{status}] {result.op_name}: max_abs={result.max_abs_err:.2e}, max_rel={result.max_rel_err:.2e}, avg_abs={result.avg_abs_err:.2e}, avg_rel={result.avg_rel_err:.2e}")
                if result.input_info:
                    print(f"      input: {result.input_info}")
            finally:
                del test_case
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        passed = sum(1 for r in all_correctness if r.passed)
        print(f"\\n  通过: {passed}/{len(all_correctness)}")

    if not args.only_correctness:
        print("\\n执行性能测试...")
        benchmark = PerfBenchmark(benchmark_iters=args.iters)

        for data in test_data:
            mapped_op = _build_mapped_op(data)
            if mapped_op is None:
                continue
            test_case = builder.build(mapped_op)
            try:
                result = benchmark.run(test_case, backend=args.backend)
                all_perf.append(result)
            finally:
                del test_case
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if all_perf:
            avg_speedup = sum(r.speedup for r in all_perf) / len(all_perf)
            print(f"\\n  平均加速比: {avg_speedup:.2f}x")

    print("\\n[报告] 生成测试报告...")
    if args.format in ("markdown", "all"):
        md_reporter = MarkdownReporter()
        md_path = args.output if args.output.endswith(".md") else args.output + ".md"
        md_reporter.generate(all_correctness, all_perf, md_path)
        print(f"  Markdown 报告: {md_path}")

    if args.format in ("html", "all"):
        if HTMLReporter is not None:
            html_reporter = HTMLReporter()
            html_path = args.output if args.output.endswith(".html") else args.output + ".html"
            html_reporter.generate(all_correctness, all_perf, html_path)
            print(f"  HTML 报告: {html_path}")
        else:
            print("  警告: HTML 报告需要 jinja2")

    if args.format in ("json", "all"):
        json_path = args.output.replace(".md", ".json").replace(".html", ".json").replace(".xlsx", ".json")
        if not json_path.endswith(".json"):
            json_path += ".json"
        import json as _json
        report_data = {
            "summary": {
                "total_ops": len(test_data),
                "correctness_passed": sum(1 for r in all_correctness if r.passed),
                "correctness_failed": sum(1 for r in all_correctness if not r.passed),
            },
            "correctness": [
                {"op_name": r.op_name, "passed": r.passed, "max_abs_err": r.max_abs_err,
                 "max_rel_err": r.max_rel_err, "error": r.error_message}
                for r in all_correctness
            ],
            "performance": [
                {"op_name": r.op_name, "category": r.category, "avg_time_ms": r.avg_time_ms,
                 "flops": r.flops, "bandwidth_gbps": r.bandwidth_gbps, "speedup": r.speedup}
                for r in all_perf
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  JSON 报告: {json_path}")

    if args.format in ("excel", "all"):
        if ExcelReporter is not None:
            excel_reporter = ExcelReporter()
            excel_path = args.output.replace(".md", ".xlsx").replace(".html", ".xlsx").replace(".json", ".xlsx")
            if not excel_path.endswith(".xlsx"):
                excel_path += ".xlsx"
            excel_reporter.generate(all_correctness, all_perf, excel_path)
            print(f"  Excel 报告: {excel_path}")
        else:
            print("  警告: Excel 报告需要 openpyxl")

    print("\\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    failed = sum(1 for r in all_correctness if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''
