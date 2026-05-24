"""命令行入口"""
import argparse
import json
import os
import sys
from typing import Optional

import torch

from op_testgen.analyzer.trace_analyzer import TraceAnalyzer
from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.config import get_settings
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.parser.trace_parser import TraceParser
from op_testgen.perf.benchmark import PerfBenchmark
from op_testgen.reporter.markdown_reporter import MarkdownReporter
from op_testgen.reporter.html_reporter import HTMLReporter
from op_testgen.reporter.excel_reporter import ExcelReporter


def _check_openpyxl() -> bool:
    try:
        import openpyxl
        return True
    except ImportError:
        return False


def cmd_analyze(args) -> int:
    """analyze 子命令：Trace 统计分析"""
    print("=" * 60)
    print("PyTorch Profiler Trace 分析器")
    print("=" * 60)

    print(f"\n[1/3] 解析 Trace 文件...")
    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse()
    print(f"  发现 {len(op_infos)} 个算子事件")

    print(f"\n[2/3] 统计分析...")
    analyzer = TraceAnalyzer(op_infos)
    print(f"  共 {len(analyzer.operators)} 个不同算子")

    # 导出
    print(f"\n[3/3] 导出报告...")
    use_excel = _check_openpyxl() and not args.csv

    if args.csv or not use_excel:
        operators_file = args.output if args.output else "cpu_operators.csv"
        if not operators_file.endswith(".csv"):
            operators_file += ".csv"
        base = operators_file.replace(".csv", "")
        shapes_file = f"{base}_shapes.csv"
        analyzer.export_to_csv(operators_file, shapes_file)
        print(f"  CSV 算子总表: {operators_file}")
        print(f"  CSV Shape统计表: {shapes_file}")
    else:
        output_file = args.output if args.output else "operator_analysis.xlsx"
        if not output_file.endswith(".xlsx"):
            output_file += ".xlsx"
        if analyzer.export_to_excel(output_file):
            print(f"  Excel 报告: {output_file}")

    if not args.no_summary:
        analyzer.print_summary()

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    return 0


def cmd_test(args) -> int:
    """test 子命令：测试生成与执行"""
    print("=" * 60)
    print("PyTorch Profiler 自动化测试生成器")
    print("=" * 60)

    # 1. 解析 Trace
    print("\n[1/6] 解析 Trace 文件...")
    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse()
    print(f"  发现 {len(op_infos)} 个算子")

    # 2. 映射算子
    print("\n[2/6] 映射算子...")
    mapper = OpMapper()
    mapped_ops = mapper.map_all(op_infos)
    print(f"  成功映射 {len(mapped_ops)} 个算子")

    # 3. 去重 + 构建测试用例
    print("\n[3/6] 去重并构建测试用例...")
    seen_keys = set()
    unique_mapped_ops = []
    for m in mapped_ops:
        info = m.op_info
        def _to_tuple(x):
            if isinstance(x, list):
                return tuple(_to_tuple(i) for i in x)
            return x
        dims_tuple = _to_tuple(info.input_dims)
        strides_tuple = _to_tuple(info.input_strides)
        types_tuple = tuple(info.input_types)
        key = (info.name, dims_tuple, strides_tuple, types_tuple)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_mapped_ops.append(m)

    print(f"  去重前: {len(mapped_ops)} 个, 去重后: {len(unique_mapped_ops)} 个")

    if args.op_filter:
        import fnmatch
        unique_mapped_ops = [m for m in unique_mapped_ops if fnmatch.fnmatch(m.op_info.name, args.op_filter)]
        print(f"  名称过滤后剩余: {len(unique_mapped_ops)} 个")

    if args.max_ops > 0 and len(unique_mapped_ops) > args.max_ops:
        print(f"  超过 --max-ops={args.max_ops}，截断至前 {args.max_ops} 个")
        unique_mapped_ops = unique_mapped_ops[:args.max_ops]

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    builder = TensorBuilder(seed=args.seed)
    test_cases = [builder.build(m) for m in unique_mapped_ops]
    print(f"  构建 {len(test_cases)} 个测试用例")

    # 确定后端
    backends = []
    if args.backend == "auto":
        backends = ["cuda"]
        if os.environ.get("SWDNN_AVAILABLE"):
            backends.append("swdnn")
    else:
        backends = [args.backend]

    all_correctness = []
    all_perf = []

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"后端: {backend.upper()}")
        print(f"{'='*60}")

        if not args.only_performance:
            print(f"\n[4/6] 执行正确性测试...")
            correctness_runner = CorrectnessRunner(fail_fast=args.fail_fast)
            correctness_results = correctness_runner.run_all(test_cases, backend=backend)
            all_correctness.extend(correctness_results)
            passed = sum(1 for r in correctness_results if r.passed)
            print(f"  通过: {passed}/{len(correctness_results)}")

        if not args.only_correctness:
            print(f"\n[5/6] 执行性能测试...")
            perf_benchmark = PerfBenchmark(benchmark_iters=args.iters)
            perf_results = perf_benchmark.run_all(test_cases, backend=backend)
            all_perf.extend(perf_results)
            avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results) if perf_results else 1.0
            print(f"  平均加速比: {avg_speedup:.2f}x")

    # 生成报告
    print(f"\n[6/6] 生成报告...")
    if args.format in ("markdown", "all"):
        md_reporter = MarkdownReporter()
        md_path = args.output if args.output.endswith(".md") else args.output + ".md"
        md_reporter.generate(all_correctness, all_perf, md_path)
        print(f"  Markdown 报告: {md_path}")

    if args.format in ("html", "all"):
        try:
            html_reporter = HTMLReporter()
            html_path = args.output if args.output.endswith(".html") else args.output + ".html"
            html_reporter.generate(all_correctness, all_perf, html_path)
            print(f"  HTML 报告: {html_path}")
        except ImportError as e:
            print(f"  警告: {e}")

    if args.format in ("json", "all"):
        json_path = args.output.replace(".md", ".json").replace(".html", ".json").replace(".xlsx", ".json")
        if not json_path.endswith(".json"):
            json_path += ".json"
        report_data = {
            "summary": {
                "total_ops": len(test_cases),
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
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  JSON 报告: {json_path}")

    if args.format in ("excel", "all"):
        try:
            excel_reporter = ExcelReporter()
            excel_path = args.output.replace(".md", ".xlsx").replace(".html", ".xlsx").replace(".json", ".xlsx")
            if not excel_path.endswith(".xlsx"):
                excel_path += ".xlsx"
            excel_reporter.generate(all_correctness, all_perf, excel_path)
            print(f"  Excel 报告: {excel_path}")
        except ImportError as e:
            print(f"  警告: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 黑名单更新提示
    if args.update_blacklist and mapper.unmapped_ops:
        print("\n" + "-" * 60)
        print(f"发现 {len(mapper.unmapped_ops)} 个未映射算子:")
        for name in sorted(mapper.unmapped_ops):
            print(f"  - {name}")
        print("-" * 60)
        try:
            answer = input(f"\n是否将这 {len(mapper.unmapped_ops)} 个算子加入黑名单? [y/N]: ")
            if answer.strip().lower() in ("y", "yes"):
                OpMapper.update_blacklist_yaml(sorted(mapper.unmapped_ops))
            else:
                print("已取消，未更新黑名单")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消，未更新黑名单")

    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler 自动化工具集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析 Trace（原 analys_trace_v4.py 功能）
  op_testgen analyze profiler_trace.json
  op_testgen analyze profiler_trace.json -o report.xlsx

  # 生成并执行测试
  op_testgen test profiler_trace.json
  op_testgen test profiler_trace.json --backend swdnn --output report.html
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # === analyze 子命令 ===
    analyze_parser = subparsers.add_parser(
        "analyze", help="分析 Trace 并输出统计报告",
        description="解析 PyTorch Profiler Chrome Trace，输出算子统计（Excel/CSV）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen analyze trace.json
  op_testgen analyze trace.json -o report.xlsx
  op_testgen analyze trace.json --csv -o operators.csv
  op_testgen analyze trace.json --no-summary
        """,
    )
    analyze_parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    analyze_parser.add_argument("-o", "--output", help="输出文件路径")
    analyze_parser.add_argument("--csv", action="store_true", help="强制输出 CSV 格式")
    analyze_parser.add_argument("--no-summary", action="store_true", help="不显示摘要")

    # === test 子命令 ===
    test_parser = subparsers.add_parser(
        "test", help="自动生成并执行算子测试",
        description="从 Trace 提取算子信息，生成正确性/性能测试用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen test profiler_trace.json
  op_testgen test profiler_trace.json --backend swdnn
  op_testgen test profiler_trace.json --only-correctness --fail-fast
        """,
    )
    test_parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    test_parser.add_argument("--backend", choices=["cuda", "swdnn", "auto", "cpu"], default="cuda",
                            help="对比后端 (默认: cuda)")
    test_parser.add_argument("-o", "--output", default="op_testgen_report.md", help="输出文件路径")
    test_parser.add_argument("--format", choices=["markdown", "html", "json", "excel", "all"], default="markdown",
                            help="输出格式 (默认: markdown)")
    test_parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    test_parser.add_argument("--iters", type=int, default=10, help="性能测试迭代次数")
    test_parser.add_argument("--max-ops", type=int, default=100, help="最大测试算子数（默认: 100）")
    test_parser.add_argument("--fail-fast", action="store_true", help="第一个失败即停止")
    test_parser.add_argument("--only-correctness", action="store_true", help="仅执行正确性测试")
    test_parser.add_argument("--only-performance", action="store_true", help="仅执行性能测试")
    test_parser.add_argument("--op-filter", help="仅测试匹配名称的算子 (支持通配符)")
    test_parser.add_argument("--update-whitelist", action="store_true", help="动态发现后更新白名单")
    test_parser.add_argument("--update-blacklist", action="store_true", help="将未映射算子加入黑名单")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
