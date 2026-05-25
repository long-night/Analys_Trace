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
from op_testgen.mapper.op_mapper import MappedOp, OpMapper
from op_testgen.parser.trace_parser import TraceParser
from op_testgen.perf.benchmark import PerfBenchmark
from op_testgen.reporter.markdown_reporter import MarkdownReporter
from op_testgen.reporter.html_reporter import HTMLReporter
from op_testgen.reporter.excel_reporter import ExcelReporter
from op_testgen.generator.test_case_generator import TestCaseGenerator


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
    op_infos = parser_obj.parse_hierarchical()
    root_count = sum(1 for op in op_infos if op.is_root)
    print(f"  发现 {len(op_infos)} 个层级节点（{root_count} 个根节点）")

    print(f"\n[2/3] 统计分析...")
    analyzer = TraceAnalyzer(op_infos, use_hierarchical=True)
    print(f"  共 {len(analyzer.operators)} 个不同层级路径")

    # 导出
    print(f"\n[3/3] 导出报告...")
    use_excel = _check_openpyxl() and not args.csv
    output_file = args.output if args.output else "operator_analysis.xlsx"

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
        if not output_file.endswith(".xlsx"):
            output_file += ".xlsx"
        if analyzer.export_to_excel(output_file):
            print(f"  Excel 报告: {output_file}")

    if not args.no_summary:
        analyzer.print_summary()

    if args.hierarchy or args.call_chains or args.detect_recursion or args.export_tree:
        analyzer.print_hierarchical_summary()

    if args.export_tree:
        analyzer._ensure_hierarchy()
        assert analyzer._hierarchy_analyzer is not None
        analyzer._hierarchy_analyzer.export_call_tree_text(args.export_tree)
        print(f"调用树已导出: {args.export_tree}")

    if args.hierarchy and not args.csv:
        base, ext = os.path.splitext(output_file)
        hierarchy_file = f"{base}_hierarchy{ext}"
        if analyzer.export_hierarchy_to_excel(output_file):
            print(f"层级数据已添加到 Excel: {output_file}")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    return 0


def _prepare_test_cases(args) -> tuple:
    """提取 test 和 generate 子命令的公共逻辑

    Returns:
        (unique_mapped_ops, test_cases, mapper) 或 (unique_mapped_ops, None, mapper)
    """
    from op_testgen.parser.trace_parser import HierarchicalOpInfo

    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse_hierarchical()

    if not getattr(args, "test_all_ops", False):
        op_infos = [op for op in op_infos if op.is_root]
        print(f"  根节点过滤: {len(op_infos)} 个根节点待测试")
    else:
        print(f"  全量模式: {len(op_infos)} 个算子待测试")

    mapper = OpMapper()
    mapped_ops = mapper.map_all(op_infos)

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
        concrete_tuple = _to_tuple(info.concrete_inputs)
        name_key = info.hierarchical_name if isinstance(info, HierarchicalOpInfo) else info.name
        key = (name_key, dims_tuple, strides_tuple, types_tuple, concrete_tuple)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_mapped_ops.append(m)

    unique_mapped_ops.sort(key=lambda m: m.op_info.name)

    if hasattr(args, "op_filter") and args.op_filter:
        import fnmatch
        unique_mapped_ops = [
            m for m in unique_mapped_ops if fnmatch.fnmatch(m.op_info.name, args.op_filter)
        ]

    if hasattr(args, "max_ops") and args.max_ops > 0 and len(unique_mapped_ops) > args.max_ops:
        unique_mapped_ops = unique_mapped_ops[:args.max_ops]

    if not unique_mapped_ops:
        return [], None, mapper

    builder = TensorBuilder(seed=args.seed)
    test_cases = [builder.build(m) for m in unique_mapped_ops]

    return unique_mapped_ops, test_cases, mapper


def cmd_test(args) -> int:
    """test 子命令：测试生成与执行"""
    print("=" * 60)
    print("PyTorch Profiler 自动化测试生成器")
    print("=" * 60)

    print("\n[1/4] 解析并准备测试用例...")
    unique_mapped_ops, test_cases, mapper = _prepare_test_cases(args)

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(unique_mapped_ops)} 个算子")
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
            correctness_results = []

            from itertools import groupby
            for op_name, group in groupby(test_cases, key=lambda tc: tc.mapped_op.op_info.name):
                group_list = list(group)
                print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
                for tc in group_list:
                    result = correctness_runner.run(tc, backend=backend)
                    correctness_results.append(result)
                    all_correctness.append(result)
                    status = "通过" if result.passed else "失败"
                    if result.error_message:
                        print(f"    [{status}] {result.op_name}: {result.error_message}")
                    else:
                        print(f"    [{status}] {result.op_name}: max_abs={result.max_abs_err:.2e}, max_rel={result.max_rel_err:.2e}, avg_abs={result.avg_abs_err:.2e}, avg_rel={result.avg_rel_err:.2e}")
                    if result.input_info:
                        print(f"      input: {result.input_info}")
            passed = sum(1 for r in correctness_results if r.passed)
            print(f"\n  通过: {passed}/{len(correctness_results)}")

        if not args.only_correctness:
            print(f"\n[5/6] 执行性能测试...")
            perf_benchmark = PerfBenchmark(benchmark_iters=args.iters)

            from itertools import groupby
            for op_name, group in groupby(test_cases, key=lambda tc: tc.mapped_op.op_info.name):
                group_list = list(group)
                print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
                perf_results = perf_benchmark.run_all(group_list, backend=backend)
                all_perf.extend(perf_results)
            avg_speedup = sum(r.speedup for r in all_perf) / len(all_perf) if all_perf else 1.0
            print(f"\n  平均加速比: {avg_speedup:.2f}x")

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


def cmd_generate(args) -> int:
    """generate 子命令：生成测试用例文件"""
    print("=" * 60)
    print("PyTorch Profiler 测试用例生成器")
    print("=" * 60)

    print("\n[1/2] 解析并准备测试用例...")
    unique_mapped_ops, _, mapper = _prepare_test_cases(args)

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(unique_mapped_ops)} 个算子")

    print("\n[2/2] 生成测试文件...")
    generator = TestCaseGenerator()
    output_path = generator.generate(
        mapped_ops=unique_mapped_ops,
        output_path=args.output,
        source_trace=args.trace_file,
        backend=args.backend,
        seed=args.seed,
        iters=args.iters,
        format=getattr(args, "format", "markdown"),
        output=getattr(args, "report_output", "op_testgen_report.md"),
    )
    print(f"  测试文件: {output_path}")

    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"运行: op_testgen run {output_path}")
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


def cmd_run(args) -> int:
    """run 子命令：在当前进程内执行生成的 .py 测试文件"""
    import importlib.util

    print("=" * 60)
    print("执行生成的测试文件")
    print("=" * 60)

    if not os.path.exists(args.test_file):
        print(f"错误：文件不存在: {args.test_file}")
        return 1

    # 构建参数列表，传递给生成的 .py 文件的 main()
    test_argv = []
    if args.backend is not None:
        test_argv.extend(["--backend", args.backend])
    if args.only_correctness:
        test_argv.append("--only-correctness")
    if args.only_performance:
        test_argv.append("--only-performance")
    if args.iters is not None:
        test_argv.extend(["--iters", str(args.iters)])
    if args.fail_fast:
        test_argv.append("--fail-fast")
    if args.op_filter is not None:
        test_argv.extend(["--op-filter", args.op_filter])
    if args.format is not None:
        test_argv.extend(["--format", args.format])
    if args.output is not None:
        test_argv.extend(["-o", args.output])

    print(f"\n加载并执行: {args.test_file}")
    try:
        spec = importlib.util.spec_from_file_location("generated_test", args.test_file)
        if spec is None or spec.loader is None:
            print(f"错误: 无法加载文件: {args.test_file}")
            return 1
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_test"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "main"):
            print("错误: 测试文件没有 main() 函数")
            return 1

        returncode = module.main(test_argv)
        return returncode if isinstance(returncode, int) else 0
    except Exception as e:
        print(f"错误: 执行测试文件失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
    analyze_parser.add_argument("--hierarchy", action="store_true", help="启用层级分析")
    analyze_parser.add_argument("--self-time-top", type=int, default=10, help="自耗时排名数量")
    analyze_parser.add_argument("--call-chains", action="store_true", help="输出高频调用链")
    analyze_parser.add_argument("--detect-recursion", action="store_true", help="检测递归模式")
    analyze_parser.add_argument("--export-tree", type=str, default=None, help="导出调用树到文件")
    analyze_parser.add_argument("--max-chain-depth", type=int, default=10, help="调用链最大深度")
    analyze_parser.add_argument("--min-chain-occurrence", type=int, default=2, help="调用链最小出现次数")

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
    test_parser.add_argument("--test-all-ops", action="store_true",
                            help="测试所有算子（默认仅测试根节点）")
    test_parser.add_argument("--update-whitelist", action="store_true", help="动态发现后更新白名单")
    test_parser.add_argument("--update-blacklist", action="store_true", help="将未映射算子加入黑名单")

    # === generate 子命令 ===
    generate_parser = subparsers.add_parser(
        "generate", help="从 Trace 生成测试用例文件",
        description="解析 PyTorch Profiler Trace，生成可执行的 Python 测试文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen generate profiler_trace.json -o tests/test_ops.py
  op_testgen generate profiler_trace.json -o tests/test_ops.py --backend cpu --max-ops 50
        """,
    )
    generate_parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    generate_parser.add_argument("-o", "--output", required=True, help="输出 .py 文件路径")
    generate_parser.add_argument("--backend", choices=["cuda", "swdnn", "auto", "cpu"], default="cuda",
                                help="默认后端 (默认: cuda)")
    generate_parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    generate_parser.add_argument("--max-ops", type=int, default=100, help="最大算子数 (默认: 100)")
    generate_parser.add_argument("--op-filter", help="算子名称过滤 (支持通配符)")
    generate_parser.add_argument("--iters", type=int, default=10, help="性能测试迭代次数 (默认: 10)")
    generate_parser.add_argument("--test-all-ops", action="store_true",
                                help="生成所有算子的测试（默认仅生成根节点）")
    generate_parser.add_argument("--update-blacklist", action="store_true", help="将未映射算子加入黑名单")
    generate_parser.add_argument("--format", choices=["markdown", "html", "json", "excel", "all"], default="markdown",
                                help="报告格式 (默认: markdown)")
    generate_parser.add_argument("--report-output", default="op_testgen_report.md",
                                help="报告输出路径 (默认: op_testgen_report.md)")

    # === run 子命令 ===
    run_parser = subparsers.add_parser(
        "run", help="执行生成的测试用例文件",
        description="运行由 generate 子命令生成的 Python 测试文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen run tests/test_ops.py
  op_testgen run tests/test_ops.py --backend cuda --only-correctness
        """,
    )
    run_parser.add_argument("test_file", help="生成的 .py 测试文件路径")
    run_parser.add_argument("--backend", choices=["cuda", "swdnn", "cpu"],
                           help="后端 (覆盖文件默认值)")
    run_parser.add_argument("--only-correctness", action="store_true", help="仅正确性测试")
    run_parser.add_argument("--only-performance", action="store_true", help="仅性能测试")
    run_parser.add_argument("--iters", type=int, help="性能测试迭代次数")
    run_parser.add_argument("--fail-fast", action="store_true", help="第一个失败即停止")
    run_parser.add_argument("--op-filter", help="仅测试匹配名称的算子 (支持通配符)")
    run_parser.add_argument("--format", choices=["markdown", "html", "json", "excel", "all"],
                           help="报告格式 (覆盖文件默认值)")
    run_parser.add_argument("-o", "--output", default="op_testgen_report.md",
                           help="报告输出路径 (默认: op_testgen_report.md)")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
