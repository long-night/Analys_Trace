"""命令行入口"""
import argparse
import json
import os
import sys
from typing import Optional

import torch

from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.config import get_settings
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.parser.trace_parser import TraceParser
from op_testgen.perf.benchmark import PerfBenchmark
from op_testgen.reporter.html_reporter import HTMLReporter
from op_testgen.reporter.excel_reporter import ExcelReporter


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler 自动化测试生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen profiler_trace.json
  op_testgen profiler_trace.json --backend swdnn --output report.html
  op_testgen profiler_trace.json --only-correctness --fail-fast
        """,
    )
    parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    parser.add_argument("--backend", choices=["cuda", "swdnn", "auto", "cpu"], default="cuda",
                       help="对比后端 (默认: cuda)")
    parser.add_argument("-o", "--output", default="op_testgen_report.html", help="输出文件路径")
    parser.add_argument("--format", choices=["html", "json", "excel", "all"], default="html",
                       help="输出格式 (默认: html)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--iters", type=int, default=10, help="性能测试迭代次数 (默认: 10)")
    parser.add_argument("--fail-fast", action="store_true", help="第一个失败即停止")
    parser.add_argument("--only-correctness", action="store_true", help="仅执行正确性测试")
    parser.add_argument("--only-performance", action="store_true", help="仅执行性能测试")
    parser.add_argument("--op-filter", help="仅测试匹配名称的算子 (支持通配符)")
    parser.add_argument("--update-whitelist", action="store_true", help="动态发现后更新白名单")
    parser.add_argument("--max-ops", type=int, default=100, help="最大测试算子数（去重后，默认: 100）")

    args = parser.parse_args(argv)

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

    # 3. 去重 + 构建测试用例（流式处理避免OOM）
    print("\n[3/6] 去重并构建测试用例...")

    # 3a. 按 (name, dims, strides, types) 去重
    seen_keys = set()
    unique_mapped_ops = []
    for m in mapped_ops:
        info = m.op_info
        # 将 dims 和 strides 转为不可变元组以便哈希
        dims_tuple = tuple(tuple(d) for d in info.input_dims)
        strides_tuple = tuple(tuple(s) if s else () for s in info.input_strides)
        types_tuple = tuple(info.input_types)
        key = (info.name, dims_tuple, strides_tuple, types_tuple)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_mapped_ops.append(m)

    print(f"  去重前: {len(mapped_ops)} 个, 去重后: {len(unique_mapped_ops)} 个")

    # 3b. 应用名称过滤
    if args.op_filter:
        import fnmatch
        unique_mapped_ops = [m for m in unique_mapped_ops if fnmatch.fnmatch(m.op_info.name, args.op_filter)]
        print(f"  名称过滤后剩余: {len(unique_mapped_ops)} 个")

    # 3c. 限制最大算子数
    if args.max_ops > 0 and len(unique_mapped_ops) > args.max_ops:
        print(f"  超过 --max-ops={args.max_ops}，截断至前 {args.max_ops} 个")
        unique_mapped_ops = unique_mapped_ops[:args.max_ops]

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    # 3d. 流式构建测试用例（避免一次性全量驻留内存）
    builder = TensorBuilder(seed=args.seed)
    test_cases = [builder.build(m) for m in unique_mapped_ops]
    print(f"  构建 {len(test_cases)} 个测试用例")

    # 确定后端列表
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

        # 4. 正确性测试
        if not args.only_performance:
            print(f"\n[4/6] 执行正确性测试...")
            correctness_runner = CorrectnessRunner(fail_fast=args.fail_fast)
            correctness_results = correctness_runner.run_all(test_cases, backend=backend)
            all_correctness.extend(correctness_results)

            passed = sum(1 for r in correctness_results if r.passed)
            print(f"  通过: {passed}/{len(correctness_results)}")

        # 5. 性能测试
        if not args.only_correctness:
            print(f"\n[5/6] 执行性能测试...")
            perf_benchmark = PerfBenchmark(benchmark_iters=args.iters)
            perf_results = perf_benchmark.run_all(test_cases, backend=backend)
            all_perf.extend(perf_results)

            avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results) if perf_results else 1.0
            print(f"  平均加速比: {avg_speedup:.2f}x")

    # 6. 生成报告
    print(f"\n[6/6] 生成报告...")

    if args.format in ("html", "all"):
        html_reporter = HTMLReporter()
        html_path = args.output if args.output.endswith(".html") else args.output + ".html"
        html_reporter.generate(all_correctness, all_perf, html_path)
        print(f"  HTML 报告: {html_path}")

    if args.format in ("json", "all"):
        json_path = args.output.replace(".html", ".json").replace(".xlsx", ".json")
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
            excel_path = args.output.replace(".html", ".xlsx").replace(".json", ".xlsx")
            if not excel_path.endswith(".xlsx"):
                excel_path += ".xlsx"
            excel_reporter.generate(all_correctness, all_perf, excel_path)
            print(f"  Excel 报告: {excel_path}")
        except ImportError as e:
            print(f"  警告: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
