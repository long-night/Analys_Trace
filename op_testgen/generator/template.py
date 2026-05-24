"""测试文件模板"""

TEST_FILE_TEMPLATE = '''#!/usr/bin/env python3
# Auto-generated test cases from trace: ${source_trace}
# Generated at: ${generated_at}
# Default options: backend=${backend}, seed=${seed}, iters=${iters}

import argparse
import importlib
import sys
from typing import Any, Dict, List

import torch
from op_testgen.builder.tensor_builder import TensorBuilder, TestCase
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.mapper.op_mapper import MappedOp
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.perf.benchmark import PerfBenchmark

TEST_CASES_DATA = ${test_cases_data_repr}


def build_test_cases(seed: int = ${seed}) -> List[TestCase]:
    """从序列化数据重建 TestCase 列表"""
    torch.manual_seed(seed)
    builder = TensorBuilder(seed=seed)
    test_cases = []

    for data in TEST_CASES_DATA:
        op_info = OpInfo(
            name=data["op_name"],
            input_dims=data["input_dims"],
            input_strides=data["input_strides"],
            input_types=data["input_types"],
            concrete_inputs=data["concrete_inputs"],
        )

        # 解析 callable
        parts = data["callable_path"].split(".")
        try:
            module = importlib.import_module(".".join(parts[:-1]))
            callable_obj = getattr(module, parts[-1])
        except (ImportError, AttributeError) as e:
            print(f"警告: 无法导入 {data['callable_path']}: {e}")
            continue

        mapped_op = MappedOp(
            op_info=op_info,
            callable=callable_obj,
            callable_path=data["callable_path"],
            namespace=data["callable_path"].split(".")[0],
            support_status="generated",
        )

        try:
            test_case = builder.build(mapped_op)
            test_cases.append(test_case)
        except Exception as e:
            print(f"警告: 构建 {data['op_name']} 失败: {e}")
            continue

    return test_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-generated operator tests")
    parser.add_argument("--backend", default="${backend}", choices=["cuda", "swdnn", "cpu"],
                        help="Backend to test against")
    parser.add_argument("--only-correctness", action="store_true",
                        help="Only run correctness tests")
    parser.add_argument("--only-performance", action="store_true",
                        help="Only run performance tests")
    parser.add_argument("--iters", type=int, default=${iters},
                        help="Performance benchmark iterations")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first failure")
    parser.add_argument("--seed", type=int, default=${seed},
                        help="Random seed for tensor generation")
    args = parser.parse_args()

    print("=" * 60)
    print("Auto-generated Operator Tests")
    print("=" * 60)

    test_cases = build_test_cases(seed=args.seed)
    print(f"\\n构建 {len(test_cases)} 个测试用例")

    if not test_cases:
        print("错误: 没有可测试的算子")
        return 1

    all_correctness = []
    all_perf = []

    if not args.only_performance:
        print("\\n执行正确性测试...")
        runner = CorrectnessRunner(fail_fast=args.fail_fast)
        for tc in test_cases:
            result = runner.run(tc, backend=args.backend)
            all_correctness.append(result)
            status = "通过" if result.passed else "失败"
            if result.error_message:
                print(f"  [{status}] {result.op_name}: {result.error_message}")
            else:
                print(f"  [{status}] {result.op_name}: max_abs={result.max_abs_err:.2e}, max_rel={result.max_rel_err:.2e}")
        passed = sum(1 for r in all_correctness if r.passed)
        print(f"  通过: {passed}/{len(all_correctness)}")

    if not args.only_correctness:
        print("\\n执行性能测试...")
        benchmark = PerfBenchmark(benchmark_iters=args.iters)
        perf_results = benchmark.run_all(test_cases, backend=args.backend)
        all_perf.extend(perf_results)
        if perf_results:
            avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results)
            print(f"  平均加速比: {avg_speedup:.2f}x")

    print("\\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 返回非零退出码如果有失败
    failed = sum(1 for r in all_correctness if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
'''
