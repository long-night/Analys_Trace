# op_testgen 按需张量构建设计文档

> **日期**: 2026-05-26
> **状态**: 待审阅
> **主题**: 将张量创建从全局一次性构建改为每个测试用例按需构建

---

## 1. 背景与目标

### 1.1 背景

当前 `op_testgen` 的 `test` 和 `generate` 子命令在 `_prepare_test_cases()` 阶段会一次性为**所有**待测算子构建输入张量（`TensorBuilder.build()`），生成 `List[TestCase]`。当 trace 中算子数量多（默认 `max-ops=100`，但 `--test-all-ops` 可覆盖）或单个张量尺寸较大时，峰值内存占用与算子总数成正比，容易导致 OOM。

### 1.2 目标

重构测试流程，实现**按需张量创建**：
- 每个测试用例独立创建张量 → 执行正确性/性能测试 → 释放张量
- 峰值内存仅取决于**单个测试用例的最大张量占用**，而非全部之和
- 同时应用于 `test`（一步到位）和 `generate`（生成 .py 文件）两种模式

### 1.3 非目标

- 不改动 `TensorBuilder` 的内部构建逻辑（`build()` 接口保持不变）
- 不改动 `CorrectnessRunner` / `PerfBenchmark` 的测试执行接口
- 不改动 `TraceParser` / `OpMapper` 的解析映射逻辑

---

## 2. 当前问题分析

### 2.1 问题代码路径

```
cli.py _prepare_test_cases()
  ├── parser.parse_hierarchical()     # List[OpInfo]
  ├── mapper.map_all()                # List[MappedOp]
  ├── 去重 / 过滤 / max-ops 限制
  ├── builder = TensorBuilder(seed)
  └── [问题] test_cases = [builder.build(m) for m in mapped_ops]  # 一次性构建所有张量

cli.py cmd_test()
  └── for tc in test_cases:
        correctness_runner.run(tc)    # 使用已构建的张量
        perf_benchmark.run(tc)        # 使用已构建的张量

cli.py cmd_generate()
  └── generator.generate(mapped_ops)  # 本应只序列化，但 _prepare_test_cases() 已浪费内存构建张量
```

### 2.2 内存占用估算

假设 100 个算子，每个输入张量平均 100MB（如大矩阵 `[1024, 10240]` float32）：
- **当前**：峰值内存 ≈ 100 × 100MB = **10GB+**（同时持有所有张量）
- **目标**：峰值内存 ≈ **100MB**（仅当前用例的张量）

### 2.3 附带问题：generate 模式的无谓构建

`cmd_generate()` 当前调用 `_prepare_test_cases()`，虽然只用到返回的 `mapped_ops`，但函数内部已经构建了所有 `TestCase` 张量，这些张量生成后立即被 GC，纯属浪费。

### 2.4 附带问题：inplace 算子的 clone 缺失

`correctness/test_runner.py` 中 CUDA/SWDNN 分支：
```python
target_positional.append(arg.cuda())  # 缺少 clone()
```
对于 inplace 算子（如 `aten::add_`），这会直接修改原始张量，导致后续性能测试的输入数据被破坏。CPU baseline 分支已有 `t.clone().cpu()`，但 CUDA 分支遗漏。

---

## 3. 架构方案

采用 **方案 A：循环内按需构建**。

### 3.1 核心思路

```
当前: _prepare_test_cases() → List[TestCase] → test loop
新设计: _prepare_mapped_ops() → List[MappedOp] → test loop (逐个 build + test + del)
```

### 3.2 数据流对比

**当前流程：**
```
Trace JSON
    │
    ▼
┌──────────────┐
│ TraceParser  │ ──► List[OpInfo]
└──────────────┘
    │
    ▼
┌──────────────┐
│   OpMapper   │ ──► List[MappedOp]
└──────────────┘
    │
    ▼
┌──────────────┐
│ TensorBuilder│ ──► [内存峰值] List[TestCase]（所有张量同时存在）
└──────────────┘
    │
    ▼
┌──────────────────┐
│ Correctness/Perf │ ──► 测试执行
└──────────────────┘
```

**新流程：**
```
Trace JSON
    │
    ▼
┌──────────────┐
│ TraceParser  │ ──► List[OpInfo]
└──────────────┘
    │
    ▼
┌──────────────┐
│   OpMapper   │ ──► List[MappedOp]
└──────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ 测试主循环                            │
│  for mapped_op in mapped_ops:        │
│      test_case = builder.build()     │ ──► [内存峰值] 仅单个用例
│      correctness_runner.run(tc)      │
│      perf_benchmark.run(tc)          │
│      del test_case                   │ ──► 立即释放
└──────────────────────────────────────┘
```

---

## 4. 详细设计

### 4.1 cli.py — 公共准备逻辑重构

#### 4.1.1 新增 `_prepare_mapped_ops()`

提取 `test` 和 `generate` 的公共逻辑，**不构建张量**：

```python
def _prepare_mapped_ops(args) -> tuple[List[MappedOp], OpMapper]:
    """解析、映射、去重、过滤，返回 MappedOp 列表（不构建张量）

    Args:
        args: 命令行参数对象

    Returns:
        (unique_mapped_ops, mapper)
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

    # 去重
    seen_keys = set()
    unique_mapped_ops = []
    for m in mapped_ops:
        info = m.op_info
        def _to_tuple(x):
            if isinstance(x, list):
                return tuple(_to_tuple(i) for i in x)
            return x
        key = (
            info.hierarchical_name if isinstance(info, HierarchicalOpInfo) else info.name,
            _to_tuple(info.input_dims),
            _to_tuple(info.input_strides),
            tuple(info.input_types),
            _to_tuple(info.concrete_inputs),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            unique_mapped_ops.append(m)

    # 过滤
    if hasattr(args, "op_filter") and args.op_filter:
        import fnmatch
        unique_mapped_ops = [
            m for m in unique_mapped_ops if fnmatch.fnmatch(m.op_info.name, args.op_filter)
        ]

    # max-ops 限制
    if hasattr(args, "max_ops") and args.max_ops > 0:
        from collections import defaultdict
        op_total_duration: dict[str, float] = defaultdict(float)
        for m in unique_mapped_ops:
            op_total_duration[m.op_info.name] += m.op_info.duration_us
        selected = set(sorted(
            op_total_duration.keys(),
            key=lambda n: op_total_duration[n],
            reverse=True
        )[:args.max_ops])
        unique_mapped_ops = [m for m in unique_mapped_ops if m.op_info.name in selected]

    unique_mapped_ops.sort(key=lambda m: (m.op_info.name, -m.op_info.duration_us))
    return unique_mapped_ops, mapper
```

#### 4.1.2 移除旧 `_prepare_test_cases()`

旧函数中 `builder.build()` 的批量构建逻辑迁移到 `cmd_test()` 和生成的 `.py` 文件的 `main()` 中。

### 4.2 cli.py — test 子命令主循环

```python
def cmd_test(args) -> int:
    print("=" * 60)
    print("PyTorch Profiler 自动化测试生成器")
    print("=" * 60)

    print("\n[1/4] 解析并准备测试用例...")
    mapped_ops, mapper = _prepare_mapped_ops(args)

    if not mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(mapped_ops)} 个算子")

    # 确定后端
    backends = []
    if args.backend == "auto":
        backends = ["cuda"]
        if os.environ.get("SWDNN_AVAILABLE"):
            backends.append("swdnn")
    else:
        backends = [args.backend]

    builder = TensorBuilder(seed=args.seed)
    all_correctness = []
    all_perf = []

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"后端: {backend.upper()}")
        print(f"{'='*60}")

        correctness_runner = CorrectnessRunner(fail_fast=args.fail_fast)
        perf_benchmark = PerfBenchmark(benchmark_iters=args.iters)

        if not args.only_performance:
            print(f"\n[2/4] 执行正确性测试...")
            from itertools import groupby
            for op_name, group in groupby(mapped_ops, key=lambda m: m.op_info.name):
                group_list = list(group)
                print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
                for mapped_op in group_list:
                    test_case = builder.build(mapped_op)
                    try:
                        result = correctness_runner.run(test_case, backend=backend)
                        all_correctness.append(result)
                        status = "通过" if result.passed else "失败"
                        if result.error_message:
                            print(f"    [{status}] {result.op_name}: {result.error_message}")
                        else:
                            print(f"    [{status}] {result.op_name}: max_abs={result.max_abs_err:.2e}, max_rel={result.max_rel_err:.2e}")
                        if result.input_info:
                            print(f"      input: {result.input_info}")
                    finally:
                        del test_case
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            passed = sum(1 for r in all_correctness if r.passed)
            total = len([r for r in all_correctness if r.backend == backend])
            print(f"\n  通过: {passed}/{total}")

        if not args.only_correctness:
            print(f"\n[3/4] 执行性能测试...")
            from itertools import groupby
            for op_name, group in groupby(mapped_ops, key=lambda m: m.op_info.name):
                group_list = list(group)
                print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
                for mapped_op in group_list:
                    test_case = builder.build(mapped_op)
                    try:
                        result = perf_benchmark.run(test_case, backend=backend)
                        all_perf.append(result)
                    finally:
                        del test_case
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            backend_perf = [r for r in all_perf if r.backend == backend]
            if backend_perf:
                avg_speedup = sum(r.speedup for r in backend_perf) / len(backend_perf)
                print(f"\n  平均加速比: {avg_speedup:.2f}x")

    # [4/4] 报告生成（与现有逻辑相同）
    ...
```

**关键变化点：**
1. 循环对象从 `List[TestCase]` 变为 `List[MappedOp]`
2. 每个 `MappedOp` 在循环内通过 `builder.build()` 创建 `TestCase`
3. `try/finally` 确保即使测试异常也能释放张量
4. `torch.cuda.empty_cache()` 可选触发 CUDA 缓存回收

### 4.3 cli.py — generate 子命令

```python
def cmd_generate(args) -> int:
    print("=" * 60)
    print("PyTorch Profiler 测试用例生成器")
    print("=" * 60)

    print("\n[1/2] 解析并准备测试用例...")
    mapped_ops, mapper = _prepare_mapped_ops(args)

    if not mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(mapped_ops)} 个算子")

    print("\n[2/2] 生成测试文件...")
    generator = TestCaseGenerator()
    output_path = generator.generate(
        mapped_ops=mapped_ops,
        output_path=args.output,
        source_trace=args.trace_file,
        backend=args.backend,
        seed=args.seed,
        iters=args.iters,
        format=getattr(args, "format", "markdown"),
        output=getattr(args, "report_output", "op_testgen_report.md"),
    )
    print(f"  测试文件: {output_path}")
    ...
```

**关键变化：**
- 调用 `_prepare_mapped_ops()` 替代 `_prepare_test_cases()`
- 不再无谓地构建张量

### 4.4 template.py — 生成的 .py 文件模板

生成的 `.py` 文件需要同步改为按需构建。关键改动：

```python
# 当前 build_test_cases() 一次性构建所有张量 —— 改为按需构建

def main(argv=None) -> int:
    ...
    
    # 不再调用 build_test_cases() 批量构建
    # 改为在循环内逐个构建
    
    builder = TensorBuilder(seed=args.seed)
    all_correctness = []
    all_perf = []

    if not args.only_performance:
        print("\n执行正确性测试...")
        runner = CorrectnessRunner(fail_fast=args.fail_fast)
        
        for data in TEST_CASES_DATA:
            mapped_op = _build_mapped_op(data)  # 辅助函数：从数据构造 _MappedOp
            test_case = builder.build(mapped_op)
            try:
                result = runner.run(test_case, backend=args.backend)
                all_correctness.append(result)
                ...
            finally:
                del test_case
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not args.only_correctness:
        print("\n执行性能测试...")
        benchmark = PerfBenchmark(benchmark_iters=args.iters)
        
        for data in TEST_CASES_DATA:
            mapped_op = _build_mapped_op(data)
            test_case = builder.build(mapped_op)
            try:
                result = benchmark.run(test_case, backend=args.backend)
                all_perf.append(result)
                ...
            finally:
                del test_case
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    ...
```

### 4.5 correctness/test_runner.py — 修复 inplace 算子 clone 缺失

在 `_run_single()` 的 CUDA/SWDNN 分支中，对输入张量统一 `clone()`：

```python
# 修改前（有 bug）
for arg in test_case.positional_args:
    if isinstance(arg, torch.Tensor):
        if device_str == "cuda":
            target_positional.append(arg.cuda())  # ← 直接移动，未 clone
        else:
            target_positional.append(arg.clone().cpu())

# 修改后（修复）
for arg in test_case.positional_args:
    if isinstance(arg, torch.Tensor):
        t = arg.clone()  # 统一 clone，保护 inplace 算子
        if device_str == "cuda":
            target_positional.append(t.cuda())
        else:
            target_positional.append(t.cpu())
    elif isinstance(arg, list):
        moved = []
        for t in arg:
            if isinstance(t, torch.Tensor):
                t_clone = t.clone()  # 列表内张量同样 clone
                moved.append(t_clone.cuda() if device_str == "cuda" else t_clone.cpu())
            else:
                moved.append(t)
        target_positional.append(moved)
    else:
        target_positional.append(arg)
```

---

## 5. 内存释放策略

| 层级 | 释放操作 | 触发时机 |
|------|----------|----------|
| Python 对象引用 | `del test_case` | 每个用例 `finally` 块 |
| PyTorch 张量内存 | Python GC 回收 | `del` 后引用计数归零 |
| CUDA 显存缓存 | `torch.cuda.empty_cache()` | `finally` 块中（可选） |
| 驱动层显存 | CUDA 驱动回收 | `empty_cache()` 触发 |

**`empty_cache()` 调用策略：**
- 默认每个用例后调用，确保显存及时回收到缓存池
- 纯 CPU 测试（`--backend cpu`）可跳过，但调用也无害
- 不引入额外 CLI 选项控制，保持简单

---

## 6. 影响范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `cli.py` | **重构** | `_prepare_test_cases()` → `_prepare_mapped_ops()`；`cmd_test()` 循环结构改动；`cmd_generate()` 调用更新 |
| `generator/template.py` | **重构** | 生成文件的 `main()` 改为按需构建；移除 `build_test_cases()` 的批量构建 |
| `correctness/test_runner.py` | **Bugfix** | CUDA/SWDNN 分支补充 `clone()`，修复 inplace 算子隐患 |
| `builder/tensor_builder.py` | 无变更 | `build()` 接口不变 |
| `perf/benchmark.py` | 无变更 | `run(tc)` 接口不变 |
| `generator/test_case_generator.py` | 无变更 | 序列化逻辑不变 |
| `generator/test_case_runner.py` | 无变更 | `InProcessRunner` 逻辑不变 |

---

## 7. 边界情况处理

### 7.1 异常安全

每个用例使用 `try/finally` 包裹：
```python
test_case = builder.build(mapped_op)
try:
    result = runner.run(test_case, backend)
finally:
    del test_case
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```
确保即使 `run()` 抛出异常，张量也能被释放。

### 7.2 正确性测试与性能测试的数据隔离

由于正确性测试内部已做 `clone()`（CPU baseline 和 CUDA/SWDNN 分支均 clone），性能测试使用的原始张量不会被修改。

### 7.3 同一算子多变体的顺序

`groupby` 在 `MappedOp` 列表上执行，按算子名分组。每个变体独立构建/测试/释放，互不干扰。

---

## 8. 验收标准

- [ ] `test` 子命令执行时，不再出现与算子总数成正比的内存峰值
- [ ] `generate` 子命令不再无谓构建张量（生成阶段内存占用显著降低）
- [ ] 生成的 `.py` 文件执行时同样采用按需构建模式
- [ ] inplace 算子（如 `aten::add_`）的正确性测试不再破坏性能测试的输入数据
- [ ] 测试通过率与重构前保持一致（功能无回归）
- [ ] `--only-correctness` / `--only-performance` 选项正常工作
- [ ] `--backend cpu` / `cuda` / `swdnn` 均正常工作
- [ ] `pytest tests/` 全部通过

---

*文档版本: v1.0*
*作者: Sisyphus*
