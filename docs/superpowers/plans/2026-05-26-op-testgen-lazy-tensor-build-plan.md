# op_testgen 按需张量构建 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 op_testgen 的 test 和 generate 模式从全局一次性张量构建改为每个测试用例按需构建、测试、释放

**Architecture:** 拆分 `_prepare_test_cases()` 为纯解析的 `_prepare_mapped_ops()`，在测试循环内逐个 `builder.build()` → 正确性测试 → 性能测试 → `del test_case` + `torch.cuda.empty_cache()`。同步修复 CUDA 分支 clone 缺失。

**Tech Stack:** Python 3.12, PyTorch, itertools.groupby

---

## 文件变更清单

| 文件 | 变更类型 | 职责 |
|------|----------|------|
| `op_testgen/cli.py` | 重构 | `_prepare_test_cases()` → `_prepare_mapped_ops()`；`cmd_test()` 和 `cmd_generate()` 循环按需构建 |
| `op_testgen/generator/template.py` | 重构 | 生成文件的 `main()` 改为按需构建，移除 `build_test_cases()` 批量构建 |
| `op_testgen/correctness/test_runner.py` | Bugfix | CUDA/SWDNN 分支补充 `clone()` |

---

## Task 1: 修复 correctness/test_runner.py CUDA 分支 clone 缺失

**Files:**
- Modify: `op_testgen/correctness/test_runner.py:173-193`

- [ ] **Step 1: 定位 CUDA/SWDNN 分支的输入处理代码**

  当前有 bug 的代码段（约 173-193 行）：
  ```python
  for arg in test_case.positional_args:
      if isinstance(arg, torch.Tensor):
          if device_str == "cuda":
              target_positional.append(arg.cuda())  # ← 缺少 clone()
          else:
              target_positional.append(arg.clone().cpu())
      elif isinstance(arg, list):
          moved = []
          for t in arg:
              if isinstance(t, torch.Tensor):
                  if device_str == "cuda":
                      moved.append(t.cuda())  # ← 缺少 clone()
                  else:
                      moved.append(t.clone().cpu())
              else:
                  moved.append(t)
          target_positional.append(moved)
      else:
          target_positional.append(arg)
  ```

- [ ] **Step 2: 应用 clone() 修复**

  替换为：
  ```python
  for arg in test_case.positional_args:
      if isinstance(arg, torch.Tensor):
          t = arg.clone()
          if device_str == "cuda":
              target_positional.append(t.cuda())
          else:
              target_positional.append(t.cpu())
      elif isinstance(arg, list):
          moved = []
          for t in arg:
              if isinstance(t, torch.Tensor):
                  t_clone = t.clone()
                  moved.append(t_clone.cuda() if device_str == "cuda" else t_clone.cpu())
              else:
                  moved.append(t)
          target_positional.append(moved)
      else:
          target_positional.append(arg)
  ```

- [ ] **Step 3: 验证修改**

  Run: `python -c "import op_testgen.correctness.test_runner"`
  Expected: 无 import 错误

- [ ] **Step 4: Commit**

  ```bash
  git add op_testgen/correctness/test_runner.py
  git commit -m "fix(correctness): clone input tensors in CUDA/SWDNN branch

  Prevents inplace operators (e.g., aten::add_) from mutating
  original tensors used by subsequent performance tests."
  ```

---

## Task 2: 重构 cli.py — 添加 _prepare_mapped_ops()

**Files:**
- Modify: `op_testgen/cli.py:91-158`

- [ ] **Step 1: 将 `_prepare_test_cases()` 重命名为 `_prepare_mapped_ops()`**

  修改函数签名和文档：
  ```python
  def _prepare_mapped_ops(args) -> tuple[List[MappedOp], OpMapper]:
      """解析、映射、去重、过滤，返回 MappedOp 列表（不构建张量）
      
      Args:
          args: 命令行参数对象
      
      Returns:
          (unique_mapped_ops, mapper)
      """
  ```

- [ ] **Step 2: 移除 TensorBuilder 和张量构建逻辑**

  删除原函数末尾的：
  ```python
  builder = TensorBuilder(seed=args.seed)
  test_cases = [builder.build(m) for m in unique_mapped_ops]
  
  return unique_mapped_ops, test_cases, mapper
  ```
  
  改为：
  ```python
  return unique_mapped_ops, mapper
  ```

- [ ] **Step 3: 移除 cli.py 顶部的 TensorBuilder 导入（如不再需要）**

  检查 `_prepare_mapped_ops()` 是否仍引用 `TensorBuilder`——移除后应该不再需要。在 `cli.py` 顶部：
  ```python
  from op_testgen.builder.tensor_builder import TensorBuilder
  ```
  这个导入保留，因为 `cmd_test()` 中仍需使用。

- [ ] **Step 4: Commit**

  ```bash
  git add op_testgen/cli.py
  git commit -m "refactor(cli): split _prepare_test_cases into _prepare_mapped_ops

  Extracts only parsing/mapping/deduplication. Removes eager tensor
  construction so memory scales with single test case, not total count."
  ```

---

## Task 3: 重构 cli.py — cmd_test() 改为按需构建

**Files:**
- Modify: `op_testgen/cli.py:160-302`

- [ ] **Step 1: 更新函数开头的准备调用**

  将：
  ```python
  unique_mapped_ops, test_cases, mapper = _prepare_test_cases(args)
  ```
  改为：
  ```python
  mapped_ops, mapper = _prepare_mapped_ops(args)
  ```

- [ ] **Step 2: 更新空值检查**

  将：
  ```python
  if not unique_mapped_ops:
  ```
  改为：
  ```python
  if not mapped_ops:
  ```

- [ ] **Step 3: 重构正确性测试循环**

  当前：
  ```python
  for op_name, group in groupby(test_cases, key=lambda tc: tc.mapped_op.op_info.name):
      group_list = list(group)
      print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
      for tc in group_list:
          result = correctness_runner.run(tc, backend=backend)
          all_correctness.append(result)
          ...
  ```
  
  改为：
  ```python
  builder = TensorBuilder(seed=args.seed)
  
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
                  print(f"    [{status}] {result.op_name}: max_abs={result.max_abs_err:.2e}, max_rel={result.max_rel_err:.2e}, avg_abs={result.avg_abs_err:.2e}, avg_rel={result.avg_rel_err:.2e}")
              if result.input_info:
                  print(f"      input: {result.input_info}")
          finally:
              del test_case
              if torch.cuda.is_available():
                  torch.cuda.empty_cache()
  ```

- [ ] **Step 4: 重构性能测试循环**

  当前：
  ```python
  for op_name, group in groupby(test_cases, key=lambda tc: tc.mapped_op.op_info.name):
      group_list = list(group)
      print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
      perf_results = perf_benchmark.run_all(group_list, backend=backend)
      all_perf.extend(perf_results)
  ```
  
  改为：
  ```python
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
  ```
  
  注意：这里不再使用 `perf_benchmark.run_all()`（批量接口），改为逐个调用 `perf_benchmark.run()`。

- [ ] **Step 5: 更新统计信息**

  正确性统计：
  ```python
  passed = sum(1 for r in all_correctness if r.passed)
  total = len([r for r in all_correctness if r.backend == backend])
  print(f"\n  通过: {passed}/{total}")
  ```

  性能统计：
  ```python
  backend_perf = [r for r in all_perf if r.backend == backend]
  if backend_perf:
      avg_speedup = sum(r.speedup for r in backend_perf) / len(backend_perf)
      print(f"\n  平均加速比: {avg_speedup:.2f}x")
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add op_testgen/cli.py
  git commit -m "refactor(cli): make test command build tensors lazily per-case

  Peak memory now proportional to single test case instead of total count.
  Uses try/finally + torch.cuda.empty_cache() for deterministic cleanup."
  ```

---

## Task 4: 重构 cli.py — cmd_generate() 使用 _prepare_mapped_ops()

**Files:**
- Modify: `op_testgen/cli.py:305-355`

- [ ] **Step 1: 更新调用**

  将：
  ```python
  unique_mapped_ops, _, mapper = _prepare_test_cases(args)
  ```
  改为：
  ```python
  mapped_ops, mapper = _prepare_mapped_ops(args)
  ```

- [ ] **Step 2: 更新后续引用**

  将 `unique_mapped_ops` 的引用改为 `mapped_ops`（`generator.generate()` 调用参数）。

- [ ] **Step 3: Commit**

  ```bash
  git add op_testgen/cli.py
  git commit -m "refactor(cli): make generate command skip eager tensor build

  Uses _prepare_mapped_ops() which no longer constructs tensors.
  Eliminates wasteful memory allocation during file generation."
  ```

---

## Task 5: 重构 template.py — 生成文件改为按需构建

**Files:**
- Modify: `op_testgen/generator/template.py`

- [ ] **Step 1: 移除 build_test_cases() 函数中的批量构建**

  当前 `build_test_cases()` 函数（约 60-96 行）：
  ```python
  def build_test_cases(seed: int = ${seed}) -> List[TestCase]:
      torch.manual_seed(seed)
      builder = TensorBuilder(seed=seed)
      test_cases = []
      
      for data in TEST_CASES_DATA:
          op_info = _OpInfo(...)
          ...
          mapped_op = _MappedOp(...)
          test_case = builder.build(mapped_op)
          test_cases.append(test_case)
      
      return test_cases
  ```
  
  改为**辅助函数 `_build_mapped_op()`**：
  ```python
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
  ```

- [ ] **Step 2: 重构 main() 中的正确性测试循环**

  当前 `main()` 中的正确性测试：
  ```python
  test_cases = build_test_cases(seed=args.seed)
  ...
  for op_name, group in groupby(test_cases, key=lambda tc: tc.mapped_op.op_info.name):
      group_list = list(group)
      print(f"\n  算子: {op_name} ({len(group_list)} 个变体)")
      for tc in group_list:
          result = runner.run(tc, backend=args.backend)
          all_correctness.append(result)
          ...
  ```
  
  改为：
  ```python
  builder = TensorBuilder(seed=args.seed)
  ...
  for data in TEST_CASES_DATA:
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
  ```

- [ ] **Step 3: 重构 main() 中的性能测试循环**

  类似地，将性能测试改为：
  ```python
  for data in TEST_CASES_DATA:
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
  ```

- [ ] **Step 4: 移除 op_filter 逻辑中对 test_cases 的依赖**

  当前 `op_filter` 过滤在 `build_test_cases()` 之后：
  ```python
  if args.op_filter:
      test_cases = [tc for tc in test_cases if fnmatch.fnmatch(tc.mapped_op.op_info.name, args.op_filter)]
  ```
  
  改为在循环前过滤 `TEST_CASES_DATA`：
  ```python
  test_data = TEST_CASES_DATA
  if args.op_filter:
      test_data = [d for d in TEST_CASES_DATA if fnmatch.fnmatch(d["op_name"], args.op_filter)]
      print(f"\n应用过滤 '{args.op_filter}' 后: {len(test_data)} 个测试用例")
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add op_testgen/generator/template.py
  git commit -m "refactor(template): make generated .py files build tensors lazily

  Generated test scripts now construct tensors inside the test loop,
  matching the lazy behavior of the direct test command."
  ```

---

## Task 6: 运行测试验证

**Files:**
- Test: `tests/` 全部测试

- [ ] **Step 1: 运行 pytest**

  Run: `pytest tests/ -v`
  Expected: 全部通过（73 个测试用例）

- [ ] **Step 2: 运行端到端 smoke test**

  Run: `python -m op_testgen.cli test profiler_trace.json --only-correctness --backend cpu --max-ops 5`
  Expected: 正常执行，无内存异常，输出报告

- [ ] **Step 3: 运行 generate smoke test**

  Run:
  ```bash
  python -m op_testgen.cli generate profiler_trace.json -o /tmp/test_ops.py --max-ops 5
  python /tmp/test_ops.py --only-correctness --backend cpu
  ```
  Expected: 生成文件成功，执行成功

- [ ] **Step 4: Commit（如有测试修复）**

  ```bash
  git add -A
  git commit -m "test: verify lazy tensor build with existing test suite"
  ```

---

## 自审检查

- [x] **Spec 覆盖**: 所有 design.md 第 4 节的修改点均已对应到 Task 1-5
- [x] **Placeholder 检查**: 无 TBD/TODO/"similar to Task N"
- [x] **类型一致性**: `MappedOp` / `TestCase` / `TensorBuilder` 接口在 design 和 plan 中一致
- [x] **文件路径**: 所有路径均使用绝对路径或从 repo root 的相对路径
