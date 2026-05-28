# op_testgen 测试流程重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 op_testgen 测试执行流程，改为算子外层循环，复用 correctness CPU baseline 时间和结果，优化默认配置。

**Architecture:** 通过调整循环顺序（算子→backend）、在 CorrectnessResult 中缓存 cpu_out/cpu_time_ms、让 PerfBenchmark 接收复用的 CPU 时间，减少重复 build 和重复 CPU baseline 执行。

**Tech Stack:** Python 3.12, PyTorch, pytest

---

## 文件变更映射

| 文件 | 变更类型 | 职责 |
|------|---------|------|
| `op_testgen/config/settings.py` | 修改 | 调整默认性能测试配置（warmup=0, cpu_iters=1, target_iters=3） |
| `op_testgen/correctness/test_runner.py` | 修改 | CorrectnessRunner 增加 CPU 计时、cpu_out 缓存、_run_target_only |
| `op_testgen/perf/benchmark.py` | 修改 | PerfBenchmark 移除 warmup、支持 CPU time 复用、拆分 iters |
| `op_testgen/cli.py` | 修改 | 执行顺序重构（算子外层→backend 内层）、新增 CLI 参数 |
| `op_testgen/generator/template.py` | 修改 | 生成的 .py 文件同步新执行顺序和参数 |
| `tests/test_correctness.py` | 修改 | 更新正确性测试用例 |
| `tests/test_perf.py` | 修改 | 更新性能测试用例 |
| `tests/test_generator.py` | 修改 | 验证生成文件包含新参数 |

---

### Task 1: 调整 Settings 默认配置

**Files:**
- Modify: `op_testgen/config/settings.py`

- [ ] **Step 1: 修改 Settings 默认值**

```python
@dataclass
class Settings:
    # ...
    perf_warmup_iters: int = 0          # 从 3 改为 0
    perf_cpu_iters: int = 1             # 新增
    perf_target_iters: int = 3          # 新增（替代原来的 perf_benchmark_iters=10）
    # 保留 perf_benchmark_iters 以兼容旧代码，但默认值改为 3
    perf_benchmark_iters: int = 3       # 从 10 改为 3
```

- [ ] **Step 2: 验证配置读取**

Run: `python -c "from op_testgen.config import get_settings; s = get_settings(); print(s.perf_warmup_iters, s.perf_cpu_iters, s.perf_target_iters, s.perf_benchmark_iters)"`
Expected: `0 1 3 3`

- [ ] **Step 3: Commit**

```bash
git add op_testgen/config/settings.py
git commit -m "config(settings): adjust perf defaults (warmup=0, cpu_iters=1, target_iters=3)"
```

---

### Task 2: CorrectnessRunner 增加 CPU 计时和缓存

**Files:**
- Modify: `op_testgen/correctness/test_runner.py`

- [ ] **Step 1: CorrectnessResult 新增字段**

在 `CorrectnessResult` dataclass 中增加：

```python
@dataclass
class CorrectnessResult:
    # ... 现有字段 ...
    cpu_time_ms: float = 0.0
    cpu_out: Optional[torch.Tensor] = None
```

- [ ] **Step 2: _run_single 中 CPU baseline 计时**

在 `_run_single` 方法的 CPU baseline 执行处添加计时：

```python
# 1. CPU baseline (SWDNN=OFF)
os.environ["SWDNN"] = "OFF"
cpu_kwargs = {k: v for k, v in test_case.kwargs.items()}

def to_cpu(args):
    return [
        t.clone().cpu() if isinstance(t, torch.Tensor) else
        [x.clone().cpu() if isinstance(x, torch.Tensor) else x for x in t] if isinstance(t, list) else
        t
        for t in args
    ]

cpu_positional = to_cpu(test_case.positional_args)

import time
start = time.perf_counter()
cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
cpu_time_ms = (time.perf_counter() - start) * 1000
```

- [ ] **Step 3: _run_single 返回时带上新字段**

在 `_run_single` 所有返回路径中，增加 `cpu_time_ms` 和 `cpu_out`：

```python
# backend == "cpu" 的返回
return CorrectnessResult(
    op_name=op_name, passed=True, backend="cpu",
    input_info=input_info,
    error_message="CPU baseline only",
    cpu_time_ms=cpu_time_ms,
    cpu_out=cpu_out,
)

# 最终返回
return CorrectnessResult(
    op_name=op_name,
    passed=passed,
    max_abs_err=errors["max_abs"],
    max_rel_err=errors["max_rel"],
    avg_abs_err=errors["avg_abs"],
    avg_rel_err=errors["avg_rel"],
    backend=backend,
    input_info=input_info,
    cpu_time_ms=cpu_time_ms,
    cpu_out=cpu_out,
)
```

- [ ] **Step 4: 新增 _run_target_only 方法**

```python
def _run_target_only(self, test_case: TestCase, backend: str,
                     cpu_out: torch.Tensor, cpu_time_ms: float) -> CorrectnessResult:
    """只执行 target 分支，复用缓存的 CPU baseline 结果"""
    op = test_case.mapped_op
    op_name = op.op_info.name
    input_info = self._format_input_info(test_case)
    prev_swdnn = os.environ.get("SWDNN", "OFF")

    try:
        # 仅执行 target 分支（与 _run_single 中 target 分支逻辑相同）
        if backend == "swdnn":
            os.environ["SWDNN"] = "ON"
            device_str = "cpu"
        else:
            os.environ["SWDNN"] = "OFF"
            device_str = "cuda"

        if device_str == "cuda" and not torch.cuda.is_available():
            return CorrectnessResult(
                op_name=op_name, passed=False, backend=backend,
                error_message="CUDA not available",
                input_info=input_info,
                cpu_time_ms=cpu_time_ms,
                cpu_out=cpu_out,
            )

        target_kwargs = {k: v for k, v in test_case.kwargs.items()}
        target_positional = []
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

        target_out = op.callable(*target_positional, **target_kwargs)

        if isinstance(target_out, tuple):
            target_out = [o for o in target_out if isinstance(o, torch.Tensor)]
            if target_out:
                target_out = target_out[0]
            else:
                return CorrectnessResult(
                    op_name=op_name, passed=True, backend=backend,
                    error_message="Non-tensor output, shape check only",
                    input_info=input_info,
                    cpu_time_ms=cpu_time_ms,
                    cpu_out=cpu_out,
                )

        target_out_cpu = target_out.cpu().to(torch.float64)

        if cpu_out.shape != target_out_cpu.shape:
            return CorrectnessResult(
                op_name=op_name, passed=False, backend=backend,
                error_message=f"Shape mismatch: CPU {cpu_out.shape} vs {backend.upper()} {target_out_cpu.shape}",
                output_shapes_match=False,
                input_info=input_info,
                cpu_time_ms=cpu_time_ms,
                cpu_out=cpu_out,
            )

        errors = self._compute_errors(cpu_out, target_out_cpu)
        dtype_str = op.op_info.input_types[0] if op.op_info.input_types else "float32"
        atol, rtol = self._get_threshold(dtype_str, op_name)
        passed = self._check_tolerance(cpu_out, target_out_cpu, atol, rtol)

        return CorrectnessResult(
            op_name=op_name,
            passed=passed,
            max_abs_err=errors["max_abs"],
            max_rel_err=errors["max_rel"],
            avg_abs_err=errors["avg_abs"],
            avg_rel_err=errors["avg_rel"],
            backend=backend,
            input_info=input_info,
            cpu_time_ms=cpu_time_ms,
            cpu_out=cpu_out,
        )

    except Exception as e:
        return CorrectnessResult(
            op_name=op_name, passed=False, backend=backend,
            error_message=str(e), input_info=input_info,
            cpu_time_ms=cpu_time_ms, cpu_out=cpu_out,
        )

    finally:
        os.environ["SWDNN"] = prev_swdnn
```

- [ ] **Step 5: run 方法支持 cpu_baseline_cache 参数**

```python
def run(self, test_case: TestCase, backend: str = "cuda",
        cpu_baseline_cache: Optional[Tuple[torch.Tensor, float]] = None) -> CorrectnessResult:
    if cpu_baseline_cache is not None:
        cpu_out, cpu_time_ms = cpu_baseline_cache
        return self._run_target_only(test_case, backend, cpu_out, cpu_time_ms)
    return self._run_single(test_case, backend)
```

- [ ] **Step 6: Commit**

```bash
git add op_testgen/correctness/test_runner.py
git commit -m "feat(correctness): add CPU baseline timing and cache for cross-backend reuse"
```

---

### Task 3: PerfBenchmark 重构

**Files:**
- Modify: `op_testgen/perf/benchmark.py`

- [ ] **Step 1: 移除 warmup，拆分 iters**

```python
class PerfBenchmark:
    def __init__(self, cpu_iters: int = 1, target_iters: int = 3):
        self.settings = get_settings()
        self.cpu_iters = cpu_iters
        self.target_iters = target_iters
        self.classifier = OpClassifier()
        self.metrics = MetricsCalculator()
```

- [ ] **Step 2: _measure_time 增加 iters 参数，移除 warmup**

```python
def _measure_time(self, test_case: TestCase, device: str, iters: int) -> float:
    op = test_case.mapped_op

    def to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.clone().to(device)
        elif isinstance(obj, list):
            return [to_device(x) for x in obj]
        return obj

    positional = [to_device(arg) for arg in test_case.positional_args]
    kwargs = {k: v for k, v in test_case.kwargs.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        op.callable(*positional, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters
```

- [ ] **Step 3: run 方法支持 cpu_time_ms 复用**

```python
def run(self, test_case: TestCase, backend: str = "cuda",
        cpu_time_ms: Optional[float] = None) -> PerfResult:
    op = test_case.mapped_op
    op_name = op.op_info.name
    category = self.classifier.classify(op_name)
    input_info = self._format_input_info(test_case)
    prev_swdnn = os.environ.get("SWDNN", "OFF")

    try:
        # CPU time：优先复用 correctness 的数据
        if cpu_time_ms is not None:
            cpu_time = cpu_time_ms / 1000.0
        else:
            os.environ["SWDNN"] = "OFF"
            cpu_time = self._measure_time(test_case, "cpu", self.cpu_iters)

        target_time = cpu_time
        if backend == "swdnn":
            os.environ["SWDNN"] = "ON"
            target_time = self._measure_time(test_case, "cpu", self.target_iters)
        elif backend == "cuda" and torch.cuda.is_available():
            os.environ["SWDNN"] = "OFF"
            target_time = self._measure_time(test_case, "cuda", self.target_iters)

        speedup = cpu_time / target_time if target_time > 0 else 1.0

        # ... flops/bandwidth 计算保持不变 ...
        flops = 0.0
        bandwidth = 0.0
        dtype = torch.float32
        if op.op_info.input_types:
            tb = TensorBuilder()
            dtype = tb._map_dtype(op.op_info.input_types[0])

        if category == "compute":
            formula = self.classifier.get_flops_formula(op_name)
            total_flops = self.metrics.compute_flops(op_name, op.op_info.input_dims, formula)
            flops = (total_flops / (target_time * 1e9)) if target_time > 0 else 0.0
        elif category in ("memory", "mixed"):
            bytes_total = self.metrics.compute_bytes(op.op_info.input_dims, dtype)
            bandwidth = (bytes_total / (target_time * 1e9)) if target_time > 0 else 0.0
        elif category == "communication":
            bytes_total = self.metrics.compute_communication_bytes(op.op_info.input_dims, dtype)
            bandwidth = (bytes_total / (target_time * 1e9)) if target_time > 0 else 0.0

        result = PerfResult(
            op_name=op_name, category=category, backend=backend,
            avg_time_ms=target_time * 1000,
            flops=flops, bandwidth_gbps=bandwidth, speedup=speedup,
            input_info=input_info,
        )
        print(f"    [性能] {op_name}: {result.avg_time_ms:.4f}ms, speedup={result.speedup:.2f}x, flops={result.flops:.2f}GFLOPS, bandwidth={result.bandwidth_gbps:.2f}GB/s")
        print(f"      input: {input_info}")
        return result

    except Exception as e:
        result = PerfResult(
            op_name=op_name, category=category, backend=backend,
            error_message=str(e), input_info=input_info,
        )
        print(f"    [性能] {op_name}: ERROR - {e}")
        print(f"      input: {input_info}")
        return result

    finally:
        os.environ["SWDNN"] = prev_swdnn
```

- [ ] **Step 4: Commit**

```bash
git add op_testgen/perf/benchmark.py
git commit -m "feat(perf): remove warmup, split cpu/target iters, support CPU time reuse"
```

---

### Task 4: CLI 执行顺序重构和参数调整

**Files:**
- Modify: `op_testgen/cli.py`

- [ ] **Step 1: 修改 CLI 参数**

在 `test_parser` 和 `generate_parser` 中：
- `--iters` 保留但语义变为 target iters（默认 3）
- 新增 `--cpu-iters`（默认 1）

```python
test_parser.add_argument("--iters", type=int, default=3, help="性能测试迭代次数（target backend，默认: 3）")
test_parser.add_argument("--cpu-iters", type=int, default=1, help="CPU baseline 迭代次数（默认: 1）")

generate_parser.add_argument("--iters", type=int, default=3, help="性能测试迭代次数（默认: 3）")
generate_parser.add_argument("--cpu-iters", type=int, default=1, help="CPU baseline 迭代次数（默认: 1）")
```

- [ ] **Step 2: 重构 cmd_test 执行顺序**

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

    # 获取性能测试配置
    cpu_iters = getattr(args, "cpu_iters", 1)
    target_iters = getattr(args, "iters", 3)

    for mapped_op in mapped_ops:                          # 外层：算子
        test_case = builder.build(mapped_op)
        cpu_baseline_cache = None

        try:
            for backend in backends:                      # 内层：backend
                print(f"\n  算子: {mapped_op.op_info.name} | 后端: {backend.upper()}")

                if not args.only_performance:
                    correctness_result = correctness_runner.run(
                        test_case, backend,
                        cpu_baseline_cache=cpu_baseline_cache
                    )
                    all_correctness.append(correctness_result)

                    if cpu_baseline_cache is None:
                        cpu_baseline_cache = (
                            correctness_result.cpu_out,
                            correctness_result.cpu_time_ms
                        )

                    status = "通过" if correctness_result.passed else "失败"
                    if correctness_result.error_message:
                        print(f"    [正确性 {status}] {correctness_result.op_name}: {correctness_result.error_message}")
                    else:
                        print(f"    [正确性 {status}] {correctness_result.op_name}: max_abs={correctness_result.max_abs_err:.2e}")

                if not args.only_correctness:
                    perf_benchmark = PerfBenchmark(cpu_iters=cpu_iters, target_iters=target_iters)
                    perf_result = perf_benchmark.run(
                        test_case, backend,
                        cpu_time_ms=correctness_result.cpu_time_ms if not args.only_performance else None
                    )
                    all_perf.append(perf_result)

                if args.fail_fast and not correctness_result.passed:
                    print("  fail-fast: 停止后续测试")
                    break

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            del test_case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 生成报告（保持不变）...
```

- [ ] **Step 3: Commit**

```bash
git add op_testgen/cli.py
git commit -m "refactor(cli): operator-outer loop, reuse CPU baseline across backends"
```

---

### Task 5: 同步重构 template.py

**Files:**
- Modify: `op_testgen/generator/template.py`

- [ ] **Step 1: 修改模板中的默认参数**

将模板中的 `${iters}` 替换为 `${target_iters}`，新增 `${cpu_iters}`：

```python
TEST_FILE_TEMPLATE = '''#!/usr/bin/env python3
# Auto-generated test cases from trace: ${source_trace}
# Default options: backend=${backend}, seed=${seed}, cpu_iters=${cpu_iters}, target_iters=${target_iters}

# ...

parser.add_argument("--backend", default="${backend}", choices=["cuda", "swdnn", "cpu"])
parser.add_argument("--cpu-iters", type=int, default=${cpu_iters})
parser.add_argument("--target-iters", type=int, default=${target_iters})
# ...
'''
```

- [ ] **Step 2: 修改模板中的执行顺序**

将模板中的 correctness 循环和 performance 循环合并为算子外层循环：

```python
def main(argv=None) -> int:
    # ... 参数解析 ...

    for data in test_data:
        mapped_op = _build_mapped_op(data)
        if mapped_op is None:
            continue
        test_case = builder.build(mapped_op)
        cpu_baseline_cache = None

        try:
            for backend in backends:
                if not args.only_performance:
                    correctness_result = runner.run(
                        test_case, backend,
                        cpu_baseline_cache=cpu_baseline_cache
                    )
                    all_correctness.append(correctness_result)
                    if cpu_baseline_cache is None:
                        cpu_baseline_cache = (
                            correctness_result.cpu_out,
                            correctness_result.cpu_time_ms
                        )
                    # ... 打印结果 ...

                if not args.only_correctness:
                    benchmark = PerfBenchmark(
                        cpu_iters=args.cpu_iters,
                        target_iters=args.target_iters
                    )
                    perf_result = benchmark.run(
                        test_case, backend,
                        cpu_time_ms=correctness_result.cpu_time_ms if not args.only_performance else None
                    )
                    all_perf.append(perf_result)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            del test_case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ... 报告生成 ...
```

- [ ] **Step 3: 修改 TestCaseGenerator 传入新参数**

```python
def generate(..., cpu_iters: int = 1, target_iters: int = 3, ...):
    options = {
        # ...
        "cpu_iters": cpu_iters,
        "target_iters": target_iters,
    }
```

- [ ] **Step 4: Commit**

```bash
git add op_testgen/generator/template.py op_testgen/generator/test_case_generator.py
git commit -m "refactor(template): sync operator-outer loop and CPU time reuse"
```

---

### Task 6: 更新单元测试

**Files:**
- Modify: `tests/test_correctness.py`
- Modify: `tests/test_perf.py`

- [ ] **Step 1: test_correctness.py 验证 CPU 计时**

```python
def test_cpu_baseline_timing():
    # 构造一个简单的测试用例
    # 调用 runner.run()
    # 验证 result.cpu_time_ms > 0
    # 验证 result.cpu_out is not None
    pass

def test_cpu_baseline_cache():
    # 构造测试用例
    # 第一次调用 runner.run(test_case, "cuda")
    # 第二次调用 runner.run(test_case, "cuda", cpu_baseline_cache=(result.cpu_out, result.cpu_time_ms))
    # 验证第二次返回的 cpu_time_ms 与第一次相同
    pass
```

- [ ] **Step 2: test_perf.py 验证 CPU time 复用**

```python
def test_cpu_time_reuse():
    # 构造测试用例
    # benchmark.run(test_case, "cpu", cpu_time_ms=100.0)
    # 验证 speedup 计算正确（不应再跑 CPU baseline）
    pass

def test_no_warmup():
    # 验证 PerfBenchmark 默认 warmup=0
    benchmark = PerfBenchmark()
    assert benchmark.cpu_iters == 1
    assert benchmark.target_iters == 3
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_correctness.py tests/test_perf.py
git commit -m "test: update correctness and perf tests for new flow"
```

---

### Task 7: 集成测试与端到端验证

**Files:**
- None（使用现有 trace 文件测试）

- [ ] **Step 1: 测试 test 命令**

```bash
python run.py test profiler_trace.json --only-correctness --backend cpu --max-ops 5
```
Expected: 正常执行，算子外层循环

- [ ] **Step 2: 测试 generate + run 分离流程**

```bash
python run.py generate profiler_trace.json -o /tmp/test_ops.py --backend cpu --max-ops 3 --cpu-iters 1 --target-iters 3
python run.py run /tmp/test_ops.py --only-correctness
```
Expected: 生成文件包含新参数，运行正常

- [ ] **Step 3: 运行全部单元测试**

```bash
pytest tests/ -v
```
Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: add integration tests for refactored test flow"
```

---

## Spec Coverage 检查

| Spec 需求 | 实现任务 |
|----------|---------|
| 算子外层 → backend 内层循环 | Task 4 (cli.py), Task 5 (template.py) |
| correctness CPU baseline 计时 | Task 2 (test_runner.py) |
| cpu_out / cpu_time_ms 缓存 | Task 2 (test_runner.py) |
| PerfBenchmark CPU time 复用 | Task 3 (benchmark.py) |
| 跨 backend 复用 CPU baseline | Task 2 (test_runner.py run() 方法) |
| warmup=0, cpu_iters=1, target_iters=3 | Task 1 (settings.py), Task 3 (benchmark.py) |
| CLI 参数调整 | Task 4 (cli.py) |
| 生成的 .py 同步 | Task 5 (template.py) |
| 边界情况处理 | Task 2, 3, 4 |

无遗漏 ✅

---

## Type 一致性检查

| 名称 | 定义位置 | 使用位置 | 一致性 |
|------|---------|---------|--------|
| `cpu_baseline_cache` | `test_runner.py:run()` | `cli.py`, `template.py` | Tuple[torch.Tensor, float] ✅ |
| `cpu_time_ms` | `CorrectnessResult` | `benchmark.py:run()` | float ✅ |
| `cpu_out` | `CorrectnessResult` | `test_runner.py:_run_target_only()` | Optional[torch.Tensor] ✅ |
| `cpu_iters` / `target_iters` | `PerfBenchmark.__init__()` | `cli.py`, `template.py` | int ✅ |

无矛盾 ✅
