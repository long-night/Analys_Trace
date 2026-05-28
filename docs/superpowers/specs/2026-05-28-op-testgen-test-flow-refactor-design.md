# op_testgen 测试流程重构设计文档

**日期**: 2026-05-28
**状态**: 已评审
**作者**: Sisyphus

---

## 1. 背景

当前 `op_testgen` 的测试执行流程存在两个问题：

1. **执行顺序低效**：先全量跑完所有算子的正确性测试，再全量跑完所有算子的性能测试。同一个算子被 `builder.build()` 两次（correctness 一次 + performance 一次），且正确性和性能数据不能即时关联。

2. **性能数据冗余**：CPU baseline 在每个 backend 的 correctness 和 performance 中各执行一次，且 performance 的 warmup + 10 次迭代开销较大。

## 2. 目标

1. **执行顺序重构**：改为"算子外层循环 → backend 内层循环 → 每个用例先测正确性再测性能"。

2. **性能数据复用**：correctness 的 CPU baseline 执行顺带计时，直接复用给 performance 计算 speedup。

3. **配置优化**：默认不需要预热，CPU baseline 只跑1遍，对比端跑3遍。

4. **跨 backend 复用**：同一个算子的 CPU baseline 结果和时间在多个 backend 之间只计算一次。

## 3. 设计方案

### 3.1 执行顺序重构

**变更前**（backend 外层）：

```python
for backend in backends:
    for mapped_op in mapped_ops:    # correctness 全量
        test_case = builder.build(mapped_op)
        correctness_runner.run(test_case, backend)
        del test_case
    
    for mapped_op in mapped_ops:    # performance 全量
        test_case = builder.build(mapped_op)
        perf_benchmark.run(test_case, backend)
        del test_case
```

**变更后**（算子外层）：

```python
for mapped_op in mapped_ops:
    test_case = builder.build(mapped_op)    # 每个算子只 build 一次
    
    cpu_baseline_cache = None
    
    for backend in backends:
        correctness_result = correctness_runner.run(
            test_case, backend, 
            cpu_baseline_cache=cpu_baseline_cache
        )
        
        if cpu_baseline_cache is None:
            cpu_baseline_cache = (
                correctness_result.cpu_out, 
                correctness_result.cpu_time_ms
            )
        
        perf_result = perf_benchmark.run(
            test_case, backend,
            cpu_time_ms=correctness_result.cpu_time_ms
        )
        
        torch.cuda.empty_cache()
    
    del test_case
```

**收益**：
- 每个算子只 `build()` 一次，减少 50% 张量分配开销
- CPU baseline 只跑 1 次（跨 backend 复用）
- correctness 和 performance 数据即时关联

### 3.2 CorrectnessRunner 修改

#### 3.2.1 CorrectnessResult 新增字段

```python
@dataclass
class CorrectnessResult:
    op_name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    avg_abs_err: float = 0.0
    avg_rel_err: float = 0.0
    backend: str = "cuda"
    error_message: Optional[str] = None
    output_shapes_match: bool = True
    input_info: str = ""
    cpu_time_ms: float = 0.0           # 新增：CPU baseline 执行时间
    cpu_out: Optional[torch.Tensor] = None  # 新增：CPU baseline 输出张量（缓存用）
```

#### 3.2.2 CPU baseline 计时

```python
# _run_single() 中
start = time.perf_counter()
cpu_out = op.callable(*cpu_positional, **cpu_kwargs)
cpu_time_ms = (time.perf_counter() - start) * 1000
```

#### 3.2.3 跨 backend 复用 CPU baseline

```python
def run(self, test_case: TestCase, backend: str = "cuda",
        cpu_baseline_cache: Optional[Tuple[torch.Tensor, float]] = None) -> CorrectnessResult:
    
    if cpu_baseline_cache is not None:
        # 复用缓存的 CPU baseline 结果和时间
        cpu_out, cpu_time_ms = cpu_baseline_cache
        # 只执行 target 分支
        return self._run_target_only(test_case, backend, cpu_out, cpu_time_ms)
    else:
        # 首次：执行完整的 correctness（含 CPU baseline）
        return self._run_single(test_case, backend)
```

#### 3.2.4 _run_target_only 方法

```python
def _run_target_only(self, test_case: TestCase, backend: str,
                     cpu_out: torch.Tensor, cpu_time_ms: float) -> CorrectnessResult:
    """只执行 target 分支，复用缓存的 CPU baseline 结果"""
    # ... target 分支逻辑（与 _run_single 相同，但跳过 CPU baseline）
    # 返回结果时带上 cpu_time_ms
```

### 3.3 PerfBenchmark 修改

#### 3.3.1 run() 方法签名变更

```python
def run(self, test_case: TestCase, backend: str = "cuda",
        cpu_time_ms: Optional[float] = None) -> PerfResult:
```

#### 3.3.2 CPU time 复用逻辑

```python
if cpu_time_ms is not None:
    cpu_time = cpu_time_ms / 1000.0
else:
    cpu_time = self._measure_time(test_case, "cpu")  # 回退
```

#### 3.3.3 _measure_time 调整

- 移除 warmup 循环（`warmup_iters = 0`）
- benchmark 循环次数由调用方控制（CPU=1, target=3）

```python
def _measure_time(self, test_case: TestCase, device: str, iters: int) -> float:
    # 移除 warmup
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        op.callable(*positional, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters
```

#### 3.3.4 run() 内部逻辑

```python
try:
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
    # ...
```

### 3.4 配置变更

#### 3.4.1 Settings

```python
@dataclass
class Settings:
    # ...
    perf_warmup_iters: int = 0          # 从 3 改为 0
    perf_cpu_iters: int = 1             # 新增：CPU baseline 迭代次数
    perf_target_iters: int = 3          # 新增：target backend 迭代次数
```

#### 3.4.2 CLI 参数调整

```bash
# 新参数
--cpu-iters     # CPU baseline 迭代次数（默认 1）
--target-iters  # target backend 迭代次数（默认 3）

# 兼容性：--iters 映射为 --target-iters
--iters 5       # 等价于 --target-iters 5
```

### 3.5 CLI 执行流程重构

```python
def cmd_test(args) -> int:
    mapped_ops, mapper = _prepare_mapped_ops(args)
    builder = TensorBuilder(seed=args.seed)
    
    all_correctness = []
    all_perf = []
    
    for mapped_op in mapped_ops:                          # 外层：算子
        test_case = builder.build(mapped_op)
        cpu_baseline_cache = None
        
        try:
            for backend in backends:                      # 内层：backend
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
                
                if not args.only_correctness:
                    perf_result = perf_benchmark.run(
                        test_case, backend,
                        cpu_time_ms=correctness_result.cpu_time_ms
                    )
                    all_perf.append(perf_result)
                
                torch.cuda.empty_cache()
        finally:
            del test_case
    
    # 生成报告...
```

### 3.6 生成的 .py 文件同步重构

`template.py` 中的循环结构同步修改：

```python
for data in test_data:
    mapped_op = _build_mapped_op(data)
    if mapped_op is None:
        continue
    test_case = builder.build(mapped_op)
    cpu_baseline_cache = None
    
    try:
        for backend in backends:
            # correctness
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
            
            # performance
            if not args.only_correctness:
                perf_result = benchmark.run(
                    test_case, backend,
                    cpu_time_ms=correctness_result.cpu_time_ms
                )
                all_perf.append(perf_result)
            
            torch.cuda.empty_cache()
    finally:
        del test_case
        torch.cuda.empty_cache()
```

## 4. 边界情况处理

| 场景 | 处理方案 |
|------|---------|
| correctness CPU baseline 失败 | performance 直接标记失败（不独立回退测量） |
| inplace 算子（如 `relu_`） | 安全：correctness/performance 内部都 clone，test_case 不被污染 |
| backend="cpu" | correctness 只跑 CPU baseline；performance CPU time = target time，speedup = 1.0 |
| CUDA 显存不足 | 每个 backend 后 `torch.cuda.empty_cache()`，算子全部 backend 测完后 `del test_case` |
| fail-fast | 保持现有语义：correctness 失败时停止后续测试 |
| 非张量输出 | 保持现有处理：跳过误差计算，标记为 shape check only |

## 5. 性能收益预估

| 优化项 | 当前开销 | 优化后 | 收益 |
|--------|---------|--------|------|
| `builder.build()` | 算子 × 2次 | 算子 × 1次 | **减少 50% build 开销** |
| CPU baseline（跨 backend）| N 次 | 1 次 | **减少 (N-1) 次 CPU 执行** |
| performance warmup | 3 次 | 0 次 | **减少 3 次预热执行** |
| performance 总迭代 | 10 次 | 3 次 | **减少 70% target 迭代** |

## 6. 测试计划

1. **单元测试**：
   - `test_correctness.py`：验证 CPU baseline 计时和缓存复用
   - `test_perf.py`：验证 warmup=0 和 iters 拆分
   - `test_cli.py`：验证执行顺序和参数解析

2. **集成测试**：
   - `python run.py test profiler_trace.json --only-correctness --backend cpu`
   - `python run.py test profiler_trace.json --backend cpu,cuda`
   - `python run.py generate + run` 分离流程

3. **边界测试**：
   - inplace 算子（`relu_`）
   - 非张量输出算子
   - fail-fast 场景
   - 多 backend 场景

## 7. 兼容性

- `--iters` 参数保持向后兼容，映射为 `--target-iters`
- 生成的 `.py` 文件行为与 `test` 命令一致
- 报告格式保持不变

## 8. 风险

| 风险 | 缓解措施 |
|------|---------|
| performance 不复测 CPU 导致 speedup 偏差 | correctness 的 CPU 计时使用 `time.perf_counter()`，精度足够 |
| inplace 算子复用 test_case 导致数据污染 | correctness/performance 内部 clone，test_case 原始张量不受影响 |
| 跨 backend CPU baseline 缓存占用内存 | `cpu_out` 是 CPU 张量，内存占用有限；测试结束后释放 |
| 生成的 .py 文件需要同步修改 | template.py 同步重构，确保行为一致 |
