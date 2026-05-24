# PyTorch Profiler 自动化测试生成器 设计文档

> **日期**: 2026-05-24
> **状态**: 待审阅

---

## 1. 背景与目标

### 1.1 背景

现有 `analys_trace_v4.py` 可解析 PyTorch Profiler 输出的 Chrome Trace JSON，提取算子的运行时信息（name、Input Dims、Input Strides、Input type、Concrete Inputs）。本方案在此基础上，进一步实现从 profiler trace 到可复现测试用例的自动生成。

### 1.2 目标

以 PyTorch Profiler JSON 为输入，自动生成并执行两类测试：

- **正确性测试**：以 CPU 版本为 baseline，对比 CUDA / SWDNN 实现（通过环境变量 `SWDNN=ON/OFF` 切换），输出绝对误差与相对误差
- **性能测试**：按算子分类（计算密集 / 访存密集 / 通信密集）分别测试 FLOPS、访存带宽、通信带宽，输出绝对指标与相对加速比

### 1.3 非目标

- 不支持需要复杂前置状态（如模型权重、自定义 autograd function）的算子
- 不生成端到端模型测试，仅针对单个算子级别
- 不处理随机性算子（如 `dropout` 的严格数值对比，仅验证输出形状与类型）

---

## 2. 架构概览

系统由六个阶段组成，每个阶段对应一个独立的 Python 模块：

```
profiler_trace.json
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ TraceParser  │ ──► │   OpMapper   │ ──► │TensorBuilder │
│  (数据解析)   │     │ (算子映射过滤)│     │ (张量参数重构)│
└──────────────┘     └──────────────┘     └──────────────┘
       │                                          │
       │         ┌─────────────────┐              │
       │         │ CorrectnessRunner│ ◄───────────┘
       │         │   (正确性测试)   │
       │         └─────────────────┘
       │                    │
       │         ┌───────────────┐
       └────────►│ PerfBenchmark  │
                 │   (性能测试)   │
                 └───────────────┘
                          │
                          ▼
                 ┌───────────────┐
                 │   Reporter    │ ──► final_report.html / .json
                 │   (报告输出)   │
                 └───────────────┘
```

---

## 3. 模块详细设计

### 3.1 TraceParser（数据解析）

**职责**：加载并解析 Chrome Trace JSON，提取算子运行时信息。

**输入**：`profiler_trace.json` 文件路径
**输出**：`List[OpInfo]`

**核心数据结构** `OpInfo`：

```python
@dataclass
class OpInfo:
    name: str                      # 算子名称，如 "aten::add"
    input_dims: List[List[int]]    # 输入张量形状列表
    input_strides: List[List[int]] # 输入张量 stride 列表
    input_types: List[str]         # 输入数据类型，如 ["float", "float"]
    concrete_inputs: List[Any]     # 标量/列表参数，如 [0, True]
    duration_us: float             # 执行耗时（微秒）
    is_communication: bool         # 是否为通信算子
```

**实现策略**：
- 复用现有 `analys_trace_v4.py` 的解析逻辑
- 同时处理 `ph: "B"/"E"`（配对事件）和 `ph: "X"`（完整事件）
- 提取字段优先级：`Input Dims` > `input_dims` > `Input Shapes`

### 3.2 OpMapper（算子映射与过滤）

**职责**：将 trace 中的算子名称映射到可执行的 Python callable，并过滤不支持的算子。

**输入**：`List[OpInfo]`
**输出**：`List[MappedOp]`

**核心数据结构** `MappedOp`：

```python
@dataclass
class MappedOp:
    op_info: OpInfo
    callable: Callable             # 映射到的 Python 函数/方法
    callable_path: str             # 如 "torch.add"
    namespace: str                 # torch / torch.nn / torch.distributed / ...
    support_status: str            # "whitelisted" | "auto_discovered" | "unsupported"
```

**映射策略（混合方案）**：

1. **白名单优先**：读取 `config/op_whitelist.yaml`，若算子在白名单中，直接使用预定义映射
2. **动态反射**：若不在白名单，尝试通过以下规则动态解析：
   - `aten::add` → `torch.add`
   - `aten::conv2d` → `torch.nn.functional.conv2d`
   - `c10d::allreduce_` → `torch.distributed.all_reduce`
   - 带 `Tensor` 后缀的去掉后缀（`aten::add.Tensor` → `aten::add`）
3. **命名空间过滤**：仅保留以下命名空间的算子：
   - `torch.*`（张量操作与数学函数）
   - `torch.nn.functional.*` / `torch.nn.*`（神经网络层）
   - `torch.linalg.*`（线性代数）
   - `torch.distributed.*`（分布式通信）
   - `torch.optim.*`（优化器算法）
4. **白名单更新**：动态发现成功后，自动追加到 `op_whitelist.yaml`（可选，需 `--update-whitelist` 标志）

**过滤规则**：
- 排除内存管理类：`aten::empty`, `aten::zeros`, `aten::ones`
- 排除数据转换类：`aten::to`, `aten::copy_`, `aten::detach`
- 排除内部调度类：`aten::call`, `aten::__interpolate`

### 3.3 TensorBuilder（张量与参数重构）

**职责**：根据 `OpInfo` 中的类型、形状、stride 信息，构建可传入 callable 的实际张量和参数。

**输入**：`List[MappedOp]`
**输出**：`List[TestCase]`

**核心数据结构** `TestCase`：

```python
@dataclass
class TestCase:
    mapped_op: MappedOp
    input_tensors: List[torch.Tensor]   # 构建好的输入张量
    kwargs: Dict[str, Any]              # 从 concrete_inputs 解析的参数
    expected_output_shapes: List[List[int]]  # 期望输出形状（可选）
```

**构建规则**：

1. **数据类型映射**（`input_types` → `torch.dtype`）：

   | Input type | torch dtype |
   |------------|-------------|
   | `float` / `float32` | `torch.float32` |
   | `double` / `float64` | `torch.float64` |
   | `half` / `float16` | `torch.float16` |
   | `bfloat16` | `torch.bfloat16` |
   | `long` / `long int` / `int64` | `torch.int64` |
   | `int` / `int32` | `torch.int32` |
   | `short` / `int16` | `torch.int16` |
   | `char` / `int8` | `torch.int8` |
   | `byte` / `uint8` | `torch.uint8` |
   | `bool` | `torch.bool` |
   | `Scalar` | 作为 Python 标量处理，不构建张量 |

2. **张量构建**：
   - 使用 `torch.randn()` / `torch.randint()` 按形状和类型生成随机数据
   - 若提供了 `input_strides`，使用 `torch.as_strided()` 重构非连续张量
   - 默认 `torch.manual_seed(42)` 保证可复现，支持 `--seed` 覆盖

3. **Concrete Inputs 解析**：
   - 根据算子签名（`inspect.signature`）自动匹配位置参数与关键字参数
   - `Scalar` 类型直接作为 Python 标量传入
   - 列表/元组参数原样传入
   - 布尔值处理：`True`/`False` 保持为 bool
   - 无法解析的参数标记为 `unresolved`，记录到日志但不阻塞执行

### 3.4 CorrectnessRunner（正确性测试）

**职责**：对每个 `TestCase`，分别在 CPU 和 CUDA/SWDNN 环境下执行，比较输出结果。

**执行流程**：

1. **CPU Baseline**：
   - 输入张量 `.cpu().float64()`（转换为高精度避免数值误差）
   - 执行 callable，记录输出 `output_cpu`

2. **CUDA/SWDNN 对比**：
   - `--backend=cuda`：设置 `SWDNN=OFF`，使用原生 CUDA
   - `--backend=swdnn`：设置 `SWDNN=ON`，使用 SWDNN 后端
   - `--backend=auto`：先后执行两次（CUDA 和 SWDNN），分别记录结果
   - 输入张量 `.cuda()`（保持原始 dtype）
   - 执行 callable，记录输出 `output_cuda`
   - 将 `output_cuda` 转回 CPU 和 `float64`

3. **误差计算**：
   - 绝对误差：`abs(output_cpu - output_cuda)`
   - 相对误差：`abs(output_cpu - output_cuda) / abs(output_cpu)`，mask 掉 `output_cpu == 0` 的位置
   - 统计最大绝对误差、最大相对误差、平均绝对误差、平均相对误差

4. **通过判定**：

   | dtype | max_abs_err | max_rel_err |
   |-------|-------------|-------------|
   | float16 | `< 1e-3` | `< 1e-2` |
   | bfloat16 | `< 5e-3` | `< 5e-2` |
   | float32 | `< 1e-5` | `< 1e-4` |
   | float64 | `< 1e-10` | `< 1e-9` |
   | int* / bool | 严格相等 | - |

5. **失败处理**：
   - 默认非 fail-fast，记录所有失败，最后汇总
   - 提供 `--fail-fast` 选项，第一个失败即停止

### 3.5 PerfBenchmark（性能测试）

**职责**：对每个算子执行性能测试，按分类计算对应指标。

**算子分类策略（混合方案）**：

1. **配置文件优先**：读取 `config/op_classification.yaml`，若算子有显式分类，直接使用
2. **启发式默认**：
   - 名称包含 `matmul`, `conv`, `mm`, `bmm`, `linear` → **计算密集型**
   - 名称包含 `allreduce`, `allgather`, `broadcast`, `send`, `recv`, `reduce` → **通信密集型**
   - 其余 → **访存密集型**（默认）
3. **运行时验证**：测试时同时估算理论 FLOPS 和访存量，若两者均显著（差距 < 10×），标记为**混合密集型**，同时计算两类指标

**性能指标计算**：

| 分类 | 指标 | 计算公式 |
|------|------|----------|
| 计算密集型 | FLOPS | `理论计算量 / 平均耗时` |
| 访存密集型 | 访存带宽 | `读写总字节数 / 平均耗时` |
| 通信密集型 | 通信带宽 | `传输数据量 / 平均耗时` |
| 混合密集型 | FLOPS + 带宽 | 同时输出两项 |

**理论计算量公式（可配置）**：

```yaml
# config/op_classification.yaml 示例
aten::conv2d:
  category: compute
  flops_formula: "2 * H * W * Cin * Cout * K * K"
  
nccl:all_reduce:
  category: communication
  bytes_formula: "2 * numel * element_size"  # 发送 + 接收
```

**性能测试执行**：
- Warm-up：3 次不计时执行
- 测量：10 次执行取平均耗时（支持 `--iters` 调整）
- 同步：CUDA 测试使用 `torch.cuda.synchronize()` 确保计时准确

### 3.6 Reporter（报告输出）

**职责**：汇总正确性测试和性能测试结果，生成结构化报告。

**输出格式**：

1. **JSON 中间报告**（供后续处理）：
   ```json
   {
     "summary": {
       "total_ops": 100,
       "passed": 95,
       "failed": 5,
       "skipped": 10
     },
     "correctness": [...],
     "performance": [...]
   }
   ```

2. **HTML 报告**（默认输出）：
   - 汇总卡片：总算子数、通过率、平均加速比
   - 正确性表格：算子名称、误差、通过状态
   - 性能表格：FLOPS / 带宽 / 加速比
   - 失败详情：展开查看具体错误信息

3. **Excel 报告**（可选，需 `openpyxl`）：
   - Sheet1: 正确性测试结果
   - Sheet2: 性能测试结果
   - Sheet3: 失败详情

---

## 4. 配置系统

### 4.1 配置文件清单

| 文件 | 用途 | 是否可自动更新 |
|------|------|----------------|
| `config/settings.py` | 全局设置（误差阈值、随机种子、迭代次数） | 否 |
| `config/op_whitelist.yaml` | 算子白名单与映射 | 是（`--update-whitelist`） |
| `config/op_classification.yaml` | 算子分类与计算公式 | 否（手动维护） |

### 4.2 命令行参数

```
op_testgen profiler_trace.json \
  --backend {cuda,swdnn,auto}     # 对比后端，auto 表示 CUDA 和 SWDNN 都测
  --output report.html            # 输出文件路径
  --format {html,json,excel,all}  # 输出格式
  --seed 42                       # 随机种子
  --iters 10                      # 性能测试迭代次数
  --fail-fast                     # 第一个失败即停止
  --update-whitelist              # 动态发现后更新白名单
  --only-correctness              # 仅执行正确性测试
  --only-performance              # 仅执行性能测试
  --op-filter "aten::conv*"       # 仅测试匹配名称的算子
```

---

## 5. 错误处理策略

| 错误类型 | 处理方式 | 日志级别 |
|----------|----------|----------|
| JSON 解析失败 | 立即退出，提示文件损坏 | ERROR |
| 算子映射失败 | 跳过该算子，记录到 `skipped` 列表 | WARNING |
| 张量构建失败（shape 不匹配） | 跳过该算子 | WARNING |
| Concrete Inputs 解析失败 | 尝试部分参数执行，记录未解析项 | WARNING |
| 正确性测试失败 | 记录误差，继续执行（非 fail-fast） | ERROR |
| CUDA 内存不足 | 跳过该算子，标记为 `oom` | ERROR |
| 性能测试异常 | 跳过该算子性能测试，不影响正确性测试 | WARNING |

---

## 6. 边界情况处理

1. **inplace 算子**（如 `aten::add_`）：
   - 构建输入张量时使用 `.clone()` 避免修改原始数据
   - 正确性测试时分别 clone 给 CPU 和 CUDA 执行

2. **多输出算子**（如 `aten::max` 返回 values + indices）：
   - 遍历所有输出张量逐一比较
   - 非张量输出（如 int）直接比较

3. **通信算子**（如 `all_reduce`）：
   - 正确性测试：初始化 `torch.distributed`，单进程模拟多进程（`init_process_group` with `gloo` backend）
   - 性能测试：记录通信数据量（基于输入张量大小 × 进程数）

4. **随机性算子**（如 `dropout`）：
   - 正确性测试：仅验证输出形状与 dtype，不验证数值
   - 性能测试：正常执行并计时

5. **无输入算子**（如 `aten::randn`）：
   - 从 `concrete_inputs` 中提取 shape 参数构建输出
   - 若无可用的 shape 信息，跳过该算子

---

## 7. 与现有代码的关系

- `analys_trace_v4.py` **保持不变**，作为独立的 trace 分析工具继续使用
- `op_testgen/parser/trace_parser.py` 将复用 `analys_trace_v4.py` 的核心解析逻辑，但拆分为更细粒度的函数
- 两者可共存：`analys_trace_v4.py` 用于快速分析，`op_testgen` 用于深度测试

---

## 8. 扩展性考虑

- **新后端支持**：未来若需支持 XPU、ROCm 等，只需在 `CorrectnessRunner` 中增加 `device = 'xpu'` 分支
- **新算子类型**：更新 `op_whitelist.yaml` 和 `op_classification.yaml` 即可，无需修改代码
- **新报告格式**：`Reporter` 采用插件式设计，新增格式只需实现 `BaseReporter` 接口

---

## 9. 验收标准

- [ ] 能成功解析 `profiler_trace.json` 并提取所有算子信息
- [ ] 至少 80% 的常见算子（add, mul, conv2d, linear, matmul）能正确映射并构建张量
- [ ] 正确性测试能输出每个算子的 abs_err 和 rel_err，并与 CPU baseline 对比
- [ ] 性能测试能正确分类算子并输出 FLOPS / 带宽 / 加速比
- [ ] 报告包含汇总统计、详细表格、失败详情三部分
- [ ] 整个流程可通过一条 CLI 命令完成

---

*文档版本: v1.0*
*作者: Sisyphus*
