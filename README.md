# PyTorch Profiler 自动化工具集

基于 PyTorch Profiler Chrome Trace JSON 的自动化工具集，包含：
- **Trace 分析器** (`op_testgen analyze`) — 算子统计分析、Excel/CSV 导出（原 `analys_trace_v4.py` 功能升级）
- **测试生成器** (`op_testgen test`) — 自动生成并执行正确性/性能测试用例，对比 CPU vs CUDA/SWDNN
- **测试用例生成** (`op_testgen generate`) — 从 Trace 生成可独立运行的 Python 测试文件
- **测试用例执行** (`op_testgen run`) — 运行由 `generate` 生成的 Python 测试文件

## 功能特性

| 模块 | 功能 |
|------|------|
| **Trace 解析** | 自动解析 PyTorch Profiler Chrome Trace JSON，支持 B/E/X 三种事件相位 |
| **算子映射** | 动态反射发现 + 可配置白名单混合策略，支持 `torch`/`torch.nn.functional`/`torch.linalg` |
| **张量重构** | 根据 Input Dims / Strides / Types 自动构建输入张量，支持标量参数解析 |
| **正确性测试** | CPU (float64) baseline vs CUDA/SWDNN，输出绝对/相对误差，mask 零值 |
| **性能测试** | 自动分类算子（计算/访存/通信密集型），计算 FLOPS / 带宽 / 加速比 |
| **统计分析** | 算子总表、Shape 统计表、TOP 排名、通信算子专项分析 |
| **OOM 防护** | 黑名单过滤（55+ 模式）、去重、max-ops 限制，确保内存安全 |
| **可配置性** | YAML 配置文件支持白名单、黑名单、分类公式自定义 |

## 安装

### 环境要求

- Python >= 3.10
- PyTorch
- PyYAML

可选依赖：
- `jinja2` — HTML 报告支持
- `openpyxl` — Excel 报告支持

### 安装方式

```bash
# 克隆仓库
git clone <repository-url>
cd Analys_Trace

# 基础安装
pip install -e .

# 带 HTML 报告支持
pip install -e ".[html]"

# 带 Excel 报告支持
pip install -e ".[excel]"

# 同时安装 HTML + Excel
pip install -e ".[html,excel]"

# 开发依赖
pip install -e ".[dev]"
```

### 安装验证

```bash
# 检查命令是否可用
op_testgen --help

# 运行测试套件
pytest tests/ -v
```

### 直接运行（无需安装）

如果你不想通过 pip 安装，可以直接使用以下两种方式运行：

**方式一：使用顶层 run.py 脚本（推荐）**

```bash
# 分析 Trace
python run.py analyze profiler_trace.json

# 生成并执行测试（一步到位）
python run.py test profiler_trace.json --backend cpu --only-correctness

# 先生成测试文件，再执行（分离模式）
python run.py generate profiler_trace.json -o tests/test_ops.py --backend cpu
python run.py run tests/test_ops.py --only-correctness
```

**方式二：使用模块方式运行**

```bash
# 分析 Trace
python -m op_testgen.cli analyze profiler_trace.json

# 生成并执行测试（一步到位）
python -m op_testgen.cli test profiler_trace.json --backend cpu --only-correctness

# 先生成测试文件，再执行（分离模式）
python -m op_testgen.cli generate profiler_trace.json -o tests/test_ops.py --backend cpu
python -m op_testgen.cli run tests/test_ops.py --only-correctness
```

> **注意**：直接运行时需要确保基础依赖已安装（`pip install torch pyyaml`）。可选：`jinja2`（HTML 报告）、`openpyxl`（Excel 报告）。

## 快速开始

### 1. 生成 Profiler Trace

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    z = x + y

prof.export_chrome_trace("profiler_trace.json")
```

### 2. Trace 分析

```bash
# 分析 Trace，输出 Excel（默认，需 openpyxl）
op_testgen analyze profiler_trace.json

# 分析并输出 CSV
op_testgen analyze profiler_trace.json --csv

# 自定义输出文件名
op_testgen analyze profiler_trace.json -o my_analysis

# 不显示摘要
op_testgen analyze profiler_trace.json --no-summary
```

### 3. 测试生成与执行

#### 方式一：一步到位（`test` 子命令）

```bash
# 完整测试（正确性 + 性能，需要 CUDA）
op_testgen test profiler_trace.json

# 仅正确性测试，CPU baseline（无需 GPU）
op_testgen test profiler_trace.json --only-correctness --backend cpu

# 仅测试特定算子（支持通配符）
op_testgen test profiler_trace.json --op-filter "aten::conv*"

# 增大测试覆盖范围（默认 100 个）
op_testgen test profiler_trace.json --max-ops 200

# 输出所有格式
op_testgen test profiler_trace.json --format all -o report

# SWDNN 对比测试
SWDNN=ON op_testgen test profiler_trace.json --backend swdnn
```

#### 方式二：先生成再执行（`generate` + `run` 子命令）

```bash
# 生成测试文件（可复用、可版本控制）
op_testgen generate profiler_trace.json -o tests/test_ops.py --backend cpu --max-ops 50

# 执行生成的测试文件
op_testgen run tests/test_ops.py --only-correctness

# 使用不同后端运行同一套测试
op_testgen run tests/test_ops.py --backend cuda --only-performance

# 生成后支持手动编辑测试文件，调整参数或添加断言
```

### 4. 查看报告

**分析模式输出：**
- Excel 模式：`operator_analysis.xlsx`（两个工作表：算子总表、Shape统计表）
- CSV 模式：`cpu_operators.csv` + `cpu_operators_shapes.csv`

**测试模式输出：**
- **Markdown 报告**（默认）：`op_testgen_report.md` — 纯文本表格，无需额外依赖
- HTML 报告：`op_testgen_report.html`（需 jinja2）
- JSON 报告：`op_testgen_report.json`
- Excel 报告：`op_testgen_report.xlsx`（需 openpyxl）

## 命令行参数

### `analyze` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON 文件路径 | *(必填)* |
| `-o, --output` | 输出文件路径 | `cpu_operators.csv` / `operator_analysis.xlsx` |
| `--csv` | 强制输出 CSV 格式 | `False` |
| `--no-summary` | 不显示终端摘要 | `False` |

### `generate` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON 文件路径 | *(必填)* |
| `-o, --output` | 输出 .py 文件路径 | *(必填)* |
| `--backend` | 默认后端: `cuda`, `swdnn`, `cpu`, `auto` | `cuda` |
| `--seed` | 随机种子（影响输入张量生成） | `42` |
| `--max-ops` | 最大测试算子数（去重后） | `100` |
| `--op-filter` | 仅生成匹配名称的算子测试（通配符） | *(无)* |
| `--iters` | 性能测试迭代次数（写入文件默认值） | `10` |
| `--update-blacklist` | 将未映射算子加入黑名单 | `False` |

### `run` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `test_file` | 生成的 .py 测试文件路径 | *(必填)* |
| `--backend` | 后端（覆盖文件默认值）: `cuda`, `swdnn`, `cpu` | 文件内默认值 |
| `--only-correctness` | 仅正确性测试 | `False` |
| `--only-performance` | 仅性能测试 | `False` |
| `--iters` | 性能测试迭代次数（覆盖文件默认值） | 文件内默认值 |
| `--fail-fast` | 第一个失败即停止 | `False` |

### `test` 子命令（一步到位）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON 文件路径 | *(必填)* |
| `--backend` | 对比后端: `cuda`, `swdnn`, `cpu`, `auto` | `cuda` |
| `-o, --output` | 输出报告文件路径 | `op_testgen_report.md` |
| `--format` | 输出格式: `markdown`, `html`, `json`, `excel`, `all` | `markdown` |
| `--seed` | 随机种子（影响输入张量生成） | `42` |
| `--iters` | 性能测试迭代次数 | `10` |
| `--max-ops` | 最大测试算子数（去重后） | `100` |
| `--fail-fast` | 第一个失败即停止 | `False` |
| `--only-correctness` | 仅执行正确性测试 | `False` |
| `--only-performance` | 仅执行性能测试 | `False` |
| `--op-filter` | 仅测试匹配名称的算子（通配符） | *(无)* |
| `--update-whitelist` | 动态发现后更新白名单 | `False` |
| `--update-blacklist` | 将未映射算子加入黑名单 | `False` |

## 配置

编辑 `op_testgen/config/` 下的 YAML 文件自定义行为：

### `op_whitelist.yaml`

算子白名单，明确声明 trace 名称到 Python callable 的映射：

```yaml
aten::add:
  callable_path: "torch.add"
  namespace: "torch"

aten::conv2d:
  callable_path: "torch.nn.functional.conv2d"
  namespace: "torch.nn.functional"
```

不在白名单中的算子将尝试**动态反射映射**（`aten::xxx` → `torch.xxx` / `torch.nn.functional.xxx`）。

### `op_blacklist.yaml`

算子黑名单，匹配的算子将被跳过不执行测试：

```yaml
patterns:
  - "aten::empty"
  - "aten::zeros"
  - "aten::copy_"
  - "aten::view"
  # ...
```

支持两种匹配模式：
- **精确匹配**： `"aten::add_"` — 仅匹配 `aten::add_`
- **前缀匹配**： `"aten::_*"` — 匹配所有 `aten::_` 开头的算子

**手动添加**：直接编辑 YAML 文件，在 "用户自定义添加区" 添加算子名称。

**自动更新**：使用 `--update-blacklist` 参数，运行结束后会提示将未映射算子加入黑名单：

```bash
op_testgen trace.json --update-blacklist
# 运行结束后提示：
# 发现 15 个未映射算子，是否加入黑名单? [y/N]: y
```

### `op_classification.yaml`

算子分类与性能指标计算公式：

```yaml
aten::matmul:
  category: compute          # 计算密集型
  flops_formula: "2 * M * N * K"

aten::add:
  category: memory           # 访存密集型

c10d::allreduce_:
  category: communication    # 通信密集型
  bytes_formula: "2 * numel * element_size"
```

可用变量: `M`, `N`, `K`, `B`, `H`, `W`, `Cin`, `Cout`, `Hout`, `Wout`, `numel`, `element_size`

### `settings.py`

全局配置（误差阈值、性能测试参数等）：

```python
# 误差阈值: {dtype: (max_abs_err, max_rel_err)}
error_thresholds = {
    "float16": (1e-3, 1e-2),
    "bfloat16": (5e-3, 5e-2),
    "float32": (1e-5, 1e-4),
    "float64": (1e-10, 1e-9),
}
```

## 架构

六阶段流水线：

```
profiler_trace.json
       │
       ▼
┌──────────────┐
│  TraceParser │  ──► 解析 JSON，提取 OpInfo(name, dims, strides, types, concrete_inputs)
└──────────────┘
       │
       ▼
┌──────────────┐
│   OpMapper   │  ──► 名称映射到 Python callable，黑名单过滤
└──────────────┘
       │
       ▼
┌──────────────┐
│ TensorBuilder│  ──► 构建输入张量，解析 concrete_inputs 为 kwargs
└──────────────┘
       │
       ├──► ┌─────────────────┐ ──► 正确性报告
       │    │ CorrectnessRunner│      (abs_err, rel_err, pass/fail)
       │    └─────────────────┘
       │
       └──► ┌───────────────┐ ──► 性能报告
            │ PerfBenchmark  │      (FLOPS, bandwidth, speedup)
            └───────────────┘
                     │
                     ▼
            ┌───────────────┐
            │   Reporter    │ ──► HTML / JSON / Excel
            └───────────────┘
```

## 注意事项

### OOM 防护

对于大规模 trace 文件（如分布式训练 Profiler 输出），工具内置多层防护：

1. **黑名单过滤**：自动过滤 `empty`, `copy_`, `to`, `select`, `view`, `unsqueeze` 等 55+ 个内存/视图/元数据算子
2. **去重**：按 `(name, dims, strides, types)` 去重，避免重复构建相同测试用例
3. **max-ops 限制**：默认最多测试 100 个唯一算子，可通过 `--max-ops` 调整

### SWDNN 支持

SWDNN 通过环境变量控制，与 PyTorch 原生 CUDA API 完全一致：

```bash
# Baseline
SWDNN=OFF op_testgen test trace.json --backend cuda

# SWDNN 实现
SWDNN=ON op_testgen test trace.json --backend swdnn
```

### 误差阈值

正确性测试使用数据类型相关的误差阈值：

| 数据类型 | 最大绝对误差 | 最大相对误差 |
|----------|-------------|-------------|
| float16 | 1e-3 | 1e-2 |
| bfloat16 | 5e-3 | 5e-2 |
| float32 | 1e-5 | 1e-4 |
| float64 | 1e-10 | 1e-9 |

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 项目结构

```
Analys_Trace/
├── op_testgen/              # 主包
│   ├── __init__.py
│   ├── cli.py               # 命令行入口（analyze / generate / run / test 子命令）
│   ├── generator/           # 测试用例生成与执行
│   │   ├── __init__.py
│   │   ├── test_case_generator.py
│   │   ├── test_case_runner.py
│   │   └── template.py
│   ├── analyzer/            # Trace 统计分析
│   │   ├── __init__.py
│   │   └── trace_analyzer.py
│   ├── config/              # 配置系统
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── op_whitelist.yaml
│   │   ├── op_blacklist.yaml
│   │   └── op_classification.yaml
│   ├── parser/              # Trace JSON 解析
│   │   ├── __init__.py
│   │   └── trace_parser.py
│   ├── mapper/              # 算子映射与过滤
│   │   ├── __init__.py
│   │   └── op_mapper.py
│   ├── builder/             # 张量与参数重构
│   │   ├── __init__.py
│   │   └── tensor_builder.py
│   ├── correctness/         # 正确性测试
│   │   ├── __init__.py
│   │   └── test_runner.py
│   ├── perf/                # 性能测试
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── metrics.py
│   │   └── benchmark.py
│   └── reporter/            # 报告生成
│       ├── __init__.py
│       ├── base.py
│       ├── html_reporter.py
│       └── excel_reporter.py
├── tests/                   # 测试套件
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_mapper.py
│   ├── test_tensor_builder.py
│   ├── test_correctness.py
│   ├── test_generator.py
│   ├── test_perf.py
│   └── test_reporter.py
├── docs/                    # 设计文档
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── analys_trace_v4.py       # 原始分析脚本（已整合为 analyze 子命令）
├── pyproject.toml           # 包配置
└── README.md                # 本文档
```

### 代码规范

项目使用以下工具保证代码质量：

```bash
# 类型检查
mypy op_testgen/ --ignore-missing-imports

# 代码格式化
ruff format op_testgen/ tests/

# 代码检查
ruff check op_testgen/ tests/
```

### 添加新算子支持

1. 编辑 `op_testgen/config/op_whitelist.yaml`，添加算子映射
2. 如需特殊 FLOPS 公式，编辑 `op_testgen/config/op_classification.yaml`
3. 运行测试验证：`pytest tests/test_mapper.py -v`

## 已知限制

- 仅支持 Chrome Trace 格式的 PyTorch Profiler 输出
- 动态反射映射可能无法覆盖所有 `aten::` 算子（需手动维护白名单）
- 性能测试的 FLOPS 估算基于启发式公式，可能与实际计算量存在偏差
- 通信算子测试需要实际的分布式环境（NCCL/Gloo/MPI）

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1.0 | 2026-05 | 初始版本：Trace 解析、算子映射、正确性/性能测试、报告生成、黑名单配置 |
| v0.2.0 | 2026-05 | 测试流程拆分：新增 `generate` 和 `run` 子命令，支持先生成测试文件再执行 |

## 贡献

欢迎提交 Issue 和 PR！请确保：
- 代码通过 `pytest tests/` 测试
- 新增功能包含对应的测试用例
- 遵循现有代码风格和类型注解规范

## License

MIT License
