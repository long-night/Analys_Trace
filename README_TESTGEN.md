# PyTorch Profiler 自动化工具集

基于 PyTorch Profiler Chrome Trace JSON 的自动化工具集，包含：
- **Trace 分析器** (`analyze`) — 算子统计分析、Excel/CSV 导出（原 `analys_trace_v4.py`）
- **测试生成器** (`test`) — 自动生成正确性/性能测试用例，对比 CPU vs CUDA/SWDNN

## 功能特性

- **自动解析** PyTorch Profiler 输出的 Chrome Trace JSON
- **算子映射** 支持动态反射发现 + 可配置白名单混合策略
- **张量重构** 根据 Input Dims / Strides / Types 自动构建输入张量
- **正确性测试** CPU (float64) baseline vs CUDA/SWDNN，输出绝对/相对误差
- **性能测试** 自动分类算子（计算/访存/通信密集型），计算 FLOPS / 带宽 / 加速比
- **OOM 防护** 内置黑名单过滤、去重、max-ops 限制，确保内存安全

## 安装

```bash
# 基础安装
pip install -e .

# 带 Excel 报告支持
pip install -e ".[excel]"

# 开发依赖
pip install -e ".[dev]"
```

依赖: Python >= 3.10, PyTorch, PyYAML, Jinja2

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

### 3. 测试生成

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

### 3. 查看报告

```bash
# HTML 报告
open op_testgen_report.html

# JSON 报告
cat op_testgen_report.json

# Excel 报告（需安装 openpyxl）
op_testgen profiler_trace.json --format excel -o report.xlsx
```

## 命令行参数

### `analyze` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON 文件路径 | *(必填)* |
| `-o, --output` | 输出文件路径 | `cpu_operators.csv` / `operator_analysis.xlsx` |
| `--csv` | 强制输出 CSV 格式 | `False` |
| `--no-summary` | 不显示终端摘要 | `False` |

### `test` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON 文件路径 | *(必填)* |
| `--backend` | 对比后端: `cuda`, `swdnn`, `cpu`, `auto` | `cuda` |
| `-o, --output` | 输出文件路径 | `op_testgen_report.html` |
| `--format` | 输出格式: `html`, `json`, `excel`, `all` | `html` |
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
op_testgen/
├── analyzer/        # Trace 统计分析（原 analys_trace_v4.py）
├── config/          # 配置系统
├── parser/          # Trace JSON 解析
├── mapper/          # 算子映射与过滤
├── builder/         # 张量与参数重构
├── correctness/     # 正确性测试
├── perf/            # 性能测试
├── reporter/        # 报告生成
└── cli.py           # 命令行入口（analyze / test 子命令）
```

## License

MIT
