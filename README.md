# PyTorch Profiler 自动化工具集

基于 PyTorch Profiler Chrome Trace JSON 的自动化分析、测试与报告工具。

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   analyze       │     │    generate      │     │      run        │
│   Trace分析     │     │  生成.py测试文件  │     │  进程内执行测试  │
│  Excel/CSV报告  │     │  精简模块依赖    │     │  importlib加载  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │    test (一步)   │
                        │ 生成+执行+报告   │
                        └─────────────────┘
```

## 功能特性

| 模块 | 功能 |
|------|------|
| **Trace 解析** | 解析 Chrome Trace JSON，支持 B/E/X 三种事件相位，提取算子名称、形状、步长、类型、标量参数 |
| **层级分析** | 调用树构建、自耗时计算、调用链提取、递归检测；支持**去重聚合视图** `[×N]`、**递归截断** `[RECURSIVE]`、树形分支符号 |
| **算子映射** | 动态反射发现 + 可配置白名单，支持 `torch` / `torch.nn.functional` / `torch.linalg` |
| **张量重构** | 根据 Input Dims / Strides / Types 自动构建输入张量，智能解析 concrete_inputs |
| **正确性测试** | CPU (float64) baseline vs CUDA/SWDNN，数据类型相关误差阈值，mask 零值 |
| **性能测试** | 自动分类算子（计算/访存/通信密集型），计算 FLOPS / 带宽 / 加速比 |
| **报告生成** | Markdown（默认）/ HTML / JSON / Excel 多格式输出 |
| **OOM 防护** | 黑名单过滤（55+ 模式）、去重、max-ops 限制 |
| **进程内执行** | `run` 子命令通过 `importlib` 在当前进程加载执行，无子进程开销 |
| **模块解耦** | 生成的 `.py` 文件内联数据类，仅依赖核心测试执行模块 |

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd Analys_Trace

# 基础安装
pip install -e .

# 带可选依赖
pip install -e ".[html,excel,dev]"
```

**环境要求：** Python >= 3.10, PyTorch, PyYAML

**可选依赖：** `jinja2` (HTML 报告), `openpyxl` (Excel 报告)

**验证安装：**

```bash
op_testgen --help
pytest tests/ -v          # 73 个测试用例
```

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
# 分析并输出 Excel（默认，需 openpyxl）
op_testgen analyze profiler_trace.json

# 输出 CSV
op_testgen analyze profiler_trace.json --csv

# 启用层级分析 + 导出调用树
op_testgen analyze profiler_trace.json --hierarchy --export-tree call_tree.txt

# 仅输出高频调用链和递归模式
op_testgen analyze profiler_trace.json --call-chains --detect-recursion
```

**分析输出：**
- Excel: `operator_analysis.xlsx`（算子总表 + Shape统计表 + 层级数据）
- CSV: `cpu_operators.csv` + `cpu_operators_shapes.csv`

**调用树格式示例：**

```
──────────────────────────────────────────────────────────────────────
根节点 #1: aten::convolution_backward
──────────────────────────────────────────────────────────────────────
  出现次数: 60
  自耗时:   2119.681 ms (平均 35.328 ms)
  总耗时:   2177.911 ms (平均 36.299 ms)

  子树结构:
  └── aten::convolution_backward [×60]  [self=35.328ms, total=36.299ms]
      ├── aten::contiguous [×18]  [self=0.006ms, total=2.972ms]
      │   └── aten::clone [×18]  [self=0.018ms, total=2.966ms]
      │       ├── aten::copy_ [×18]  [self=2.922ms, total=2.922ms]
      │       └── aten::empty_like [×18]  [self=0.007ms, total=0.026ms]
      ├── aten::empty [×174]  [self=0.018ms, total=0.018ms]
      └── aten::resize_ [×117]  [self=0.002ms, total=0.002ms]
```

### 3. 测试（一步到位）

```bash
# 完整测试（正确性 + 性能）
# 默认仅测试根节点，减少用例数量
op_testgen test profiler_trace.json

# 仅正确性测试，CPU baseline（无需 GPU）
op_testgen test profiler_trace.json --only-correctness --backend cpu

# 测试所有算子（包括子算子）
op_testgen test profiler_trace.json --test-all-ops

# 过滤特定算子
op_testgen test profiler_trace.json --op-filter "aten::conv*"

# 增大覆盖范围
op_testgen test profiler_trace.json --max-ops 200
```

**测试输出：** `op_testgen_report.md`（默认 Markdown，支持 HTML/JSON/Excel）

### 4. 测试（生成 + 执行分离）

```bash
# 生成独立 .py 测试文件（可复用、可版本控制）
op_testgen generate profiler_trace.json -o tests/test_ops.py --backend cpu --max-ops 50

# 在当前进程内执行（无子进程开销）
op_testgen run tests/test_ops.py --only-correctness

# 使用不同后端运行同一套测试
op_testgen run tests/test_ops.py --backend cuda --only-performance

# 生成后支持手动编辑 .py 文件，调整参数或添加断言
```

## 命令行参数

### `analyze` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON | 必填 |
| `-o, --output` | 输出文件路径 | `operator_analysis.xlsx` |
| `--csv` | 强制输出 CSV | `False` |
| `--no-summary` | 不显示终端摘要 | `False` |
| `--hierarchy` | 启用层级分析 | `False` |
| `--self-time-top` | 自耗时排名 TOP N | `10` |
| `--call-chains` | 输出高频调用链 | `False` |
| `--detect-recursion` | 检测递归/循环调用 | `False` |
| `--export-tree` | 导出调用树到文件 | *(无)* |
| `--max-chain-depth` | 调用链最大深度 | `10` |
| `--min-chain-occurrence` | 调用链最小出现次数 | `2` |

### `generate` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON | 必填 |
| `-o, --output` | 输出 `.py` 文件路径 | 必填 |
| `--backend` | 默认后端 | `cuda` |
| `--seed` | 随机种子 | `42` |
| `--max-ops` | 最大测试算子数（去重后） | `100` |
| `--op-filter` | 算子名称过滤（通配符） | *(无)* |
| `--iters` | 性能测试迭代次数 | `10` |
| `--test-all-ops` | 生成所有算子（默认仅根节点） | `False` |
| `--update-blacklist` | 将未映射算子加入黑名单 | `False` |

### `run` 子命令

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `test_file` | 生成的 `.py` 测试文件路径 | 必填 |
| `--backend` | 后端（覆盖文件默认值） | 文件内默认值 |
| `--only-correctness` | 仅正确性测试 | `False` |
| `--only-performance` | 仅性能测试 | `False` |
| `--iters` | 性能测试迭代次数（覆盖） | 文件内默认值 |
| `--fail-fast` | 第一个失败即停止 | `False` |
| `--op-filter` | 算子名称过滤（通配符） | *(无)* |
| `--format` | 报告格式（覆盖） | 文件内默认值 |
| `-o, --output` | 报告输出路径（覆盖） | 文件内默认值 |

### `test` 子命令（一步到位）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | PyTorch Profiler Chrome Trace JSON | 必填 |
| `--backend` | 对比后端 | `cuda` |
| `-o, --output` | 输出报告文件路径 | `op_testgen_report.md` |
| `--format` | 输出格式 | `markdown` |
| `--seed` | 随机种子 | `42` |
| `--iters` | 性能测试迭代次数 | `10` |
| `--max-ops` | 最大测试算子数 | `100` |
| `--fail-fast` | 第一个失败即停止 | `False` |
| `--only-correctness` | 仅正确性测试 | `False` |
| `--only-performance` | 仅性能测试 | `False` |
| `--op-filter` | 算子名称过滤（通配符） | *(无)* |
| `--test-all-ops` | 测试所有算子 | `False` |
| `--update-whitelist` | 动态发现后更新白名单 | `False` |
| `--update-blacklist` | 将未映射算子加入黑名单 | `False` |

## 架构

### 数据流

```
profiler_trace.json
       │
       ▼
┌──────────────┐
│  TraceParser │  ──► 解析 JSON，提取 OpInfo
└──────────────┘
       │
       ▼
┌──────────────┐
│   OpMapper   │  ──► 名称映射到 Python callable，黑名单过滤
└──────────────┘
       │
       ▼
┌──────────────┐
│ TensorBuilder│  ──► 构建输入张量，解析 concrete_inputs
└──────────────┘
       │
       ├──► ┌─────────────────┐ ──► 正确性报告
       │    │ CorrectnessRunner│
       │    └─────────────────┘
       │
       └──► ┌───────────────┐ ──► 性能报告
            │ PerfBenchmark  │
            └───────────────┘
                     │
                     ▼
            ┌───────────────┐
            │   Reporter    │ ──► Markdown / HTML / JSON / Excel
            └───────────────┘
```

### 测试文件生成与执行

```
generate ──► .py 文件（含测试数据和执行逻辑）
                  │
                  │  内联 _OpInfo / _MappedOp（不依赖 parser/mapper）
                  │  仅依赖核心模块：tensor_builder / correctness / perf / reporter
                  │
                  ▼
run ─────────► importlib 进程内加载
               调用 main(argv) 执行
               无 subprocess 开销
```

## 配置

编辑 `op_testgen/config/` 下的 YAML 文件：

### `op_whitelist.yaml`

```yaml
aten::add:
  callable_path: "torch.add"
  namespace: "torch"

aten::conv2d:
  callable_path: "torch.nn.functional.conv2d"
  namespace: "torch.nn.functional"
```

### `op_blacklist.yaml`

```yaml
patterns:
  - "aten::empty"
  - "aten::zeros"
  - "aten::copy_"
  - "aten::_*"      # 前缀匹配
```

### `op_classification.yaml`

```yaml
aten::matmul:
  category: compute
  flops_formula: "2 * M * N * K"

c10d::allreduce_:
  category: communication
  bytes_formula: "2 * numel * element_size"
```

### `settings.py`

```python
error_thresholds = {
    "float16": (1e-3, 1e-2),
    "bfloat16": (5e-3, 5e-2),
    "float32": (1e-5, 1e-4),
    "float64": (1e-10, 1e-9),
}
```

## 项目结构

```
Analys_Trace/
├── op_testgen/              # 主包
│   ├── cli.py               # 命令行入口（analyze / generate / run / test）
│   ├── generator/           # 测试用例生成与进程内执行
│   │   ├── test_case_generator.py   # .py 文件生成器（模板渲染）
│   │   ├── test_case_runner.py      # InProcessRunner（importlib 加载）
│   │   └── template.py              # 测试文件模板（精简依赖）
│   ├── analyzer/            # Trace 统计分析与层级分析
│   ├── config/              # YAML 配置（白名单/黑名单/分类公式）
│   ├── parser/              # Trace JSON 解析
│   ├── mapper/              # 算子映射与过滤
│   ├── builder/             # 张量与参数重构
│   ├── correctness/         # 正确性测试执行
│   ├── perf/                # 性能测试与分类
│   └── reporter/            # 报告生成（Markdown/HTML/JSON/Excel）
├── tests/                   # 测试套件（73 个用例）
├── docs/                    # 设计文档
├── analys_trace_v4.py       # 原始分析脚本
├── pyproject.toml           # 包配置
└── README.md
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 类型检查
mypy op_testgen/ --ignore-missing-imports

# 格式化
ruff format op_testgen/ tests/

# 代码检查
ruff check op_testgen/ tests/
```

### 添加新算子支持

1. 编辑 `op_testgen/config/op_whitelist.yaml`
2. 如需 FLOPS 公式，编辑 `op_testgen/config/op_classification.yaml`
3. 运行 `pytest tests/test_mapper.py -v` 验证

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1.0 | 2026-05 | 初始版本：Trace 解析、算子映射、正确性/性能测试、报告生成 |
| v0.2.0 | 2026-05 | 测试流程拆分：`generate` + `run` 子命令 |
| v0.2.1 | 2026-05 | `run` 改为进程内执行（`InProcessRunner` / `importlib`），`generate` 精简 op_testgen 依赖（内联数据类，移除 parser/mapper 导入） |
| v0.3.0 | 2026-05 | 层级调用分析：调用树、自耗时、调用链、递归检测 |
| v0.3.1 | 2026-05 | 调用树去重聚合视图：`[×N]` 合并、`[RECURSIVE]` 截断、树形符号 |
| v0.4.0 | 2026-05 | 父子算子层级化：默认仅测试根节点，CSV/Excel 使用层级路径 `root/child1/child2` |

## License

MIT License
