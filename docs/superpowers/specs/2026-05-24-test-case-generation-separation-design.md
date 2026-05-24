# 测试用例生成与执行分离设计

## 背景与目标

当前 `op_testgen test` 子命令将"测试用例生成"和"测试执行"耦合在一起：解析 Trace → 去重映射 → 构建张量 → 立即执行正确性/性能测试 → 输出报告。用户希望将这一过程拆分为两个独立阶段：

1. **生成阶段**：从 Trace 文件生成可独立运行的 Python 测试文件（`.py`）
2. **执行阶段**：运行已生成的 Python 测试文件，输出报告

同时保留现有的 `test` 子命令作为快捷方式（一步到位）。

## 方案选择：方案 C（混合模式）

经过对比，选择**方案 C（混合模式）**：生成单个自包含 `.py` 文件，测试数据以 Python 字面量内联，重建逻辑通过调用 `op_testgen` 包的公共 API 实现。

**决策理由**：
- 单文件即可运行，符合直觉
- 复用现有 `TensorBuilder`、`CorrectnessRunner`、`PerfBenchmark`，无需重写核心逻辑
- 实现复杂度适中，改动面可控
- 支持分离模式和快捷模式两种工作流

## 新增模块

### 1. `op_testgen/generator/__init__.py`

包初始化文件。

### 2. `op_testgen/generator/test_case_generator.py`

**职责**：将 `MappedOp` + `TestCase` 序列化为可自包含的 Python 源文件。

**核心类**：
- `TestCaseGenerator`
  - `generate(mapped_ops, output_path, **options)`：生成 `.py` 文件
  - `_serialize_test_case(mapped_op)`：将单个算子序列化为 dict
  - `_render_template(data, options)`：渲染模板

**序列化数据格式**（Python 字面量）：
```python
TEST_CASES_DATA = [
    {
        "op_name": "aten::add",
        "callable_path": "torch.add",
        "input_dims": [[3, 4], [3, 4]],
        "input_strides": [[4, 1], [4, 1]],
        "input_types": ["float", "float"],
        "concrete_inputs": [],
    },
    # ...
]
```

**模板结构**：
```python
# 头部注释（来源 trace、生成时间、命令参数）
# import 语句
# TEST_CASES_DATA 常量
# build_test_cases() 函数：使用 TensorBuilder 重建 TestCase 列表
# main() 函数：解析参数 → 重建 → 执行 → 输出报告
```

### 3. `op_testgen/generator/template.py`

**职责**：存放测试文件模板字符串。

**模板变量**：
- `{{ header_comment }}`：文件头注释
- `{{ imports }}`：import 语句
- `{{ test_cases_data }}`：序列化后的测试数据
- `{{ cli_options }}`：命令行参数默认值

### 4. `op_testgen/generator/test_case_runner.py`

**职责**：供 `run` 子命令调用，加载并执行生成的 `.py` 文件。

**核心类**：
- `TestCaseRunner`
  - `run(file_path, **options)`：执行生成的测试文件
  - 支持 `--backend`、`--only-correctness`、`--only-performance`、`--iters`、`--fail-fast` 等参数

**执行方式**：使用 `subprocess.run` 调用 `python <file_path>`，传递参数。

## CLI 扩展

### 新增 `generate` 子命令

```bash
op_testgen generate profiler_trace.json -o tests/test_add.py --backend cuda --max-ops 50
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trace_file` | Trace 文件路径 | *(必填)* |
| `-o, --output` | 输出 .py 文件路径 | *(必填)* |
| `--backend` | 默认后端（写入文件头部注释和默认值） | `cuda` |
| `--seed` | 随机种子 | `42` |
| `--max-ops` | 最大算子数 | `100` |
| `--op-filter` | 算子名称过滤 | *(无)* |
| `--only-correctness` | 默认仅正确性测试 | `False` |
| `--only-performance` | 默认仅性能测试 | `False` |
| `--iters` | 性能测试迭代次数 | `10` |

### 新增 `run` 子命令

```bash
op_testgen run tests/test_add.py --backend cuda --only-correctness
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `test_file` | 生成的 .py 文件路径 | *(必填)* |
| `--backend` | 后端（覆盖文件默认值） | 文件内默认值 |
| `--only-correctness` | 仅正确性测试 | 文件内默认值 |
| `--only-performance` | 仅性能测试 | 文件内默认值 |
| `--iters` | 性能测试迭代次数 | 文件内默认值 |
| `--fail-fast` | 第一个失败即停止 | `False` |

### 保留 `test` 子命令（快捷方式）

```bash
op_testgen test profiler_trace.json --backend cuda --max-ops 50
```

行为不变：解析 → 映射 → 去重 → 构建 → 直接执行 → 输出报告（不写中间文件）。

## 数据流

### 分离模式（generate + run）

```
profiler_trace.json
       │
       ▼
┌──────────────┐
│ TraceParser  │ ──► 解析 JSON，提取 OpInfo
└──────────────┘
       │
       ▼
┌──────────────┐
│   OpMapper   │ ──► 名称映射到 Python callable，黑名单过滤
└──────────────┘
       │
       ▼
┌─────────────────────┐
│ TestCaseGenerator   │ ──► 序列化数据 + 渲染模板 ──► test_add.py
└─────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ TestRunner   │ ──► python test_add.py ──► 报告
                       └──────────────┘
```

### 快捷模式（test）

```
profiler_trace.json
       │
       ▼
┌──────────────┐
│ TraceParser  │
└──────────────┘
       │
       ▼
┌──────────────┐
│   OpMapper   │
└──────────────┘
       │
       ▼
┌──────────────┐
│ TensorBuilder│ ──► 直接构建 TestCase 列表
└──────────────┘
       │
       ├──► ┌─────────────────┐ ──► 正确性报告
       │    │ CorrectnessRunner│
       │    └─────────────────┘
       │
       └──► ┌───────────────┐ ──► 性能报告
            │ PerfBenchmark  │
            └───────────────┘
```

## 关键约束与假设

1. **生成的文件依赖 `op_testgen` 包**：假设目标环境已安装 `op_testgen`（通过 `pip install -e .` 或其他方式）
2. **向后兼容**：`test` 子命令的行为和参数保持不变
3. **模板渲染**：使用 Python 字符串模板（无需引入 Jinja2 等额外依赖）
4. **序列化安全**：所有内联数据均为 JSON-safe 的 Python 字面量（str, int, float, list, dict, bool, None）
5. **路径处理**：`callable_path` 使用 `"torch.add"` 格式，运行时通过 `OpMapper` 动态导入

## 文件变更清单

### 新增文件
- `op_testgen/generator/__init__.py`
- `op_testgen/generator/test_case_generator.py`
- `op_testgen/generator/template.py`
- `op_testgen/generator/test_case_runner.py`

### 修改文件
- `op_testgen/cli.py`：新增 `generate` 和 `run` 子命令
- `op_testgen/mapper/op_mapper.py`：可能需要暴露 `resolve_callable` 工具函数供模板使用
- `tests/`：新增 `test_generator.py` 测试

## 测试策略

1. **单元测试**：`test_generator.py`
   - 测试序列化/反序列化一致性
   - 测试模板渲染输出
   - 测试生成文件可语法解析（`ast.parse`）

2. **集成测试**：
   - `generate` + `run` 端到端流程
   - 生成文件实际可执行并输出报告
   - `test` 子命令行为不变

3. **验证命令**：
   ```bash
   python run.py generate profiler_trace.json -o /tmp/test_generated.py
   python run.py run /tmp/test_generated.py --only-correctness
   ```

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 生成的文件过大 | 单个 trace 可能有数百个算子 | 支持 `--max-ops` 限制；去重后通常 < 100 个 |
| callable_path 解析失败 | 运行时无法导入 callable | 模板中使用 try/except 包装，失败时打印警告并跳过 |
| 模板维护成本 | 模板变更需同步 | 模板与 generator 放在同一目录，单元测试覆盖渲染输出 |
| `test` 子命令行为漂移 | 用户依赖现有行为 | `test` 子命令逻辑不变，仅提取公共函数供 generator 复用 |

## 实现计划

1. 提取 `cmd_test` 中的公共逻辑（解析 → 映射 → 去重 → 构建）到独立函数
2. 实现 `TestCaseGenerator` 和模板渲染
3. 实现 `TestCaseRunner`
4. 扩展 CLI，新增 `generate` 和 `run` 子命令
5. 编写单元测试和集成测试
6. 验证现有 `test` 子命令行为不变
