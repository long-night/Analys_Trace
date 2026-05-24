# 父子算子（递归调用）层级化设计文档

**日期**: 2026-05-24  
**作者**: Sisyphus  
**状态**: 已确认（方案2：统一层级化视图）  
**相关模块**: parser, cli, analyzer, generator  

---

## 1. 背景与动机

当前 `op_testgen` 的 trace 分析以**平面视角**处理所有算子，每个算子独立统计、独立测试。然而，PyTorch Profiler Chrome Trace 天然包含**调用层级关系**（B/E 事件的时间戳重叠可构建调用树）。

引入父子算子关系可以带来两个核心价值：

1. **减少测试用例数量**：默认只测试根节点（入口算子），子算子在根算子执行过程中被递归调用，从而间接完成测试
2. **增强可追溯性**：CSV/Excel 输出中引入层级路径，用户可直观理解算子在整个调用链中的位置

---

## 2. 设计目标

| 目标 | 优先级 | 说明 |
|------|--------|------|
| 默认仅测试根节点 | P0 | `test` / `generate` 子命令默认只选择 `is_root=True` 的算子 |
| 算子名称层级化 | P0 | CSV/Excel 中使用层级路径格式 `root/child1/child2` |
| 向后兼容 | P1 | `analys_trace_v4.py` 保持原有行为不变 |
| 摘要信息增强 | P1 | 统计信息按根节点聚合，展示层级深度分布 |

---

## 3. 方案概述：统一层级化视图（方案2）

**核心原则**：默认进入层级化模式，所有算子统一使用层级路径格式呈现。

### 3.1 与现有层级分析的关系

项目已具备成熟的层级分析基础设施：
- `TraceParser.parse_hierarchical()`：构建 `HierarchicalOpInfo` 调用树
- `HierarchyAnalyzer`：自耗时计算、递归检测、去重聚合

本设计**复用**上述基础设施，将层级化从「可选分析功能」提升为「默认数据视图」。

---

## 4. 模块详细设计

### 4.1 算子名称层级化

**变更文件**: `op_testgen/parser/trace_parser.py`

#### 4.1.1 HierarchicalOpInfo 增强

```python
@dataclass
class HierarchicalOpInfo(OpInfo):
    parent: Optional['HierarchicalOpInfo'] = None
    children: List['HierarchicalOpInfo'] = field(default_factory=list)
    depth: int = 0
    is_root: bool = False

    @property
    def hierarchical_name(self) -> str:
        """完整层级路径，如 aten::convolution_backward/aten::contiguous"""
        return "/".join(self.get_call_chain())
```

**说明**：
- `name` 保持原始算子名（用于 `OpMapper` 映射到 Python callable）
- `hierarchical_name` 为新增属性，用于展示和去重

#### 4.1.2 TraceParser 默认返回层级节点

```python
def parse(self) -> List[HierarchicalOpInfo]:
    """解析 trace 并返回层级节点列表（默认行为）"""
    flat_ops = self._parse_flat()  # 原 parse() 逻辑
    return self._build_hierarchy(flat_ops)
```

### 4.2 仅测试根节点（默认开启）

**变更文件**: `op_testgen/cli.py`

#### 4.2.1 新增 CLI 参数

| 参数 | 子命令 | 类型 | 默认值 | 说明 |
|------|--------|------|--------|------|
| `--test-all-ops` | `test`, `generate` | flag | False | 关闭仅根节点模式，测试所有算子 |

#### 4.2.2 _prepare_test_cases 过滤逻辑

```python
def _prepare_test_cases(args) -> tuple:
    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse()  # 返回 HierarchicalOpInfo 列表
    
    # 默认仅保留根节点
    if not getattr(args, 'test_all_ops', False):
        op_infos = [op for op in op_infos if op.is_root]
    
    # ... 后续映射、去重逻辑不变
```

**效果**：
- 默认情况下，测试用例数量 = 根节点数量（通常为总算子数的 5%~20%）
- 根算子执行时会递归调用子算子，间接完成子算子测试

### 4.3 CSV/Excel 层级化输出

**变更文件**: `op_testgen/analyzer/trace_analyzer.py`

#### 4.3.1 算子总表新增列

| 列名 | 数据类型 | 说明 |
|------|----------|------|
| 算子名称 | str | 层级路径 `root/child1/child2` |
| **深度** | int | **新增**：该算子在调用树中的层级（根节点=0） |
| **根算子** | str | **新增**：该算子所属的根节点名称 |
| 输入Shapes | str | 保持不变 |
| 输入Strides | str | 保持不变 |
| 输入数据类型 | str | 保持不变 |
| Concrete Inputs | str | 保持不变 |
| 调用次数 | int | 保持不变 |
| 总执行时间(ms) | float | 保持不变 |
| 平均执行时间(ms) | float | 保持不变 |

#### 4.3.2 Shape 统计表新增列

| 列名 | 数据类型 | 说明 |
|------|----------|------|
| 算子名称 | str | 层级路径 |
| **深度** | int | **新增** |
| **根算子** | str | **新增** |
| ... | ... | 其他列保持不变 |

#### 4.3.3 去重逻辑调整

去重 key 从 `(name, dims, strides, types, concrete)` 扩展为 `(hierarchical_name, dims, strides, types, concrete)`，确保同一算子在**不同调用链中**被视为不同测试用例。

**示例**：
- `aten::convolution_backward/aten::contiguous` → shape A
- `aten::convolution_backward/aten::empty` → shape B

这两个层级路径不同，即使底层算子名相同，也作为不同测试用例。

### 4.4 分析摘要层级化

**变更文件**: `op_testgen/analyzer/trace_analyzer.py`

#### 4.4.1 TOP 排名按根节点聚合

```python
# 按根节点聚合算子统计
def get_root_aggregated_stats(operators: List[OperatorInfo]) -> Dict[str, OperatorStats]:
    """将层级路径按根节点聚合"""
    root_stats: Dict[str, OperatorStats] = {}
    for op in operators:
        root_name = op.name.split('/')[0]
        if root_name not in root_stats:
            root_stats[root_name] = OperatorStats(root_name)
        root_stats[root_name].call_count += op.call_count
        root_stats[root_name].total_duration_us += op.total_duration_us
    return root_stats
```

#### 4.4.2 新增统计维度

```
【层级统计】
  根节点数: X
  总层级路径数: Y
  平均深度: Z

【最耗时的根算子 TOP 10】
  1. aten::convolution_backward: 2177.91 ms (包含 5 个子算子)
  2. aten::matmul: 1200.50 ms (包含 2 个子算子)
  ...

【调用深度分布】
  深度 0: 120 个
  深度 1: 350 个
  深度 2: 200 个
  深度 3: 50 个
```

### 4.5 向后兼容

| 组件 | 兼容策略 |
|------|----------|
| `analys_trace_v4.py` | **不改动**。保持原有平面分析行为，作为独立脚本继续可用 |
| `op_testgen analyze` | 默认层级化，新增 `--flat` 参数可切回平面模式 |
| 现有测试脚本 | `test` / `generate` 默认行为改变，用户可通过 `--test-all-ops` 恢复旧行为 |
| Excel/CSV 格式 | 新增列不影响旧列顺序，旧列保持原有位置 |

---

## 5. 数据流图

```
profiler_trace.json
       │
       ▼
┌──────────────────┐
│   TraceParser    │  ──► parse() 返回 HierarchicalOpInfo[]
│  (parse 默认层级) │      hierarchical_name = "root/child1/child2"
└──────────────────┘
       │
       ├──► 分支1: analyze 子命令
       │           │
       │           ▼
       │     ┌──────────────┐
       │     │ TraceAnalyzer│  ──► CSV/Excel 含层级路径 + 深度 + 根算子
       │     │  (层级化输出) │
       │     └──────────────┘
       │
       └──► 分支2: test/generate 子命令
                   │
                   ▼
             ┌──────────────┐
             │ 根节点过滤    │  ──► 默认仅保留 is_root=True
             │ (默认开启)    │
             └──────────────┘
                   │
                   ▼
             ┌──────────────┐
             │   OpMapper   │  ──► 层级路径不参与映射，name 属性映射到 callable
             └──────────────┘
```

---

## 6. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 根节点数量过少导致测试覆盖不足 | 中 | 高 | 提供 `--test-all-ops` 显式恢复全量测试 |
| 层级路径过长导致 CSV 可读性差 | 低 | 中 | 新增「根算子」列辅助分组；Excel 中支持筛选 |
| 去重 key 变长导致性能下降 | 低 | 低 | 仅影响内存占用，无显著性能瓶颈 |
| 与现有 `analys_trace_v4.py` 行为冲突 | 低 | 低 | 独立脚本不改动，op_testgen CLI 与旧脚本共存 |

---

## 7. 验收标准

- [ ] `op_testgen test trace.json` 默认只测试根节点，测试用例数量 < 总算子数
- [ ] `op_testgen test trace.json --test-all-ops` 测试所有算子（与旧行为一致）
- [ ] `op_testgen generate trace.json -o test.py` 生成的测试文件只包含根节点
- [ ] CSV 算子总表中算子名称显示为层级路径，新增「深度」和「根算子」列
- [ ] Excel 算子总表与 CSV 格式一致
- [ ] 分析摘要新增「层级统计」和「调用深度分布」
- [ ] `analys_trace_v4.py` 行为保持不变
- [ ] 所有现有测试通过（`pytest tests/`）

---

## 8. 附录：术语表

| 术语 | 定义 |
|------|------|
| 根节点 (Root) | 调用树中没有父节点的算子，即顶层入口算子 |
| 层级路径 (Hierarchical Name) | 从根节点到当前节点的完整调用链，用 `/` 分隔 |
| 自耗时 (Self Time) | 算子总耗时减去所有直接子算子耗时 |
| 去重 (Deduplication) | 按 (算子标识, 输入参数) 去重，避免重复测试相同用例 |

---

*文档版本: v1.0*  
*最后更新: 2026-05-24*
