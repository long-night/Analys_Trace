# 父子算子与层级调用分析设计文档

**日期**: 2026-05-24  
**作者**: op_testgen 开发团队  
**状态**: 待审核  

---

## 1. 背景与动机

当前 `op_testgen` 的 Trace 分析基于**平铺的算子列表**：`TraceParser` 解析 Chrome Trace JSON 后输出 `List[OpInfo]`，`TraceAnalyzer` 对每个算子独立聚合统计。这种设计能够准确回答"某个算子总共被调用了多少次、耗时多久"，但无法回答以下关键问题：

- **自耗时 vs 总耗时**：`aten::conv2d` 的 100ms 中，有多少是它自身的计算，有多少是派发给子算子（如 `aten::convolution`）的开销？
- **调用链模式**：模型中频繁出现 `conv2d → convolution → _convolution` 这样的三层调用，其模式频率和耗时分布如何？
- **递归/循环调用**：是否存在算子 A 调用算子 B、B 又回调 A 的循环依赖？
- **入口算子识别**：哪些算子是用户代码直接调用的"根节点"，哪些是内部派发的"叶子节点"？

这些问题对于性能瓶颈定位（区分"自身慢"还是"子算子慢"）和测试生成策略（优先测试高频入口算子）至关重要。

---

## 2. 目标

1. **精确构建调用树**：基于 Chrome Trace 的 B/E 事件对，在单线程内构建完整的父子层级关系。
2. **自耗时计算**：每个算子的 `self_duration = total_duration - sum(children_durations)`，消除子算子耗时污染。
3. **调用链提取**：识别并统计常见的调用路径模式（如 `conv2d → convolution → _convolution`）。
4. **递归检测**：发现循环调用模式（A→B→A 或自递归 A→A）。
5. **向后兼容**：现有平铺流水线（`parse() → analyze() → generate()`）不受影响，层级能力作为**可选扩展**提供。
6. **渐进式测试集成**：短期内层级分析仅用于统计报告，为未来支持"嵌套调用式测试生成"预留接口。

---

## 3. 非目标

- **多线程跨线程父子关系**：Chrome Trace 中不同 `tid` 的事件不存在父子关系，本设计仅处理单线程内的层级。
- **GPU Kernel 层级**：仅处理 CPU 算子（`aten::`, `torch::`, 通信算子）的 B/E 事件，不追踪 CUDA Kernel 的启动关系。
- **动态图/控制流**：不分析 Python 层面的 `if/else`、`for` 循环，仅分析算子调用栈。
- **测试生成重构**：本次设计不改变 `TestCaseGenerator` 的平铺生成逻辑，仅增加分析能力。

---

## 4. 数据模型

### 4.1 OpInfo 增强

为支持层级构建，在 `OpInfo` 中增加**时间戳字段**（向后兼容，使用 keyword args 的现有代码不受影响）：

```python
@dataclass
class OpInfo:
    """算子运行时信息"""
    name: str
    input_dims: List[List[int]] = field(default_factory=list)
    input_strides: List[List[int]] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    concrete_inputs: List[Any] = field(default_factory=list)
    duration_us: float = 0.0
    is_communication: bool = False
    # 新增字段
    start_ts: float = 0.0  # 事件开始时间戳（微秒）
    tid: int = 0           # 线程 ID
```

### 4.2 层级数据模型

新增 `HierarchicalOpInfo` 继承自 `OpInfo`：

```python
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HierarchicalOpInfo(OpInfo):
    """带层级关系的算子信息（继承 OpInfo 保持完全兼容）"""
    
    # 层级关系
    parent: Optional['HierarchicalOpInfo'] = None
    children: List['HierarchicalOpInfo'] = field(default_factory=list)
    depth: int = 0  # 调用深度（根节点=0）
    
    # 标识
    is_root: bool = False  # 是否为根节点（无父节点）
    
    @property
    def self_duration_us(self) -> float:
        """自耗时 = 总耗时 - 所有直接子算子耗时总和"""
        children_total = sum(c.duration_us for c in self.children)
        return max(0.0, self.duration_us - children_total)
    
    @property
    def total_duration_us_recursive(self) -> float:
        """递归总耗时（包含所有后代节点）"""
        return self.duration_us + sum(
            c.total_duration_us_recursive for c in self.children
        )
    
    @property
    def descendant_count(self) -> int:
        """后代节点总数（不含自身）"""
        return len(self.children) + sum(
            c.descendant_count for c in self.children
        )
    
    def get_call_chain(self) -> List[str]:
        """获取从根节点到自身的完整调用链"""
        chain = []
        current: Optional[HierarchicalOpInfo] = self
        while current is not None:
            chain.append(current.name)
            current = current.parent
        return list(reversed(chain))
```

### 4.3 设计决策

- **继承而非包装**：`HierarchicalOpInfo` 继承 `OpInfo`，确保任何接收 `OpInfo` 的函数都能直接处理 `HierarchicalOpInfo`（Liskov 替换原则）。
- **延迟计算**：`self_duration_us`、`total_duration_us_recursive` 等属性为 `@property`，在访问时实时计算，避免解析时的性能开销。
- **可变性**：`children` 列表在树构建后可变，便于后续分析器进行剪枝或重组。

---

## 5. 解析器增强

### 5.1 现有解析流程

当前 `TraceParser.parse()` 输出 `List[OpInfo]`：

```
Chrome Trace JSON
    │
    ▼
TraceParser.parse() ──► List[OpInfo]（平铺）
```

### 5.2 增强后解析流程

新增 `parse_hierarchical()` 方法：

```
Chrome Trace JSON
    │
    ├──► TraceParser.parse() ──► List[OpInfo]（平铺，向后兼容）
    │         │
    │         └──► 每个 OpInfo 包含 start_ts + duration_us + tid
    │
    └──► TraceParser.parse_hierarchical() 
             │
             ├──► _parse_flat_with_ts() ──► List[OpInfo]（含时间戳）
             │
             └──► _build_hierarchy() ──► List[HierarchicalOpInfo]（根节点列表）
```

### 5.3 层级构建算法

```python
def _build_hierarchy(self, flat_ops: List[OpInfo]) -> List[HierarchicalOpInfo]:
    """
    基于时间戳范围重叠构建调用树。
    
    前置条件：flat_ops 中每个 OpInfo 必须包含 start_ts、duration_us、tid
    
    算法（按线程独立处理）：
    1. 按 tid 分组，每组按 start_ts 排序
    2. 维护调用栈（单调递增的 start_ts）
    3. 对于每个算子 op：
       - 当栈顶算子的结束时间 ≤ op.start_ts 时，弹出栈顶（栈顶已结束）
       - 若栈非空，栈顶即为 op 的父节点
       - 将 op 压入栈
    4. 从未被作为子节点的节点即为根节点
    
    时间复杂度：O(N log N)（排序）+ O(N)（栈遍历）= O(N log N)
    空间复杂度：O(N)
    """
    # 实现细节详见实现计划
    ...
```

### 5.4 关键边界条件

| 场景 | 处理策略 |
|------|----------|
| **未匹配的 E 事件**（无对应 B） | 视为根节点（孤儿节点），记录警告 |
| **未匹配的 B 事件**（无对应 E） | 使用文件结束时间作为 E 的时间戳，记录警告 |
| **X 事件**（完整事件，含 dur） | 独立处理，通过时间范围查找父节点（若存在时间重叠的父级 B 事件） |
| **同名算子嵌套**（A→A） | 栈使用 `(name, start_ts)` 作为唯一键，确保正确匹配 |
| **零耗时子算子** | `self_duration` 最小为 0（`max(0, total - children)`） |

---

## 6. 层级分析器（HierarchyAnalyzer）

### 6.1 职责

`HierarchyAnalyzer` 是新增的核心分析模块，专门处理层级调用关系，与现有 `TraceAnalyzer` 互补。

### 6.2 API 设计

```python
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CallChain:
    """调用链模式"""
    chain: Tuple[str, ...]  # 如 ("aten::conv2d", "aten::convolution", "aten::_convolution")
    occurrence_count: int
    total_duration_ms: float
    avg_duration_ms: float


@dataclass
class RecursivePattern:
    """递归/循环模式"""
    cycle: Tuple[str, ...]  # 如 ("aten::add", "aten::mul", "aten::add")
    occurrence_count: int
    depth: int  # 最大递归深度


class HierarchyAnalyzer:
    """层级调用分析器"""
    
    def __init__(self, root_ops: List[HierarchicalOpInfo]):
        self.root_ops = root_ops
        self._all_nodes: Optional[List[HierarchicalOpInfo]] = None
    
    @property
    def all_nodes(self) -> List[HierarchicalOpInfo]:
        """扁平化遍历所有节点（含根和后代）"""
        if self._all_nodes is None:
            self._all_nodes = []
            self._dfs_collect(self.root_ops)
        return self._all_nodes
    
    def _dfs_collect(self, nodes: List[HierarchicalOpInfo]):
        """深度优先收集所有节点"""
        for node in nodes:
            self._all_nodes.append(node)
            self._dfs_collect(node.children)
    
    # ─── 统计方法 ───
    
    def get_self_time_stats(self) -> Dict[str, 'OperatorStats']:
        """基于自耗时的算子统计（排除子算子耗时）"""
        ...
    
    def get_total_time_stats(self) -> Dict[str, 'OperatorStats']:
        """基于总耗时的算子统计（含子算子）"""
        ...
    
    def get_call_chains(self, min_occurrence: int = 2, max_depth: int = 10) -> List[CallChain]:
        """提取高频调用链模式"""
        ...
    
    def find_recursive_patterns(self, max_cycle_length: int = 5) -> List[RecursivePattern]:
        """识别递归/循环调用模式"""
        ...
    
    def get_root_ops_stats(self) -> Dict[str, 'OperatorStats']:
        """根节点（入口算子）统计"""
        ...
    
    def get_depth_distribution(self) -> Dict[int, int]:
        """调用深度分布（depth → 节点数）"""
        ...
    
    # ─── 导出方法 ───
    
    def export_call_tree_text(self, output_file: str, max_depth: int = 10):
        """导出文本格式的调用树"""
        ...
    
    def export_call_tree_json(self, output_file: str):
        """导出 JSON 格式的调用树（含完整层级）"""
        ...
```

### 6.3 统计指标

#### 自耗时排名（关键指标）

```
【自耗时最高的算子 TOP 10】（排除子算子影响）
  1. aten::conv2d          自耗时: 45.2 ms (总耗时: 100.1 ms, 子算子: 54.9 ms)
  2. aten::matmul          自耗时: 38.7 ms (总耗时: 38.7 ms, 无子算子)
  ...
```

#### 调用链分析

```
【高频调用链 TOP 5】
  1. conv2d → convolution → _convolution (出现 128 次, 平均耗时: 15.3 ms)
  2. linear → addmm → mm (出现 64 次, 平均耗时: 8.7 ms)
  ...
```

#### 递归检测

```
【检测到的循环调用模式】
  1. add → mul → add (出现 32 次, 最大深度: 8)
     警告: 可能存在不必要的递归计算
```

#### 深度分布

```
【调用深度分布】
  深度 0 (根): 15 个算子
  深度 1:     42 个算子
  深度 2:     89 个算子
  深度 3:     156 个算子
  深度 4+:    203 个算子
```

---

## 7. TraceAnalyzer 扩展

现有 `TraceAnalyzer` 增加层级视图支持，保持所有现有方法不变：

```python
class TraceAnalyzer:
    def __init__(self, op_infos: List[OpInfo]):
        self.op_infos = op_infos
        self.operators: Dict[str, OperatorStats] = {}
        self._hierarchy: Optional[List[HierarchicalOpInfo]] = None
        self._hierarchy_analyzer: Optional[HierarchyAnalyzer] = None
        self._analyze()  # 现有平铺分析
    
    # ─── 现有方法保持不变 ───
    # print_summary(), export_to_csv(), export_to_excel() ...
    
    # ─── 新增层级方法 ───
    
    def _ensure_hierarchy(self):
        """懒加载层级数据"""
        if self._hierarchy is None:
            parser = TraceParser("")  # 复用构建逻辑
            self._hierarchy = parser._build_hierarchy(self.op_infos)
            self._hierarchy_analyzer = HierarchyAnalyzer(self._hierarchy)
    
    def print_hierarchical_summary(self):
        """打印层级统计摘要"""
        self._ensure_hierarchy()
        # 输出自耗时排名、调用链、递归检测、深度分布
        ...
    
    def export_hierarchy_to_csv(self, output_file: str):
        """导出层级统计到 CSV"""
        self._ensure_hierarchy()
        ...
    
    def export_hierarchy_to_excel(self, output_file: str) -> bool:
        """导出层级统计到 Excel（新增工作表）"""
        self._ensure_hierarchy()
        # 新增工作表：
        # - "自耗时排名"
        # - "调用链分析"  
        # - "层级结构"
        ...
```

---

## 8. CLI 扩展

### 8.1 `analyze` 子命令新增选项

```bash
# 现有用法（完全不受影响）
op_testgen analyze trace.json

# 新增层级分析选项
op_testgen analyze trace.json --hierarchy              # 启用层级分析并输出摘要
op_testgen analyze trace.json --self-time-top 20       # 自耗时 TOP N（默认 10）
op_testgen analyze trace.json --call-chains            # 输出高频调用链
op_testgen analyze trace.json --detect-recursion       # 检测递归模式
op_testgen analyze trace.json --export-tree tree.txt   # 导出文本调用树

# 输出格式扩展
op_testgen analyze trace.json -o report --hierarchy    # Excel/CSV 包含层级工作表
```

### 8.2 参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hierarchy` | flag | False | 启用层级分析 |
| `--self-time-top` | int | 10 | 自耗时排名数量 |
| `--call-chains` | flag | False | 输出调用链分析 |
| `--detect-recursion` | flag | False | 检测递归/循环模式 |
| `--export-tree` | str | None | 导出调用树到文件 |
| `--max-chain-depth` | int | 10 | 调用链最大深度 |
| `--min-chain-occurrence` | int | 2 | 调用链最小出现次数阈值 |

---

## 9. 向后兼容性

### 9.1 零破坏保证

| 组件 | 变更类型 | 兼容性 |
|------|----------|--------|
| `TraceParser.parse()` | 无变更 | ✅ 100% 兼容 |
| `OpInfo` 数据类 | 无变更 | ✅ 100% 兼容 |
| `TraceAnalyzer` 现有方法 | 无变更 | ✅ 100% 兼容 |
| `TestCaseGenerator` | 无变更 | ✅ 100% 兼容 |
| CLI `analyze` 默认行为 | 无变更 | ✅ 100% 兼容 |
| 现有测试套件 | 无变更 | ✅ 全部通过 |

### 9.2 渐进式采用

用户无需任何迁移即可使用新版本。层级功能完全可选：

- **不启用**：行为与 v0.2.0 完全一致
- **启用**：通过 `--hierarchy` 等参数访问新功能
- **混合**：平铺报告和层级报告可同时输出

---

## 10. 测试策略

### 10.1 单元测试

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_hierarchy_parser.py` | `_build_hierarchy()` 算法正确性（简单嵌套、深层嵌套、未匹配事件） |
| `tests/test_hierarchy_analyzer.py` | `HierarchyAnalyzer` 所有统计方法 |
| `tests/test_self_time.py` | 自耗时计算边界条件（零耗时子算子、单节点、深层树） |
| `tests/test_call_chains.py` | 调用链提取（简单链、分支、重复模式） |
| `tests/test_recursion_detect.py` | 递归检测（自递归、双向循环、多节点循环） |

### 10.2 集成测试

- **端到端测试**：`op_testgen analyze profiler_trace.json --hierarchy` 完整流程
- **向后兼容测试**：不带 `--hierarchy` 时输出与 v0.2.0 完全一致
- **大规模测试**：使用 `George_*.pt.trace.json`（150MB+）验证性能和内存

### 10.3 性能基准

| 指标 | 目标 |
|------|------|
| 层级构建耗时 | < 平铺解析耗时的 50% |
| 内存开销 | < 平铺数据内存的 30%（仅增加 parent/children 指针） |
| 百万级事件处理 | 能在 30 秒内完成层级构建 |

---

## 11. 未来扩展

### 11.1 短期（v0.3.x）

- [ ] 实现本设计文档中的所有功能
- [ ] 支持导出调用树为 Graphviz DOT 格式（可视化）
- [ ] CLI 支持 `--format tree` 文本树输出

### 11.2 中期（v0.4.x）

- [ ] **嵌套测试生成**：`TestCaseGenerator` 支持生成模拟原始调用层级的测试代码
  ```python
  # 生成的测试代码示例
  def test_conv2d_with_children():
      x = torch.randn(1, 3, 224, 224)
      with profiler.record_function("aten::conv2d"):
          y = F.conv2d(x, ...)  # 父算子
          # 内部自动派发子算子（convolution, _convolution）
  ```
- [ ] **调用链模板**：识别高频调用链后，生成针对整条链的集成测试

### 11.3 长期（v0.5.x）

- [ ] **跨线程分析**：利用 Chrome Trace 的 `flow` 事件追踪跨线程的算子依赖
- [ ] **时间线可视化**：生成 HTML 时间线，可展开/折叠调用树
- [ ] **智能根节点推荐**：基于调用频率和自耗时，推荐最值得测试的入口算子

---

## 12. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Chrome Trace B/E 事件不严格嵌套 | 中 | 高 | 增加鲁棒性处理（未匹配事件警告+兜底逻辑） |
| 层级构建内存占用过高 | 低 | 中 | 使用生成器延迟加载，支持大文件分块处理 |
| 自耗时计算为负（精度问题） | 低 | 中 | `max(0, self_duration)` 兜底，记录警告 |
| 现有用户依赖平铺输出格式 | 低 | 高 | 保证默认行为不变，层级功能完全可选 |

---

## 13. 附录

### 13.1 术语表

| 术语 | 定义 |
|------|------|
| **自耗时（Self Time）** | 算子总耗时减去所有直接子算子耗时之和，反映算子自身的净开销 |
| **总耗时（Total Time）** | 算子从 B 到 E 的完整时间跨度（含子算子） |
| **根节点（Root）** | 调用树中无父节点的算子，通常是用户代码直接调用的入口 |
| **调用链（Call Chain）** | 从根节点到叶子节点的完整路径，如 `conv2d → convolution → _convolution` |
| **调用深度（Depth）** | 算子在调用树中的层级，根节点 depth=0 |

### 13.2 参考实现

- PyTorch Profiler Chrome Trace 格式规范：[https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)（内部格式）
- Chrome Trace Event Format：[https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)

---

**审核意见栏**

| 审核人 | 日期 | 意见 | 状态 |
|--------|------|------|------|
| | | | |

---

*本文档遵循 op_testgen 项目的设计规范。如有疑问，请联系开发团队。*
