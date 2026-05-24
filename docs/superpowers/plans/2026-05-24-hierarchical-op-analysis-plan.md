# 父子算子与层级调用分析实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 op_testgen 增加层级调用分析能力，支持自耗时计算、调用链提取、递归检测，同时保持100%向后兼容。

**Architecture:** 采用"混合架构"（方案C）：保留现有平铺流水线，新增独立层级构建器和分析器。通过 `HierarchicalOpInfo` 继承 `OpInfo` 保持兼容，所有层级功能通过 `--hierarchy` 等CLI参数显式启用。

**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|------|
| `op_testgen/parser/trace_parser.py` | 解析器：增强 OpInfo 时间戳，新增 `_build_hierarchy()` | 修改 |
| `op_testgen/analyzer/trace_analyzer.py` | 分析器：新增 `_ensure_hierarchy()` 和层级导出方法 | 修改 |
| `op_testgen/analyzer/hierarchy_analyzer.py` | **新增**：层级调用分析器 `HierarchyAnalyzer` | 创建 |
| `op_testgen/cli.py` | CLI：`analyze` 子命令新增层级参数 | 修改 |
| `tests/test_hierarchy_parser.py` | 层级构建算法测试 | 创建 |
| `tests/test_hierarchy_analyzer.py` | 层级分析器统计方法测试 | 创建 |
| `tests/test_self_time.py` | 自耗时计算边界条件测试 | 创建 |
| `tests/test_call_chains.py` | 调用链提取测试 | 创建 |
| `tests/test_recursion_detect.py` | 递归检测测试 | 创建 |

---

## Task 1: 数据模型增强（OpInfo + HierarchicalOpInfo）

**Files:**
- Modify: `op_testgen/parser/trace_parser.py:9-21`
- Test: `tests/test_hierarchy_parser.py`（将在 Task 2 创建，此处仅验证类型）

- [ ] **Step 1: 修改 OpInfo 增加时间戳字段**

修改 `op_testgen/parser/trace_parser.py` 中的 `OpInfo`：

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
    # 新增：层级分析必需字段
    start_ts: float = 0.0
    tid: int = 0

    def __repr__(self) -> str:
        return f"OpInfo(name={self.name}, dims={self.input_dims}, types={self.input_types}, start_ts={self.start_ts})"
```

- [ ] **Step 2: 在 `trace_parser.py` 末尾新增 HierarchicalOpInfo**

在 `op_testgen/parser/trace_parser.py` 文件末尾（`TraceParser` 类之后）添加：

```python

@dataclass
class HierarchicalOpInfo(OpInfo):
    """带层级关系的算子信息"""
    
    parent: Optional['HierarchicalOpInfo'] = None
    children: List['HierarchicalOpInfo'] = field(default_factory=list)
    depth: int = 0
    is_root: bool = False
    
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

- [ ] **Step 3: 运行现有测试确保向后兼容**

Run: `pytest tests/test_parser.py -v`
Expected: ALL PASS（OpInfo 新增字段有默认值，不影响现有构造）

- [ ] **Step 4: Commit**

```bash
git add op_testgen/parser/trace_parser.py
git commit -m "feat(parser): add start_ts/tid to OpInfo and HierarchicalOpInfo dataclass"
```

---

## Task 2: TraceParser 时间戳收集

**Files:**
- Modify: `op_testgen/parser/trace_parser.py:97-157`
- Test: `tests/test_parser.py`

- [ ] **Step 1: 修改 parse() 方法收集 start_ts 和 tid**

修改 `op_testgen/parser/trace_parser.py` 中 `parse()` 方法的 B/E/X 事件处理：

**B 事件处理（约第115-120行）**，将：
```python
pending[tid][name] = (ts, shapes, strides, types, concrete)
```
改为：
```python
pending[tid][name] = (ts, shapes, strides, types, concrete)
```
（保持原样，但需要记录 tid）

**E 事件处理（约第122-137行）**，将：
```python
op = OpInfo(
    name=name,
    input_dims=shapes,
    input_strides=strides,
    input_types=types,
    concrete_inputs=concrete,
    duration_us=duration,
    is_communication=self._is_communication(name),
)
```
改为：
```python
op = OpInfo(
    name=name,
    input_dims=shapes,
    input_strides=strides,
    input_types=types,
    concrete_inputs=concrete,
    duration_us=duration,
    is_communication=self._is_communication(name),
    start_ts=start_ts,
    tid=tid,
)
```

**X 事件处理（约第139-155行）**，将：
```python
op = OpInfo(
    name=name,
    input_dims=shapes,
    input_strides=strides,
    input_types=types,
    concrete_inputs=concrete,
    duration_us=duration,
    is_communication=self._is_communication(name),
)
```
改为：
```python
op = OpInfo(
    name=name,
    input_dims=shapes,
    input_strides=strides,
    input_types=types,
    concrete_inputs=concrete,
    duration_us=duration,
    is_communication=self._is_communication(name),
    start_ts=event.get("ts", 0),
    tid=tid,
)
```

- [ ] **Step 2: 运行现有测试**

Run: `pytest tests/test_parser.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add op_testgen/parser/trace_parser.py
git commit -m "feat(parser): collect start_ts and tid for hierarchy analysis"
```

---

## Task 3: 层级构建算法（_build_hierarchy）

**Files:**
- Modify: `op_testgen/parser/trace_parser.py`
- Create: `tests/test_hierarchy_parser.py`

- [ ] **Step 1: 在 TraceParser 中新增 _build_hierarchy 方法**

在 `TraceParser` 类中 `parse()` 方法之后添加：

```python
    def _build_hierarchy(self, flat_ops: List[OpInfo]) -> List[HierarchicalOpInfo]:
        """基于时间戳范围重叠构建调用树
        
        前置条件：flat_ops 中每个 OpInfo 必须包含 start_ts、duration_us、tid
        
        算法（按线程独立处理）：
        1. 按 tid 分组，每组按 start_ts 排序
        2. 维护调用栈（单调递增的 start_ts）
        3. 对于每个算子 op：
           - 当栈顶算子的结束时间 <= op.start_ts 时，弹出栈顶（栈顶已结束）
           - 若栈非空，栈顶即为 op 的父节点
           - 将 op 压入栈
        4. 从未被作为子节点的节点即为根节点
        """
        from collections import defaultdict
        
        # 按 tid 分组
        by_tid: Dict[int, List[OpInfo]] = defaultdict(list)
        for op in flat_ops:
            by_tid[op.tid].append(op)
        
        # 转换并构建层级
        node_map: Dict[int, HierarchicalOpInfo] = {}  # id(op) -> HierarchicalOpInfo
        all_roots: List[HierarchicalOpInfo] = []
        
        for tid, ops in by_tid.items():
            # 按 start_ts 排序
            ops.sort(key=lambda o: o.start_ts)
            
            stack: List[HierarchicalOpInfo] = []
            roots_for_tid: List[HierarchicalOpInfo] = []
            
            for op in ops:
                end_ts = op.start_ts + op.duration_us
                
                # 弹出已结束的节点
                while stack and (stack[-1].start_ts + stack[-1].duration_us) <= op.start_ts:
                    stack.pop()
                
                # 创建层级节点
                h_op = HierarchicalOpInfo(
                    name=op.name,
                    input_dims=op.input_dims,
                    input_strides=op.input_strides,
                    input_types=op.input_types,
                    concrete_inputs=op.concrete_inputs,
                    duration_us=op.duration_us,
                    is_communication=op.is_communication,
                    start_ts=op.start_ts,
                    tid=op.tid,
                )
                node_map[id(op)] = h_op
                
                if stack:
                    # 有父节点
                    parent = stack[-1]
                    h_op.parent = parent
                    h_op.depth = parent.depth + 1
                    parent.children.append(h_op)
                else:
                    # 根节点
                    h_op.is_root = True
                    roots_for_tid.append(h_op)
                
                stack.append(h_op)
            
            all_roots.extend(roots_for_tid)
        
        return all_roots
```

- [ ] **Step 2: 在 TraceParser 中新增 parse_hierarchical 方法**

在 `_build_hierarchy` 之后添加：

```python
    def parse_hierarchical(self) -> List[HierarchicalOpInfo]:
        """解析 trace 并返回带层级关系的根节点列表"""
        flat_ops = self.parse()
        return self._build_hierarchy(flat_ops)
```

- [ ] **Step 3: 创建测试文件 `tests/test_hierarchy_parser.py`**

```python
"""层级构建算法测试"""
import pytest
from op_testgen.parser.trace_parser import OpInfo, HierarchicalOpInfo, TraceParser


class TestBuildHierarchy:
    """测试 _build_hierarchy 算法"""
    
    def test_simple_nested(self):
        """简单嵌套：parent -> child"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="parent", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="child", start_ts=10, duration_us=50, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        assert len(roots) == 1
        assert roots[0].name == "parent"
        assert roots[0].is_root is True
        assert len(roots[0].children) == 1
        assert roots[0].children[0].name == "child"
        assert roots[0].children[0].parent == roots[0]
        assert roots[0].children[0].depth == 1
    
    def test_deep_nesting(self):
        """深层嵌套：A -> B -> C"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=10, duration_us=80, tid=1),
            OpInfo(name="C", start_ts=20, duration_us=60, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        assert roots[0].name == "A"
        assert roots[0].depth == 0
        assert roots[0].children[0].name == "B"
        assert roots[0].children[0].depth == 1
        assert roots[0].children[0].children[0].name == "C"
        assert roots[0].children[0].children[0].depth == 2
    
    def test_sibling_ops(self):
        """同级算子：A -> [B, C]（顺序执行）"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=10, duration_us=20, tid=1),
            OpInfo(name="C", start_ts=40, duration_us=20, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        assert len(roots) == 1
        assert roots[0].name == "A"
        assert len(roots[0].children) == 2
        assert roots[0].children[0].name == "B"
        assert roots[0].children[1].name == "C"
        # B 和 C 同级
        assert roots[0].children[0].parent == roots[0]
        assert roots[0].children[1].parent == roots[0]
    
    def test_multi_tid(self):
        """多线程独立处理"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="B", start_ts=0, duration_us=50, tid=2),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        assert len(roots) == 2
        assert all(r.is_root for r in roots)
    
    def test_unmatched_events(self):
        """未匹配事件（孤儿节点）"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="orphan", start_ts=50, duration_us=10, tid=1),
            OpInfo(name="parent", start_ts=0, duration_us=100, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        # 孤儿节点也成为根节点
        assert len(roots) == 2
        root_names = {r.name for r in roots}
        assert root_names == {"orphan", "parent"}
    
    def test_self_nested(self):
        """同名算子嵌套：A -> A"""
        parser = TraceParser("")
        flat_ops = [
            OpInfo(name="A", start_ts=0, duration_us=100, tid=1),
            OpInfo(name="A", start_ts=10, duration_us=50, tid=1),
        ]
        roots = parser._build_hierarchy(flat_ops)
        
        assert roots[0].name == "A"
        assert roots[0].is_root is True
        assert roots[0].children[0].name == "A"
        assert roots[0].children[0].depth == 1
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_hierarchy_parser.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add op_testgen/parser/trace_parser.py tests/test_hierarchy_parser.py
git commit -m "feat(parser): implement hierarchy building algorithm with tests"
```

---

## Task 4: HierarchyAnalyzer 核心实现

**Files:**
- Create: `op_testgen/analyzer/hierarchy_analyzer.py`
- Test: `tests/test_hierarchy_analyzer.py`

- [ ] **Step 1: 创建 `op_testgen/analyzer/hierarchy_analyzer.py`**

```python
"""层级调用分析器"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.trace_analyzer import OperatorStats


@dataclass
class CallChain:
    """调用链模式"""
    chain: Tuple[str, ...]
    occurrence_count: int
    total_duration_ms: float
    avg_duration_ms: float


@dataclass
class RecursivePattern:
    """递归/循环模式"""
    cycle: Tuple[str, ...]
    occurrence_count: int
    depth: int


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
    
    def get_self_time_stats(self) -> Dict[str, OperatorStats]:
        """基于自耗时的算子统计（排除子算子耗时）"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.all_nodes:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.self_duration_us
        return stats
    
    def get_total_time_stats(self) -> Dict[str, OperatorStats]:
        """基于总耗时的算子统计（含子算子）"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.all_nodes:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.duration_us
        return stats
    
    def get_call_chains(self, min_occurrence: int = 2, max_depth: int = 10) -> List[CallChain]:
        """提取高频调用链模式"""
        chain_counts: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
        
        for node in self.all_nodes:
            chain = tuple(node.get_call_chain())
            if len(chain) <= max_depth:
                chain_counts[chain].append(node.duration_us)
        
        results = []
        for chain, durations in chain_counts.items():
            if len(durations) >= min_occurrence:
                total_ms = sum(durations) / 1000.0
                avg_ms = total_ms / len(durations)
                results.append(CallChain(
                    chain=chain,
                    occurrence_count=len(durations),
                    total_duration_ms=total_ms,
                    avg_duration_ms=avg_ms,
                ))
        
        results.sort(key=lambda x: x.occurrence_count, reverse=True)
        return results
    
    def find_recursive_patterns(self, max_cycle_length: int = 5) -> List[RecursivePattern]:
        """识别递归/循环调用模式"""
        cycle_counts: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        
        for node in self.all_nodes:
            chain = node.get_call_chain()
            # 检查链中是否存在循环
            for i in range(len(chain)):
                for j in range(i + 1, min(i + max_cycle_length + 1, len(chain))):
                    cycle = tuple(chain[i:j])
                    if len(cycle) >= 2 and cycle[0] == cycle[-1]:
                        cycle_counts[cycle].append(node.depth)
        
        results = []
        for cycle, depths in cycle_counts.items():
            results.append(RecursivePattern(
                cycle=cycle,
                occurrence_count=len(depths),
                depth=max(depths),
            ))
        
        results.sort(key=lambda x: x.occurrence_count, reverse=True)
        return results
    
    def get_root_ops_stats(self) -> Dict[str, OperatorStats]:
        """根节点（入口算子）统计"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.root_ops:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.duration_us
        return stats
    
    def get_depth_distribution(self) -> Dict[int, int]:
        """调用深度分布（depth -> 节点数）"""
        dist: Dict[int, int] = defaultdict(int)
        for node in self.all_nodes:
            dist[node.depth] += 1
        return dict(sorted(dist.items()))
    
    def export_call_tree_text(self, output_file: str, max_depth: int = 10):
        """导出文本格式的调用树"""
        with open(output_file, "w", encoding="utf-8") as f:
            for root in self.root_ops:
                self._write_tree_node(f, root, 0, max_depth)
    
    def _write_tree_node(self, f, node: HierarchicalOpInfo, indent: int, max_depth: int):
        """递归写入树节点"""
        if indent > max_depth:
            return
        prefix = "  " * indent
        self_dur = node.self_duration_us / 1000.0
        total_dur = node.duration_us / 1000.0
        f.write(f"{prefix}{node.name}  [self={self_dur:.3f}ms, total={total_dur:.3f}ms, depth={node.depth}]\n")
        for child in node.children:
            self._write_tree_node(f, child, indent + 1, max_depth)
    
    def export_call_tree_json(self, output_file: str):
        """导出 JSON 格式的调用树"""
        import json
        
        def node_to_dict(node: HierarchicalOpInfo) -> dict:
            return {
                "name": node.name,
                "duration_us": node.duration_us,
                "self_duration_us": node.self_duration_us,
                "depth": node.depth,
                "is_root": node.is_root,
                "children": [node_to_dict(c) for c in node.children],
            }
        
        tree = [node_to_dict(r) for r in self.root_ops]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: 创建测试文件 `tests/test_hierarchy_analyzer.py`**

```python
"""层级分析器测试"""
import pytest
import tempfile
import json

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer, CallChain, RecursivePattern


@pytest.fixture
def sample_tree():
    """构建样本调用树：
    root1 (conv2d)
      ├── child1 (convolution)
      │     └── grandchild1 (_convolution)
      └── child2 (add)
    root2 (matmul)
    """
    root1 = HierarchicalOpInfo(name="aten::conv2d", start_ts=0, duration_us=100, tid=1, is_root=True)
    child1 = HierarchicalOpInfo(name="aten::convolution", start_ts=10, duration_us=60, tid=1)
    grandchild1 = HierarchicalOpInfo(name="aten::_convolution", start_ts=20, duration_us=40, tid=1)
    child2 = HierarchicalOpInfo(name="aten::add", start_ts=80, duration_us=10, tid=1)
    root2 = HierarchicalOpInfo(name="aten::matmul", start_ts=0, duration_us=50, tid=1, is_root=True)
    
    root1.children = [child1, child2]
    child1.parent = root1
    child2.parent = root1
    child1.children = [grandchild1]
    grandchild1.parent = child1
    
    # 设置 depth
    root1.depth = 0
    child1.depth = 1
    grandchild1.depth = 2
    child2.depth = 1
    root2.depth = 0
    
    return [root1, root2]


class TestHierarchyAnalyzer:
    def test_all_nodes(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        nodes = analyzer.all_nodes
        assert len(nodes) == 5
        names = {n.name for n in nodes}
        assert names == {"aten::conv2d", "aten::convolution", "aten::_convolution", "aten::add", "aten::matmul"}
    
    def test_self_time_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_self_time_stats()
        
        # conv2d: 100 - 60 - 10 = 30 (child1=60, child2=10)
        assert stats["aten::conv2d"].total_duration_us == 30.0
        # convolution: 60 - 40 = 20
        assert stats["aten::convolution"].total_duration_us == 20.0
        # _convolution: 40 (no children)
        assert stats["aten::_convolution"].total_duration_us == 40.0
        # matmul: 50 (no children)
        assert stats["aten::matmul"].total_duration_us == 50.0
    
    def test_total_time_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_total_time_stats()
        
        assert stats["aten::conv2d"].total_duration_us == 100.0
        assert stats["aten::_convolution"].total_duration_us == 40.0
    
    def test_get_call_chains(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        chains = analyzer.get_call_chains(min_occurrence=1)
        
        # 应该有 5 条不同的调用链
        assert len(chains) == 5
        
        # 找到 conv2d -> convolution -> _convolution 链
        conv_chain = next((c for c in chains if c.chain == ("aten::conv2d", "aten::convolution", "aten::_convolution")), None)
        assert conv_chain is not None
        assert conv_chain.occurrence_count == 1
    
    def test_get_root_ops_stats(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        stats = analyzer.get_root_ops_stats()
        
        assert len(stats) == 2
        assert "aten::conv2d" in stats
        assert "aten::matmul" in stats
    
    def test_get_depth_distribution(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        dist = analyzer.get_depth_distribution()
        
        assert dist[0] == 2  # root1, root2
        assert dist[1] == 2  # child1, child2
        assert dist[2] == 1  # grandchild1
    
    def test_export_call_tree_text(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        analyzer.export_call_tree_text(temp_path)
        
        with open(temp_path, "r") as f:
            content = f.read()
        
        assert "aten::conv2d" in content
        assert "aten::convolution" in content
        assert "aten::_convolution" in content
        assert "self=" in content
    
    def test_export_call_tree_json(self, sample_tree):
        analyzer = HierarchyAnalyzer(sample_tree)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        analyzer.export_call_tree_json(temp_path)
        
        with open(temp_path, "r") as f:
            tree = json.load(f)
        
        assert len(tree) == 2
        assert tree[0]["name"] == "aten::conv2d"
        assert len(tree[0]["children"]) == 2
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_hierarchy_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add op_testgen/analyzer/hierarchy_analyzer.py tests/test_hierarchy_analyzer.py
git commit -m "feat(analyzer): implement HierarchyAnalyzer with full test coverage"
```

---

## Task 5: 自耗时边界条件测试

**Files:**
- Create: `tests/test_self_time.py`

- [ ] **Step 1: 创建测试文件 `tests/test_self_time.py`**

```python
"""自耗时计算边界条件测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo


class TestSelfTime:
    def test_single_node_no_children(self):
        """单节点，无子节点：自耗时 = 总耗时"""
        node = HierarchicalOpInfo(name="A", duration_us=100)
        assert node.self_duration_us == 100.0
    
    def test_parent_with_children(self):
        """父节点有子节点：自耗时 = 总 - 子之和"""
        parent = HierarchicalOpInfo(name="parent", duration_us=100)
        child1 = HierarchicalOpInfo(name="child1", duration_us=30)
        child2 = HierarchicalOpInfo(name="child2", duration_us=40)
        parent.children = [child1, child2]
        
        assert parent.self_duration_us == 30.0  # 100 - 30 - 40
    
    def test_zero_duration_children(self):
        """子节点零耗时"""
        parent = HierarchicalOpInfo(name="parent", duration_us=100)
        child = HierarchicalOpInfo(name="child", duration_us=0)
        parent.children = [child]
        
        assert parent.self_duration_us == 100.0
    
    def test_negative_self_time_clamped(self):
        """子节点耗时超过父节点时，自耗时最小为 0"""
        parent = HierarchicalOpInfo(name="parent", duration_us=50)
        child = HierarchicalOpInfo(name="child", duration_us=60)
        parent.children = [child]
        
        assert parent.self_duration_us == 0.0
    
    def test_deep_tree_cumulative(self):
        """深层树：自耗时只减直接子节点"""
        root = HierarchicalOpInfo(name="root", duration_us=100)
        child = HierarchicalOpInfo(name="child", duration_us=80)
        grandchild = HierarchicalOpInfo(name="grandchild", duration_us=50)
        
        root.children = [child]
        child.children = [grandchild]
        
        assert root.self_duration_us == 20.0  # 100 - 80
        assert child.self_duration_us == 30.0  # 80 - 50
        assert grandchild.self_duration_us == 50.0
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_self_time.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_self_time.py
git commit -m "test(self-time): add boundary condition tests for self duration calculation"
```

---

## Task 6: 调用链提取测试

**Files:**
- Create: `tests/test_call_chains.py`

- [ ] **Step 1: 创建测试文件 `tests/test_call_chains.py`**

```python
"""调用链提取测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer


class TestCallChains:
    def test_simple_chain(self):
        """简单调用链：A -> B -> C"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        
        root.children = [b]
        b.parent = root
        b.children = [c]
        c.parent = b
        
        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1)
        
        chain_tuples = [c.chain for c in chains]
        assert ("A", "B", "C") in chain_tuples
        assert ("A", "B") in chain_tuples
        assert ("A",) in chain_tuples
    
    def test_branching_tree(self):
        """分支树：A -> [B, C]"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=40)
        c = HierarchicalOpInfo(name="C", start_ts=60, duration_us=30)
        
        root.children = [b, c]
        b.parent = root
        c.parent = root
        
        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1)
        
        chain_tuples = [c.chain for c in chains]
        assert ("A", "B") in chain_tuples
        assert ("A", "C") in chain_tuples
    
    def test_min_occurrence_filter(self):
        """最小出现次数过滤"""
        # 创建两个相同的调用链
        root1 = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b1 = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        root1.children = [b1]
        b1.parent = root1
        
        root2 = HierarchicalOpInfo(name="A", start_ts=200, duration_us=100, is_root=True)
        b2 = HierarchicalOpInfo(name="B", start_ts=210, duration_us=80)
        root2.children = [b2]
        b2.parent = root2
        
        analyzer = HierarchyAnalyzer([root1, root2])
        
        # min_occurrence=2 应该返回 (A, B)
        chains = analyzer.get_call_chains(min_occurrence=2)
        assert len(chains) >= 1
        chain = next(c for c in chains if c.chain == ("A", "B"))
        assert chain.occurrence_count == 2
        
        # min_occurrence=3 应该过滤掉
        chains = analyzer.get_call_chains(min_occurrence=3)
        assert not any(c.chain == ("A", "B") for c in chains)
    
    def test_max_depth_filter(self):
        """最大深度过滤"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        d = HierarchicalOpInfo(name="D", start_ts=30, duration_us=40)
        
        root.children = [b]
        b.children = [c]
        c.children = [d]
        b.parent = root
        c.parent = b
        d.parent = c
        
        analyzer = HierarchyAnalyzer([root])
        chains = analyzer.get_call_chains(min_occurrence=1, max_depth=2)
        
        # 深度超过 2 的链 (A, B, C, D) 应该被过滤
        chain_tuples = [c.chain for c in chains]
        assert ("A", "B", "C", "D") not in chain_tuples
        assert ("A", "B") in chain_tuples
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_call_chains.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_call_chains.py
git commit -m "test(call-chains): add call chain extraction tests"
```

---

## Task 7: 递归检测测试

**Files:**
- Create: `tests/test_recursion_detect.py`

- [ ] **Step 1: 创建测试文件 `tests/test_recursion_detect.py`**

```python
"""递归/循环调用检测测试"""
import pytest

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer


class TestRecursionDetection:
    def test_no_recursion(self):
        """无递归：A -> B -> C"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        
        root.children = [b]
        b.children = [c]
        b.parent = root
        c.parent = b
        
        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()
        
        # A -> B -> C 没有循环
        assert len(patterns) == 0
    
    def test_self_recursion(self):
        """自递归：A -> A"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        a2 = HierarchicalOpInfo(name="A", start_ts=10, duration_us=80)
        
        root.children = [a2]
        a2.parent = root
        
        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()
        
        # 应该检测到 (A, A) 循环
        assert len(patterns) >= 1
        cycles = [p.cycle for p in patterns]
        assert ("A", "A") in cycles
    
    def test_mutual_recursion(self):
        """双向递归：A -> B -> A"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        a2 = HierarchicalOpInfo(name="A", start_ts=20, duration_us=60)
        
        root.children = [b]
        b.children = [a2]
        b.parent = root
        a2.parent = b
        
        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()
        
        # 应该检测到 (A, B, A) 循环
        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "A") in cycles
    
    def test_longer_cycle(self):
        """多节点循环：A -> B -> C -> A"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        a2 = HierarchicalOpInfo(name="A", start_ts=30, duration_us=40)
        
        root.children = [b]
        b.children = [c]
        c.children = [a2]
        b.parent = root
        c.parent = b
        a2.parent = c
        
        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns()
        
        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "C", "A") in cycles
    
    def test_max_cycle_length_filter(self):
        """最大循环长度过滤"""
        root = HierarchicalOpInfo(name="A", start_ts=0, duration_us=100, is_root=True)
        b = HierarchicalOpInfo(name="B", start_ts=10, duration_us=80)
        c = HierarchicalOpInfo(name="C", start_ts=20, duration_us=60)
        d = HierarchicalOpInfo(name="D", start_ts=30, duration_us=40)
        a2 = HierarchicalOpInfo(name="A", start_ts=40, duration_us=30)
        
        root.children = [b]
        b.children = [c]
        c.children = [d]
        d.children = [a2]
        b.parent = root
        c.parent = b
        d.parent = c
        a2.parent = d
        
        analyzer = HierarchyAnalyzer([root])
        patterns = analyzer.find_recursive_patterns(max_cycle_length=3)
        
        # (A, B, C, D, A) 长度=5 > 3，应该被过滤
        cycles = [p.cycle for p in patterns]
        assert ("A", "B", "C", "D", "A") not in cycles
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_recursion_detect.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_recursion_detect.py
git commit -m "test(recursion): add recursive pattern detection tests"
```

---

## Task 8: TraceAnalyzer 层级扩展

**Files:**
- Modify: `op_testgen/analyzer/trace_analyzer.py`

- [ ] **Step 1: 在 trace_analyzer.py 顶部导入新增模块**

在 `op_testgen/analyzer/trace_analyzer.py` 的导入区（第1-8行）增加：

```python
from typing import Dict, List, Optional

from op_testgen.parser.trace_parser import OpInfo, TraceParser
from op_testgen.analyzer.hierarchy_analyzer import HierarchyAnalyzer, HierarchicalOpInfo
```

（注意：`Optional` 可能已导入，如有重复则跳过）

- [ ] **Step 2: 在 TraceAnalyzer 类中新增层级属性和方法**

在 `TraceAnalyzer.__init__` 中（第120-123行），将：
```python
    def __init__(self, op_infos: List[OpInfo]):
        self.op_infos = op_infos
        self.operators: Dict[str, OperatorStats] = {}
        self._analyze()
```
改为：
```python
    def __init__(self, op_infos: List[OpInfo]):
        self.op_infos = op_infos
        self.operators: Dict[str, OperatorStats] = {}
        self._hierarchy: Optional[List[HierarchicalOpInfo]] = None
        self._hierarchy_analyzer: Optional[HierarchyAnalyzer] = None
        self._analyze()
```

在 `TraceAnalyzer` 类末尾（`export_to_excel` 方法之后）添加：

```python
    # ─── 层级分析扩展 ───
    
    def _ensure_hierarchy(self):
        """懒加载层级数据"""
        if self._hierarchy is None:
            parser = TraceParser("")
            self._hierarchy = parser._build_hierarchy(self.op_infos)
            self._hierarchy_analyzer = HierarchyAnalyzer(self._hierarchy)
    
    def print_hierarchical_summary(self):
        """打印层级统计摘要"""
        self._ensure_hierarchy()
        assert self._hierarchy_analyzer is not None
        
        print("\n" + "=" * 60)
        print("层级分析摘要".center(60))
        print("=" * 60)
        
        # 自耗时排名 TOP 10
        self_time_stats = self._hierarchy_analyzer.get_self_time_stats()
        total_self_time = sum(s.total_duration_us for s in self_time_stats.values())
        
        print("\n【自耗时最高的算子 TOP 10】（排除子算子影响）")
        top_self = sorted(self_time_stats.values(), key=lambda x: x.total_duration_us, reverse=True)[:10]
        for i, op in enumerate(top_self, 1):
            pct = (op.total_duration_us / total_self_time * 100) if total_self_time > 0 else 0
            print(f"  {i:2d}. {op.name}")
            print(f"      自耗时: {op.total_duration_ms:.2f} ms ({pct:.1f}%), 调用: {op.call_count} 次")
        
        # 调用链分析
        chains = self._hierarchy_analyzer.get_call_chains()
        if chains:
            print("\n【高频调用链 TOP 5】")
            for i, chain in enumerate(chains[:5], 1):
                chain_str = " -> ".join(chain.chain)
                print(f"  {i}. {chain_str} (出现 {chain.occurrence_count} 次, 平均耗时: {chain.avg_duration_ms:.2f} ms)")
        
        # 递归检测
        patterns = self._hierarchy_analyzer.find_recursive_patterns()
        if patterns:
            print("\n【检测到的循环调用模式】")
            for i, pattern in enumerate(patterns[:5], 1):
                cycle_str = " -> ".join(pattern.cycle)
                print(f"  {i}. {cycle_str} (出现 {pattern.occurrence_count} 次, 最大深度: {pattern.depth})")
        
        # 深度分布
        depth_dist = self._hierarchy_analyzer.get_depth_distribution()
        if depth_dist:
            print("\n【调用深度分布】")
            for depth, count in depth_dist.items():
                label = "根" if depth == 0 else f"深度 {depth}"
                print(f"  {label}: {count} 个算子")
        
        print("=" * 60)
    
    def export_hierarchy_to_csv(self, output_file: str):
        """导出层级统计到 CSV"""
        self._ensure_hierarchy()
        assert self._hierarchy_analyzer is not None
        
        import csv
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "算子名称", "调用深度", "是否为根", "自耗时(ms)", "总耗时(ms)",
                "子算子数", "调用次数"
            ])
            writer.writeheader()
            
            for node in self._hierarchy_analyzer.all_nodes:
                writer.writerow({
                    "算子名称": node.name,
                    "调用深度": node.depth,
                    "是否为根": "是" if node.is_root else "否",
                    "自耗时(ms)": round(node.self_duration_us / 1000.0, 4),
                    "总耗时(ms)": round(node.duration_us / 1000.0, 4),
                    "子算子数": len(node.children),
                    "调用次数": 1,  # 每个节点代表一次调用
                })
    
    def export_hierarchy_to_excel(self, output_file: str) -> bool:
        """导出层级统计到 Excel（新增工作表）"""
        self._ensure_hierarchy()
        assert self._hierarchy_analyzer is not None
        
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment, PatternFill
            from openpyxl.utils import get_column_letter
        except ImportError:
            print("错误: openpyxl 未安装，无法导出 Excel")
            return False
        
        wb = openpyxl.load_workbook(output_file)
        
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center")
        
        # 新增工作表：自耗时排名
        ws_self = wb.create_sheet("自耗时排名")
        headers = ["算子名称", "调用深度", "是否为根", "自耗时(ms)", "总耗时(ms)", "子算子数"]
        ws_self.append(headers)
        for col_num, _ in enumerate(headers, 1):
            cell = ws_self.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
        
        for node in self._hierarchy_analyzer.all_nodes:
            ws_self.append([
                node.name, node.depth, "是" if node.is_root else "否",
                round(node.self_duration_us / 1000.0, 4),
                round(node.duration_us / 1000.0, 4),
                len(node.children),
            ])
        
        # 自动调整列宽
        for col in ws_self.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            ws_self.column_dimensions[col_letter].width = min(max_length + 2, 80)
        
        wb.save(output_file)
        return True
```

- [ ] **Step 3: 运行现有测试确保向后兼容**

Run: `pytest tests/ -v --ignore=tests/test_hierarchy_analyzer.py --ignore=tests/test_self_time.py --ignore=tests/test_call_chains.py --ignore=tests/test_recursion_detect.py`
Expected: ALL PASS（现有功能不受影响）

- [ ] **Step 4: Commit**

```bash
git add op_testgen/analyzer/trace_analyzer.py
git commit -m "feat(analyzer): add hierarchical analysis methods to TraceAnalyzer"
```

---

## Task 9: CLI 扩展

**Files:**
- Modify: `op_testgen/cli.py`
- Test: 手动验证

- [ ] **Step 1: 修改 `op_testgen/cli.py` 的 analyze 子命令**

找到 `analyze` 子命令的 argparse 定义，添加新参数：

```python
    analyze_parser.add_argument("--hierarchy", action="store_true", help="启用层级分析")
    analyze_parser.add_argument("--self-time-top", type=int, default=10, help="自耗时排名数量")
    analyze_parser.add_argument("--call-chains", action="store_true", help="输出高频调用链")
    analyze_parser.add_argument("--detect-recursion", action="store_true", help="检测递归模式")
    analyze_parser.add_argument("--export-tree", type=str, default=None, help="导出调用树到文件")
    analyze_parser.add_argument("--max-chain-depth", type=int, default=10, help="调用链最大深度")
    analyze_parser.add_argument("--min-chain-occurrence", type=int, default=2, help="调用链最小出现次数")
```

- [ ] **Step 2: 修改 analyze 命令的处理逻辑**

在 `analyze` 命令处理函数中（调用 `analyzer.print_summary()` 之后）添加：

```python
        # 层级分析（可选）
        if args.hierarchy or args.call_chains or args.detect_recursion or args.export_tree:
            analyzer.print_hierarchical_summary()
        
        if args.export_tree:
            analyzer._ensure_hierarchy()
            assert analyzer._hierarchy_analyzer is not None
            analyzer._hierarchy_analyzer.export_call_tree_text(args.export_tree)
            print(f"调用树已导出: {args.export_tree}")
        
        # 导出层级数据（如果启用了层级分析且输出格式支持）
        if args.hierarchy and not args.csv:
            base, ext = os.path.splitext(output_file)
            hierarchy_file = f"{base}_hierarchy{ext}"
            if args.csv:
                analyzer.export_hierarchy_to_csv(hierarchy_file)
            else:
                # Excel 模式下尝试添加工作表
                if analyzer.export_hierarchy_to_excel(output_file):
                    print(f"层级数据已添加到 Excel: {output_file}")
```

（注意：需要处理 `args` 对象中可能不存在新参数的情况，或者确保 argparse 定义已更新）

- [ ] **Step 3: 手动验证 CLI**

Run: `python -m op_testgen.cli analyze profiler_trace.json --hierarchy`
Expected: 正常输出平铺摘要 + 层级摘要

Run: `python -m op_testgen.cli analyze profiler_trace.json --hierarchy --export-tree tree.txt`
Expected: 生成 tree.txt 文件

- [ ] **Step 4: Commit**

```bash
git add op_testgen/cli.py
git commit -m "feat(cli): add hierarchy analysis options to analyze subcommand"
```

---

## Task 10: 集成测试与端到端验证

**Files:**
- 无新增文件

- [ ] **Step 1: 运行完整测试套件**

Run: `pytest tests/ -v`
Expected: ALL PASS（包括新测试和现有测试）

- [ ] **Step 2: 端到端验证（使用真实 trace 文件）**

Run: `python -m op_testgen.cli analyze profiler_trace.json --hierarchy`
Expected:
- 平铺摘要正常输出
- 层级摘要正常输出（自耗时排名、调用链、深度分布）
- 无异常

Run: `python -m op_testgen.cli analyze profiler_trace.json --hierarchy --export-tree call_tree.txt`
Expected: `call_tree.txt` 文件生成，内容包含调用树

Run: `python -m op_testgen.cli analyze profiler_trace.json`
Expected: 仅输出平铺摘要（向后兼容验证）

- [ ] **Step 3: Commit**

```bash
git commit -m "test(e2e): validate hierarchy analysis with real trace file"
```

---

## Self-Review 检查清单

- [x] **Spec 覆盖检查**：
  - 自耗时计算 → Task 4 (HierarchyAnalyzer.get_self_time_stats) + Task 5 (边界测试)
  - 调用链提取 → Task 4 (get_call_chains) + Task 6 (测试)
  - 递归检测 → Task 4 (find_recursive_patterns) + Task 7 (测试)
  - 层级构建 → Task 3 (_build_hierarchy)
  - CLI 扩展 → Task 9
  - 向后兼容 → 贯穿所有 Task（不修改现有接口）

- [x] **Placeholder 扫描**：
  - 无 "TBD"、"TODO"、"实现细节详见..." 等占位符
  - 每个步骤都有具体代码和命令

- [x] **类型一致性**：
  - `self_duration_us` 在所有文件中一致使用
  - `HierarchicalOpInfo` 继承关系一致
  - `HierarchyAnalyzer` API 签名与设计文档一致

---

## 执行方式选择

**计划已保存到 `docs/superpowers/plans/2026-05-24-hierarchical-op-analysis-plan.md`**

两个执行选项：

**1. Subagent-Driven（推荐）** — 每个 Task 由一个独立子代理执行，我在 Task 间审查结果

**2. Inline Execution** — 在当前会话中按顺序执行所有 Task

你希望采用哪种方式？
