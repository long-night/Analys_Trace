"""Chrome Trace 解析模块"""
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from collections import defaultdict


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
    start_ts: float = 0.0
    tid: int = 0

    def __repr__(self) -> str:
        return f"OpInfo(name={self.name}, dims={self.input_dims}, types={self.input_types})"


@dataclass
class HierarchicalOpInfo(OpInfo):
    """带层级关系的算子信息"""

    parent: Optional['HierarchicalOpInfo'] = None
    children: List['HierarchicalOpInfo'] = field(default_factory=list)
    depth: int = 0
    is_root: bool = False

    @property
    def hierarchical_name(self) -> str:
        """完整层级路径，如 aten::convolution_backward/aten::contiguous"""
        return "/".join(self.get_call_chain())

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


class TraceParser:
    """Chrome Trace 解析器"""

    COMMUNICATION_PREFIXES = ("c10d::", "nccl:", "gloo:", "mpi:")

    def __init__(self, trace_file: str):
        self.trace_file = trace_file

    def load_trace(self) -> dict:
        """加载 trace JSON 文件"""
        try:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"错误：无法解析 JSON 文件: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print(f"错误：文件不存在: {self.trace_file}", file=sys.stderr)
            sys.exit(1)

    def _is_cpu_operator(self, event: dict) -> bool:
        """判断是否为 CPU 算子或通信算子"""
        name = event.get("name", "")
        cat = event.get("cat", "")

        if name.startswith(self.COMMUNICATION_PREFIXES):
            return True

        return (
            name.startswith(("aten::", "torch::"))
            and ("cpu" in cat.lower() or cat == "cpu_op" or "kernel" not in cat.lower())
        )

    def _extract_field(self, event: dict, keys: List[str]) -> Optional[Any]:
        """从 event args 中提取字段，尝试多个 key"""
        args = event.get("args", {})
        for key in keys:
            if key in args:
                return args[key]
        return None

    def _extract_shapes(self, event: dict) -> List[List[int]]:
        """提取输入 shapes"""
        result = self._extract_field(event, ["Input Dims", "input_dims", "Input Shapes"])
        if isinstance(result, list):
            return result
        return []

    def _extract_strides(self, event: dict) -> List[List[int]]:
        """提取输入 strides"""
        result = self._extract_field(event, ["Input Strides", "input_strides"])
        if isinstance(result, list):
            return result
        return []

    def _extract_types(self, event: dict) -> List[str]:
        """提取输入数据类型"""
        result = self._extract_field(event, ["Input type", "input_type"])
        if isinstance(result, list):
            return [str(t) for t in result]
        return []

    def _extract_concrete(self, event: dict) -> List[Any]:
        """提取 concrete inputs"""
        result = self._extract_field(event, ["Concrete Inputs", "concrete_inputs"])
        if isinstance(result, list):
            return result
        return []

    def _is_communication(self, name: str) -> bool:
        return name.startswith(self.COMMUNICATION_PREFIXES)

    def parse(self) -> List[OpInfo]:
        """解析 trace 并返回算子信息列表"""
        data = self.load_trace()
        events = data.get("traceEvents", [])

        # 记录每个线程的 pending B 事件
        pending: Dict[int, Dict[str, tuple]] = defaultdict(dict)
        ops: List[OpInfo] = []

        for event in events:
            if not self._is_cpu_operator(event):
                continue

            name = event.get("name", "")
            tid = event.get("tid", 0)
            ph = event.get("ph", "")
            ts = event.get("ts", 0)

            if ph == "B":
                shapes = self._extract_shapes(event)
                strides = self._extract_strides(event)
                types = self._extract_types(event)
                concrete = self._extract_concrete(event)
                pending[tid][name] = (ts, shapes, strides, types, concrete)

            elif ph == "E":
                if name in pending[tid]:
                    start_ts, shapes, strides, types, concrete = pending[tid][name]
                    duration = ts - start_ts

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
                    ops.append(op)
                    del pending[tid][name]

            elif ph == "X":
                duration = event.get("dur", 0.0)
                shapes = self._extract_shapes(event)
                strides = self._extract_strides(event)
                types = self._extract_types(event)
                concrete = self._extract_concrete(event)

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
                ops.append(op)

        return ops

    def _build_hierarchy(self, flat_ops: Sequence[OpInfo]) -> List[HierarchicalOpInfo]:
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
        by_tid: Dict[int, List[OpInfo]] = defaultdict(list)
        for op in flat_ops:
            by_tid[op.tid].append(op)

        node_map: Dict[int, HierarchicalOpInfo] = {}
        all_roots: List[HierarchicalOpInfo] = []

        for tid, ops in by_tid.items():
            ops.sort(key=lambda o: o.start_ts)

            stack: List[HierarchicalOpInfo] = []
            roots_for_tid: List[HierarchicalOpInfo] = []

            for op in ops:
                while stack and (stack[-1].start_ts + stack[-1].duration_us) <= op.start_ts:
                    stack.pop()

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
                    parent = stack[-1]
                    h_op.parent = parent
                    h_op.depth = parent.depth + 1
                    parent.children.append(h_op)
                else:
                    h_op.is_root = True
                    roots_for_tid.append(h_op)

                stack.append(h_op)

            all_roots.extend(roots_for_tid)

        return all_roots

    def parse_hierarchical(self) -> List[HierarchicalOpInfo]:
        """解析 trace 并返回带层级关系的根节点列表"""
        flat_ops = self.parse()
        return self._build_hierarchy(flat_ops)
