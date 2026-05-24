"""Chrome Trace 解析模块"""
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
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

    def __repr__(self) -> str:
        return f"OpInfo(name={self.name}, dims={self.input_dims}, types={self.input_types})"


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
                )
                ops.append(op)

        return ops
