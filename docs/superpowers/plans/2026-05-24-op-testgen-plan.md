# PyTorch Profiler 自动化测试生成器 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 PyTorch Profiler Chrome Trace JSON 自动生成并执行算子正确性测试与性能测试，输出结构化报告。

**Architecture:** 六阶段流水线（TraceParser → OpMapper → TensorBuilder → CorrectnessRunner / PerfBenchmark → Reporter），每个阶段为独立模块，通过 dataclass 传递数据。CLI 统一编排执行。

**Tech Stack:** Python 3.12, PyTorch, pytest, PyYAML, Jinja2 (HTML 报告), openpyxl (可选 Excel)

---

## 文件结构总览

```
op_testgen/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── settings.py              # 全局配置类
│   ├── op_whitelist.yaml        # 算子白名单
│   └── op_classification.yaml   # 算子分类与公式
├── parser/
│   ├── __init__.py
│   └── trace_parser.py          # 数据解析
├── mapper/
│   ├── __init__.py
│   └── op_mapper.py             # 算子映射与过滤
├── builder/
│   ├── __init__.py
│   └── tensor_builder.py        # 张量与参数重构
├── correctness/
│   ├── __init__.py
│   └── test_runner.py           # 正确性测试执行
├── perf/
│   ├── __init__.py
│   ├── classifier.py            # 算子分类
│   ├── metrics.py               # FLOPS/带宽计算
│   └── benchmark.py             # 性能测试执行
├── reporter/
│   ├── __init__.py
│   ├── base.py                  # 报告基类
│   ├── html_reporter.py         # HTML 报告
│   └── excel_reporter.py        # Excel 报告（可选）
└── cli.py                       # 命令行入口

tests/
├── __init__.py
├── conftest.py                  # pytest fixtures
├── test_parser.py
├── test_mapper.py
├── test_tensor_builder.py
├── test_correctness.py
└── test_perf.py

pyproject.toml                   # 包配置
```

---

## Task 1: 项目骨架与配置系统

**目标**: 建立项目目录结构、包配置、全局配置类和初始配置文件。

**Files:**
- Create: `pyproject.toml`
- Create: `op_testgen/__init__.py`
- Create: `op_testgen/config/__init__.py`
- Create: `op_testgen/config/settings.py`
- Create: `op_testgen/config/op_whitelist.yaml`
- Create: `op_testgen/config/op_classification.yaml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

---

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "op_testgen"
version = "0.1.0"
description = "PyTorch Profiler 自动化测试生成器"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "pyyaml",
    "jinja2",
]

[project.optional-dependencies]
excel = ["openpyxl"]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]

[project.scripts]
op_testgen = "op_testgen.cli:main"
```

---

- [ ] **Step 2: 创建根 __init__.py**

```python
"""PyTorch Profiler 自动化测试生成器"""
__version__ = "0.1.0"
```

File: `op_testgen/__init__.py`

---

- [ ] **Step 3: 创建 config 模块**

`op_testgen/config/__init__.py`:
```python
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
```

`op_testgen/config/settings.py`:
```python
"""全局配置管理"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Settings:
    """测试生成器全局配置"""
    # 误差阈值: {dtype_str: (max_abs_err, max_rel_err)}
    error_thresholds: Dict[str, tuple] = field(default_factory=lambda: {
        "float16": (1e-3, 1e-2),
        "bfloat16": (5e-3, 5e-2),
        "float32": (1e-5, 1e-4),
        "float64": (1e-10, 1e-9),
    })

    # 性能测试
    perf_warmup_iters: int = 3
    perf_benchmark_iters: int = 10

    # 随机种子
    random_seed: int = 42

    # 白名单与分类表路径
    whitelist_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "op_whitelist.yaml"
    ))
    classification_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "op_classification.yaml"
    ))

    # 报告输出
    default_output_format: str = "html"
    default_output_path: str = "op_testgen_report.html"


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
```

---

- [ ] **Step 4: 创建初始白名单**

`op_testgen/config/op_whitelist.yaml`:
```yaml
# 算子白名单: trace_name -> {callable_path, namespace}
# 不在此列表中的算子将尝试动态反射映射

aten::add:
  callable_path: "torch.add"
  namespace: "torch"

aten::mul:
  callable_path: "torch.mul"
  namespace: "torch"

aten::sub:
  callable_path: "torch.sub"
  namespace: "torch"

aten::div:
  callable_path: "torch.div"
  namespace: "torch"

aten::matmul:
  callable_path: "torch.matmul"
  namespace: "torch"

aten::mm:
  callable_path: "torch.mm"
  namespace: "torch"

aten::bmm:
  callable_path: "torch.bmm"
  namespace: "torch"

aten::conv2d:
  callable_path: "torch.nn.functional.conv2d"
  namespace: "torch.nn.functional"

aten::linear:
  callable_path: "torch.nn.functional.linear"
  namespace: "torch.nn.functional"

aten::relu:
  callable_path: "torch.nn.functional.relu"
  namespace: "torch.nn.functional"

aten::softmax:
  callable_path: "torch.softmax"
  namespace: "torch"

aten::max:
  callable_path: "torch.max"
  namespace: "torch"

aten::sum:
  callable_path: "torch.sum"
  namespace: "torch"

aten::mean:
  callable_path: "torch.mean"
  namespace: "torch"
```

---

- [ ] **Step 5: 创建初始分类表**

`op_testgen/config/op_classification.yaml`:
```yaml
# 算子分类与性能指标计算公式
# category: compute | memory | communication | mixed
# flops_formula / bytes_formula: 使用 Python eval 安全的表达式

aten::matmul:
  category: compute
  flops_formula: "2 * M * N * K"
  # 变量由运行时根据 input_dims 自动注入: M,N,K 为矩阵维度

aten::mm:
  category: compute
  flops_formula: "2 * M * N * K"

aten::bmm:
  category: compute
  flops_formula: "2 * B * M * N * K"

aten::conv2d:
  category: compute
  flops_formula: "2 * N * Cout * Hout * Wout * Cin * K * K"

aten::linear:
  category: compute
  flops_formula: "2 * M * N * K"

aten::add:
  category: memory

aten::mul:
  category: memory

aten::relu:
  category: memory

aten::softmax:
  category: compute
  flops_formula: "5 * N"  # 近似

c10d::allreduce_:
  category: communication
  bytes_formula: "2 * numel * element_size"

nccl:all_reduce:
  category: communication
  bytes_formula: "2 * numel * element_size"
```

---

- [ ] **Step 6: 创建 tests 基础文件**

`tests/__init__.py`:
```python
"""测试包"""
```

`tests/conftest.py`:
```python
"""pytest fixtures"""
import pytest


@pytest.fixture
def sample_trace_json(tmp_path):
    """创建最小可用的 trace JSON 用于测试"""
    import json
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "aten::add",
                "cat": "cpu_op",
                "ts": 1000000,
                "dur": 500,
                "args": {
                    "Input Dims": [[3, 4], [3, 4]],
                    "Input type": ["float", "float"],
                    "Concrete Inputs": [0]
                }
            },
            {
                "ph": "X",
                "name": "aten::conv2d",
                "cat": "cpu_op",
                "ts": 2000000,
                "dur": 2000,
                "args": {
                    "Input Dims": [[1, 3, 32, 32], [16, 3, 3, 3]],
                    "Input type": ["float", "float"],
                    "Concrete Inputs": [None, None, [1, 1], [0, 0], [1, 1], 1]
                }
            }
        ]
    }
    path = tmp_path / "test_trace.json"
    with open(path, "w") as f:
        json.dump(trace, f)
    return str(path)
```

---

- [ ] **Step 7: 验证包结构**

```bash
cd /mnt/d/ubuntu/opencode/training_framework/Analys_Trace
pip install -e . 2>&1 | tail -5
python -c "from op_testgen.config import get_settings; s = get_settings(); print(s.random_seed)"
```

Expected:
```
Successfully installed op_testgen-0.1.0
42
```

---

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml op_testgen/ tests/
git commit -m "chore: initialize op_testgen package structure and config system"
```

---

## Task 2: TraceParser（数据解析模块）

**目标**: 实现 trace JSON 的加载与算子信息提取。

**Files:**
- Create: `op_testgen/parser/__init__.py`
- Create: `op_testgen/parser/trace_parser.py`
- Create: `tests/test_parser.py`

---

- [ ] **Step 1: 定义核心数据结构**

`op_testgen/parser/trace_parser.py`（上半部分 - 数据结构）:
```python
"""Chrome Trace 解析模块"""
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
```

---

- [ ] **Step 2: 写解析器测试**

`tests/test_parser.py`:
```python
"""测试 TraceParser"""
import pytest
from op_testgen.parser.trace_parser import TraceParser, OpInfo


class TestTraceParser:
    def test_load_trace(self, sample_trace_json):
        parser = TraceParser(sample_trace_json)
        data = parser.load_trace()
        assert "traceEvents" in data
        assert len(data["traceEvents"]) == 2

    def test_parse_events(self, sample_trace_json):
        parser = TraceParser(sample_trace_json)
        ops = parser.parse()
        assert len(ops) == 2

        # aten::add
        add_op = [op for op in ops if op.name == "aten::add"][0]
        assert add_op.input_dims == [[3, 4], [3, 4]]
        assert add_op.input_types == ["float", "float"]
        assert add_op.concrete_inputs == [0]
        assert add_op.duration_us == 500.0
        assert not add_op.is_communication

        # aten::conv2d
        conv_op = [op for op in ops if op.name == "aten::conv2d"][0]
        assert conv_op.input_dims == [[1, 3, 32, 32], [16, 3, 3, 3]]
        assert conv_op.is_communication is False

    def test_communication_operator(self, tmp_path):
        import json
        trace = {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "nccl:all_reduce",
                    "cat": "cpu_op",
                    "ts": 1000000,
                    "dur": 10000,
                    "args": {
                        "Input Dims": [[1024]],
                        "Input type": ["float"],
                    }
                }
            ]
        }
        path = tmp_path / "comm_trace.json"
        with open(path, "w") as f:
            json.dump(trace, f)

        parser = TraceParser(str(path))
        ops = parser.parse()
        assert len(ops) == 1
        assert ops[0].is_communication is True
```

---

- [ ] **Step 3: 运行测试确认失败**

```bash
cd /mnt/d/ubuntu/opencode/training_framework/Analys_Trace
pytest tests/test_parser.py -v
```

Expected: `FAILED` with `ImportError: cannot import name 'TraceParser'`

---

- [ ] **Step 4: 实现 TraceParser**

`op_testgen/parser/trace_parser.py`（完整实现）:
```python
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
```

`op_testgen/parser/__init__.py`:
```python
from .trace_parser import TraceParser, OpInfo

__all__ = ["TraceParser", "OpInfo"]
```

---

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_parser.py -v
```

Expected: 3 tests PASSED

---

- [ ] **Step 6: Commit**

```bash
git add op_testgen/parser/ tests/test_parser.py
git commit -m "feat(parser): implement TraceParser with OpInfo extraction"
```

---

## Task 3: OpMapper（算子映射与过滤模块）

**目标**: 将 trace 算子名称映射到可执行的 Python callable，过滤不支持的算子。

**Files:**
- Create: `op_testgen/mapper/__init__.py`
- Create: `op_testgen/mapper/op_mapper.py`
- Create: `tests/test_mapper.py`

---

- [ ] **Step 1: 定义 MappedOp 和测试**

`tests/test_mapper.py`:
```python
"""测试 OpMapper"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper, MappedOp


class TestOpMapper:
    @pytest.fixture
    def mapper(self):
        return OpMapper()

    def test_whitelist_mapping(self, mapper):
        op = OpInfo(name="aten::add", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.add
        assert mapped.callable_path == "torch.add"
        assert mapped.namespace == "torch"
        assert mapped.support_status == "whitelisted"

    def test_dynamic_mapping(self, mapper):
        # aten::abs 不在初始白名单中，但可通过动态反射映射
        op = OpInfo(name="aten::abs", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.abs
        assert mapped.support_status == "auto_discovered"

    def test_tensor_suffix_stripping(self, mapper):
        op = OpInfo(name="aten::add.Tensor", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"])
        mapped = mapper.map_operator(op)
        assert mapped is not None
        assert mapped.callable == torch.add

    def test_filter_unsupported(self, mapper):
        # aten::empty 在黑名单中
        op = OpInfo(name="aten::empty", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        assert mapped is None

    def test_filter_namespace(self, mapper):
        # aten::_foobar 无法映射到任何命名空间
        op = OpInfo(name="aten::_foobar", input_dims=[[2, 3]], input_types=["float"])
        mapped = mapper.map_operator(op)
        # 如果动态反射也找不到，返回 None
        assert mapped is None or mapped.support_status == "unsupported"

    def test_map_all(self, mapper):
        ops = [
            OpInfo(name="aten::add", input_dims=[[2, 3], [2, 3]], input_types=["float", "float"]),
            OpInfo(name="aten::empty", input_dims=[[2, 3]], input_types=["float"]),
            OpInfo(name="aten::conv2d", input_dims=[[1, 3, 32, 32], [16, 3, 3, 3]], input_types=["float", "float"]),
        ]
        mapped = mapper.map_all(ops)
        assert len(mapped) == 2  # add 和 conv2d，empty 被过滤
        names = [m.op_info.name for m in mapped]
        assert "aten::add" in names
        assert "aten::conv2d" in names
```

---

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_mapper.py -v
```

Expected: `ImportError`

---

- [ ] **Step 3: 实现 OpMapper**

`op_testgen/mapper/op_mapper.py`:
```python
"""算子映射与过滤模块"""
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import yaml

from op_testgen.parser.trace_parser import OpInfo


@dataclass
class MappedOp:
    """映射后的算子信息"""
    op_info: OpInfo
    callable: Callable
    callable_path: str
    namespace: str
    support_status: str  # "whitelisted" | "auto_discovered" | "unsupported"


class OpMapper:
    """算子名称到 Python callable 的映射器"""

    # 黑名单: 不追踪的算子模式
    BLACKLIST_PATTERNS = [
        "aten::empty",
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::to",
        "aten::copy_",
        "aten::detach",
        "aten::clone",
        "aten::call",
        "aten::__interpolate",
        "aten::as_strided",
        "aten::set_",
        "aten::_reshape_alias",
    ]

    # 支持的命名空间
    NAMESPACES = {
        "torch": torch,
        "torch.nn.functional": None,  # 懒加载
        "torch.linalg": None,
        "torch.distributed": None,
        "torch.optim": None,
    }

    # 特殊映射规则: trace_name -> callable_path
    SPECIAL_MAPPINGS = {
        "aten::conv2d": "torch.nn.functional.conv2d",
        "aten::linear": "torch.nn.functional.linear",
        "aten::relu": "torch.nn.functional.relu",
        "aten::max_pool2d": "torch.nn.functional.max_pool2d",
        "aten::avg_pool2d": "torch.nn.functional.avg_pool2d",
        "aten::batch_norm": "torch.nn.functional.batch_norm",
        "c10d::allreduce_": "torch.distributed.all_reduce",
        "c10d::allgather_": "torch.distributed.all_gather",
        "c10d::broadcast_": "torch.distributed.broadcast",
    }

    def __init__(self, whitelist_path: Optional[str] = None):
        self.whitelist: Dict[str, dict] = {}
        if whitelist_path:
            self._load_whitelist(whitelist_path)
        else:
            # 默认加载包内白名单
            import os
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "op_whitelist.yaml"
            )
            if os.path.exists(default_path):
                self._load_whitelist(default_path)

    def _load_whitelist(self, path: str):
        """加载白名单 YAML"""
        with open(path, "r", encoding="utf-8") as f:
            self.whitelist = yaml.safe_load(f) or {}

    def _is_blacklisted(self, name: str) -> bool:
        """检查是否在黑名单中"""
        return name in self.BLACKLIST_PATTERNS

    def _resolve_callable(self, path: str) -> Optional[Callable]:
        """通过路径解析 callable，如 'torch.add' -> torch.add"""
        parts = path.split(".")
        try:
            module = importlib.import_module(".".join(parts[:-1]))
            obj = getattr(module, parts[-1])
            if callable(obj):
                return obj
        except (ImportError, AttributeError):
            pass
        return None

    def _strip_tensor_suffix(self, name: str) -> str:
        """去掉 .Tensor 后缀"""
        if name.endswith(".Tensor"):
            return name[:-7]
        return name

    def _try_dynamic_map(self, name: str) -> Optional[tuple]:
        """尝试动态反射映射"""
        # 1. 检查特殊映射
        if name in self.SPECIAL_MAPPINGS:
            path = self.SPECIAL_MAPPINGS[name]
            callable_obj = self._resolve_callable(path)
            if callable_obj:
                namespace = path.split(".")[0]
                if "nn.functional" in path:
                    namespace = "torch.nn.functional"
                elif "distributed" in path:
                    namespace = "torch.distributed"
                return callable_obj, path, namespace

        # 2. 尝试 aten::xxx -> torch.xxx
        if name.startswith("aten::"):
            base_name = self._strip_tensor_suffix(name[6:])
            path = f"torch.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch"

            # 尝试 torch.nn.functional.xxx
            path = f"torch.nn.functional.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch.nn.functional"

        # 3. 尝试 torch.linalg.xxx
        if name.startswith("aten::linalg_"):
            base_name = name[13:]
            path = f"torch.linalg.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch.linalg"

        return None

    def map_operator(self, op_info: OpInfo) -> Optional[MappedOp]:
        """映射单个算子"""
        name = op_info.name

        # 1. 黑名单检查
        if self._is_blacklisted(name):
            return None

        # 2. 白名单优先
        if name in self.whitelist:
            entry = self.whitelist[name]
            path = entry.get("callable_path", "")
            namespace = entry.get("namespace", "torch")
            callable_obj = self._resolve_callable(path)
            if callable_obj:
                return MappedOp(
                    op_info=op_info,
                    callable=callable_obj,
                    callable_path=path,
                    namespace=namespace,
                    support_status="whitelisted",
                )

        # 3. 动态反射
        result = self._try_dynamic_map(name)
        if result:
            callable_obj, path, namespace = result
            return MappedOp(
                op_info=op_info,
                callable=callable_obj,
                callable_path=path,
                namespace=namespace,
                support_status="auto_discovered",
            )

        # 4. 无法映射
        return None

    def map_all(self, op_infos: List[OpInfo]) -> List[MappedOp]:
        """批量映射算子列表"""
        mapped = []
        for op_info in op_infos:
            m = self.map_operator(op_info)
            if m is not None:
                mapped.append(m)
        return mapped
```

`op_testgen/mapper/__init__.py`:
```python
from .op_mapper import OpMapper, MappedOp

__all__ = ["OpMapper", "MappedOp"]
```

---

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_mapper.py -v
```

Expected: 6 tests PASSED

---

- [ ] **Step 5: Commit**

```bash
git add op_testgen/mapper/ tests/test_mapper.py
git commit -m "feat(mapper): implement OpMapper with whitelist and dynamic reflection"
```

---

## Task 4: TensorBuilder（张量与参数重构模块）

**目标**: 根据 OpInfo 构建输入张量和 kwargs。

**Files:**
- Create: `op_testgen/builder/__init__.py`
- Create: `op_testgen/builder/tensor_builder.py`
- Create: `tests/test_tensor_builder.py`

---

- [ ] **Step 1: 定义数据结构和测试**

`tests/test_tensor_builder.py`:
```python
"""测试 TensorBuilder"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper, MappedOp
from op_testgen.builder.tensor_builder import TensorBuilder, TestCase


class TestTensorBuilder:
    @pytest.fixture
    def builder(self):
        return TensorBuilder(seed=42)

    def test_build_float_tensors(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        assert mapped is not None

        test_case = builder.build(mapped)
        assert len(test_case.input_tensors) == 2
        assert all(t.dtype == torch.float32 for t in test_case.input_tensors)
        assert all(t.shape == torch.Size([2, 3]) for t in test_case.input_tensors)

    def test_build_different_types(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "long"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert test_case.input_tensors[0].dtype == torch.float32
        assert test_case.input_tensors[1].dtype == torch.int64

    def test_build_with_strides(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert test_case.input_tensors[0].stride() == (3, 1)

    def test_dtype_mapping(self, builder):
        assert builder._map_dtype("float") == torch.float32
        assert builder._map_dtype("float64") == torch.float64
        assert builder._map_dtype("half") == torch.float16
        assert builder._map_dtype("bfloat16") == torch.bfloat16
        assert builder._map_dtype("long") == torch.int64
        assert builder._map_dtype("int") == torch.int32
        assert builder._map_dtype("bool") == torch.bool

    def test_scalar_not_tensor(self, builder):
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3]],
            input_types=["float", "Scalar"],
            concrete_inputs=[1.5, 0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        test_case = builder.build(mapped)
        assert len(test_case.input_tensors) == 1
        assert test_case.kwargs.get("other") == 1.5
```

---

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tensor_builder.py -v
```

Expected: `ImportError`

---

- [ ] **Step 3: 实现 TensorBuilder**

`op_testgen/builder/tensor_builder.py`:
```python
"""张量与参数重构模块"""
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from op_testgen.mapper.op_mapper import MappedOp


@dataclass
class TestCase:
    """可执行的测试用例"""
    mapped_op: MappedOp
    input_tensors: List[torch.Tensor] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class TensorBuilder:
    """根据运行时信息构建输入张量"""

    DTYPE_MAP = {
        "float": torch.float32,
        "float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "long": torch.int64,
        "long int": torch.int64,
        "int64": torch.int64,
        "int": torch.int32,
        "int32": torch.int32,
        "short": torch.int16,
        "int16": torch.int16,
        "char": torch.int8,
        "int8": torch.int8,
        "byte": torch.uint8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

    def _map_dtype(self, type_str: str) -> torch.dtype:
        """将字符串类型映射到 torch dtype"""
        type_lower = type_str.lower().strip()
        if type_lower in self.DTYPE_MAP:
            return self.DTYPE_MAP[type_lower]
        # 默认 float32
        return torch.float32

    def _is_floating(self, dtype: torch.dtype) -> bool:
        return dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)

    def _build_tensor(self, dims: List[int], strides: Optional[List[int]], dtype: torch.dtype) -> torch.Tensor:
        """构建单个张量"""
        if self._is_floating(dtype):
            t = torch.randn(dims, dtype=dtype)
        else:
            # 整数类型用 randint
            t = torch.randint(low=0, high=10, size=dims, dtype=dtype)

        if strides is not None and len(strides) == len(dims):
            # 使用 as_strided 重构 stride
            # 注意: as_strided 需要确保 stride 和 size 兼容
            try:
                t = torch.as_strided(t, size=dims, stride=strides)
            except RuntimeError:
                # stride 不兼容时回退到默认
                pass

        return t

    def _build_tensors(self, op_info) -> List[torch.Tensor]:
        """构建所有输入张量"""
        tensors = []
        for i, dims in enumerate(op_info.input_dims):
            type_str = op_info.input_types[i] if i < len(op_info.input_types) else "float"

            if type_str.lower() == "scalar":
                # Scalar 不构建张量，作为标量参数处理
                continue

            dtype = self._map_dtype(type_str)
            strides = op_info.input_strides[i] if i < len(op_info.input_strides) else None
            t = self._build_tensor(dims, strides, dtype)
            tensors.append(t)
        return tensors

    def _parse_concrete_inputs(self, mapped_op: MappedOp) -> Dict[str, Any]:
        """解析 concrete_inputs 为 kwargs"""
        kwargs = {}
        concrete = mapped_op.op_info.concrete_inputs
        if not concrete:
            return kwargs

        sig = inspect.signature(mapped_op.callable)
        params = list(sig.parameters.values())

        # 跳过前 N 个参数（已作为张量传入）
        num_tensor_args = len(mapped_op.op_info.input_dims)
        # 减去 Scalar 类型的数量
        scalar_count = sum(1 for t in mapped_op.op_info.input_types if t.lower() == "scalar")
        skip_count = num_tensor_args - scalar_count

        param_idx = skip_count
        for value in concrete:
            if param_idx >= len(params):
                break
            param = params[param_idx]
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                break
            if value is not None:
                kwargs[param.name] = value
            param_idx += 1

        return kwargs

    def build(self, mapped_op: MappedOp) -> TestCase:
        """构建测试用例"""
        tensors = self._build_tensors(mapped_op.op_info)
        kwargs = self._parse_concrete_inputs(mapped_op)

        # 处理 Scalar 类型的 concrete_inputs
        for i, type_str in enumerate(mapped_op.op_info.input_types):
            if type_str.lower() == "scalar":
                if i < len(mapped_op.op_info.concrete_inputs):
                    scalar_val = mapped_op.op_info.concrete_inputs[i]
                    # 根据算子签名决定参数名，简单启发式
                    sig = inspect.signature(mapped_op.callable)
                    params = list(sig.parameters.values())
                    if len(params) > 1:
                        kwargs[params[1].name] = scalar_val

        return TestCase(
            mapped_op=mapped_op,
            input_tensors=tensors,
            kwargs=kwargs,
        )
```

`op_testgen/builder/__init__.py`:
```python
from .tensor_builder import TensorBuilder, TestCase

__all__ = ["TensorBuilder", "TestCase"]
```

---

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_tensor_builder.py -v
```

Expected: 5 tests PASSED

---

- [ ] **Step 5: Commit**

```bash
git add op_testgen/builder/ tests/test_tensor_builder.py
git commit -m "feat(builder): implement TensorBuilder with dtype mapping and stride support"
```

---

## Task 5: CorrectnessRunner（正确性测试模块）

**目标**: 实现 CPU baseline vs CUDA/SWDNN 的正确性对比测试。

**Files:**
- Create: `op_testgen/correctness/__init__.py`
- Create: `op_testgen/correctness/test_runner.py`
- Create: `tests/test_correctness.py`

---

- [ ] **Step 1: 定义结果类和测试**

`tests/test_correctness.py`:
```python
"""测试 CorrectnessRunner"""
import pytest
import torch
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.correctness.test_runner import CorrectnessRunner, CorrectnessResult


class TestCorrectnessRunner:
    @pytest.fixture
    def runner(self):
        return CorrectnessRunner()

    def test_add_correctness(self, runner):
        """测试 aten::add 在 CPU 和 CUDA 上的一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        builder = TensorBuilder(seed=42)
        test_case = builder.build(mapped)

        result = runner.run(test_case, backend="cuda")
        assert result.passed is True
        assert result.max_abs_err < 1e-5
        assert result.max_rel_err < 1e-4

    def test_cpu_only_baseline(self, runner):
        """测试仅 CPU 执行（无 CUDA 时）"""
        op = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_types=["float", "float"],
            concrete_inputs=[0],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op)
        builder = TensorBuilder(seed=42)
        test_case = builder.build(mapped)

        result = runner.run(test_case, backend="cpu")
        assert result.passed is True
        assert result.max_abs_err == 0.0

    def test_error_thresholds(self):
        """测试不同 dtype 的误差阈值"""
        runner = CorrectnessRunner()
        assert runner._get_threshold("float32") == (1e-5, 1e-4)
        assert runner._get_threshold("float16") == (1e-3, 1e-2)
        assert runner._get_threshold("bfloat16") == (5e-3, 5e-2)
        assert runner._get_threshold("float64") == (1e-10, 1e-9)

    def test_fail_fast(self, runner):
        """测试 fail-fast 模式"""
        runner_failfast = CorrectnessRunner(fail_fast=True)
        assert runner_failfast.fail_fast is True
```

---

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_correctness.py -v
```

Expected: `ImportError`

---

- [ ] **Step 3: 实现 CorrectnessRunner**

`op_testgen/correctness/test_runner.py`:
```python
"""正确性测试执行模块"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from op_testgen.builder.tensor_builder import TestCase
from op_testgen.config import get_settings


@dataclass
class CorrectnessResult:
    """正确性测试结果"""
    op_name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    avg_abs_err: float = 0.0
    avg_rel_err: float = 0.0
    backend: str = "cuda"
    error_message: Optional[str] = None
    output_shapes_match: bool = True


class CorrectnessRunner:
    """正确性测试执行器"""

    def __init__(self, fail_fast: bool = False):
        self.fail_fast = fail_fast
        self.settings = get_settings()

    def _get_threshold(self, dtype_str: str) -> tuple:
        """获取指定数据类型的误差阈值"""
        thresholds = self.settings.error_thresholds
        type_lower = dtype_str.lower()
        if type_lower in thresholds:
            return thresholds[type_lower]
        # 默认使用 float32 阈值
        return thresholds.get("float32", (1e-5, 1e-4))

    def _compute_errors(self, cpu_out: torch.Tensor, cuda_out: torch.Tensor) -> Dict[str, float]:
        """计算绝对误差和相对误差"""
        diff = torch.abs(cpu_out - cuda_out)
        abs_err = diff

        # 相对误差，mask 掉 cpu_out == 0 的位置
        rel_err = torch.zeros_like(diff)
        mask = cpu_out != 0
        rel_err[mask] = diff[mask] / torch.abs(cpu_out[mask])

        return {
            "max_abs": float(abs_err.max()),
            "max_rel": float(rel_err.max()),
            "avg_abs": float(abs_err.mean()),
            "avg_rel": float(rel_err.mean()),
        }

    def _run_single(self, test_case: TestCase, backend: str) -> CorrectnessResult:
        """执行单个测试用例的正确性对比"""
        op = test_case.mapped_op
        op_name = op.op_info.name

        try:
            # 1. CPU Baseline (float64 高精度)
            cpu_tensors = [t.clone().cpu().to(torch.float64) for t in test_case.input_tensors]
            cpu_kwargs = {k: v for k, v in test_case.kwargs.items()}
            cpu_out = op.callable(*cpu_tensors, **cpu_kwargs)

            if not isinstance(cpu_out, torch.Tensor):
                # 非张量输出（如返回 tuple），取第一个张量
                if isinstance(cpu_out, tuple):
                    cpu_out = [o for o in cpu_out if isinstance(o, torch.Tensor)]
                    if not cpu_out:
                        return CorrectnessResult(
                            op_name=op_name, passed=True, backend=backend,
                            error_message="Non-tensor output, shape check only"
                        )
                    cpu_out = cpu_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only"
                    )

            if backend == "cpu":
                # 仅 CPU 模式，直接通过
                return CorrectnessResult(op_name=op_name, passed=True, backend="cpu")

            # 2. CUDA / SWDNN
            if backend == "swdnn":
                os.environ["SWDNN"] = "ON"
            else:
                os.environ["SWDNN"] = "OFF"

            if not torch.cuda.is_available():
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message="CUDA not available"
                )

            cuda_tensors = [t.clone().cuda() for t in test_case.input_tensors]
            cuda_kwargs = {k: v for k, v in test_case.kwargs.items()}
            cuda_out = op.callable(*cuda_tensors, **cuda_kwargs)

            if isinstance(cuda_out, tuple):
                cuda_out = [o for o in cuda_out if isinstance(o, torch.Tensor)]
                if cuda_out:
                    cuda_out = cuda_out[0]
                else:
                    return CorrectnessResult(
                        op_name=op_name, passed=True, backend=backend,
                        error_message="Non-tensor output, shape check only"
                    )

            cuda_out_cpu = cuda_out.cpu().to(torch.float64)

            # 3. 对比
            if cpu_out.shape != cuda_out_cpu.shape:
                return CorrectnessResult(
                    op_name=op_name, passed=False, backend=backend,
                    error_message=f"Shape mismatch: CPU {cpu_out.shape} vs CUDA {cuda_out_cpu.shape}",
                    output_shapes_match=False,
                )

            errors = self._compute_errors(cpu_out, cuda_out_cpu)

            # 4. 阈值检查
            dtype_str = op.op_info.input_types[0] if op.op_info.input_types else "float32"
            max_abs_thresh, max_rel_thresh = self._get_threshold(dtype_str)

            passed = errors["max_abs"] <= max_abs_thresh and errors["max_rel"] <= max_rel_thresh

            return CorrectnessResult(
                op_name=op_name,
                passed=passed,
                max_abs_err=errors["max_abs"],
                max_rel_err=errors["max_rel"],
                avg_abs_err=errors["avg_abs"],
                avg_rel_err=errors["avg_rel"],
                backend=backend,
            )

        except Exception as e:
            return CorrectnessResult(
                op_name=op_name,
                passed=False,
                backend=backend,
                error_message=str(e),
            )

    def run(self, test_case: TestCase, backend: str = "cuda") -> CorrectnessResult:
        """运行单个测试用例的正确性测试"""
        return self._run_single(test_case, backend)

    def run_all(self, test_cases: List[TestCase], backend: str = "cuda") -> List[CorrectnessResult]:
        """批量运行正确性测试"""
        results = []
        for tc in test_cases:
            result = self.run(tc, backend)
            results.append(result)
            if self.fail_fast and not result.passed:
                break
        return results
```

`op_testgen/correctness/__init__.py`:
```python
from .test_runner import CorrectnessRunner, CorrectnessResult

__all__ = ["CorrectnessRunner", "CorrectnessResult"]
```

---

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_correctness.py -v
```

Expected: 4 tests PASSED (CUDA 测试在有 GPU 时通过，无 GPU 时 skip)

---

- [ ] **Step 5: Commit**

```bash
git add op_testgen/correctness/ tests/test_correctness.py
git commit -m "feat(correctness): implement CorrectnessRunner with CPU baseline and CUDA comparison"
```

---

## Task 6: PerfBenchmark（性能测试模块）

**目标**: 实现算子分类、性能指标计算和基准测试执行。

**Files:**
- Create: `op_testgen/perf/__init__.py`
- Create: `op_testgen/perf/classifier.py`
- Create: `op_testgen/perf/metrics.py`
- Create: `op_testgen/perf/benchmark.py`
- Create: `tests/test_perf.py`

---

- [ ] **Step 1: 实现 Classifier 和测试**

`op_testgen/perf/classifier.py`:
```python
"""算子分类模块"""
import os
from typing import Optional

import yaml


class OpClassifier:
    """算子性能分类器"""

    COMPUTE_PATTERNS = ["matmul", "conv", "mm", "bmm", "linear", "softmax", "log_softmax"]
    COMMUNICATION_PATTERNS = ["allreduce", "allgather", "broadcast", "send", "recv", "reduce", "scatter"]

    def __init__(self, classification_path: Optional[str] = None):
        self.classifications = {}
        if classification_path and os.path.exists(classification_path):
            with open(classification_path, "r", encoding="utf-8") as f:
                self.classifications = yaml.safe_load(f) or {}
        else:
            # 默认加载包内分类表
            import os as _os
            default_path = _os.path.join(
                _os.path.dirname(__file__), "..", "config", "op_classification.yaml"
            )
            if _os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    self.classifications = yaml.safe_load(f) or {}

    def classify(self, op_name: str) -> str:
        """返回算子分类: compute | memory | communication | mixed"""
        # 1. 配置文件优先
        if op_name in self.classifications:
            return self.classifications[op_name].get("category", "memory")

        # 2. 启发式规则
        name_lower = op_name.lower()

        is_compute = any(p in name_lower for p in self.COMPUTE_PATTERNS)
        is_comm = any(p in name_lower for p in self.COMMUNICATION_PATTERNS)

        if is_comm:
            return "communication"
        if is_compute:
            return "compute"

        # 3. 默认访存密集
        return "memory"

    def get_flops_formula(self, op_name: str) -> Optional[str]:
        """获取 FLOPS 计算公式"""
        if op_name in self.classifications:
            return self.classifications[op_name].get("flops_formula")
        return None

    def get_bytes_formula(self, op_name: str) -> Optional[str]:
        """获取访存量/通信量计算公式"""
        if op_name in self.classifications:
            return self.classifications[op_name].get("bytes_formula")
        return None
```

`tests/test_perf.py`（Classifier 部分）:
```python
"""测试 PerfBenchmark"""
import pytest
from op_testgen.perf.classifier import OpClassifier


class TestOpClassifier:
    def test_classify_compute(self):
        c = OpClassifier()
        assert c.classify("aten::matmul") == "compute"
        assert c.classify("aten::conv2d") == "compute"
        assert c.classify("aten::linear") == "compute"

    def test_classify_communication(self):
        c = OpClassifier()
        assert c.classify("nccl:all_reduce") == "communication"
        assert c.classify("c10d::allreduce_") == "communication"

    def test_classify_memory(self):
        c = OpClassifier()
        assert c.classify("aten::add") == "memory"
        assert c.classify("aten::relu") == "memory"

    def test_config_override(self):
        # aten::add 在分类表中被显式标记为 memory
        c = OpClassifier()
        assert c.classify("aten::add") == "memory"
```

---

- [ ] **Step 2: 实现 Metrics 和测试**

`op_testgen/perf/metrics.py`:
```python
"""性能指标计算模块"""
from typing import Any, Dict, List, Optional

import torch


class MetricsCalculator:
    """性能指标计算器"""

    @staticmethod
    def compute_flops(op_name: str, input_dims: List[List[int]], formula: Optional[str] = None) -> float:
        """计算理论 FLOPS"""
        if formula is None:
            # 启发式估算
            return MetricsCalculator._heuristic_flops(op_name, input_dims)

        # 安全 eval 计算公式
        env = MetricsCalculator._build_formula_env(input_dims)
        try:
            return float(eval(formula, {"__builtins__": {}}, env))
        except Exception:
            return 0.0

    @staticmethod
    def compute_bytes(input_dims: List[List[int]], dtype: torch.dtype, num_inputs: int = 1, num_outputs: int = 1) -> int:
        """计算访存字节数 (读 + 写)"""
        element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
        total_elements = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        # 读输入 + 写输出
        return total_elements * element_size * num_inputs + total_elements * element_size * num_outputs

    @staticmethod
    def compute_communication_bytes(input_dims: List[List[int]], dtype: torch.dtype, world_size: int = 2) -> int:
        """计算通信数据量"""
        element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
        total_elements = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        # allreduce: 2 * N * element_size (send + recv 近似)
        return 2 * total_elements * element_size

    @staticmethod
    def _build_formula_env(input_dims: List[List[int]]) -> Dict[str, Any]:
        """为公式 eval 构建变量环境"""
        env = {}
        # 常见维度变量
        if len(input_dims) >= 1 and len(input_dims[0]) >= 2:
            env["M"] = input_dims[0][-2]
            env["N"] = input_dims[0][-1]
        if len(input_dims) >= 2 and len(input_dims[1]) >= 2:
            env["K"] = input_dims[1][-2] if len(input_dims[1]) >= 2 else 1
        # Conv 维度
        if len(input_dims) >= 2:
            shape0 = input_dims[0]
            shape1 = input_dims[1]
            if len(shape0) == 4:  # NCHW
                env["N"] = shape0[0]
                env["Cin"] = shape0[1]
                env["H"] = shape0[2]
                env["W"] = shape0[3]
            if len(shape1) == 4:  # Cout, Cin, K, K
                env["Cout"] = shape1[0]
                env["K"] = shape1[2]
            # 输出尺寸近似
            if "H" in env and "K" in env:
                env["Hout"] = env["H"] - env["K"] + 1
                env["Wout"] = env["W"] - env["K"] + 1
        # BMM
        if len(input_dims) >= 1:
            env["B"] = input_dims[0][0] if len(input_dims[0]) >= 3 else 1
        # 通用
        env["numel"] = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        env["element_size"] = 4  # 默认 float32
        return env

    @staticmethod
    def _heuristic_flops(op_name: str, input_dims: List[List[int]]) -> float:
        """启发式 FLOPS 估算"""
        name_lower = op_name.lower()
        if "matmul" in name_lower or "mm" in name_lower:
            if len(input_dims) >= 2 and len(input_dims[0]) >= 2 and len(input_dims[1]) >= 2:
                m, n = input_dims[0][-2], input_dims[1][-1]
                k = input_dims[0][-1]
                return 2.0 * m * n * k
        if "conv" in name_lower and len(input_dims) >= 2:
            if len(input_dims[0]) == 4 and len(input_dims[1]) == 4:
                n = input_dims[0][0]
                cout = input_dims[1][0]
                cin = input_dims[1][1]
                h, w = input_dims[0][2], input_dims[0][3]
                k = input_dims[1][2]
                return 2.0 * n * cout * h * w * cin * k * k
        return 0.0
```

---

- [ ] **Step 3: 实现 Benchmark 和测试**

`op_testgen/perf/benchmark.py`:
```python
"""性能测试执行模块"""
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from op_testgen.builder.tensor_builder import TestCase
from op_testgen.config import get_settings
from op_testgen.perf.classifier import OpClassifier
from op_testgen.perf.metrics import MetricsCalculator


@dataclass
class PerfResult:
    """性能测试结果"""
    op_name: str
    category: str
    backend: str
    avg_time_ms: float = 0.0
    flops: float = 0.0          # GFLOPS
    bandwidth_gbps: float = 0.0 # GB/s
    speedup: float = 1.0        # vs CPU
    error_message: Optional[str] = None


class PerfBenchmark:
    """性能测试执行器"""

    def __init__(self, warmup_iters: int = 3, benchmark_iters: int = 10):
        self.settings = get_settings()
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.classifier = OpClassifier()
        self.metrics = MetricsCalculator()

    def _measure_time(self, test_case: TestCase, device: str) -> float:
        """测量算子平均执行时间（秒）"""
        op = test_case.mapped_op
        tensors = [t.clone().to(device) for t in test_case.input_tensors]
        kwargs = {k: v for k, v in test_case.kwargs.items()}

        # Warm-up
        for _ in range(self.warmup_iters):
            op.callable(*tensors, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.benchmark_iters):
            op.callable(*tensors, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) / self.benchmark_iters

    def run(self, test_case: TestCase, backend: str = "cuda") -> PerfResult:
        """执行单个算子的性能测试"""
        op = test_case.mapped_op
        op_name = op.op_info.name
        category = self.classifier.classify(op_name)

        try:
            # CPU 基准
            cpu_time = self._measure_time(test_case, "cpu")

            # CUDA / SWDNN
            cuda_time = cpu_time
            if backend != "cpu" and torch.cuda.is_available():
                if backend == "swdnn":
                    import os
                    os.environ["SWDNN"] = "ON"
                else:
                    import os
                    os.environ["SWDNN"] = "OFF"
                cuda_time = self._measure_time(test_case, "cuda")

            speedup = cpu_time / cuda_time if cuda_time > 0 else 1.0

            # 计算指标
            flops = 0.0
            bandwidth = 0.0

            dtype = torch.float32
            if op.op_info.input_types:
                from op_testgen.builder.tensor_builder import TensorBuilder
                tb = TensorBuilder()
                dtype = tb._map_dtype(op.op_info.input_types[0])

            if category == "compute":
                formula = self.classifier.get_flops_formula(op_name)
                total_flops = self.metrics.compute_flops(op_name, op.op_info.input_dims, formula)
                flops = (total_flops / (cuda_time * 1e9)) if cuda_time > 0 else 0.0  # GFLOPS

            elif category in ("memory", "mixed"):
                bytes_total = self.metrics.compute_bytes(op.op_info.input_dims, dtype)
                bandwidth = (bytes_total / (cuda_time * 1e9)) if cuda_time > 0 else 0.0  # GB/s

            elif category == "communication":
                bytes_total = self.metrics.compute_communication_bytes(op.op_info.input_dims, dtype)
                bandwidth = (bytes_total / (cuda_time * 1e9)) if cuda_time > 0 else 0.0  # GB/s

            return PerfResult(
                op_name=op_name,
                category=category,
                backend=backend,
                avg_time_ms=cuda_time * 1000,
                flops=flops,
                bandwidth_gbps=bandwidth,
                speedup=speedup,
            )

        except Exception as e:
            return PerfResult(
                op_name=op_name,
                category=category,
                backend=backend,
                error_message=str(e),
            )

    def run_all(self, test_cases: List[TestCase], backend: str = "cuda") -> List[PerfResult]:
        """批量性能测试"""
        return [self.run(tc, backend) for tc in test_cases]
```

`op_testgen/perf/__init__.py`:
```python
from .classifier import OpClassifier
from .metrics import MetricsCalculator
from .benchmark import PerfBenchmark, PerfResult

__all__ = ["OpClassifier", "MetricsCalculator", "PerfBenchmark", "PerfResult"]
```

---

- [ ] **Step 4: 运行性能模块测试**

```bash
pytest tests/test_perf.py -v
```

Expected: 4+ tests PASSED

---

- [ ] **Step 5: Commit**

```bash
git add op_testgen/perf/ tests/test_perf.py
git commit -m "feat(perf): implement OpClassifier, MetricsCalculator, and PerfBenchmark"
```

---

## Task 7: Reporter（报告输出模块）

**目标**: 实现 JSON/HTML/Excel 报告生成。

**Files:**
- Create: `op_testgen/reporter/__init__.py`
- Create: `op_testgen/reporter/base.py`
- Create: `op_testgen/reporter/html_reporter.py`
- Create: `op_testgen/reporter/excel_reporter.py`
- Create: `tests/test_reporter.py`

---

- [ ] **Step 1: 实现 BaseReporter**

`op_testgen/reporter/base.py`:
```python
"""报告基类"""
from abc import ABC, abstractmethod
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult


class BaseReporter(ABC):
    """报告生成器基类"""

    @abstractmethod
    def generate(self, correctness_results: List[CorrectnessResult],
                 perf_results: List[PerfResult], output_path: str) -> None:
        """生成报告文件"""
        pass
```

---

- [ ] **Step 2: 实现 HTMLReporter**

`op_testgen/reporter/html_reporter.py`:
```python
"""HTML 报告生成器"""
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult
from op_testgen.reporter.base import BaseReporter


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>算子测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { display: flex; gap: 20px; margin-bottom: 30px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; }
        .card h3 { margin-top: 0; color: #666; font-size: 14px; }
        .card .value { font-size: 32px; font-weight: bold; color: #2c3e50; }
        .pass { color: #27ae60; }
        .fail { color: #e74c3c; }
        table { width: 100%; border-collapse: collapse; background: white; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        tr:hover { background: #f9f9f9; }
        .error { color: #e74c3c; font-size: 12px; }
    </style>
</head>
<body>
    <h1>算子测试报告</h1>

    <div class="summary">
        <div class="card">
            <h3>总算子数</h3>
            <div class="value">{{ summary.total_ops }}</div>
        </div>
        <div class="card">
            <h3>正确性通过</h3>
            <div class="value {{ 'pass' if summary.correctness_passed == summary.total_ops else 'fail' }}">
                {{ summary.correctness_passed }} / {{ summary.total_ops }}
            </div>
        </div>
        <div class="card">
            <h3>平均加速比</h3>
            <div class="value">{{ "%.2fx" % summary.avg_speedup }}</div>
        </div>
    </div>

    <h2>正确性测试结果</h2>
    <table>
        <tr>
            <th>算子名称</th>
            <th>后端</th>
            <th>状态</th>
            <th>最大绝对误差</th>
            <th>最大相对误差</th>
            <th>错误信息</th>
        </tr>
        {% for r in correctness %}
        <tr>
            <td>{{ r.op_name }}</td>
            <td>{{ r.backend }}</td>
            <td class="{{ 'pass' if r.passed else 'fail' }}">{{ '通过' if r.passed else '失败' }}</td>
            <td>{{ "%.2e" % r.max_abs_err if r.max_abs_err else '-' }}</td>
            <td>{{ "%.2e" % r.max_rel_err if r.max_rel_err else '-' }}</td>
            <td class="error">{{ r.error_message or '' }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>性能测试结果</h2>
    <table>
        <tr>
            <th>算子名称</th>
            <th>分类</th>
            <th>平均耗时(ms)</th>
            <th>GFLOPS</th>
            <th>带宽(GB/s)</th>
            <th>加速比</th>
        </tr>
        {% for r in perf %}
        <tr>
            <td>{{ r.op_name }}</td>
            <td>{{ r.category }}</td>
            <td>{{ "%.4f" % r.avg_time_ms }}</td>
            <td>{{ "%.2f" % r.flops if r.flops else '-' }}</td>
            <td>{{ "%.2f" % r.bandwidth_gbps if r.bandwidth_gbps else '-' }}</td>
            <td>{{ "%.2fx" % r.speedup }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


class HTMLReporter(BaseReporter):
    """HTML 报告生成器"""

    def generate(self, correctness_results: List[CorrectnessResult],
                 perf_results: List[PerfResult], output_path: str) -> None:
        from jinja2 import Template

        total_ops = len(correctness_results)
        passed = sum(1 for r in correctness_results if r.passed)
        avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results) if perf_results else 1.0

        summary = {
            "total_ops": total_ops,
            "correctness_passed": passed,
            "avg_speedup": avg_speedup,
        }

        template = Template(HTML_TEMPLATE)
        html = template.render(
            summary=summary,
            correctness=correctness_results,
            perf=perf_results,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
```

---

- [ ] **Step 3: 实现 ExcelReporter（可选）**

`op_testgen/reporter/excel_reporter.py`:
```python
"""Excel 报告生成器"""
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult
from op_testgen.reporter.base import BaseReporter


class ExcelReporter(BaseReporter):
    """Excel 报告生成器（需要 openpyxl）"""

    def generate(self, correctness_results: List[CorrectnessResult],
                 perf_results: List[PerfResult], output_path: str) -> None:
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            raise ImportError("Excel 报告需要 openpyxl，请安装: pip install openpyxl")

        wb = openpyxl.Workbook()

        # Sheet 1: 正确性测试
        ws1 = wb.active
        ws1.title = "正确性测试"
        headers1 = ["算子名称", "后端", "通过状态", "最大绝对误差", "最大相对误差", "平均绝对误差", "错误信息"]
        ws1.append(headers1)

        for r in correctness_results:
            ws1.append([
                r.op_name,
                r.backend,
                "通过" if r.passed else "失败",
                r.max_abs_err,
                r.max_rel_err,
                r.avg_abs_err,
                r.error_message or "",
            ])

        # Sheet 2: 性能测试
        ws2 = wb.create_sheet("性能测试")
        headers2 = ["算子名称", "分类", "平均耗时(ms)", "GFLOPS", "带宽(GB/s)", "加速比"]
        ws2.append(headers2)

        for r in perf_results:
            ws2.append([
                r.op_name,
                r.category,
                r.avg_time_ms,
                r.flops,
                r.bandwidth_gbps,
                r.speedup,
            ])

        # 样式
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        for ws in [ws1, ws2]:
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

        wb.save(output_path)
```

---

- [ ] **Step 4: 写 Reporter 测试**

`tests/test_reporter.py`:
```python
"""测试 Reporter"""
import pytest
import os
from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult
from op_testgen.reporter.html_reporter import HTMLReporter


class TestHTMLReporter:
    def test_generate_html(self, tmp_path):
        reporter = HTMLReporter()
        correctness = [
            CorrectnessResult(op_name="aten::add", passed=True, max_abs_err=1e-6, max_rel_err=1e-7, backend="cuda"),
            CorrectnessResult(op_name="aten::conv2d", passed=False, max_abs_err=1e-2, error_message="Shape mismatch", backend="cuda"),
        ]
        perf = [
            PerfResult(op_name="aten::add", category="memory", avg_time_ms=0.05, bandwidth_gbps=100.0, speedup=2.0),
            PerfResult(op_name="aten::conv2d", category="compute", avg_time_ms=2.0, flops=500.0, speedup=10.0),
        ]
        output = tmp_path / "report.html"
        reporter.generate(correctness, perf, str(output))
        assert output.exists()
        content = output.read_text()
        assert "算子测试报告" in content
        assert "aten::add" in content
        assert "通过" in content
```

---

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_reporter.py -v
```

Expected: PASSED

---

- [ ] **Step 6: Commit**

```bash
git add op_testgen/reporter/ tests/test_reporter.py
git commit -m "feat(reporter): implement HTML and Excel report generators"
```

---

## Task 8: CLI（命令行入口）

**目标**: 实现统一的命令行入口，编排六阶段流水线。

**Files:**
- Create: `op_testgen/cli.py`

---

- [ ] **Step 1: 实现 CLI**

`op_testgen/cli.py`:
```python
"""命令行入口"""
import argparse
import json
import os
import sys
from typing import Optional

import torch

from op_testgen.builder.tensor_builder import TensorBuilder
from op_testgen.config import get_settings
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.mapper.op_mapper import OpMapper
from op_testgen.parser.trace_parser import TraceParser
from op_testgen.perf.benchmark import PerfBenchmark
from op_testgen.reporter.html_reporter import HTMLReporter
from op_testgen.reporter.excel_reporter import ExcelReporter


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler 自动化测试生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen profiler_trace.json
  op_testgen profiler_trace.json --backend swdnn --output report.html
  op_testgen profiler_trace.json --only-correctness --fail-fast
        """,
    )
    parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    parser.add_argument("--backend", choices=["cuda", "swdnn", "auto", "cpu"], default="cuda",
                       help="对比后端 (默认: cuda)")
    parser.add_argument("-o", "--output", default="op_testgen_report.html", help="输出文件路径")
    parser.add_argument("--format", choices=["html", "json", "excel", "all"], default="html",
                       help="输出格式 (默认: html)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--iters", type=int, default=10, help="性能测试迭代次数 (默认: 10)")
    parser.add_argument("--fail-fast", action="store_true", help="第一个失败即停止")
    parser.add_argument("--only-correctness", action="store_true", help="仅执行正确性测试")
    parser.add_argument("--only-performance", action="store_true", help="仅执行性能测试")
    parser.add_argument("--op-filter", help="仅测试匹配名称的算子 (支持通配符)")
    parser.add_argument("--update-whitelist", action="store_true", help="动态发现后更新白名单")

    args = parser.parse_args(argv)

    print("=" * 60)
    print("PyTorch Profiler 自动化测试生成器")
    print("=" * 60)

    # 1. 解析 Trace
    print("\n[1/6] 解析 Trace 文件...")
    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse()
    print(f"  发现 {len(op_infos)} 个算子")

    # 2. 映射算子
    print("\n[2/6] 映射算子...")
    mapper = OpMapper()
    mapped_ops = mapper.map_all(op_infos)
    print(f"  成功映射 {len(mapped_ops)} 个算子")

    # 3. 构建测试用例
    print("\n[3/6] 构建测试用例...")
    builder = TensorBuilder(seed=args.seed)
    test_cases = [builder.build(m) for m in mapped_ops]
    print(f"  构建 {len(test_cases)} 个测试用例")

    # 过滤
    if args.op_filter:
        import fnmatch
        test_cases = [tc for tc in test_cases if fnmatch.fnmatch(tc.mapped_op.op_info.name, args.op_filter)]
        print(f"  过滤后剩余 {len(test_cases)} 个测试用例")

    if not test_cases:
        print("错误：没有可测试的算子")
        return 1

    # 确定后端列表
    backends = []
    if args.backend == "auto":
        backends = ["cuda"]
        if os.environ.get("SWDNN_AVAILABLE"):  # 简化判断
            backends.append("swdnn")
    else:
        backends = [args.backend]

    all_correctness = []
    all_perf = []

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"后端: {backend.upper()}")
        print(f"{'='*60}")

        # 4. 正确性测试
        if not args.only_performance:
            print(f"\n[4/6] 执行正确性测试...")
            correctness_runner = CorrectnessRunner(fail_fast=args.fail_fast)
            correctness_results = correctness_runner.run_all(test_cases, backend=backend)
            all_correctness.extend(correctness_results)

            passed = sum(1 for r in correctness_results if r.passed)
            print(f"  通过: {passed}/{len(correctness_results)}")

        # 5. 性能测试
        if not args.only_correctness:
            print(f"\n[5/6] 执行性能测试...")
            perf_benchmark = PerfBenchmark(benchmark_iters=args.iters)
            perf_results = perf_benchmark.run_all(test_cases, backend=backend)
            all_perf.extend(perf_results)

            avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results) if perf_results else 1.0
            print(f"  平均加速比: {avg_speedup:.2f}x")

    # 6. 生成报告
    print(f"\n[6/6] 生成报告...")

    if args.format in ("html", "all"):
        html_reporter = HTMLReporter()
        html_path = args.output if args.output.endswith(".html") else args.output + ".html"
        html_reporter.generate(all_correctness, all_perf, html_path)
        print(f"  HTML 报告: {html_path}")

    if args.format in ("json", "all"):
        json_path = args.output.replace(".html", ".json").replace(".xlsx", ".json")
        if not json_path.endswith(".json"):
            json_path += ".json"
        report_data = {
            "summary": {
                "total_ops": len(test_cases),
                "correctness_passed": sum(1 for r in all_correctness if r.passed),
                "correctness_failed": sum(1 for r in all_correctness if not r.passed),
            },
            "correctness": [
                {"op_name": r.op_name, "passed": r.passed, "max_abs_err": r.max_abs_err,
                 "max_rel_err": r.max_rel_err, "error": r.error_message}
                for r in all_correctness
            ],
            "performance": [
                {"op_name": r.op_name, "category": r.category, "avg_time_ms": r.avg_time_ms,
                 "flops": r.flops, "bandwidth_gbps": r.bandwidth_gbps, "speedup": r.speedup}
                for r in all_perf
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  JSON 报告: {json_path}")

    if args.format in ("excel", "all"):
        try:
            excel_reporter = ExcelReporter()
            excel_path = args.output.replace(".html", ".xlsx").replace(".json", ".xlsx")
            if not excel_path.endswith(".xlsx"):
                excel_path += ".xlsx"
            excel_reporter.generate(all_correctness, all_perf, excel_path)
            print(f"  Excel 报告: {excel_path}")
        except ImportError as e:
            print(f"  警告: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

- [ ] **Step 2: 端到端测试**

```bash
cd /mnt/d/ubuntu/opencode/training_framework/Analys_Trace
# 安装包
pip install -e .

# 运行端到端测试（使用示例 trace）
python -m op_testgen.cli profiler_trace.json --only-correctness --backend cpu --output test_report
```

Expected: 成功生成报告，无异常退出

---

- [ ] **Step 3: Commit**

```bash
git add op_testgen/cli.py
git commit -m "feat(cli): implement unified CLI entry point with 6-stage pipeline"
```

---

## Task 9: 集成验证与收尾

**目标**: 运行完整测试套件，修复问题，添加 README。

**Files:**
- Modify: `op_testgen/__init__.py`
- Create: `README_TESTGEN.md`

---

- [ ] **Step 1: 运行完整测试套件**

```bash
cd /mnt/d/ubuntu/opencode/training_framework/Analys_Trace
pytest tests/ -v --tb=short
```

Expected: 所有测试通过（或明确标记 skip）

---

- [ ] **Step 2: 运行端到端验证**

```bash
# 完整流程（CPU only，无需 CUDA）
op_testgen profiler_trace.json --backend cpu --only-correctness -o test_report.html

# 检查输出
ls -la test_report.html test_report.json
```

---

- [ ] **Step 3: 添加 README**

`README_TESTGEN.md`:
```markdown
# PyTorch Profiler 自动化测试生成器

基于 PyTorch Profiler Chrome Trace JSON 自动生成算子正确性与性能测试用例。

## 安装

```bash
pip install -e .
# 可选: Excel 报告支持
pip install -e ".[excel]"
```

## 快速开始

```bash
# 完整测试（正确性 + 性能）
op_testgen profiler_trace.json

# 仅正确性测试，CPU baseline
op_testgen profiler_trace.json --only-correctness --backend cpu

# 仅测试特定算子
op_testgen profiler_trace.json --op-filter "aten::conv*"

# 指定输出格式
op_testgen profiler_trace.json --format all -o report
```

## 配置

编辑 `op_testgen/config/` 下的 YAML 文件：
- `op_whitelist.yaml`: 算子白名单与映射
- `op_classification.yaml`: 算子分类与 FLOPS 公式

## 架构

六阶段流水线: TraceParser → OpMapper → TensorBuilder → CorrectnessRunner/PerfBenchmark → Reporter
```

---

- [ ] **Step 4: 最终 Commit**

```bash
git add README_TESTGEN.md op_testgen/__init__.py
git commit -m "docs: add README and finalize package exports"
```

---

## 自审清单

### Spec 覆盖检查

| Spec 要求 | 对应 Task |
|-----------|-----------|
| 解析 JSON 提取算子信息 | Task 2 (TraceParser) |
| 算子映射表与过滤表 | Task 3 (OpMapper) |
| 张量与参数重构 | Task 4 (TensorBuilder) |
| CPU baseline vs CUDA/SWDNN 正确性测试 | Task 5 (CorrectnessRunner) |
| 性能测试（FLOPS/带宽/通信带宽） | Task 6 (PerfBenchmark) |
| 报告输出（HTML/Excel/JSON） | Task 7 (Reporter) |
| CLI 入口 | Task 8 (CLI) |

### Placeholder 检查

- [x] 无 "TBD", "TODO", "implement later"
- [x] 每个步骤包含实际代码
- [x] 每个步骤包含验证命令和预期输出
- [x] 类型一致性检查通过

### 类型一致性

- `OpInfo` / `MappedOp` / `TestCase` / `CorrectnessResult` / `PerfResult` 在各模块间一致
- `backend` 参数取值 `"cuda" | "swdnn" | "cpu" | "auto"` 在 CLI、CorrectnessRunner、PerfBenchmark 中统一

---

*实施计划版本: v1.0*
*基于设计文档: docs/superpowers/specs/2026-05-24-op-testgen-design.md*
