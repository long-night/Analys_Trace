# 测试用例生成与执行分离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `op_testgen test` 拆分为 `generate`（生成 `.py` 测试文件）和 `run`（执行 `.py` 文件）两个阶段，同时保留 `test` 作为快捷方式。

**Architecture:** 新增 `op_testgen/generator/` 模块，包含 `TestCaseGenerator`（序列化 + 模板渲染生成 `.py` 文件）和 `TestCaseRunner`（通过 `subprocess` 执行 `.py` 文件）。CLI 新增 `generate` 和 `run` 子命令，`test` 子命令逻辑不变。

**Tech Stack:** Python 3.10+, pytest, standard library (`string.Template`, `subprocess`, `ast`)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `op_testgen/generator/__init__.py` | 包初始化，暴露公共 API |
| `op_testgen/generator/test_case_generator.py` | 将 `MappedOp` 列表序列化为 `.py` 文件 |
| `op_testgen/generator/template.py` | 测试文件模板字符串 |
| `op_testgen/generator/test_case_runner.py` | 执行生成的 `.py` 文件 |
| `op_testgen/cli.py` | 新增 `generate` 和 `run` 子命令 |
| `tests/test_generator.py` | generator 模块的单元测试和集成测试 |

---

## Task 1: 创建 generator 包 + 序列化测试

**Files:**
- Create: `op_testgen/generator/__init__.py`
- Create: `tests/test_generator.py`
- Create: `op_testgen/generator/test_case_generator.py`

### Step 1.1: 创建包初始化文件

```bash
touch op_testgen/generator/__init__.py
```

内容：
```python
"""测试用例生成器模块"""
from op_testgen.generator.test_case_generator import TestCaseGenerator
from op_testgen.generator.test_case_runner import TestCaseRunner

__all__ = ["TestCaseGenerator", "TestCaseRunner"]
```

### Step 1.2: 写序列化单元测试

在 `tests/test_generator.py` 中：

```python
"""测试 generator 模块"""
import pytest
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.mapper.op_mapper import OpMapper, MappedOp
from op_testgen.generator.test_case_generator import TestCaseGenerator


class TestSerializeTestCase:
    def test_serialize_basic_op(self):
        """测试基本算子的序列化"""
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[3, 4], [3, 4]],
            input_strides=[[4, 1], [4, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        data = gen._serialize_test_case(mapped)

        assert data["op_name"] == "aten::add"
        assert data["callable_path"] == "torch.add"
        assert data["input_dims"] == [[3, 4], [3, 4]]
        assert data["input_strides"] == [[4, 1], [4, 1]]
        assert data["input_types"] == ["float", "float"]
        assert data["concrete_inputs"] == []

    def test_serialize_op_with_concrete_inputs(self):
        """测试带 concrete_inputs 的算子序列化"""
        op_info = OpInfo(
            name="aten::sum",
            input_dims=[[2, 3, 4]],
            input_strides=[[12, 4, 1]],
            input_types=["float"],
            concrete_inputs=[0, True],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        data = gen._serialize_test_case(mapped)

        assert data["op_name"] == "aten::sum"
        assert data["concrete_inputs"] == [0, True]
```

### Step 1.3: 运行测试确认失败

```bash
pytest tests/test_generator.py::TestSerializeTestCase -v
```

**Expected:** FAIL — `TestCaseGenerator` 类不存在

### Step 1.4: 实现最小序列化逻辑

在 `op_testgen/generator/test_case_generator.py` 中：

```python
"""测试用例生成器"""
from typing import Any, Dict, List

from op_testgen.mapper.op_mapper import MappedOp


class TestCaseGenerator:
    """将 MappedOp 序列化为可执行的 Python 测试文件"""

    def __init__(self):
        pass

    def _serialize_test_case(self, mapped_op: MappedOp) -> Dict[str, Any]:
        """将单个 MappedOp 序列化为字典"""
        op_info = mapped_op.op_info
        return {
            "op_name": op_info.name,
            "callable_path": mapped_op.callable_path,
            "input_dims": op_info.input_dims,
            "input_strides": op_info.input_strides,
            "input_types": op_info.input_types,
            "concrete_inputs": op_info.concrete_inputs,
        }
```

### Step 1.5: 运行测试确认通过

```bash
pytest tests/test_generator.py::TestSerializeTestCase -v
```

**Expected:** PASS

### Step 1.6: 提交

```bash
git add op_testgen/generator/ tests/test_generator.py
git commit -m "feat(generator): add TestCaseGenerator with serialization"
```

---

## Task 2: 模板渲染

**Files:**
- Create: `op_testgen/generator/template.py`
- Modify: `tests/test_generator.py`
- Modify: `op_testgen/generator/test_case_generator.py`

### Step 2.1: 写模板测试

在 `tests/test_generator.py` 中添加：

```python
import re


class TestTemplateRendering:
    def test_rendered_file_contains_header(self):
        """测试生成的文件包含头部注释"""
        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        assert "Auto-generated test cases" in content
        assert "test.json" in content

    def test_rendered_file_contains_test_data(self):
        """测试生成的文件包含 TEST_CASES_DATA"""
        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        assert "TEST_CASES_DATA" in content
        assert "aten::add" in content
        assert "torch.add" in content

    def test_rendered_file_is_valid_python(self):
        """测试生成的文件是合法 Python 语法"""
        import ast

        gen = TestCaseGenerator()
        data = [
            {
                "op_name": "aten::add",
                "callable_path": "torch.add",
                "input_dims": [[2, 3], [2, 3]],
                "input_strides": [[3, 1], [3, 1]],
                "input_types": ["float", "float"],
                "concrete_inputs": [],
            }
        ]
        options = {"source_trace": "test.json", "backend": "cpu", "seed": 42}
        content = gen._render_template(data, options)

        # 验证语法正确
        ast.parse(content)
```

### Step 2.2: 运行测试确认失败

```bash
pytest tests/test_generator.py::TestTemplateRendering -v
```

**Expected:** FAIL — `_render_template` 方法不存在

### Step 2.3: 实现模板模块

创建 `op_testgen/generator/template.py`：

```python
"""测试文件模板"""

TEST_FILE_TEMPLATE = '''#!/usr/bin/env python3
# Auto-generated test cases from trace: {source_trace}
# Generated at: {generated_at}
# Default options: backend={backend}, seed={seed}, iters={iters}

import argparse
import importlib
import sys
from typing import Any, Dict, List

import torch
from op_testgen.builder.tensor_builder import TensorBuilder, TestCase
from op_testgen.correctness.test_runner import CorrectnessRunner
from op_testgen.mapper.op_mapper import MappedOp
from op_testgen.parser.trace_parser import OpInfo
from op_testgen.perf.benchmark import PerfBenchmark

TEST_CASES_DATA = {test_cases_data_repr}


def build_test_cases(seed: int = {seed}) -> List[TestCase]:
    """从序列化数据重建 TestCase 列表"""
    torch.manual_seed(seed)
    builder = TensorBuilder(seed=seed)
    test_cases = []

    for data in TEST_CASES_DATA:
        op_info = OpInfo(
            name=data["op_name"],
            input_dims=data["input_dims"],
            input_strides=data["input_strides"],
            input_types=data["input_types"],
            concrete_inputs=data["concrete_inputs"],
        )

        # 解析 callable
        parts = data["callable_path"].split(".")
        try:
            module = importlib.import_module(".".join(parts[:-1]))
            callable_obj = getattr(module, parts[-1])
        except (ImportError, AttributeError) as e:
            print(f"警告: 无法导入 {{data['callable_path']}}: {{e}}")
            continue

        mapped_op = MappedOp(
            op_info=op_info,
            callable=callable_obj,
            callable_path=data["callable_path"],
            namespace=data["callable_path"].split(".")[0],
            support_status="generated",
        )

        try:
            test_case = builder.build(mapped_op)
            test_cases.append(test_case)
        except Exception as e:
            print(f"警告: 构建 {{data['op_name']}} 失败: {{e}}")
            continue

    return test_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-generated operator tests")
    parser.add_argument("--backend", default="{backend}", choices=["cuda", "swdnn", "cpu"],
                        help="Backend to test against")
    parser.add_argument("--only-correctness", action="store_true",
                        help="Only run correctness tests")
    parser.add_argument("--only-performance", action="store_true",
                        help="Only run performance tests")
    parser.add_argument("--iters", type=int, default={iters},
                        help="Performance benchmark iterations")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first failure")
    parser.add_argument("--seed", type=int, default={seed},
                        help="Random seed for tensor generation")
    args = parser.parse_args()

    print("=" * 60)
    print("Auto-generated Operator Tests")
    print("=" * 60)

    test_cases = build_test_cases(seed=args.seed)
    print(f"\\n构建 {{len(test_cases)}} 个测试用例")

    if not test_cases:
        print("错误: 没有可测试的算子")
        return 1

    all_correctness = []
    all_perf = []

    if not args.only_performance:
        print("\\n执行正确性测试...")
        runner = CorrectnessRunner(fail_fast=args.fail_fast)
        for tc in test_cases:
            result = runner.run(tc, backend=args.backend)
            all_correctness.append(result)
            status = "通过" if result.passed else "失败"
            if result.error_message:
                print(f"  [{{status}}] {{result.op_name}}: {{result.error_message}}")
            else:
                print(f"  [{{status}}] {{result.op_name}}: max_abs={{result.max_abs_err:.2e}}, max_rel={{result.max_rel_err:.2e}}")
        passed = sum(1 for r in all_correctness if r.passed)
        print(f"  通过: {{passed}}/{{len(all_correctness)}}")

    if not args.only_correctness:
        print("\\n执行性能测试...")
        benchmark = PerfBenchmark(benchmark_iters=args.iters)
        perf_results = benchmark.run_all(test_cases, backend=args.backend)
        all_perf.extend(perf_results)
        if perf_results:
            avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results)
            print(f"  平均加速比: {{avg_speedup:.2f}}x")

    print("\\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 返回非零退出码如果有失败
    failed = sum(1 for r in all_correctness if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
'''
```

### Step 2.4: 在 generator 中实现 `_render_template`

修改 `op_testgen/generator/test_case_generator.py`：

```python
"""测试用例生成器"""
from datetime import datetime
from typing import Any, Dict, List

from op_testgen.mapper.op_mapper import MappedOp
from op_testgen.generator.template import TEST_FILE_TEMPLATE


class TestCaseGenerator:
    """将 MappedOp 序列化为可执行的 Python 测试文件"""

    def __init__(self):
        pass

    def _serialize_test_case(self, mapped_op: MappedOp) -> Dict[str, Any]:
        """将单个 MappedOp 序列化为字典"""
        op_info = mapped_op.op_info
        return {
            "op_name": op_info.name,
            "callable_path": mapped_op.callable_path,
            "input_dims": op_info.input_dims,
            "input_strides": op_info.input_strides,
            "input_types": op_info.input_types,
            "concrete_inputs": op_info.concrete_inputs,
        }

    def _render_template(self, data: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """渲染模板生成 Python 源码"""
        from string import Template
        
        template = Template(TEST_FILE_TEMPLATE)
        return template.substitute(
            source_trace=options.get("source_trace", "unknown"),
            generated_at=datetime.now().isoformat(),
            backend=options.get("backend", "cuda"),
            seed=options.get("seed", 42),
            iters=options.get("iters", 10),
            test_cases_data_repr=repr(data),
        )
```

### Step 2.5: 运行测试确认通过

```bash
pytest tests/test_generator.py::TestTemplateRendering -v
```

**Expected:** PASS

### Step 2.6: 提交

```bash
git add op_testgen/generator/template.py op_testgen/generator/test_case_generator.py tests/test_generator.py
git commit -m "feat(generator): add template rendering for test file generation"
```

---

## Task 3: 实现完整的 `generate` 方法 + 集成测试

**Files:**
- Modify: `tests/test_generator.py`
- Modify: `op_testgen/generator/test_case_generator.py`

### Step 3.1: 写集成测试

在 `tests/test_generator.py` 中添加：

```python
import os
import tempfile


class TestGenerateMethod:
    def test_generate_creates_file(self):
        """测试 generate 方法创建文件"""
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped], output_path, source_trace="test.json", backend="cpu", seed=42)
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "TEST_CASES_DATA" in content
            assert "aten::add" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_generate_with_multiple_ops(self):
        """测试生成包含多个算子的文件"""
        op1 = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        op2 = OpInfo(
            name="aten::mul",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped1 = mapper.map_operator(op1)
        mapped2 = mapper.map_operator(op2)
        assert mapped1 is not None
        assert mapped2 is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped1, mapped2], output_path, source_trace="test.json")
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content.count("aten::add") >= 1
            assert content.count("aten::mul") >= 1
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
```

### Step 3.2: 运行测试确认失败

```bash
pytest tests/test_generator.py::TestGenerateMethod -v
```

**Expected:** FAIL — `generate` 方法不存在

### Step 3.3: 实现 `generate` 方法

修改 `op_testgen/generator/test_case_generator.py`，在 `TestCaseGenerator` 类中添加：

```python
    def generate(
        self,
        mapped_ops: List[MappedOp],
        output_path: str,
        source_trace: str = "",
        backend: str = "cuda",
        seed: int = 42,
        iters: int = 10,
    ) -> str:
        """生成测试文件

        Args:
            mapped_ops: 映射后的算子列表
            output_path: 输出 .py 文件路径
            source_trace: 来源 trace 文件路径（用于注释）
            backend: 默认后端
            seed: 随机种子
            iters: 性能测试迭代次数

        Returns:
            生成的文件路径
        """
        data = [self._serialize_test_case(m) for m in mapped_ops]
        options = {
            "source_trace": source_trace,
            "backend": backend,
            "seed": seed,
            "iters": iters,
        }
        content = self._render_template(data, options)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
```

### Step 3.4: 运行测试确认通过

```bash
pytest tests/test_generator.py::TestGenerateMethod -v
```

**Expected:** PASS

### Step 3.5: 提交

```bash
git add op_testgen/generator/test_case_generator.py tests/test_generator.py
git commit -m "feat(generator): implement full generate() method with file output"
```

---

## Task 4: 实现 TestCaseRunner

**Files:**
- Create: `op_testgen/generator/test_case_runner.py`
- Modify: `tests/test_generator.py`

### Step 4.1: 写 runner 测试

在 `tests/test_generator.py` 中添加：

```python
import subprocess


class TestTestCaseRunner:
    def test_run_generated_file(self):
        """测试执行生成的文件"""
        op_info = OpInfo(
            name="aten::add",
            input_dims=[[2, 3], [2, 3]],
            input_strides=[[3, 1], [3, 1]],
            input_types=["float", "float"],
            concrete_inputs=[],
        )
        mapper = OpMapper()
        mapped = mapper.map_operator(op_info)
        assert mapped is not None

        gen = TestCaseGenerator()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            gen.generate([mapped], output_path, source_trace="test.json", backend="cpu", seed=42)

            runner = TestCaseRunner()
            returncode = runner.run(output_path, backend="cpu", only_correctness=True)

            assert returncode == 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_run_with_fail_fast(self):
        """测试 fail-fast 参数传递"""
        # 创建一个会失败的测试文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name
            f.write('''
import sys

def main():
    return 1

if __name__ == "__main__":
    sys.exit(main())
''')

        try:
            runner = TestCaseRunner()
            returncode = runner.run(output_path)
            assert returncode == 1
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
```

### Step 4.2: 运行测试确认失败

```bash
pytest tests/test_generator.py::TestTestCaseRunner -v
```

**Expected:** FAIL — `TestCaseRunner` 类不存在

### Step 4.3: 实现 TestCaseRunner

创建 `op_testgen/generator/test_case_runner.py`：

```python
"""测试用例执行器"""
import subprocess
import sys
from typing import Optional


class TestCaseRunner:
    """执行生成的测试文件"""

    def run(
        self,
        file_path: str,
        backend: Optional[str] = None,
        only_correctness: bool = False,
        only_performance: bool = False,
        iters: Optional[int] = None,
        fail_fast: bool = False,
    ) -> int:
        """运行生成的测试文件

        Args:
            file_path: 生成的 .py 文件路径
            backend: 后端（覆盖文件默认值）
            only_correctness: 仅正确性测试
            only_performance: 仅性能测试
            iters: 性能测试迭代次数
            fail_fast: 第一个失败即停止

        Returns:
            子进程返回码
        """
        cmd = [sys.executable, file_path]

        if backend is not None:
            cmd.extend(["--backend", backend])
        if only_correctness:
            cmd.append("--only-correctness")
        if only_performance:
            cmd.append("--only-performance")
        if iters is not None:
            cmd.extend(["--iters", str(iters)])
        if fail_fast:
            cmd.append("--fail-fast")

        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode
```

### Step 4.4: 运行测试确认通过

```bash
pytest tests/test_generator.py::TestTestCaseRunner -v
```

**Expected:** PASS

### Step 4.5: 提交

```bash
git add op_testgen/generator/test_case_runner.py tests/test_generator.py
git commit -m "feat(runner): add TestCaseRunner to execute generated test files"
```

---

## Task 5: 扩展 CLI

**Files:**
- Modify: `op_testgen/cli.py`
- Modify: `tests/test_generator.py`（添加端到端测试）

### Step 5.1: 提取公共逻辑

当前 `cmd_test` 中第 80-128 行（解析 → 映射 → 去重 → 构建）需要提取为公共函数，供 `cmd_generate` 复用。

在 `op_testgen/cli.py` 中，`cmd_test` 函数之前添加：

```python
def _prepare_test_cases(args) -> tuple:
    """提取 test 和 generate 子命令的公共逻辑

    Returns:
        (unique_mapped_ops, test_cases) 或 (unique_mapped_ops, None)
    """
    # 1. 解析 Trace
    parser_obj = TraceParser(args.trace_file)
    op_infos = parser_obj.parse()

    # 2. 映射算子
    mapper = OpMapper()
    mapped_ops = mapper.map_all(op_infos)

    # 3. 去重
    seen_keys = set()
    unique_mapped_ops = []
    for m in mapped_ops:
        info = m.op_info
        def _to_tuple(x):
            if isinstance(x, list):
                return tuple(_to_tuple(i) for i in x)
            return x
        dims_tuple = _to_tuple(info.input_dims)
        strides_tuple = _to_tuple(info.input_strides)
        types_tuple = tuple(info.input_types)
        concrete_tuple = _to_tuple(info.concrete_inputs)
        key = (info.name, dims_tuple, strides_tuple, types_tuple, concrete_tuple)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_mapped_ops.append(m)

    # 4. 过滤
    if hasattr(args, "op_filter") and args.op_filter:
        import fnmatch
        unique_mapped_ops = [
            m for m in unique_mapped_ops if fnmatch.fnmatch(m.op_info.name, args.op_filter)
        ]

    # 5. 截断
    if hasattr(args, "max_ops") and args.max_ops > 0 and len(unique_mapped_ops) > args.max_ops:
        unique_mapped_ops = unique_mapped_ops[:args.max_ops]

    if not unique_mapped_ops:
        return [], None

    # 6. 构建测试用例
    builder = TensorBuilder(seed=args.seed)
    test_cases = [builder.build(m) for m in unique_mapped_ops]

    return unique_mapped_ops, test_cases
```

### Step 5.2: 重构 `cmd_test` 使用 `_prepare_test_cases`

替换 `cmd_test` 中第 80-128 行：

```python
def cmd_test(args) -> int:
    """test 子命令：测试生成与执行"""
    print("=" * 60)
    print("PyTorch Profiler 自动化测试生成器")
    print("=" * 60)

    print("\n[1/4] 解析并准备测试用例...")
    unique_mapped_ops, test_cases = _prepare_test_cases(args)

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(unique_mapped_ops)} 个算子")
    print(f"  构建 {len(test_cases)} 个测试用例")

    # 确定后端
    backends = []
    if args.backend == "auto":
        backends = ["cuda"]
        if os.environ.get("SWDNN_AVAILABLE"):
            backends.append("swdnn")
    else:
        backends = [args.backend]

    all_correctness = []
    all_perf = []

    for backend in backends:
        # ... 剩余逻辑不变
```

**注意**：保留 `cmd_test` 的其余逻辑（第 139-246 行）不变，只替换前面的准备逻辑。

### Step 5.3: 实现 `cmd_generate`

在 `op_testgen/cli.py` 中，在 `cmd_test` 之后添加：

```python
def cmd_generate(args) -> int:
    """generate 子命令：生成测试用例文件"""
    print("=" * 60)
    print("PyTorch Profiler 测试用例生成器")
    print("=" * 60)

    print("\n[1/2] 解析并准备测试用例...")
    unique_mapped_ops, _ = _prepare_test_cases(args)

    if not unique_mapped_ops:
        print("错误：没有可测试的算子")
        return 1

    print(f"  成功映射并去重: {len(unique_mapped_ops)} 个算子")

    print("\n[2/2] 生成测试文件...")
    generator = TestCaseGenerator()
    output_path = generator.generate(
        mapped_ops=unique_mapped_ops,
        output_path=args.output,
        source_trace=args.trace_file,
        backend=args.backend,
        seed=args.seed,
        iters=args.iters,
    )
    print(f"  测试文件: {output_path}")

    print("\n" + "=" * 60)
    print("生成完成!")
    print(f"运行: op_testgen run {output_path}")
    print("=" * 60)

    return 0
```

### Step 5.4: 实现 `cmd_run`

在 `op_testgen/cli.py` 中，在 `cmd_generate` 之后添加：

```python
def cmd_run(args) -> int:
    """run 子命令：执行生成的测试文件"""
    print("=" * 60)
    print("执行生成的测试文件")
    print("=" * 60)

    if not os.path.exists(args.test_file):
        print(f"错误：文件不存在: {args.test_file}")
        return 1

    runner = TestCaseRunner()
    returncode = runner.run(
        file_path=args.test_file,
        backend=args.backend,
        only_correctness=args.only_correctness,
        only_performance=args.only_performance,
        iters=args.iters,
        fail_fast=args.fail_fast,
    )

    print("\n" + "=" * 60)
    print("执行完成!")
    print("=" * 60)

    return returncode
```

### Step 5.5: 在 `main()` 中注册子命令

在 `test_parser` 之后添加：

```python
    # === generate 子命令 ===
    generate_parser = subparsers.add_parser(
        "generate", help="从 Trace 生成测试用例文件",
        description="解析 PyTorch Profiler Trace，生成可执行的 Python 测试文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen generate profiler_trace.json -o tests/test_ops.py
  op_testgen generate profiler_trace.json -o tests/test_ops.py --backend cpu --max-ops 50
        """,
    )
    generate_parser.add_argument("trace_file", help="PyTorch Profiler Chrome Trace JSON 文件路径")
    generate_parser.add_argument("-o", "--output", required=True, help="输出 .py 文件路径")
    generate_parser.add_argument("--backend", choices=["cuda", "swdnn", "auto", "cpu"], default="cuda",
                                help="默认后端 (默认: cuda)")
    generate_parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    generate_parser.add_argument("--max-ops", type=int, default=100, help="最大算子数 (默认: 100)")
    generate_parser.add_argument("--op-filter", help="算子名称过滤 (支持通配符)")
    generate_parser.add_argument("--iters", type=int, default=10, help="性能测试迭代次数 (默认: 10)")

    # === run 子命令 ===
    run_parser = subparsers.add_parser(
        "run", help="执行生成的测试用例文件",
        description="运行由 generate 子命令生成的 Python 测试文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  op_testgen run tests/test_ops.py
  op_testgen run tests/test_ops.py --backend cuda --only-correctness
        """,
    )
    run_parser.add_argument("test_file", help="生成的 .py 测试文件路径")
    run_parser.add_argument("--backend", choices=["cuda", "swdnn", "cpu"],
                           help="后端 (覆盖文件默认值)")
    run_parser.add_argument("--only-correctness", action="store_true", help="仅正确性测试")
    run_parser.add_argument("--only-performance", action="store_true", help="仅性能测试")
    run_parser.add_argument("--iters", type=int, help="性能测试迭代次数")
    run_parser.add_argument("--fail-fast", action="store_true", help="第一个失败即停止")
```

在 `args = parser.parse_args(argv)` 之后，修改命令分发逻辑：

```python
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 0
```

### Step 5.6: 添加 CLI 导入

在 `op_testgen/cli.py` 顶部添加导入：

```python
from op_testgen.generator.test_case_generator import TestCaseGenerator
from op_testgen.generator.test_case_runner import TestCaseRunner
```

### Step 5.7: 写端到端测试

在 `tests/test_generator.py` 中添加：

```python
class TestCLIE2E:
    def test_generate_subcommand(self):
        """测试 generate 子命令"""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                ["python", "run.py", "generate", "profiler_trace.json", "-o", output_path, "--max-ops", "5", "--backend", "cpu"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "TEST_CASES_DATA" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_run_subcommand(self):
        """测试 run 子命令"""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            output_path = f.name

        try:
            # 先 generate
            result = subprocess.run(
                ["python", "run.py", "generate", "profiler_trace.json", "-o", output_path, "--max-ops", "5", "--backend", "cpu"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

            # 再 run
            result = subprocess.run(
                ["python", "run.py", "run", output_path, "--only-correctness"],
                capture_output=True,
                text=True,
            )
            # run 的返回码取决于测试结果，不强制为 0
            assert "测试完成" in result.stdout or "测试完成" in result.stderr
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
```

### Step 5.8: 运行所有测试

```bash
pytest tests/test_generator.py -v
```

**Expected:** PASS

### Step 5.9: 验证现有 test 子命令未破坏

```bash
python run.py test profiler_trace.json --only-correctness --backend cpu --max-ops 5
```

**Expected:** 正常输出正确性测试结果

### Step 5.10: 提交

```bash
git add op_testgen/cli.py tests/test_generator.py
git commit -m "feat(cli): add generate and run subcommands"
```

---

## Task 6: 集成验证

**Files:** 无新增文件

### Step 6.1: 完整流程验证

```bash
# 1. 生成测试文件
python run.py generate profiler_trace.json -o /tmp/test_ops.py --max-ops 10 --backend cpu

# 2. 检查文件内容
head -n 30 /tmp/test_ops.py

# 3. 执行测试文件
python run.py run /tmp/test_ops.py --only-correctness
```

**Expected:**
- generate 成功创建 `/tmp/test_ops.py`
- 文件内容包含 `TEST_CASES_DATA` 和 `main()` 函数
- run 成功执行并输出正确性测试结果

### Step 6.2: 运行全部测试套件

```bash
pytest tests/ -v
```

**Expected:** 所有测试通过（包括新增和原有测试）

### Step 6.3: 最终提交

```bash
git add .
git commit -m "feat: separate test case generation and execution into generate/run commands"
```

---

## Spec Coverage Check

| Spec Requirement | Task | Status |
|-----------------|------|--------|
| 新增 `generate` 子命令 | Task 5 | ✅ |
| 新增 `run` 子命令 | Task 5 | ✅ |
| 保留 `test` 子命令快捷方式 | Task 5 | ✅ |
| 生成自包含 `.py` 文件 | Task 1-3 | ✅ |
| 文件包含 `TEST_CASES_DATA` 常量 | Task 2 | ✅ |
| 文件包含 `build_test_cases()` 重建逻辑 | Task 2 | ✅ |
| 文件包含 `main()` 执行逻辑 | Task 2 | ✅ |
| `TestCaseGenerator` 类 | Task 1, 3 | ✅ |
| `TestCaseRunner` 类 | Task 4 | ✅ |
| 使用 `subprocess` 执行生成文件 | Task 4 | ✅ |
| 向后兼容 `test` 子命令 | Task 5 | ✅ |
| 单元测试覆盖序列化 | Task 1 | ✅ |
| 单元测试覆盖模板渲染 | Task 2 | ✅ |
| 单元测试覆盖文件生成 | Task 3 | ✅ |
| 单元测试覆盖 runner | Task 4 | ✅ |
| 端到端测试覆盖 CLI | Task 5 | ✅ |
