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
            format=options.get("format", "markdown"),
            output=options.get("output", "op_testgen_report.md"),
            test_cases_data_repr=repr(data),
        )

    def generate(
        self,
        mapped_ops: List[MappedOp],
        output_path: str,
        source_trace: str = "",
        backend: str = "cuda",
        seed: int = 42,
        cpu_iters: int = 1,
        target_iters: int = 3,
        format: str = "markdown",
        output: str = "op_testgen_report.md",
    ) -> str:
        """生成测试文件

        生成的文件为独立的 .py 文件，仅依赖 op_testgen 核心测试执行模块。

        Args:
            mapped_ops: 映射后的算子列表
            output_path: 输出 .py 文件路径
            source_trace: 来源 trace 文件路径（用于注释）
            backend: 默认后端
            seed: 随机种子
            cpu_iters: CPU baseline 迭代次数
            target_iters: target backend 迭代次数
            format: 报告格式
            output: 报告输出路径

        Returns:
            生成的文件路径
        """
        data = [self._serialize_test_case(m) for m in mapped_ops]
        options = {
            "source_trace": source_trace,
            "backend": backend,
            "seed": seed,
            "cpu_iters": cpu_iters,
            "target_iters": target_iters,
            "format": format,
            "output": output,
        }
        content = self._render_template(data, options)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
