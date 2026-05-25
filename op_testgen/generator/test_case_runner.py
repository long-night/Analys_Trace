"""测试用例进程内执行器"""
import importlib.util
import sys
from typing import List, Optional


class InProcessRunner:
    """在当前 Python 进程内加载并执行生成的 .py 测试文件"""

    def run(
        self,
        file_path: str,
        backend: Optional[str] = None,
        only_correctness: bool = False,
        only_performance: bool = False,
        iters: Optional[int] = None,
        fail_fast: bool = False,
        op_filter: Optional[str] = None,
        format: Optional[str] = None,
        output: Optional[str] = None,
    ) -> int:
        """在当前进程内执行生成的测试文件

        通过 importlib 动态加载 .py 文件并调用其 main() 函数，
        避免使用 subprocess 创建新进程。

        Args:
            file_path: 生成的 .py 文件路径
            backend: 后端（覆盖文件默认值）
            only_correctness: 仅正确性测试
            only_performance: 仅性能测试
            iters: 性能测试迭代次数
            fail_fast: 第一个失败即停止
            op_filter: 算子名称过滤
            format: 报告格式（覆盖文件默认值）
            output: 报告输出路径（覆盖文件默认值）

        Returns:
            main() 函数的返回值
        """
        spec = importlib.util.spec_from_file_location("generated_test", file_path)
        if spec is None or spec.loader is None:
            print(f"错误: 无法加载文件: {file_path}")
            return 1

        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_test"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "main"):
            print("错误: 测试文件没有 main() 函数")
            return 1

        argv: List[str] = []
        if backend is not None:
            argv.extend(["--backend", backend])
        if only_correctness:
            argv.append("--only-correctness")
        if only_performance:
            argv.append("--only-performance")
        if iters is not None:
            argv.extend(["--iters", str(iters)])
        if fail_fast:
            argv.append("--fail-fast")
        if op_filter is not None:
            argv.extend(["--op-filter", op_filter])
        if format is not None:
            argv.extend(["--format", format])
        if output is not None:
            argv.extend(["-o", output])

        returncode = module.main(argv)
        return returncode if isinstance(returncode, int) else 0
