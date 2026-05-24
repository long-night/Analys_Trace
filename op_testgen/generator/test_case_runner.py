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
