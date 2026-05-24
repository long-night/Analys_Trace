"""Markdown 报告生成器（默认格式，无需外部依赖）"""
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult
from op_testgen.reporter.base import BaseReporter


def _escape_md(text: str) -> str:
    """转义 Markdown 表格中的特殊字符，防止破坏表格格式"""
    if not text:
        return ""
    # 转义管道符（表格分隔符）和反斜杠
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    # 转义 Markdown 格式字符：* _ ~ ` 等
    for ch in ("*", "_", "~", "`"):
        text = text.replace(ch, "\\" + ch)
    # 将换行替换为空格，防止破坏单行表格
    text = text.replace("\n", " ").replace("\r", " ")
    return text


class MarkdownReporter(BaseReporter):
    """Markdown 报告生成器 —— 纯文本，无需 jinja2"""

    def generate(self, correctness_results: List[CorrectnessResult],
                 perf_results: List[PerfResult], output_path: str) -> None:
        total_ops = len(correctness_results)
        passed = sum(1 for r in correctness_results if r.passed)
        avg_speedup = sum(r.speedup for r in perf_results) / len(perf_results) if perf_results else 1.0

        lines: List[str] = []
        lines.append("# 算子测试报告\n")

        # 汇总卡片
        lines.append("## 汇总\n")
        lines.append(f"- **总算子数**: {total_ops}")
        lines.append(f"- **正确性通过**: {passed}/{total_ops}")
        lines.append(f"- **正确性通过率**: {passed/total_ops*100:.1f}%" if total_ops > 0 else "- **正确性通过率**: N/A")
        lines.append(f"- **平均加速比**: {avg_speedup:.2f}x\n")

        # 正确性测试
        lines.append("## 正确性测试结果\n")
        lines.append("| 算子名称 | 后端 | 状态 | 最大绝对误差 | 最大相对误差 | 平均绝对误差 | 平均相对误差 | 输入参数 | 错误信息 |")
        lines.append("|---------|------|------|-------------|-------------|-------------|-------------|---------|---------|")
        for r in correctness_results:
            status = "✅ 通过" if r.passed else "❌ 失败"
            max_abs = f"{r.max_abs_err:.2e}" if r.max_abs_err else "-"
            max_rel = f"{r.max_rel_err:.2e}" if r.max_rel_err else "-"
            avg_abs = f"{r.avg_abs_err:.2e}" if r.avg_abs_err else "-"
            avg_rel = f"{r.avg_rel_err:.2e}" if r.avg_rel_err else "-"
            input_info = _escape_md(r.input_info) if r.input_info else "-"
            err_msg = _escape_md(r.error_message) if r.error_message else "-"
            lines.append(f"| {r.op_name} | {r.backend} | {status} | {max_abs} | {max_rel} | {avg_abs} | {avg_rel} | {input_info} | {err_msg} |")
        lines.append("")

        # 性能测试
        lines.append("## 性能测试结果\n")
        lines.append("| 算子名称 | 分类 | 平均耗时(ms) | GFLOPS | 带宽(GB/s) | 加速比 | 输入参数 |")
        lines.append("|---------|------|------------|--------|-----------|--------|---------|")
        for r in perf_results:
            flops = f"{r.flops:.2f}" if r.flops else "-"
            bw = f"{r.bandwidth_gbps:.2f}" if r.bandwidth_gbps else "-"
            input_info = _escape_md(r.input_info) if r.input_info else "-"
            lines.append(f"| {r.op_name} | {r.category} | {r.avg_time_ms:.4f} | {flops} | {bw} | {r.speedup:.2f}x | {input_info} |")
        lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
