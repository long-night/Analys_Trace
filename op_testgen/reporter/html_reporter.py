"""HTML 报告生成器 —— 可选功能，需要 jinja2"""
from typing import List

from op_testgen.correctness.test_runner import CorrectnessResult
from op_testgen.perf.benchmark import PerfResult
from op_testgen.reporter.base import BaseReporter


def _check_jinja2():
    try:
        from jinja2 import Template  # noqa: F401
    except ImportError:
        raise ImportError(
            "HTML 报告需要 jinja2。请安装: pip install jinja2"
        )


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
        _check_jinja2()
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
