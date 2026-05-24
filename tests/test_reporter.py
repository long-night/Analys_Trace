"""测试 Reporter"""
import pytest
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
            PerfResult(op_name="aten::add", category="memory", backend="cuda", avg_time_ms=0.05, bandwidth_gbps=100.0, speedup=2.0),
            PerfResult(op_name="aten::conv2d", category="compute", backend="cuda", avg_time_ms=2.0, flops=500.0, speedup=10.0),
        ]
        output = tmp_path / "report.html"
        reporter.generate(correctness, perf, str(output))
        assert output.exists()
        content = output.read_text()
        assert "算子测试报告" in content
        assert "aten::add" in content
        assert "通过" in content
