#!/usr/bin/env python3
"""Tests for analys_trace_v4.py - verifying bug fixes"""
import json
import os
import sys
import tempfile
import unittest
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analys_trace_v4 import ChromeTraceAnalyzer, OperatorInfo, ShapeStats


class TestExtractShapesBug(unittest.TestCase):
    """P0: extract_shapes should NOT return Input type as shape"""

    def test_extract_shapes_should_not_use_input_type(self):
        """When only Input type exists (no Input Dims), should return None"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "args": {
                "Input type": ["float", "int"],
            }
        }
        result = analyzer.extract_shapes(event)
        self.assertIsNone(result, "extract_shapes should NOT return Input type as shape")

    def test_extract_shapes_prefers_input_dims(self):
        """Should return Input Dims when both exist"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "args": {
                "Input Dims": [[2, 3], [4, 5]],
                "Input type": ["float", "int"],
            }
        }
        result = analyzer.extract_shapes(event)
        self.assertEqual(result, [[2, 3], [4, 5]])

    def test_extract_shapes_with_input_shapes_alias(self):
        """Should accept Input Shapes alias"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "args": {
                "Input Shapes": [[2, 3]],
            }
        }
        result = analyzer.extract_shapes(event)
        self.assertEqual(result, [[2, 3]])


class TestNestedBEEvents(unittest.TestCase):
    """P0: B/E events with same name should be handled via stack"""

    def test_nested_same_name_events(self):
        """Nested aten::add should not lose outer event stats"""
        trace_data = {
            "traceEvents": [
                {"ph": "B", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 100, 
                 "args": {"Input Dims": [[2, 3]], "Input type": ["float"]}},
                {"ph": "B", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 150, 
                 "args": {"Input Dims": [[4, 5]], "Input type": ["float"]}},
                {"ph": "E", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 200},
                {"ph": "E", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 300},
            ]
        }
        analyzer = ChromeTraceAnalyzer("dummy")
        analyzer.operators = {}  # Reset
        
        # Simulate analysis
        events = trace_data["traceEvents"]
        pending_events = analyzer.pending_events if hasattr(analyzer, 'pending_events') else {}
        
        # Create a temp file and run analyze
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            analyzer.analyze()
            
            op = analyzer.operators.get("aten::add")
            self.assertIsNotNone(op, "aten::add should be recorded")
            self.assertEqual(op.call_count, 2, "Should have 2 calls (outer + inner)")
            
            # Check both shapes are present
            shape_keys = list(op.shape_stats.keys())
            self.assertEqual(len(shape_keys), 2, "Should have 2 different shapes")
            
            # Verify durations: outer = 200, inner = 50
            total_ms = op.get_total_duration_ms()
            self.assertAlmostEqual(total_ms, 0.25, places=4, msg="Total duration should be 250us = 0.25ms")
        finally:
            os.unlink(temp_path)


class TestIsCpuOperator(unittest.TestCase):
    """P1: is_cpu_operator should not match empty category"""

    def test_empty_category_should_not_match(self):
        """Empty cat should NOT be treated as CPU operator"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "name": "aten::add",
            "cat": ""
        }
        result = analyzer.is_cpu_operator(event)
        self.assertFalse(result, "Empty category should NOT match CPU operator")

    def test_kernel_category_should_not_match_aten(self):
        """Kernel category should NOT match aten operators"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "name": "aten::add",
            "cat": "kernel"
        }
        result = analyzer.is_cpu_operator(event)
        self.assertFalse(result, "Kernel category should NOT match CPU operator")

    def test_cpu_op_category_should_match(self):
        """cpu_op category should match"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "name": "aten::add",
            "cat": "cpu_op"
        }
        result = analyzer.is_cpu_operator(event)
        self.assertTrue(result, "cpu_op category should match")

    def test_cpu_in_category_should_match(self):
        """Category containing 'cpu' should match"""
        analyzer = ChromeTraceAnalyzer("dummy")
        event = {
            "name": "aten::add",
            "cat": "cpu_function_call"
        }
        result = analyzer.is_cpu_operator(event)
        self.assertTrue(result, "Category containing 'cpu' should match")


class TestNegativeDuration(unittest.TestCase):
    """P1: Negative duration should be rejected"""

    def test_negative_dur_x_event(self):
        """X event with negative dur should be skipped"""
        trace_data = {
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "aten::add", "tid": 1, 
                 "ts": 100, "dur": -50, "args": {"Input Dims": [[2, 3]]}},
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            analyzer.analyze()
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Should either skip the event or print warning
            op = analyzer.operators.get("aten::add")
            if op:
                self.assertEqual(op.call_count, 0, "Negative duration event should be skipped")
        finally:
            os.unlink(temp_path)

    def test_be_negative_duration(self):
        """B/E event where E < B should be skipped"""
        trace_data = {
            "traceEvents": [
                {"ph": "B", "cat": "cpu_op", "name": "aten::add", "tid": 1, 
                 "ts": 200, "args": {"Input Dims": [[2, 3]]}},
                {"ph": "E", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 100},
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            analyzer.analyze()
            
            op = analyzer.operators.get("aten::add")
            if op:
                self.assertEqual(op.call_count, 0, "Negative duration B/E should be skipped")
        finally:
            os.unlink(temp_path)


class TestBEMismatch(unittest.TestCase):
    """P2: Unmatched E event should produce warning"""

    def test_unmatched_e_event_warning(self):
        """E without matching B should print warning"""
        trace_data = {
            "traceEvents": [
                {"ph": "E", "cat": "cpu_op", "name": "aten::add", "tid": 1, "ts": 100},
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            analyzer.analyze()
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            self.assertIn("未匹配", output, "Should warn about unmatched E event")
        finally:
            os.unlink(temp_path)


class TestCommunicationOperatorReuse(unittest.TestCase):
    """P2: print_summary should reuse is_communication_operator"""

    def test_communication_op_detection(self):
        """Communication operators should be detected consistently"""
        analyzer = ChromeTraceAnalyzer("dummy")
        
        # Test all prefixes
        prefixes = ['c10d::', 'nccl:', 'gloo:', 'mpi:']
        for prefix in prefixes:
            event = {"name": f"{prefix}test_op"}
            self.assertTrue(
                analyzer.is_communication_operator(event),
                f"{prefix} should be detected as communication operator"
            )


class TestRegression(unittest.TestCase):
    """Ensure fixes don't break existing functionality"""

    def test_x_event_basic(self):
        """X events should still work"""
        trace_data = {
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "aten::add", "tid": 1, 
                 "ts": 100, "dur": 50, "args": {"Input Dims": [[2, 3]], "Input type": ["float"]}},
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            analyzer.analyze()
            
            op = analyzer.operators["aten::add"]
            self.assertEqual(op.call_count, 1)
            self.assertAlmostEqual(op.get_total_duration_ms(), 0.05, places=4)
            self.assertIn("[(2, 3)]", op.get_all_shapes_str())
        finally:
            os.unlink(temp_path)

    def test_be_event_basic(self):
        """B/E events should still work"""
        trace_data = {
            "traceEvents": [
                {"ph": "B", "cat": "cpu_op", "name": "aten::mul", "tid": 1, 
                 "ts": 100, "args": {"Input Dims": [[3, 4]], "Input type": ["float"]}},
                {"ph": "E", "cat": "cpu_op", "name": "aten::mul", "tid": 1, "ts": 150},
            ]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            temp_path = f.name
        
        try:
            analyzer = ChromeTraceAnalyzer(temp_path)
            analyzer.analyze()
            
            op = analyzer.operators["aten::mul"]
            self.assertEqual(op.call_count, 1)
            self.assertAlmostEqual(op.get_total_duration_ms(), 0.05, places=4)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
