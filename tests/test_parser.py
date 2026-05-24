"""测试 TraceParser"""
import json
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
