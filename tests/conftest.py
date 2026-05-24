"""pytest fixtures"""
import json
import pytest


@pytest.fixture
def sample_trace_json(tmp_path):
    """创建最小可用的 trace JSON 用于测试"""
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
