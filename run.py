#!/usr/bin/env python3
"""
PyTorch Profiler 自动化工具集 — 便捷入口脚本

无需 pip 安装，直接运行：
    python run.py analyze trace.json
    python run.py test trace.json

等价于：
    op_testgen analyze trace.json
    op_testgen test trace.json
"""
import sys
from op_testgen.cli import main

if __name__ == "__main__":
    sys.exit(main())
