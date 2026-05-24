"""算子分类模块"""
import os
from typing import Optional

import yaml


class OpClassifier:
    """算子性能分类器"""

    COMPUTE_PATTERNS = ["matmul", "conv", "mm", "bmm", "linear", "softmax", "log_softmax"]
    COMMUNICATION_PATTERNS = ["allreduce", "allgather", "broadcast", "send", "recv", "reduce", "scatter"]

    def __init__(self, classification_path: Optional[str] = None):
        self.classifications = {}
        if classification_path and os.path.exists(classification_path):
            with open(classification_path, "r", encoding="utf-8") as f:
                self.classifications = yaml.safe_load(f) or {}
        else:
            import os as _os
            default_path = _os.path.join(
                _os.path.dirname(__file__), "..", "config", "op_classification.yaml"
            )
            if _os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    self.classifications = yaml.safe_load(f) or {}

    def classify(self, op_name: str) -> str:
        """返回算子分类: compute | memory | communication | mixed"""
        if op_name in self.classifications:
            return self.classifications[op_name].get("category", "memory")

        name_lower = op_name.lower()
        is_compute = any(p in name_lower for p in self.COMPUTE_PATTERNS)
        is_comm = any(p in name_lower for p in self.COMMUNICATION_PATTERNS)

        if is_comm:
            return "communication"
        if is_compute:
            return "compute"

        return "memory"

    def get_flops_formula(self, op_name: str) -> Optional[str]:
        """获取 FLOPS 计算公式"""
        if op_name in self.classifications:
            return self.classifications[op_name].get("flops_formula")
        return None

    def get_bytes_formula(self, op_name: str) -> Optional[str]:
        """获取访存量/通信量计算公式"""
        if op_name in self.classifications:
            return self.classifications[op_name].get("bytes_formula")
        return None
