"""性能指标计算模块"""
from typing import Any, Dict, List, Optional

import torch


class MetricsCalculator:
    """性能指标计算器"""

    @staticmethod
    def compute_flops(op_name: str, input_dims: List[List[int]], formula: Optional[str] = None) -> float:
        """计算理论 FLOPS"""
        if formula is None:
            return MetricsCalculator._heuristic_flops(op_name, input_dims)

        env = MetricsCalculator._build_formula_env(input_dims)
        try:
            return float(eval(formula, {"__builtins__": {}}, env))
        except Exception:
            return 0.0

    @staticmethod
    def compute_bytes(input_dims: List[List[int]], dtype: torch.dtype, num_inputs: int = 1, num_outputs: int = 1) -> int:
        """计算访存字节数 (读 + 写)"""
        if dtype == torch.bool:
            element_size = 1
        elif dtype.is_floating_point:
            element_size = torch.finfo(dtype).bits // 8
        else:
            element_size = torch.iinfo(dtype).bits // 8
        total_elements = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        return total_elements * element_size * num_inputs + total_elements * element_size * num_outputs

    @staticmethod
    def compute_communication_bytes(input_dims: List[List[int]], dtype: torch.dtype, world_size: int = 2) -> int:
        """计算通信数据量"""
        if dtype == torch.bool:
            element_size = 1
        elif dtype.is_floating_point:
            element_size = torch.finfo(dtype).bits // 8
        else:
            element_size = torch.iinfo(dtype).bits // 8
        total_elements = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        return 2 * total_elements * element_size

    @staticmethod
    def _build_formula_env(input_dims: List[List[int]]) -> Dict[str, Any]:
        """为公式 eval 构建变量环境"""
        env = {}
        if len(input_dims) >= 1 and len(input_dims[0]) >= 2:
            env["M"] = input_dims[0][-2]
            env["N"] = input_dims[0][-1]
        if len(input_dims) >= 2 and len(input_dims[1]) >= 2:
            env["K"] = input_dims[1][-2] if len(input_dims[1]) >= 2 else 1
        if len(input_dims) >= 2:
            shape0 = input_dims[0]
            shape1 = input_dims[1]
            if len(shape0) == 4:
                env["N"] = shape0[0]
                env["Cin"] = shape0[1]
                env["H"] = shape0[2]
                env["W"] = shape0[3]
            if len(shape1) == 4:
                env["Cout"] = shape1[0]
                env["K"] = shape1[2]
            if "H" in env and "K" in env:
                env["Hout"] = env["H"] - env["K"] + 1
                env["Wout"] = env["W"] - env["K"] + 1
        if len(input_dims) >= 1:
            env["B"] = input_dims[0][0] if len(input_dims[0]) >= 3 else 1
        env["numel"] = sum(int(torch.prod(torch.tensor(d))) for d in input_dims)
        env["element_size"] = 4
        return env

    @staticmethod
    def _heuristic_flops(op_name: str, input_dims: List[List[int]]) -> float:
        """启发式 FLOPS 估算"""
        name_lower = op_name.lower()
        if "matmul" in name_lower or "mm" in name_lower:
            if len(input_dims) >= 2 and len(input_dims[0]) >= 2 and len(input_dims[1]) >= 2:
                m, n = input_dims[0][-2], input_dims[1][-1]
                k = input_dims[0][-1]
                return 2.0 * m * n * k
        if "conv" in name_lower and len(input_dims) >= 2:
            if len(input_dims[0]) == 4 and len(input_dims[1]) == 4:
                n = input_dims[0][0]
                cout = input_dims[1][0]
                cin = input_dims[1][1]
                h, w = input_dims[0][2], input_dims[0][3]
                k = input_dims[1][2]
                return 2.0 * n * cout * h * w * cin * k * k
        return 0.0
