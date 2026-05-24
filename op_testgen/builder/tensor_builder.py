"""张量与参数重构模块"""
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from op_testgen.mapper.op_mapper import MappedOp


@dataclass
class TestCase:
    """可执行的测试用例"""
    mapped_op: MappedOp
    input_tensors: List[torch.Tensor] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class TensorBuilder:
    """根据运行时信息构建输入张量"""

    DTYPE_MAP = {
        "float": torch.float32,
        "float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "long": torch.int64,
        "long int": torch.int64,
        "int64": torch.int64,
        "int": torch.int32,
        "int32": torch.int32,
        "short": torch.int16,
        "int16": torch.int16,
        "char": torch.int8,
        "int8": torch.int8,
        "byte": torch.uint8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

    def _map_dtype(self, type_str: str) -> torch.dtype:
        """将字符串类型映射到 torch dtype"""
        type_lower = type_str.lower().strip()
        if type_lower in self.DTYPE_MAP:
            return self.DTYPE_MAP[type_lower]
        return torch.float32

    def _is_floating(self, dtype: torch.dtype) -> bool:
        return dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)

    def _build_tensor(self, dims: List[int], strides: Optional[List[int]], dtype: torch.dtype) -> torch.Tensor:
        """构建单个张量"""
        if self._is_floating(dtype):
            t = torch.randn(dims, dtype=dtype)
        else:
            t = torch.randint(low=0, high=10, size=dims, dtype=dtype)

        if strides is not None and len(strides) == len(dims):
            try:
                t = torch.as_strided(t, size=dims, stride=strides)
            except RuntimeError:
                pass
        return t

    def _build_tensors(self, op_info) -> List[torch.Tensor]:
        """构建所有输入张量"""
        tensors = []
        for i, dims in enumerate(op_info.input_dims):
            type_str = op_info.input_types[i] if i < len(op_info.input_types) else "float"
            if type_str.lower() == "scalar":
                continue
            dtype = self._map_dtype(type_str)
            strides = op_info.input_strides[i] if i < len(op_info.input_strides) else None
            t = self._build_tensor(dims, strides, dtype)
            tensors.append(t)
        return tensors

    def _parse_concrete_inputs(self, mapped_op: MappedOp) -> Dict[str, Any]:
        """解析 concrete_inputs 为 kwargs"""
        kwargs = {}
        concrete = mapped_op.op_info.concrete_inputs
        if not concrete:
            return kwargs

        params = []
        try:
            sig = inspect.signature(mapped_op.callable)
            params = list(sig.parameters.values())
        except ValueError:
            # torch 内置函数没有 signature，使用启发式
            pass

        num_tensor_args = len(mapped_op.op_info.input_dims)
        scalar_count = sum(1 for t in mapped_op.op_info.input_types if t.lower() == "scalar")
        skip_count = num_tensor_args - scalar_count

        param_idx = skip_count
        for value in concrete:
            if param_idx >= len(params):
                break
            param = params[param_idx]
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                break
            if value is not None:
                kwargs[param.name] = value
            param_idx += 1

        return kwargs

    def build(self, mapped_op: MappedOp) -> TestCase:
        """构建测试用例"""
        tensors = self._build_tensors(mapped_op.op_info)
        kwargs = self._parse_concrete_inputs(mapped_op)

        # Scalar 参数: concrete_inputs 按非 tensor 参数顺序存储
        concrete_idx = 0
        for i, type_str in enumerate(mapped_op.op_info.input_types):
            if type_str.lower() == "scalar":
                if concrete_idx < len(mapped_op.op_info.concrete_inputs):
                    scalar_val = mapped_op.op_info.concrete_inputs[concrete_idx]
                    try:
                        sig = inspect.signature(mapped_op.callable)
                        params = list(sig.parameters.values())
                        if len(params) > 1:
                            kwargs[params[1].name] = scalar_val
                    except ValueError:
                        if "other" not in kwargs:
                            kwargs["other"] = scalar_val
                    concrete_idx += 1

        return TestCase(
            mapped_op=mapped_op,
            input_tensors=tensors,
            kwargs=kwargs,
        )
