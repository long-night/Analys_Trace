"""张量与参数重构模块"""
import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

from op_testgen.mapper.op_mapper import MappedOp


@dataclass
class TestCase:
    """可执行的测试用例"""
    mapped_op: MappedOp
    input_tensors: List[torch.Tensor] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    positional_args: List[Any] = field(default_factory=list)


# 表示张量类型的字符串集合
_TENSOR_TYPES = {
    "float", "float32", "double", "float64",
    "half", "float16", "bfloat16", "c10::bfloat16",
    "long", "long int", "int64",
    "int", "int32", "short", "int16",
    "char", "int8", "byte", "uint8", "bool",
}


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
        type_lower = type_str.lower().strip()
        if type_lower in self.DTYPE_MAP:
            return self.DTYPE_MAP[type_lower]
        return torch.float32

    def _is_floating(self, dtype: torch.dtype) -> bool:
        return dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)

    _HARDCODED_KWARG_NAMES = {
        "log_softmax": {"dtype"},
    }

    def _extract_kwarg_names_from_doc(self, func) -> Set[str]:
        kwarg_names = set()

        func_name = getattr(func, "__name__", "")
        if func_name in self._HARDCODED_KWARG_NAMES:
            return self._HARDCODED_KWARG_NAMES[func_name].copy()
        
        doc = getattr(func, "__doc__", None) or ""
        if not doc:
            return kwarg_names

        lines = [line.strip() for line in doc.split("\n") if line.strip()]
        first_line = lines[0] if lines else ""
        match = re.search(r"\*\s*,\s*(.*?)(?:\)|->)", first_line)
        if match:
            kw_part = match.group(1)
            for param_match in re.finditer(r"(\w+)\s*=", kw_part):
                kwarg_names.add(param_match.group(1))

        return kwarg_names

    def _build_tensor(self, dims, strides, dtype: torch.dtype, op_name: str = "") -> torch.Tensor:
        if isinstance(dims, list) and len(dims) > 0 and isinstance(dims[0], list):
            dims = dims[0]
        dims_tuple = tuple(dims)
        base_name = op_name.replace("aten::", "").replace("aten::_", "")
        if self._is_floating(dtype):
            if base_name in ("rsqrt", "sqrt", "log", "log1p", "reciprocal"):
                t = torch.rand(dims_tuple, dtype=dtype) + 0.1
            elif base_name in ("asin", "acos", "atan"):
                t = torch.rand(dims_tuple, dtype=dtype) * 2 - 1
            elif base_name in ("acos"):
                t = torch.rand(dims_tuple, dtype=dtype) * 1.8 - 0.9
            elif base_name in ("atanh"):
                t = torch.rand(dims_tuple, dtype=dtype) * 1.8 - 0.9
            elif base_name in ("div", "true_divide", "floor_divide"):
                t = torch.randn(dims_tuple, dtype=dtype)
                t = torch.where(t == 0, torch.ones_like(t), t)
            elif base_name in ("pow"):
                t = torch.randn(dims_tuple, dtype=dtype).abs() + 0.1
            else:
                t = torch.randn(dims_tuple, dtype=dtype)
        elif dtype == torch.bool:
            t = torch.randint(low=0, high=2, size=dims_tuple, dtype=dtype)
        else:
            t = torch.randint(low=0, high=10, size=dims_tuple, dtype=dtype)

        if strides is not None and len(strides) == len(dims_tuple):
            try:
                if isinstance(strides, list) and len(strides) > 0 and isinstance(strides[0], list):
                    strides = strides[0]
                t = torch.as_strided(t, size=dims_tuple, stride=tuple(strides))
            except RuntimeError:
                pass
        return t

    def _build_tensors(self, op_info) -> Dict[int, torch.Tensor]:
        tensors: Dict[int, torch.Tensor] = {}
        op_name = getattr(op_info, "name", "")
        for i, dims in enumerate(op_info.input_dims):
            type_str = op_info.input_types[i] if i < len(op_info.input_types) else "float"
            type_lower = type_str.lower().strip()
            if type_lower == "tensorlist":
                if dims and isinstance(dims, list) and isinstance(dims[0], list):
                    tensor_list = []
                    for sub_dims in dims:
                        if sub_dims:
                            t = self._build_tensor(sub_dims, None, torch.float32, op_name)
                            tensor_list.append(t)
                    if tensor_list:
                        tensors[i] = tensor_list
                elif not dims:
                    default_dims = [2, 3]
                    t1 = self._build_tensor(default_dims, None, torch.float32, op_name)
                    t2 = self._build_tensor(default_dims, None, torch.float32, op_name)
                    tensors[i] = [t1, t2]
                continue
            if type_lower in _TENSOR_TYPES:
                dtype = self._map_dtype(type_str)
                if not dims:
                    # 空维度可能是0维标量张量（scalar tensor）
                    t = torch.tensor(0, dtype=dtype)
                    tensors[i] = t
                    continue
                strides = op_info.input_strides[i] if i < len(op_info.input_strides) else None
                t = self._build_tensor(dims, strides, dtype, op_name)
                tensors[i] = t
        return tensors

    def _parse_value(self, value: Any) -> Any:
        if isinstance(value, str):
            v = value.strip()
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            if v.startswith("[") or v.startswith("("):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, list):
                        return tuple(parsed)
                    return parsed
                except (ValueError, SyntaxError):
                    pass
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except ValueError:
                pass
        elif isinstance(value, list):
            return tuple(value)
        return value

    def _build_args_and_kwargs(
        self, mapped_op: MappedOp
    ) -> tuple[List[Any], Dict[str, Any]]:
        """解析 concrete_inputs 为 args 和 kwargs。

        对可获取签名的函数使用 kwargs；对 built-in 函数（无签名）使用 args。
        """
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        concrete = mapped_op.op_info.concrete_inputs
        if not concrete:
            return args, kwargs

        params = []
        has_signature = False
        try:
            sig = inspect.signature(mapped_op.callable)
            params = list(sig.parameters.values())
            if params and not all(
                p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in params
            ):
                has_signature = True
        except ValueError:
            pass

        non_tensor_types = [
            i for i, t in enumerate(mapped_op.op_info.input_types)
            if t.lower().strip() not in _TENSOR_TYPES and t.lower().strip() != "tensorlist"
        ]
        use_index = (
            len(concrete) == len(mapped_op.op_info.input_types)
            and len(concrete) > 0
            and all(
                concrete[i] != "" or mapped_op.op_info.input_types[i].lower().strip() in _TENSOR_TYPES
                for i in non_tensor_types
            )
        )
        concrete_idx = 0
        for i, type_str in enumerate(mapped_op.op_info.input_types):
            if use_index:
                if i >= len(concrete):
                    break
                value = concrete[i]
            else:
                if concrete_idx >= len(concrete):
                    break
                if type_str.lower().strip() in _TENSOR_TYPES:
                    continue
                value = concrete[concrete_idx]
                concrete_idx += 1

            if type_str.lower().strip() in _TENSOR_TYPES:
                continue

            parsed_value = self._parse_value(value) if value != "" else None

            if has_signature and i < len(params):
                param = params[i]
                if param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    if parsed_value is not None:
                        kwargs[param.name] = parsed_value
            elif not has_signature:
                args.append(parsed_value)

        return args, kwargs

    def build(self, mapped_op: MappedOp) -> TestCase:
        tensor_map = self._build_tensors(mapped_op.op_info)
        args, kwargs = self._build_args_and_kwargs(mapped_op)

        positional_args = []
        concrete = mapped_op.op_info.concrete_inputs
        use_index = len(concrete) == len(mapped_op.op_info.input_types)
        concrete_idx = 0

        input_dims = mapped_op.op_info.input_dims
        for i, type_str in enumerate(mapped_op.op_info.input_types):
            type_lower = type_str.lower().strip()
            if type_lower in _TENSOR_TYPES or type_lower == "tensorlist":
                if i in tensor_map:
                    positional_args.append(tensor_map[i])
                elif i < len(input_dims) and input_dims[i]:
                    positional_args.append(None)
                else:
                    if use_index:
                        if i < len(concrete):
                            value = concrete[i]
                            parsed = self._parse_value(value) if value != "" else None
                            positional_args.append(parsed)
                        else:
                            positional_args.append(None)
                    else:
                        if concrete_idx < len(concrete):
                            value = concrete[concrete_idx]
                            concrete_idx += 1
                            parsed = self._parse_value(value) if value != "" else None
                            positional_args.append(parsed)
                        else:
                            positional_args.append(None)
            else:
                if use_index:
                    if i < len(concrete):
                        value = concrete[i]
                        parsed = self._parse_value(value) if value != "" else None
                        positional_args.append(parsed)
                    else:
                        positional_args.append(None)
                else:
                    if concrete_idx < len(concrete):
                        value = concrete[concrete_idx]
                        concrete_idx += 1
                        parsed = self._parse_value(value) if value != "" else None
                        positional_args.append(parsed)
                    else:
                        positional_args.append(None)

        base_name = mapped_op.op_info.name.replace("aten::", "").replace("aten::_", "")
        base_name = base_name.split(".")[0]

        if base_name in ("sum", "mean"):
            while len(positional_args) > 3:
                positional_args.pop()
        elif base_name == "arange":
            while len(positional_args) > 3:
                positional_args.pop()

        if base_name == "add" and len(positional_args) >= 3 and positional_args[-1] is not None:
            kwargs["alpha"] = positional_args[-1]
            positional_args = positional_args[:-1]
        elif base_name == "baddbmm" and len(positional_args) >= 5 and positional_args[-1] is not None and positional_args[-2] is not None:
            kwargs["beta"] = positional_args[-2]
            kwargs["alpha"] = positional_args[-1]
            positional_args = positional_args[:-2]
        elif base_name == "addmm" and len(positional_args) >= 5 and positional_args[-1] is not None and positional_args[-2] is not None:
            kwargs["beta"] = positional_args[-2]
            kwargs["alpha"] = positional_args[-1]
            positional_args = positional_args[:-2]
        elif base_name == "log_softmax" and len(positional_args) >= 3 and positional_args[-1] is not None:
            last_arg = positional_args[-1]
            if isinstance(last_arg, int) and not isinstance(last_arg, bool):
                positional_args = positional_args[:-1]
            else:
                kwargs["dtype"] = last_arg
                positional_args = positional_args[:-1]
        elif base_name == "div" and len(positional_args) >= 3 and positional_args[-1] is not None:
            kwargs["rounding_mode"] = positional_args[-1]
            positional_args = positional_args[:-1]
        elif base_name == "sum":
            if len(positional_args) >= 2 and positional_args[-1] is not None:
                last_arg = positional_args[-1]
                if isinstance(last_arg, int) and not isinstance(last_arg, bool):
                    positional_args = positional_args[:-1]
                elif isinstance(last_arg, bool) and len(positional_args) >= 3:
                    second_last = positional_args[-2]
                    if isinstance(second_last, int) and not isinstance(second_last, bool):
                        positional_args = positional_args[:-2] + positional_args[-1:]
                elif len(positional_args) == 2 and isinstance(last_arg, int) and not isinstance(last_arg, bool):
                    positional_args = positional_args[:-1]

        if base_name in ("add", "mul", "sub") and len(positional_args) == 1:
            if concrete and len(concrete) >= 2 and concrete[1] != "":
                second = self._parse_value(concrete[1])
                if second is not None:
                    positional_args.append(second)
            else:
                positional_args.append(1.0)

        positional_args = [a for a in positional_args if a is not None]

        if base_name == "div" and len(positional_args) == 1:
            positional_args.append(1.0)

        if base_name == "arange":
            positional_args = [a for a in positional_args if a is not None]

        if base_name == "exp" and len(positional_args) > 1:
            positional_args = positional_args[:1]

        if base_name == "linalg_vector_norm" and len(positional_args) >= 3:
            dim_arg = positional_args[2]
            if isinstance(dim_arg, bool):
                positional_args = positional_args[:2]
            elif isinstance(dim_arg, int):
                positional_args[2] = (dim_arg,)

        if base_name.endswith("_") and base_name not in ("_", ""):
            return TestCase(
                mapped_op=mapped_op,
                input_tensors=list(tensor_map.values()),
                args=[],
                kwargs={},
                positional_args=[],
            )

        if not positional_args and not tensor_map:
            return TestCase(
                mapped_op=mapped_op,
                input_tensors=list(tensor_map.values()),
                args=[],
                kwargs={},
                positional_args=[],
            )

        args = []

        return TestCase(
            mapped_op=mapped_op,
            input_tensors=list(tensor_map.values()),
            args=args,
            kwargs=kwargs,
            positional_args=positional_args,
        )
