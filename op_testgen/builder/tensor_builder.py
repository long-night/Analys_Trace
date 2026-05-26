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
    "char", "int8", "byte", "uint8", "unsigned char", "bool",
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
        "unsigned char": torch.uint8,
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

    @staticmethod
    def _compute_storage_requirements(size: tuple, stride: tuple) -> tuple[int, int]:
        """计算 as_strided 所需的底层存储大小和 offset。

        原理：
        - 每个维度的索引范围是 [0, size[i]-1]
        - 该维度的 offset 范围取决于 stride 的正负
        - 总 offset 的最小值可能为负，需要 storage_offset 平移到非负
        - 存储大小 = 最大总 offset - 最小总 offset + 1

        Args:
            size: 张量各维度大小
            stride: 张量各维度步长

        Returns:
            (storage_size, storage_offset)
        """
        dim_offsets = []
        for s, st in zip(size, stride):
            if st >= 0:
                start, end = 0, (s - 1) * st
            else:
                start, end = (s - 1) * st, 0
            dim_offsets.append((start, end))

        total_start = sum(start for start, _ in dim_offsets)
        total_end = sum(end for _, end in dim_offsets)
        storage_size = total_end - total_start + 1
        storage_offset = -total_start

        return storage_size, storage_offset

    def _create_data_tensor(self, size, dtype: torch.dtype, op_name: str = "") -> torch.Tensor:
        """根据算子特性创建带适当初始值的张量。

        Args:
            size: 张量大小（int 或 tuple）
            dtype: 数据类型
            op_name: 算子名称，用于特殊初始化

        Returns:
            初始化后的张量
        """
        if isinstance(size, int):
            size = (size,)

        base_name = op_name.replace("aten::", "").replace("aten::_", "")

        if self._is_floating(dtype):
            if base_name in ("rsqrt", "sqrt", "log", "log1p", "reciprocal"):
                t = torch.rand(size, dtype=dtype) + 0.1
            elif base_name in ("asin", "acos", "atan"):
                t = torch.rand(size, dtype=dtype) * 2 - 1
            elif base_name in ("acos",):
                t = torch.rand(size, dtype=dtype) * 1.8 - 0.9
            elif base_name in ("atanh",):
                t = torch.rand(size, dtype=dtype) * 1.8 - 0.9
            elif base_name in ("div", "true_divide", "floor_divide"):
                t = torch.randn(size, dtype=dtype)
                t = torch.where(t == 0, torch.ones_like(t), t)
            elif base_name in ("pow",):
                t = torch.randn(size, dtype=dtype).abs() + 0.1
            elif base_name in ("softmax",):
                t = torch.randn(size, dtype=dtype) * 20 - 10
            else:
                t = torch.randn(size, dtype=dtype) * 2 - 1
        elif dtype == torch.bool:
            t = torch.randint(low=0, high=2, size=size, dtype=dtype)
        else:
            t = torch.randint(low=0, high=10, size=size, dtype=dtype)

        return t

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
        # 解析 dims（处理嵌套列表的情况）
        if isinstance(dims, list) and len(dims) > 0 and isinstance(dims[0], list):
            dims = dims[0]
        size_tuple = tuple(dims)

        # 无 stride 信息或长度不匹配：创建连续张量（正常路径，不警告）
        if strides is None or len(strides) != len(size_tuple):
            return self._create_data_tensor(size_tuple, dtype, op_name)

        # 解析 strides（处理嵌套列表的情况）
        if isinstance(strides, list) and len(strides) > 0 and isinstance(strides[0], list):
            strides = strides[0]
        stride_tuple = tuple(strides)

        try:
            # 计算底层存储大小（使用 abs(stride) 确保大小正确）
            storage_size = sum((s - 1) * abs(st) for s, st in zip(size_tuple, stride_tuple)) + 1

            # 创建足够大的底层张量
            base_tensor = self._create_data_tensor(storage_size, dtype, op_name)

            # 检查是否有负 stride
            has_negative = any(st < 0 for st in stride_tuple)

            if not has_negative:
                # 标准路径：正 stride，直接 as_strided
                return torch.as_strided(
                    base_tensor,
                    size=size_tuple,
                    stride=stride_tuple,
                    storage_offset=0
                )
            else:
                # 负 stride 模拟路径：用 abs(stride) + flip
                abs_strides = tuple(abs(st) for st in stride_tuple)
                t = torch.as_strided(
                    base_tensor,
                    size=size_tuple,
                    stride=abs_strides,
                    storage_offset=0
                )

                # 对负 stride 维度 flip，使数据遍历顺序一致
                flip_dims = [i for i, st in enumerate(stride_tuple) if st < 0]
                t = torch.flip(t, dims=flip_dims)

                return t

        except Exception as e:
            # 明确记录失败信息，不静默回退
            import warnings
            warnings.warn(
                f"TensorBuilder 无法为算子 '{op_name}' 创建非连续张量: {e}. "
                f"size={size_tuple}, stride={stride_tuple}. "
                f"回退到连续张量。",
                RuntimeWarning,
                stacklevel=2
            )
            return self._create_data_tensor(size_tuple, dtype, op_name)

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
                        else:
                            t = self._build_tensor([2, 3], None, torch.float32, op_name)
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
                    if i < len(op_info.concrete_inputs) and op_info.concrete_inputs[i] != "":
                        continue
                    t = torch.tensor(1, dtype=dtype)
                    tensors[i] = t
                    continue
                strides = op_info.input_strides[i] if i < len(op_info.input_strides) else None
                t = self._build_tensor(dims, strides, dtype, op_name)
                tensors[i] = t
        return tensors

    def _parse_value(self, value: Any, target_dtype: Optional[torch.dtype] = None) -> Any:
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
                v = v.replace("\u2212", "-")
                is_scientific = (
                    ("e" in v.lower())
                    and v.lower() not in ("true", "false")
                    and any(ch.isdigit() for ch in v)
                )
                if "." in v or is_scientific:
                    parsed = float(v)
                else:
                    parsed = int(v)
                if target_dtype is not None and isinstance(parsed, float):
                    if target_dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                        parsed = float(parsed)
                    elif target_dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
                        parsed = int(parsed)
                return parsed
            except ValueError:
                pass
        elif isinstance(value, list):
            return tuple(value)
        return value

    def build(self, mapped_op: MappedOp) -> TestCase:
        tensor_map = self._build_tensors(mapped_op.op_info)

        positional_args: List[Any] = []
        kwargs: Dict[str, Any] = {}
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
                        if i < len(concrete) and concrete[i] != "":
                            positional_args.append(self._parse_value(concrete[i]))
                        else:
                            positional_args.append(None)
                    else:
                        if concrete_idx < len(concrete) and concrete[concrete_idx] != "":
                            positional_args.append(self._parse_value(concrete[concrete_idx]))
                            concrete_idx += 1
                        else:
                            positional_args.append(None)
            else:
                if use_index:
                    if i < len(concrete) and concrete[i] != "":
                        positional_args.append(self._parse_value(concrete[i]))
                    else:
                        positional_args.append(None)
                else:
                    if concrete_idx < len(concrete) and concrete[concrete_idx] != "":
                        positional_args.append(self._parse_value(concrete[concrete_idx]))
                        concrete_idx += 1
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
            if len(positional_args) >= 2:
                start, end = positional_args[0], positional_args[1]
                step = positional_args[2] if len(positional_args) > 2 else 1
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    if start > end and (step is None or (isinstance(step, (int, float)) and step > 0)):
                        positional_args[0], positional_args[1] = end, start
        elif base_name == "batch_norm":
            while len(positional_args) > 8:
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
        elif base_name in ("sub", "sub_") and len(positional_args) >= 3 and positional_args[-1] is not None:
            kwargs["alpha"] = positional_args[-1]
            positional_args = positional_args[:-1]

        if base_name in ("add", "mul", "sub") and len(positional_args) == 1:
            if concrete and len(concrete) >= 2 and concrete[1] != "":
                second = self._parse_value(concrete[1])
                if second is not None:
                    positional_args.append(second)
            else:
                positional_args.append(1.0)

        while positional_args and positional_args[-1] is None:
            positional_args.pop()

        if base_name == "index_put_" and len(positional_args) >= 3 and positional_args[1] is None:
            input_tensor = positional_args[0]
            values_tensor = positional_args[2]
            if isinstance(input_tensor, torch.Tensor) and isinstance(values_tensor, torch.Tensor):
                indices = []
                for i in range(input_tensor.dim()):
                    shape = [1] * input_tensor.dim()
                    shape[i] = input_tensor.shape[i]
                    idx = torch.arange(input_tensor.shape[i]).view(shape)
                    indices.append(idx)
                positional_args[1] = tuple(indices)
                if values_tensor.dim() < input_tensor.dim() or values_tensor.shape != input_tensor.shape:
                    try:
                        positional_args[2] = values_tensor.expand_as(input_tensor)
                    except RuntimeError:
                        expanded = values_tensor
                        for _ in range(input_tensor.dim() - values_tensor.dim()):
                            expanded = expanded.unsqueeze(-1)
                        positional_args[2] = expanded.expand_as(input_tensor)

        if base_name == "div" and len(positional_args) == 1:
            positional_args.append(1.0)

        if base_name == "exp" and len(positional_args) > 1:
            positional_args = positional_args[:1]

        if base_name == "linalg_vector_norm" and len(positional_args) >= 3:
            dim_arg = positional_args[2]
            if isinstance(dim_arg, bool):
                positional_args = positional_args[:2]
            elif isinstance(dim_arg, int):
                positional_args[2] = (dim_arg,)

        if not positional_args and not tensor_map:
            return TestCase(
                mapped_op=mapped_op,
                input_tensors=list(tensor_map.values()),
                args=[],
                kwargs={},
                positional_args=[],
            )

        return TestCase(
            mapped_op=mapped_op,
            input_tensors=list(tensor_map.values()),
            args=[],
            kwargs=kwargs,
            positional_args=positional_args,
        )
