"""算子映射与过滤模块"""
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import yaml

from op_testgen.parser.trace_parser import OpInfo


@dataclass
class MappedOp:
    """映射后的算子信息"""
    op_info: OpInfo
    callable: Callable
    callable_path: str
    namespace: str
    support_status: str  # "whitelisted" | "auto_discovered" | "unsupported"


class OpMapper:
    """算子名称到 Python callable 的映射器"""

    # 黑名单: 不追踪的算子模式
    BLACKLIST_PATTERNS = [
        "aten::empty",
        "aten::zeros",
        "aten::ones",
        "aten::full",
        "aten::to",
        "aten::copy_",
        "aten::detach",
        "aten::clone",
        "aten::call",
        "aten::__interpolate",
        "aten::as_strided",
        "aten::set_",
        "aten::_reshape_alias",
    ]

    # 特殊映射规则: trace_name -> callable_path
    SPECIAL_MAPPINGS = {
        "aten::conv2d": "torch.nn.functional.conv2d",
        "aten::linear": "torch.nn.functional.linear",
        "aten::relu": "torch.nn.functional.relu",
        "aten::max_pool2d": "torch.nn.functional.max_pool2d",
        "aten::avg_pool2d": "torch.nn.functional.avg_pool2d",
        "aten::batch_norm": "torch.nn.functional.batch_norm",
        "c10d::allreduce_": "torch.distributed.all_reduce",
        "c10d::allgather_": "torch.distributed.all_gather",
        "c10d::broadcast_": "torch.distributed.broadcast",
    }

    def __init__(self, whitelist_path: Optional[str] = None):
        self.whitelist: Dict[str, dict] = {}
        if whitelist_path:
            self._load_whitelist(whitelist_path)
        else:
            import os
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "op_whitelist.yaml"
            )
            if os.path.exists(default_path):
                self._load_whitelist(default_path)

    def _load_whitelist(self, path: str):
        """加载白名单 YAML"""
        with open(path, "r", encoding="utf-8") as f:
            self.whitelist = yaml.safe_load(f) or {}

    def _is_blacklisted(self, name: str) -> bool:
        """检查是否在黑名单中"""
        return name in self.BLACKLIST_PATTERNS

    def _resolve_callable(self, path: str) -> Optional[Callable]:
        """通过路径解析 callable，如 'torch.add' -> torch.add"""
        parts = path.split(".")
        try:
            module = importlib.import_module(".".join(parts[:-1]))
            obj = getattr(module, parts[-1])
            if callable(obj):
                return obj
        except (ImportError, AttributeError):
            pass
        return None

    def _strip_tensor_suffix(self, name: str) -> str:
        """去掉 .Tensor 后缀"""
        if name.endswith(".Tensor"):
            return name[:-7]
        return name

    def _try_dynamic_map(self, name: str) -> Optional[tuple]:
        """尝试动态反射映射"""
        # 1. 检查特殊映射
        if name in self.SPECIAL_MAPPINGS:
            path = self.SPECIAL_MAPPINGS[name]
            callable_obj = self._resolve_callable(path)
            if callable_obj:
                namespace = path.split(".")[0]
                if "nn.functional" in path:
                    namespace = "torch.nn.functional"
                elif "distributed" in path:
                    namespace = "torch.distributed"
                return callable_obj, path, namespace

        # 2. 尝试 aten::xxx -> torch.xxx
        if name.startswith("aten::"):
            base_name = self._strip_tensor_suffix(name[6:])
            # 跳过私有方法（以下划线开头）
            if base_name.startswith("_"):
                return None
            path = f"torch.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch"

            # 尝试 torch.nn.functional.xxx
            path = f"torch.nn.functional.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch.nn.functional"

        # 3. 尝试 torch.linalg.xxx
        if name.startswith("aten::linalg_"):
            base_name = name[13:]
            path = f"torch.linalg.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch.linalg"

        return None

    def map_operator(self, op_info: OpInfo) -> Optional[MappedOp]:
        """映射单个算子"""
        name = op_info.name

        # 1. 黑名单检查
        if self._is_blacklisted(name):
            return None

        # 2. 白名单优先
        if name in self.whitelist:
            entry = self.whitelist[name]
            path = entry.get("callable_path", "")
            namespace = entry.get("namespace", "torch")
            callable_obj = self._resolve_callable(path)
            if callable_obj:
                return MappedOp(
                    op_info=op_info,
                    callable=callable_obj,
                    callable_path=path,
                    namespace=namespace,
                    support_status="whitelisted",
                )

        # 3. 动态反射
        result = self._try_dynamic_map(name)
        if result:
            callable_obj, path, namespace = result
            return MappedOp(
                op_info=op_info,
                callable=callable_obj,
                callable_path=path,
                namespace=namespace,
                support_status="auto_discovered",
            )

        # 4. 无法映射
        return None

    def map_all(self, op_infos: List[OpInfo]) -> List[MappedOp]:
        """批量映射算子列表"""
        mapped = []
        for op_info in op_infos:
            m = self.map_operator(op_info)
            if m is not None:
                mapped.append(m)
        return mapped
