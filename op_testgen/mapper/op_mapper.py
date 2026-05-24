"""算子映射与过滤模块"""
import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

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

    def __init__(
        self,
        whitelist_path: Optional[str] = None,
        blacklist_path: Optional[str] = None,
    ):
        self.whitelist: Dict[str, dict] = {}
        self.blacklist: List[str] = []
        self.unmapped_ops: Set[str] = set()

        self._load_whitelist(whitelist_path)
        self._load_blacklist(blacklist_path)

    def _default_config_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "config", filename)

    def _load_whitelist(self, path: Optional[str]):
        """加载白名单 YAML"""
        if path is None:
            path = self._default_config_path("op_whitelist.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.whitelist = yaml.safe_load(f) or {}

    def _load_blacklist(self, path: Optional[str]):
        """加载黑名单 YAML"""
        if path is None:
            path = self._default_config_path("op_blacklist.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.blacklist = data.get("patterns", [])

    def _is_blacklisted(self, name: str) -> bool:
        """检查是否在黑名单中（支持精确匹配和前缀匹配）"""
        for pattern in self.blacklist:
            if pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True
            elif name == pattern:
                return True
        return False

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

        if name.startswith("aten::"):
            base_name = self._strip_tensor_suffix(name[6:])
            if base_name.startswith("_"):
                return None
            path = f"torch.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch"

            path = f"torch.nn.functional.{base_name}"
            callable_obj = self._resolve_callable(path)
            if callable_obj and callable(callable_obj):
                return callable_obj, path, "torch.nn.functional"

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

        if self._is_blacklisted(name):
            return None

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

        self.unmapped_ops.add(name)
        return None

    def map_all(self, op_infos: List[OpInfo]) -> List[MappedOp]:
        """批量映射算子列表"""
        mapped = []
        for op_info in op_infos:
            m = self.map_operator(op_info)
            if m is not None:
                mapped.append(m)
        return mapped

    @staticmethod
    def update_blacklist_yaml(
        new_patterns: List[str],
        blacklist_path: Optional[str] = None,
    ) -> bool:
        """将新的模式追加到黑名单 YAML 文件中"""
        if blacklist_path is None:
            blacklist_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "op_blacklist.yaml"
            )

        if not os.path.exists(blacklist_path):
            print(f"错误: 黑名单文件不存在: {blacklist_path}")
            return False

        with open(blacklist_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        existing = set(data.get("patterns", []))
        added = []
        for p in new_patterns:
            if p not in existing:
                existing.add(p)
                added.append(p)

        if not added:
            print("没有新的算子需要加入黑名单")
            return False

        data["patterns"] = list(existing)

        with open(blacklist_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        print(f"已更新黑名单，新增 {len(added)} 个算子:")
        for p in added:
            print(f"  - {p}")
        return True
