"""层级调用分析器"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from op_testgen.parser.trace_parser import HierarchicalOpInfo
from op_testgen.analyzer.trace_analyzer import OperatorStats


@dataclass
class CallChain:
    """调用链模式"""
    chain: Tuple[str, ...]
    occurrence_count: int
    total_duration_ms: float
    avg_duration_ms: float


@dataclass
class RecursivePattern:
    """递归/循环模式"""
    cycle: Tuple[str, ...]
    occurrence_count: int
    depth: int


class HierarchyAnalyzer:
    """层级调用分析器"""

    def __init__(self, root_ops: List[HierarchicalOpInfo]):
        self.root_ops = root_ops
        self._all_nodes: List[HierarchicalOpInfo] = []

    @property
    def all_nodes(self) -> List[HierarchicalOpInfo]:
        if not self._all_nodes:
            self._dfs_collect(self.root_ops)
        return self._all_nodes

    def _dfs_collect(self, nodes: List[HierarchicalOpInfo]):
        """深度优先收集所有节点"""
        for node in nodes:
            self._all_nodes.append(node)
            self._dfs_collect(node.children)

    def get_self_time_stats(self) -> Dict[str, OperatorStats]:
        """基于自耗时的算子统计（排除子算子耗时）"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.all_nodes:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.self_duration_us
        return stats

    def get_total_time_stats(self) -> Dict[str, OperatorStats]:
        """基于总耗时的算子统计（含子算子）"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.all_nodes:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.duration_us
        return stats

    def get_call_chains(self, min_occurrence: int = 2, max_depth: int = 10) -> List[CallChain]:
        """提取高频调用链模式"""
        chain_counts: Dict[Tuple[str, ...], List[float]] = defaultdict(list)

        for node in self.all_nodes:
            chain = tuple(node.get_call_chain())
            if len(chain) <= max_depth:
                chain_counts[chain].append(node.duration_us)

        results = []
        for chain, durations in chain_counts.items():
            if len(durations) >= min_occurrence:
                total_ms = sum(durations) / 1000.0
                avg_ms = total_ms / len(durations)
                results.append(CallChain(
                    chain=chain,
                    occurrence_count=len(durations),
                    total_duration_ms=total_ms,
                    avg_duration_ms=avg_ms,
                ))

        results.sort(key=lambda x: x.occurrence_count, reverse=True)
        return results

    def find_recursive_patterns(self, max_cycle_length: int = 5) -> List[RecursivePattern]:
        """识别递归/循环调用模式"""
        cycle_counts: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

        for node in self.all_nodes:
            chain = node.get_call_chain()
            for i in range(len(chain)):
                for j in range(i + 1, min(i + max_cycle_length + 1, len(chain))):
                    cycle = tuple(chain[i:j])
                    if len(cycle) >= 2 and cycle[0] == cycle[-1]:
                        cycle_counts[cycle].append(node.depth)

        results = []
        for cycle, depths in cycle_counts.items():
            results.append(RecursivePattern(
                cycle=cycle,
                occurrence_count=len(depths),
                depth=max(depths),
            ))

        results.sort(key=lambda x: x.occurrence_count, reverse=True)
        return results

    def get_root_ops_stats(self) -> Dict[str, OperatorStats]:
        """根节点（入口算子）统计"""
        stats: Dict[str, OperatorStats] = {}
        for node in self.root_ops:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.duration_us
        return stats

    def get_depth_distribution(self) -> Dict[int, int]:
        """调用深度分布（depth -> 节点数）"""
        dist: Dict[int, int] = defaultdict(int)
        for node in self.all_nodes:
            dist[node.depth] += 1
        return dict(sorted(dist.items()))

    def export_call_tree_text(self, output_file: str, max_depth: int = 10):
        """导出文本格式的调用树"""
        with open(output_file, "w", encoding="utf-8") as f:
            for root in self.root_ops:
                self._write_tree_node(f, root, 0, max_depth)

    def _write_tree_node(self, f, node: HierarchicalOpInfo, indent: int, max_depth: int):
        """递归写入树节点"""
        if indent > max_depth:
            return
        prefix = "  " * indent
        self_dur = node.self_duration_us / 1000.0
        total_dur = node.duration_us / 1000.0
        f.write(f"{prefix}{node.name}  [self={self_dur:.3f}ms, total={total_dur:.3f}ms, depth={node.depth}]\n")
        for child in node.children:
            self._write_tree_node(f, child, indent + 1, max_depth)

    def export_call_tree_json(self, output_file: str):
        """导出 JSON 格式的调用树（含完整层级）"""
        def node_to_dict(node: HierarchicalOpInfo) -> dict:
            return {
                "name": node.name,
                "duration_us": node.duration_us,
                "self_duration_us": node.self_duration_us,
                "depth": node.depth,
                "is_root": node.is_root,
                "children": [node_to_dict(c) for c in node.children],
            }

        tree = [node_to_dict(r) for r in self.root_ops]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
