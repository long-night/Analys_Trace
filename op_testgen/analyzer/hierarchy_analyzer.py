"""层级调用分析器"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from op_testgen.parser.trace_parser import HierarchicalOpInfo


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


class _TreePattern:
    """用于去重聚合的树形模式节点"""

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.self_duration_us = 0.0
        self.total_duration_us = 0.0
        self.children: Dict[str, '_TreePattern'] = {}
        self.is_recursive = False


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

    def get_self_time_stats(self):
        """基于自耗时的算子统计（排除子算子耗时）"""
        from op_testgen.analyzer.trace_analyzer import OperatorStats
        stats: Dict[str, OperatorStats] = {}
        for node in self.all_nodes:
            name = node.name
            if name not in stats:
                stats[name] = OperatorStats(name)
            stats[name].call_count += 1
            stats[name].total_duration_us += node.self_duration_us
        return stats

    def get_total_time_stats(self):
        """基于总耗时的算子统计（含子算子）"""
        from op_testgen.analyzer.trace_analyzer import OperatorStats
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
                for j in range(i + 2, min(i + max_cycle_length + 2, len(chain) + 1)):
                    cycle = tuple(chain[i:j])
                    if cycle[0] == cycle[-1]:
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

    def get_root_ops_stats(self):
        """根节点（入口算子）统计"""
        from op_testgen.analyzer.trace_analyzer import OperatorStats
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

    def _build_pattern_tree(self) -> Dict[str, _TreePattern]:
        """构建去重聚合后的模式树"""
        roots: Dict[str, _TreePattern] = {}

        for root in self.root_ops:
            if root.name not in roots:
                roots[root.name] = _TreePattern(root.name)
            rp = roots[root.name]
            rp.count += 1
            rp.self_duration_us += root.self_duration_us
            rp.total_duration_us += root.duration_us

            self._merge_into_pattern(rp, root, set())

        return roots

    def _merge_into_pattern(self, pattern: _TreePattern, node: HierarchicalOpInfo,
                            seen: set):
        for child in node.children:
            child_name = child.name
            if child_name in seen:
                if child_name not in pattern.children:
                    pattern.children[child_name] = _TreePattern(child_name)
                pattern.children[child_name].is_recursive = True
                pattern.children[child_name].count += 1
                continue

            if child_name not in pattern.children:
                pattern.children[child_name] = _TreePattern(child_name)

            cp = pattern.children[child_name]
            cp.count += 1
            cp.self_duration_us += child.self_duration_us
            cp.total_duration_us += child.duration_us

            self._merge_into_pattern(cp, child, seen | {node.name})

    def export_call_tree_text(self, output_file: str, max_depth: int = 10,
                              deduplicate: bool = True):
        """导出文本格式的调用树

        Args:
            output_file: 输出文件路径
            max_depth:   最大展开深度
            deduplicate: 是否对相同父子关系进行聚合去重（默认 True）
        """
        with open(output_file, "w", encoding="utf-8") as f:
            if deduplicate:
                self._write_deduplicated_tree(f, max_depth)
            else:
                for root in self.root_ops:
                    self._write_raw_tree_node(f, root, 0, max_depth)

    def _write_deduplicated_tree(self, f, max_depth: int = 10):
        """写入去重聚合后的树形视图"""
        pattern_roots = self._build_pattern_tree()

        f.write("=" * 70 + "\n")
        f.write("调用树分析（去重聚合视图）\n")
        f.write("=" * 70 + "\n")
        f.write(f"\n说明：\n")
        f.write("  • 相同父子关系已聚合，显示 [出现次数] 和 [平均耗时]\n")
        f.write("  • [RECURSIVE] 标记表示检测到递归/循环调用，已截断展开\n")
        f.write("  • 不同根节点之间用分隔线 '---' 分开\n\n")

        # 按总耗时排序根节点
        sorted_roots = sorted(
            pattern_roots.items(),
            key=lambda x: x[1].total_duration_us,
            reverse=True
        )

        for idx, (root_name, root_pat) in enumerate(sorted_roots, 1):
            f.write(f"\n{'─' * 70}\n")
            f.write(f"根节点 #{idx}: {root_name}\n")
            f.write(f"{'─' * 70}\n")

            avg_self = root_pat.self_duration_us / root_pat.count / 1000.0
            avg_total = root_pat.total_duration_us / root_pat.count / 1000.0
            f.write(f"  出现次数: {root_pat.count}\n")
            f.write(f"  自耗时:   {root_pat.self_duration_us / 1000.0:.3f} ms "
                    f"(平均 {avg_self:.3f} ms)\n")
            f.write(f"  总耗时:   {root_pat.total_duration_us / 1000.0:.3f} ms "
                    f"(平均 {avg_total:.3f} ms)\n")
            f.write(f"\n  子树结构:\n")

            self._write_pattern_node(
                f, root_pat, prefix="  ", is_last=True,
                depth=0, max_depth=max_depth
            )

        rec_patterns = self.find_recursive_patterns()
        if rec_patterns:
            f.write(f"\n\n{'=' * 70}\n")
            f.write("递归 / 循环调用模式汇总\n")
            f.write(f"{'=' * 70}\n\n")
            for i, pat in enumerate(rec_patterns[:10], 1):
                cycle_str = " -> ".join(pat.cycle)
                f.write(f"  {i}. {cycle_str}\n")
                f.write(f"     出现次数: {pat.occurrence_count}, 最大深度: {pat.depth}\n\n")

        chains = self.get_call_chains(min_occurrence=2)
        if chains:
            f.write(f"\n{'=' * 70}\n")
            f.write("高频调用链汇总（出现 >= 2 次）\n")
            f.write(f"{'=' * 70}\n\n")
            for i, chain in enumerate(chains[:10], 1):
                chain_str = " -> ".join(chain.chain)
                f.write(f"  {i}. {chain_str}\n")
                f.write(f"     出现 {chain.occurrence_count} 次, "
                        f"平均耗时: {chain.avg_duration_ms:.3f} ms, "
                        f"累计: {chain.total_duration_ms:.3f} ms\n\n")

    def _write_pattern_node(self, f, pattern: _TreePattern, prefix: str,
                            is_last: bool, depth: int, max_depth: int):
        if depth > max_depth:
            return

        branch = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        avg_self = pattern.self_duration_us / pattern.count / 1000.0
        avg_total = pattern.total_duration_us / pattern.count / 1000.0

        if pattern.is_recursive:
            rec_tag = " [RECURSIVE]"
        else:
            rec_tag = ""

        if pattern.count > 1:
            count_tag = f" [×{pattern.count}]"
        else:
            count_tag = ""

        line = (f"{prefix}{branch}{pattern.name}{count_tag}{rec_tag}  "
                f"[self={avg_self:.3f}ms, total={avg_total:.3f}ms]")
        f.write(line + "\n")

        if pattern.is_recursive:
            return

        children = list(pattern.children.values())
        children.sort(key=lambda c: c.total_duration_us, reverse=True)

        for i, child in enumerate(children):
            last = (i == len(children) - 1)
            self._write_pattern_node(
                f, child, child_prefix, last,
                depth + 1, max_depth
            )

    def _write_raw_tree_node(self, f, node: HierarchicalOpInfo, indent: int,
                             max_depth: int):
        """原始未去重的树节点写入（保留兼容）"""
        if indent > max_depth:
            return
        spaces = "  " * indent
        self_dur = node.self_duration_us / 1000.0
        total_dur = node.duration_us / 1000.0
        f.write(f"{spaces}{node.name}  "
                f"[self={self_dur:.3f}ms, total={total_dur:.3f}ms, "
                f"depth={node.depth}]\n")
        for child in node.children:
            self._write_raw_tree_node(f, child, indent + 1, max_depth)

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
