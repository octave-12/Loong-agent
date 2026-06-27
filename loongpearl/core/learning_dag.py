#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2-B: 学习依赖 DAG (Learning Dependency Graph)

从概念图的关系中推导"要理解X，必须先理解哪些概念"。

推导规则:
  1. "X IS_A Y"      → Y 是 X 的前置概念 (先懂通类，再懂特例)
  2. "X PART_OF Y"    → X 是 Y 的前置概念 (先懂部分，再懂整体)
  3. "X DEFINED_AS Y" → 若 Y 引用其他概念，这些概念是前置

输出:
  - DAG 邻接表: concept → [prerequisites]
  - 拓扑排序: 学习顺序
  - 学习路径: 从已知概念到目标概念的最短路径

使用:
  dag = LearningDAG("data/models/concept_graph.db")
  prereqs = dag.get_prerequisites("量子力学")  # 学量子力学前需要先懂什么
  path = dag.learning_path("原子", "量子力学") # 从原子学到量子力学的路径
"""

import sqlite3
import os
import logging
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

log = logging.getLogger(__name__)

# 元分类标签（不应产生学习依赖）
_META_LABELS = {"中文词条", "成语", "词语", "词条", "词汇"}


class LearningDAG:
    """
    学习依赖有向无环图。

    边 X→Y 含义: "要学 X，必须先学 Y"
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "data", "models", "concept_graph.db"
            )
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._adj: Dict[str, Set[str]] = {}        # concept → {prerequisites}
        self._reverse: Dict[str, Set[str]] = {}    # concept → {dependents}
        self._built = False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # ═══════════════════════════════════════════════════════
    # 构建
    # ═══════════════════════════════════════════════════════

    def build(self, min_conf: float = 0.5) -> 'LearningDAG':
        """
        从概念图构建学习依赖 DAG。

        Args:
            min_conf: IS_A 最低置信度（PART_OF 用 0.7）
        """
        self._adj.clear()
        self._reverse.clear()

        # 规则1: X IS_A Y → Y 是 X 的前置（过滤元分类）
        rows = self.conn.execute(
            "SELECT s, o, c FROM triples WHERE r='IS_A' AND c >= ?",
            (min_conf,)
        ).fetchall()

        for s, o, c in rows:
            if o in _META_LABELS:
                continue
            self._add_edge(s, o, confidence=c, rule="IS_A")

        log.info(f"  IS_A 边: {sum(1 for r in rows if r[1] not in _META_LABELS)} 条 "
                 f"(过滤 {sum(1 for r in rows if r[1] in _META_LABELS)} 条元分类)")

        # 规则2: X PART_OF Y → 双向弱学习依赖（part↔whole 学习顺序不确定）
        #   数据中方向不一致，因此两个方向都加边（置信度低于 IS_A）
        rows = self.conn.execute(
            "SELECT s, o, c FROM triples WHERE r='PART_OF' AND c >= 0.7"
        ).fetchall()

        for s, o, c in rows:
            # 双向: 学 part 有助于学 whole，学 whole 也有助于理解 part
            self._add_edge(s, o, confidence=c * 0.5, rule="PART_OF")
            self._add_edge(o, s, confidence=c * 0.5, rule="PART_OF")

        log.info(f"  PART_OF 边: {len(rows)} 条")

        # 规则3: DEFINED_AS 中引用其他概念 → 弱前置依赖
        # （暂不实现——需要中文分词解析定义文本）

        self._built = True
        log.info(f"DAG 构建完成: {len(self._adj)} 个节点, "
                 f"{sum(len(v) for v in self._adj.values())} 条边")
        return self

    def _add_edge(self, dependent: str, prerequisite: str,
                  confidence: float, rule: str):
        """添加学习依赖边: dependent 依赖 prerequisite"""
        if dependent == prerequisite:
            return
        if dependent not in self._adj:
            self._adj[dependent] = set()
        if prerequisite not in self._reverse:
            self._reverse[prerequisite] = set()

        # 存储为带权边
        self._adj[dependent].add(prerequisite)
        self._reverse[prerequisite].add(dependent)

    # ═══════════════════════════════════════════════════════
    # 查询
    # ═══════════════════════════════════════════════════════

    def get_prerequisites(self, concept: str,
                          max_depth: int = 3) -> Dict[str, List[str]]:
        """
        获取学习某个概念所需的前置知识，按层级组织。

        Returns:
            {0: [concept], 1: [直接前置], 2: [二级前置], ...}
        """
        if not self._built:
            self.build()

        result = {0: [concept]}
        visited = {concept}
        frontier = {concept}

        for depth in range(1, max_depth + 1):
            next_frontier = set()
            for node in frontier:
                for prereq in self._adj.get(node, set()):
                    if prereq not in visited:
                        visited.add(prereq)
                        next_frontier.add(prereq)
            if not next_frontier:
                break
            result[depth] = sorted(next_frontier)
            frontier = next_frontier

        return result

    def get_dependents(self, concept: str) -> List[str]:
        """获取依赖此概念的其他概念（这个学会了，哪些也能学会）"""
        if not self._built:
            self.build()
        return sorted(self._reverse.get(concept, set()))

    def learning_path(self, from_concept: str, to_concept: str) -> Optional[List[str]]:
        """
        从 from_concept 到 to_concept 的最短学习路径（BFS）。

        路径方向：from → ... → to，每个箭头 = "学完这个就可以学下一个"
        """
        if not self._built:
            self.build()

        if from_concept == to_concept:
            return [from_concept]

        # BFS：从 from 出发，沿反向边（dependents）走
        queue = deque([[from_concept]])
        visited = {from_concept}

        while queue:
            path = queue.popleft()
            current = path[-1]

            for dependent in self._reverse.get(current, set()):
                if dependent == to_concept:
                    return path + [to_concept]
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append(path + [dependent])

        return None  # 不可达

    def topological_order(self, concepts: List[str] = None) -> List[str]:
        """
        拓扑排序——给出概念的学习顺序。
        若指定 concepts，只对这些概念排序；否则对所有有依赖关系的概念排序。
        """
        if not self._built:
            self.build()

        # Kahn 算法
        in_degree = defaultdict(int)
        target_nodes = set(concepts) if concepts else set()

        if concepts:
            # 只对指定概念子图做拓扑排序
            subgraph_nodes = set()
            for c in concepts:
                subgraph_nodes.add(c)
                for prereq in self._get_all_prereqs(c):
                    subgraph_nodes.add(prereq)
            nodes = subgraph_nodes & set(self._adj.keys()) | set(self._reverse.keys())
        else:
            nodes = set(self._adj.keys()) | set(self._reverse.keys())

        for node in nodes:
            in_degree.setdefault(node, 0)
            for prereq in self._adj.get(node, set()):
                if prereq in nodes:
                    in_degree[node] += 1

        queue = deque([n for n in nodes if in_degree[n] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in self._reverse.get(node, set()):
                if dependent in nodes:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) < len(nodes):
            log.warning(f"检测到环路! {len(nodes) - len(result)} 个节点无法排序")
            # 追加剩余节点
            remaining = nodes - set(result)
            result.extend(remaining)

        return result

    def _get_all_prereqs(self, concept: str) -> Set[str]:
        """获取所有前置概念（递归）"""
        result = set()
        stack = [concept]
        while stack:
            node = stack.pop()
            for prereq in self._adj.get(node, set()):
                if prereq not in result:
                    result.add(prereq)
                    stack.append(prereq)
        return result

    def stats(self) -> Dict:
        """DAG 统计信息"""
        if not self._built:
            self.build()

        nodes = set(self._adj.keys()) | set(self._reverse.keys())
        in_degrees = [len(self._adj.get(n, set())) for n in nodes]
        out_degrees = [len(self._reverse.get(n, set())) for n in nodes]

        # 找根节点（无前置，即基础概念）
        roots = [n for n in nodes if len(self._adj.get(n, set())) == 0]
        # 找叶子（无依赖者，即终端概念）
        leaves = [n for n in nodes if len(self._reverse.get(n, set())) == 0]

        return {
            "total_nodes": len(nodes),
            "total_edges": sum(len(v) for v in self._adj.values()),
            "roots": len(roots),
            "leaves": len(leaves),
            "root_examples": roots[:10],
            "leaf_examples": leaves[:10],
            "max_depth": max(in_degrees) if in_degrees else 0,
            "avg_prereqs": sum(in_degrees) / len(nodes) if nodes else 0,
            "avg_dependents": sum(out_degrees) / len(nodes) if nodes else 0,
        }


# ── CLI ──
if __name__ == "__main__":
    import sys

    dag = LearningDAG()
    dag.build()

    stats = dag.stats()
    print("=== DAG 统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== 示例: 前置依赖 ===")
    for concept in ["有机物", "原子", "细胞", "实数", "恒星"]:
        prereqs = dag.get_prerequisites(concept, max_depth=2)
        for depth, items in prereqs.items():
            print(f"  {concept} L{depth}: {items[:5]}")

    print("\n=== 示例: 学习路径 ===")
    for frm, to in [("原子", "物质"), ("电子", "原子"), ("实数", "复数")]:
        path = dag.learning_path(frm, to)
        print(f"  {frm} → {to}: {' → '.join(path) if path else '不可达'}")

    print("\n=== 拓扑排序 (前30个) ===")
    order = dag.topological_order()
    print(f"  {order[:30]}")

    if len(sys.argv) > 1:
        concept = sys.argv[1]
        print(f"\n=== 学习 [{concept}] 的完整前置链 ===")
        prereqs = dag.get_prerequisites(concept, max_depth=5)
        for depth, items in prereqs.items():
            if items:
                print(f"  L{depth}: {', '.join(items[:10])}")
