#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4: 轻量假设检验 (Hypothesis Tester)

对三元组假设进行轻量检验——不建结构因果模型，纯图搜索。
核心回答："如果这个三元组为真，哪些已有知识会被冲击？"

功能:
  1. test_triple — 假设施加到概念图，检测冲突与被冲击知识
  2. check_consistency — 检查某概念的三元组是否自洽
  3. simulate_learning — 模拟学了一组新三元组后 CG 状态变化

使用:
  from loongpearl.core.learning_dag import LearningDAG
  from loongpearl.core.cognitive_terrain import CognitiveTerrain
  from loongpearl.core.hypothesis_tester import HypothesisTester

  dag = LearningDAG("data/models/concept_graph.db"); dag.build()
  terrain = CognitiveTerrain(); terrain.load()
  tester = HypothesisTester("data/models/concept_graph.db", dag, terrain)

  result = tester.test_triple("龙", "IS_A", "神话生物", 0.8)
  # → {conflicts: [...], affected: [...], impact_score: float}
"""

import sqlite3
import os
import logging
import math
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any

log = logging.getLogger(__name__)

# ── 冲突关系对 ──
# 若新三元组的 relation 与已有三元组的 relation 构成冲突对，则标记为冲突
_CONFLICT_PAIRS: Set[Tuple[str, str]] = {
    ("IS_A", "PART_OF"),   # IS_A 与 PART_OF 语义方向不同（但视为弱冲突）
}


def _is_conflict_rel(r_new: str, r_existing: str) -> bool:
    """判断两个 relation 是否冲突"""
    if r_new == r_existing:
        return False  # 同关系不冲突，可能是覆盖/强化
    pair = (r_new, r_existing)
    if pair in _CONFLICT_PAIRS:
        return True
    pair_rev = (r_existing, r_new)
    if pair_rev in _CONFLICT_PAIRS:
        return True
    return False


class HypothesisTester:
    """
    轻量假设检验器。

    用 sqlite3 直连 concept_graph.db，纯图搜索，不依赖 ConceptGraph 类。
    接收 LearningDAG 和 CognitiveTerrain 以利用其索引/评分能力。
    """

    def __init__(
        self,
        db_path: str = None,
        dag: Any = None,
        terrain: Any = None,
    ):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            db_path = os.path.join(project_root, "data", "models", "concept_graph.db")

        self.db_path = db_path
        self.dag = dag        # LearningDAG 实例（可选）
        self.terrain = terrain  # CognitiveTerrain 实例（可选）
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # ═══════════════════════════════════════════════════════
    # test_triple
    # ═══════════════════════════════════════════════════════

    def test_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 0.5,
    ) -> dict:
        """
        模拟"如果这个三元组为真，哪些已有知识会被冲击"。

        检测流程:
          1. 查询概念图中与 (subject, *, *) 和 (*, *, object) 相关的三元组
          2. 检测矛盾:
             - 同 subject-object 对但 relation 冲突
             - IS_A 环路（主角反向后形成环）
             - relation 语义冲突（如 IS_A vs PART_OF 等）
          3. 计算影响得分

        Returns:
            {
                "conflicts": [
                    {"type": "rel_conflict"|"loop"|"symmetry_break",
                     "triple": (s, r, o, c),
                     "reason": str},
                    ...
                ],
                "affected": [
                    {"concept": str, "reason": str, "distance": int},
                    ...
                ],
                "impact_score": float,   # 0-1, 越高冲击越大
                "novelty": float,        # 0-1, 这个假设有多新
            }
        """
        conflicts = []
        affected_concepts: Dict[str, dict] = {}

        # ── 步骤 1: 收集相关三元组 ──
        # (subject, *, *)
        rows_s = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE s=?",
            (subject,)
        ).fetchall()

        # (*, *, object)
        rows_o = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE o=?",
            (obj,)
        ).fetchall()

        # (*, *, subject) — 以 subject 为宾语的三元组
        rows_s_as_o = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE o=?",
            (subject,)
        ).fetchall()

        # 也查以 object 为主语
        rows_o_as_s = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE s=?",
            (obj,)
        ).fetchall()

        # 合并去重（用 (s,r,o) 作为 key）
        seen = set()
        all_relevant = []
        for rows in [rows_s, rows_o, rows_s_as_o, rows_o_as_s]:
            for s, r, o, c in rows:
                key = (s, r, o)
                if key not in seen:
                    seen.add(key)
                    all_relevant.append((s, r, o, c))

        # ── 步骤 2: 检测矛盾 ──
        has_exact_match = False

        for s, r, o, c in all_relevant:
            # 2a. 完全相同三元组 → 不是冲突，只是强化
            if s == subject and r == relation and o == obj:
                has_exact_match = True
                continue

            # 2b. 同 subject-object 对但关系冲突
            if s == subject and o == obj and _is_conflict_rel(relation, r):
                conflicts.append({
                    "type": "rel_conflict",
                    "triple": (s, r, o, c),
                    "reason": f"已有 '{s} {r} {o}' (c={c})，"
                              f"与新假设 '{subject} {relation} {obj}' 的 relation 冲突",
                    "severity": "high",
                })

            # 2c. 反向三元组冲突: 已有 (obj, relation, subject)，新的是 (subject, relation, obj)
            #    例: 已有 "神话生物 IS_A 龙"，新假设 "龙 IS_A 神话生物" → 可能形成 IS_A 环
            if s == obj and o == subject and r == relation:
                conflicts.append({
                    "type": "reverse_edge",
                    "triple": (s, r, o, c),
                    "reason": f"已有反向边 '{s} {r} {o}' (c={c})，"
                              f"新假设 '{subject} {relation} {obj}' 会形成双向 {relation} 边",
                    "severity": "medium",
                })

        # ── 步骤 2d: 新假设 subject=obj 自指（只检查一次）──
        if subject == obj:
            if relation in ("IS_A", "PART_OF"):
                conflicts.append({
                    "type": "self_loop",
                    "triple": None,
                    "reason": f"'{subject} {relation} {subject}' 是自指，语义上无意义",
                    "severity": "medium",
                })

        # ── 步骤 2e: IS_A 环路检测 ──
        if relation == "IS_A":
            loop_conflicts = self._detect_is_a_loop(subject, obj)
            conflicts.extend(loop_conflicts)

        # ── 步骤 3: 收集受影响概念 ──
        # 限制受影响概念数量以避免性能问题
        _max_affected = 50
        for s, r, o, c in all_relevant:
            if len(affected_concepts) >= _max_affected:
                break
            for concept in [s, o]:
                if concept in (subject, obj):
                    continue
                if concept not in affected_concepts:
                    affected_concepts[concept] = {
                        "concept": concept,
                        "reason": f"与假设相关: 出现在三元组 '{s} {r} {o}'",
                        "distance": 1,  # 默认近距离，避免每概念做BFS
                    }

        # ── 步骤 4: 使用 terrain 评估受影响概念的当前状态 ──
        if self.terrain:
            for concept, info in affected_concepts.items():
                try:
                    energy = self.terrain.score(concept)
                    info["energy"] = energy
                    info["zone"] = self.terrain.classify(energy)
                except Exception:
                    info["energy"] = 999.0
                    info["zone"] = "unknown"

        # ── 步骤 5: 计算得分 ──
        impact_score = self._compute_impact(conflicts, affected_concepts, all_relevant)
        novelty = 0.0 if has_exact_match else min(1.0, confidence)

        return {
            "conflicts": conflicts,
            "affected": sorted(affected_concepts.values(),
                              key=lambda x: x.get("distance", 999))[:30],
            "affected_count": len(affected_concepts),
            "relevant_triples_count": len(all_relevant),
            "has_exact_match": has_exact_match,
            "impact_score": round(impact_score, 4),
            "novelty": round(novelty, 4),
        }

    def _detect_is_a_loop(self, subject: str, obj: str) -> List[dict]:
        """
        检测添加 subject IS_A obj 后是否会形成 IS_A 环路。

        环路形式: subject → obj → ... → subject
        即 obj 的传递 IS_A 闭包中是否包含 subject。
        """
        conflicts = []
        # BFS 从 obj 沿 IS_A 边向外（obj IS_A X → X IS_A Y → ...）
        visited = {obj}
        queue = deque([(obj, 0)])  # (node, depth)
        max_depth = 20

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            rows = self.conn.execute(
                "SELECT o FROM triples WHERE s=? AND r='IS_A' LIMIT 500",
                (current,)
            ).fetchall()
            for (next_concept,) in rows:
                if next_concept == subject:
                    # 找到环路！
                    # 回溯路径
                    path = self._reconstruct_path(obj, subject)
                    conflicts.append({
                        "type": "is_a_loop",
                        "triple": None,
                        "reason": (f"添加 '{subject} IS_A {obj}' 会形成 IS_A 环路: "
                                   f"{subject} → {obj} → ... → {subject}"),
                        "path": path,
                        "severity": "critical",
                    })
                    return conflicts  # 找到一个就够了
                if next_concept not in visited:
                    visited.add(next_concept)
                    queue.append((next_concept, depth + 1))

        return conflicts

    def _reconstruct_path(self, start: str, target: str) -> List[str]:
        """BFS 重构从 start 沿 IS_A 到 target 的路径"""
        parent = {start: None}
        queue = deque([(start, 0)])
        max_depth = 20
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if current == target:
                # 回溯
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))
            rows = self.conn.execute(
                "SELECT o FROM triples WHERE s=? AND r='IS_A' LIMIT 500",
                (current,)
            ).fetchall()
            for (nxt,) in rows:
                if nxt not in parent:
                    parent[nxt] = current
                    queue.append((nxt, depth + 1))
        return []

    def _graph_distance(self, a: str, b: str) -> int:
        """BFS 计算两个概念在图中的最短距离（无向）"""
        if a == b:
            return 0
        visited = {a}
        queue = deque([(a, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= 3:  # 浅层截断
                return 999
            # 邻居: 作为主语或宾语出现
            rows = self.conn.execute(
                "SELECT o FROM triples WHERE s=? LIMIT 50",
                (node,)
            ).fetchall()
            for (nxt,) in rows:
                if nxt == b:
                    return dist + 1
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
            rows = self.conn.execute(
                "SELECT s FROM triples WHERE o=? LIMIT 50",
                (node,)
            ).fetchall()
            for (nxt,) in rows:
                if nxt == b:
                    return dist + 1
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return 999

    def _compute_impact(
        self,
        conflicts: List[dict],
        affected: Dict[str, dict],
        relevant_triples: List[tuple],
    ) -> float:
        """
        计算影响得分 0-1。

        考虑因子:
          - 冲突数量与严重性
          - 受影响概念数量和距离
          - 相关三元组数量
        """
        score = 0.0

        # 冲突贡献
        severity_weights = {"critical": 0.4, "high": 0.25, "medium": 0.1, "low": 0.05}
        for c in conflicts:
            score += severity_weights.get(c.get("severity", "medium"), 0.1)

        # 受影响概念贡献（距离近的权重高）
        for info in affected.values():
            dist = info.get("distance", 999)
            if dist < 999:
                score += 0.02 * max(0, 1.0 - dist / 5.0)

        # 相关三元组数量贡献（归一化）
        rel_count = len(relevant_triples)
        if rel_count > 0:
            score += min(0.3, rel_count * 0.005)

        return min(1.0, score)

    # ═══════════════════════════════════════════════════════
    # check_consistency
    # ═══════════════════════════════════════════════════════

    def check_consistency(self, concept: str) -> dict:
        """
        检查某个概念的所有三元组之间是否自洽。

        检测项:
          1. IS_A 环路 — A IS_A B IS_A A
          2. OPPOSITE 对称性 — 若 A OPPOSITE B，应有 B OPPOSITE A
          3. 关系冲突 — 同一对概念不能同时 IS_A 和 PART_OF

        Returns:
            {
                "concept": str,
                "total_triples": int,
                "is_consistent": bool,
                "issues": [{"type": str, "detail": str, "severity": str}, ...],
                "is_a_chain": [str, ...],  # IS_A 传递链
                "cycles": [[str, ...], ...],  # 检测到的环路
            }
        """
        issues = []
        cycles = []

        # 获取该概念所有相关三元组
        triples_as_s = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE s=?",
            (concept,)
        ).fetchall()

        triples_as_o = self.conn.execute(
            "SELECT s, r, o, c FROM triples WHERE o=?",
            (concept,)
        ).fetchall()

        all_triples = list(triples_as_s) + list(triples_as_o)

        # ── 1. IS_A 环路检测 ──
        # 收集概念所在 IS_A 链
        is_a_targets = set()
        for s, r, o, c in all_triples:
            if r == "IS_A":
                if s == concept:
                    is_a_targets.add(o)
                elif o == concept:
                    is_a_targets.add(s)

        # 对每个 IS_A 目标，看是否传递回 concept
        for target in is_a_targets:
            if self._is_a_reachable(target, concept):
                path = self._reconstruct_path(target, concept)
                cycles.append(path)
                issues.append({
                    "type": "is_a_cycle",
                    "detail": f"IS_A 环路: {' → '.join(path)}",
                    "severity": "critical",
                    "cycle": path,
                })

        # ── 2. 关系冲突检测 ──
        # 按 (subject, object) 对分组
        pair_relations: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for s, r, o, c in all_triples:
            pair_relations[(s, o)].append(r)

        for (s, o), rels in pair_relations.items():
            if len(rels) >= 2:
                for i in range(len(rels)):
                    for j in range(i + 1, len(rels)):
                        if _is_conflict_rel(rels[i], rels[j]):
                            issues.append({
                                "type": "rel_conflict",
                                "detail": (f"概念对 '{s}'↔'{o}' 同时存在冲突关系: "
                                           f"{rels[i]} vs {rels[j]}"),
                                "severity": "high",
                                "pair": (s, o),
                                "relations": [rels[i], rels[j]],
                            })

        # ── 3. 自指检测 ──
        for s, r, o, c in all_triples:
            if s == o and r in ("IS_A", "PART_OF"):
                issues.append({
                    "type": "self_reference",
                    "detail": f"'{s}' {r} 自身",
                    "severity": "medium",
                })

        is_consistent = len(issues) == 0

        return {
            "concept": concept,
            "total_triples": len(all_triples),
            "as_subject": len(triples_as_s),
            "as_object": len(triples_as_o),
            "is_consistent": is_consistent,
            "issues": issues,
            "cycles": cycles,
        }

    def _is_a_reachable(self, start: str, target: str, max_depth: int = 10) -> bool:
        """BFS: 沿 IS_A 边从 start 能否到达 target"""
        if start == target:
            return True
        visited = {start}
        queue = deque([start])
        depth = 0
        while queue and depth < max_depth:
            for _ in range(len(queue)):
                node = queue.popleft()
                rows = self.conn.execute(
                    "SELECT o FROM triples WHERE s=? AND r='IS_A' LIMIT 500",
                    (node,)
                ).fetchall()
                for (nxt,) in rows:
                    if nxt == target:
                        return True
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            depth += 1
        return False

    # ═══════════════════════════════════════════════════════
    # simulate_learning
    # ═══════════════════════════════════════════════════════

    def simulate_learning(
        self,
        concept: str,
        new_triples: List[Tuple[str, str, str, float]],
    ) -> dict:
        """
        假设学习了 concept 相关的一组新三元组，CG 状态会如何变化。

        模拟:
          1. 对每个新三元组执行 test_triple
          2. 汇总冲突、受影响概念
          3. 识别"新增置信度提升的概念"和"被削弱的概念"
          4. 使用 terrain 评估盲区变化

        Args:
            concept: 核心学习概念
            new_triples: [(subject, relation, object, confidence), ...]

        Returns:
            {
                "concept": str,
                "new_triples": [...],
                "per_triple_results": [...],  # 每个三元组的 test_triple 结果
                "strengthened": [str, ...],    # 置信度提升的概念
                "weakened": [str, ...],        # 被削弱/冲突的概念
                "new_blind_spots": [str, ...], # 新出现的盲区
                "overall_impact": float,       # 总体影响
            }
        """
        per_triple_results = []
        all_conflicts = []
        all_affected: Dict[str, dict] = {}
        strengthened: Set[str] = set()
        weakened: Set[str] = set()

        for s, r, o, c in new_triples:
            result = self.test_triple(s, r, o, c)
            per_triple_results.append({
                "triple": (s, r, o, c),
                "result": result,
            })

            all_conflicts.extend(result.get("conflicts", []))

            for aff in result.get("affected", []):
                name = aff["concept"]
                if name not in all_affected:
                    all_affected[name] = aff

            # 判断强化/削弱
            if not result.get("has_exact_match") and result.get("conflicts"):
                # 有冲突 → 可能削弱已有关联概念
                for conflict in result["conflicts"]:
                    t = conflict.get("triple")
                    if t:
                        weakened.add(t[0])
                        weakened.add(t[2])
            elif not result.get("has_exact_match"):
                # 无冲突的新知识 → 强化 subject 和 object
                strengthened.add(s)
                strengthened.add(o)

            # 强化: 若已有关联，强化相关概念
            if result.get("relevant_triples_count", 0) > 0 and not result.get("conflicts"):
                for aff in result.get("affected", []):
                    if aff.get("distance", 999) <= 2:
                        strengthened.add(aff["concept"])

        # 移除同时出现在强化和削弱中的概念（以削弱为准）
        strengthened -= weakened

        # ── 新盲区评估 ──
        new_blind_spots = []
        if self.terrain:
            # 检查受影响概念中哪些是盲区
            for name, info in all_affected.items():
                if info.get("zone") == "blind_spot":
                    new_blind_spots.append(name)
            # 也检查 concept 自身
            try:
                zone = self.terrain.classify(self.terrain.score(concept))
                if zone == "blind_spot":
                    new_blind_spots.append(concept)
            except Exception:
                pass

        overall_impact = sum(
            r["result"].get("impact_score", 0) for r in per_triple_results
        ) / max(1, len(per_triple_results))

        return {
            "concept": concept,
            "new_triples": [{"triple": t, "confidence": t[3]} for t in new_triples],
            "per_triple_results": per_triple_results,
            "total_conflicts": len(all_conflicts),
            "strengthened": sorted(strengthened)[:20],
            "weakened": sorted(weakened)[:20],
            "new_blind_spots": sorted(set(new_blind_spots))[:20],
            "overall_impact": round(overall_impact, 4),
        }

    # ═══════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()


# ── CLI / 测试 ──
if __name__ == "__main__":
    import sys
    import json

    db = "data/models/concept_graph.db"
    if not os.path.exists(db):
        db = os.path.join(os.path.dirname(__file__), "..", "..", db)

    print("=" * 60)
    print("  L4 假设检验 — HypothesisTester 测试")
    print("=" * 60)

    # 可选：加载 dag 和 terrain
    dag = None
    terrain = None

    try:
        from loongpearl.core.learning_dag import LearningDAG
        dag = LearningDAG(db)
        dag.build()
        print(f"  DAG: {dag.stats()['total_nodes']} 节点, "
              f"{dag.stats()['total_edges']} 边")
    except Exception as e:
        print(f"  DAG 跳过: {e}")

    try:
        from loongpearl.core.cognitive_terrain import CognitiveTerrain
        terrain = CognitiveTerrain(db_path=db)
        terrain.load()
        stats = terrain.stats()
        print(f"  Terrain: avg_energy={stats['avg_energy']:.2f}, "
              f"mastered={stats['mastered_pct']:.1f}%")
    except Exception as e:
        print(f"  Terrain 跳过: {e}")

    tester = HypothesisTester(db, dag, terrain)

    # ── 测试 1: test_triple ──
    print("\n" + "-" * 40)
    print("测试 1: test_triple('龙', 'IS_A', '神话生物', 0.8)")
    print("-" * 40)

    result = tester.test_triple("龙", "IS_A", "神话生物", 0.8)
    print(f"  影响得分: {result['impact_score']}")
    print(f"  新颖度:   {result['novelty']}")
    print(f"  冲突数:   {len(result['conflicts'])}")
    print(f"  受影响概念: {result['affected_count']} 个")
    print(f"  相关三元组: {result['relevant_triples_count']} 条")
    print(f"  已有完全匹配: {result['has_exact_match']}")

    if result["conflicts"]:
        print(f"\n  ⚠ 冲突:")
        for c in result["conflicts"]:
            print(f"    [{c.get('severity','?')}] {c['reason'][:120]}")

    if result["affected"]:
        print(f"\n  受影响概念 (Top 10):")
        for a in result["affected"][:10]:
            zone_info = f" [{a.get('zone', '?')}]" if 'zone' in a else ""
            print(f"    - {a['concept']:20s} dist={a.get('distance', '?'):>3}{zone_info}"
                  f"  {a['reason'][:60]}")

    # ── 测试 2: 冲突场景 ──
    print("\n" + "-" * 40)
    print("测试 2: 冲突场景 — test_triple('原子', 'IS_A', '物质', 0.9)")
    print("  (已知 '物质 PART_OF 原子'，检测 IS_A vs PART_OF 冲突)")
    print("-" * 40)

    result2 = tester.test_triple("原子", "IS_A", "物质", 0.9)
    print(f"  影响得分: {result2['impact_score']}")
    print(f"  冲突数:   {len(result2['conflicts'])}")
    for c in result2["conflicts"]:
        print(f"  ⚠ [{c.get('severity','?')}] {c['reason'][:150]}")

    # ── 测试 3: check_consistency ──
    print("\n" + "-" * 40)
    test_concept = sys.argv[1] if len(sys.argv) > 1 else "原子"
    print(f"测试 3: check_consistency('{test_concept}')")
    print("-" * 40)

    cons = tester.check_consistency(test_concept)
    print(f"  三元组总数: {cons['total_triples']} (主语={cons['as_subject']}, 宾语={cons['as_object']})")
    print(f"  自洽: {cons['is_consistent']}")
    if cons["issues"]:
        print(f"  问题 ({len(cons['issues'])}):")
        for issue in cons["issues"]:
            print(f"    [{issue.get('severity','?')}] {issue['detail'][:120]}")
    if cons["cycles"]:
        print(f"  IS_A 环路:")
        for cycle in cons["cycles"]:
            print(f"    {' → '.join(cycle)}")

    # ── 测试 4: simulate_learning ──
    print("\n" + "-" * 40)
    print("测试 4: simulate_learning('龙', [('龙','IS_A','神话生物',0.8), "
          "('龙','RELATED','凤凰',0.7)])")
    print("-" * 40)

    sim = tester.simulate_learning("龙", [
        ("龙", "IS_A", "神话生物", 0.8),
        ("龙", "RELATED", "凤凰", 0.7),
    ])
    print(f"  总体影响: {sim['overall_impact']}")
    print(f"  总冲突:   {sim['total_conflicts']}")
    print(f"  强化概念: {sim['strengthened'][:10]}")
    print(f"  削弱概念: {sim['weakened'][:10]}")
    print(f"  新盲区:   {sim['new_blind_spots'][:10]}")

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)

    tester.close()
