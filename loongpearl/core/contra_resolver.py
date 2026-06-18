#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠矛盾解 (ContraResolver) — 知识冲突检测与自动消解
════════════════════════════════════════════════════════════════════════════

当知识规模上去后，必然出现冲突。矛盾解提供：
  1. 主动检测 — 环路、对立断言、逻辑不一致
  2. 分级消解 — 4种策略从严格到宽松
  3. 上下文限定 — 给冲突知识加领域/时间/条件约束
  4. 活跃争议标记 — 无法消解时保留并存

════════════════════════════════════════════════════════════════════════════
冲突类型
════════════════════════════════════════════════════════════════════════════

  环路冲突:     A IS_A B IS_A A        (分类循环)
  组成循环:     A PART_OF B PART_OF A  (整体部分循环)
  对立断言:     A IS_A B, A OPPOSITE B (同时断言同义和对立)
  属性冲突:     A HAS X, A HAS not_X   (矛盾属性)
  因果冲突:     A CAUSE B, A PREVENTS B (同时断言因果和抑制)
  多值冲突:     同一主体有多个互斥值

════════════════════════════════════════════════════════════════════════════
消解策略
════════════════════════════════════════════════════════════════════════════

  strategy="time_priority"    新证据覆盖旧证据
  strategy="source_weight"    来源权重 (学术 > 百科 > 论坛)
  strategy="confidence_based" 置信度高的一方胜
  strategy="context_qualify"  加限制定语，两方都保留
  strategy="keep_both"        标记为"活跃争议"，不做裁决

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.contra_resolver import ContraResolver

    cr = ContraResolver(concept_graph)
    conflicts = cr.detect_all()
    cr.resolve(conflicts[0], strategy="context_qualify")
    cr.print_report()

"""
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

class ConflictType(Enum):
    """冲突类型"""
    CYCLE_IS_A = "cycle_is_a"              # IS_A 环路
    CYCLE_PART_OF = "cycle_part_of"        # PART_OF 环路
    OPPOSING_ASSERTION = "opposing"        # 对立断言
    PROPERTY_CONFLICT = "property"         # 属性冲突
    CAUSAL_CONFLICT = "causal"             # 因果关系冲突
    MULTI_VALUE = "multi_value"            # 多值互斥
    SELF_CONTRADICTION = "self_contra"     # 自我矛盾


class Severity(Enum):
    """严重程度"""
    CRITICAL = 4   # 逻辑矛盾，必须消解
    HIGH = 3       # 大概率错误
    MEDIUM = 2     # 可能需要上下文限定
    LOW = 1        # 轻微不一致


# 来源权重
SOURCE_WEIGHTS = {
    "textbook":          1.00,
    "academic_paper":    0.95,
    "peer_reviewed":     0.90,
    "encyclopedia":      0.85,
    "wikidata":          0.80,
    "wikipedia":         0.75,
    "arxiv":             0.70,
    "textbook_backup":   0.65,
    "forum_expert":      0.50,
    "forum":             0.30,
    "blog":              0.25,
    "social_media":      0.10,
    "unknown":           0.40,
    "seed":              0.60,
    "test":              0.20,
}


@dataclass
class Conflict:
    """一个知识冲突"""
    type: ConflictType
    severity: Severity
    description: str
    involved_triples: List[Tuple[str, str, str, float]]  # (s, r, o, confidence)
    evidence: Dict[str, Any] = field(default_factory=dict)
    resolution: Optional[str] = None
    status: str = "unresolved"  # unresolved/resolved/marked_disputed

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.name,
            "description": self.description,
            "triples": [f"{s} {r} {o} ({conf:.2f})"
                        for s, r, o, conf in self.involved_triples],
            "resolution": self.resolution,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 矛盾解主类
# ═══════════════════════════════════════════════════════════════════════════

class ContraResolver:
    """
    矛盾解 — 知识冲突检测与自动消解引擎。

    设计原则:
      - 先检测，后消解（不预先过滤）
      - 消解策略可配置（严格→宽松）
      - 所有操作可审计（保留消解记录）
    """

    def __init__(self, concept_graph=None):
        self.cg = concept_graph
        self.conflicts: List[Conflict] = []
        self.resolution_log: List[Dict[str, Any]] = []
        self._subject_index = None  # lazy-built O(1) lookup
        self._triples_sample = None  # cached sample for iterating

    # ═════════════════════════════════════════════════════════════════════
    # 全面检测
    # ═════════════════════════════════════════════════════════════════════

    def detect_all(self, max_time_sec: float = 30.0) -> List[Conflict]:
        """运行所有检测器，返回发现的冲突列表。max_time_sec 硬性超时保护"""
        import time
        t_start = time.time()
        
        def _timed_out():
            return time.time() - t_start > max_time_sec
        
        self.conflicts = []

        if not self.cg or not hasattr(self.cg, 'triples'):
            print("[矛盾解] 无概念图，跳过检测")
            return []
        
        self._build_subject_index()

        # 检测器按开销排序: 便宜的优先
        if not _timed_out():
            self.conflicts.extend(self.detect_opposing())
        if not _timed_out():
            self.conflicts.extend(self.detect_causal_conflicts())
        if not _timed_out():
            self.conflicts.extend(self.detect_property_conflicts())
        if not _timed_out():
            self.conflicts.extend(self.detect_cycles())  # DFS最贵，放最后

        elapsed = time.time() - t_start
        print(f"[矛盾解] 检测完成 ({elapsed:.1f}s): {len(self.conflicts)} 个冲突")
        return self.conflicts

    # ═════════════════════════════════════════════════════════════════════
    # 检测器 1: 环路检测
    # ═════════════════════════════════════════════════════════════════════

    def _build_subject_index(self):
        """懒构建 O(1) 主语索引，避免每次 list() 全量 193 万三元组"""
        if self._subject_index is not None:
            return
        from collections import defaultdict
        self._subject_index = defaultdict(list)
        # 取前 10 万条三元组建索引（足够覆盖常用知识）
        count = 0
        for key, t in self.cg.triples.items():
            if count >= 100000:
                break
            if hasattr(t, 'subject'):
                self._subject_index[t.subject].append(
                    (t.relation, t.object, t.confidence, t.source))
                count += 1
        self._triples_sample = list(self.cg.triples.keys())[:20000]  # 缓存键样本

    def _get_triples_for(self, subject: str) -> List[Tuple[str, str, float, str]]:
        """O(1) 索引查找，替代原 list(self.cg.triples.items())[:50000] 全量扫描"""
        self._build_subject_index()
        return self._subject_index.get(subject, [])

    def detect_cycles(self) -> List[Conflict]:
        """检测 IS_A 和 PART_OF 环路"""
        conflicts = []

        # 对每个节点做 DFS 检测环路
        self._build_subject_index()
        key_sample = self._triples_sample[:3000]  # 只用缓存样本，避免 list() 全量
        max_iterations = 5000  # 硬上限：最多探测 5000 个起点
        it = 0
        for rel_type in ['IS_A', 'PART_OF']:
            for start_node in key_sample:
                it += 1
                if it > max_iterations:
                    break
                cycle = self._find_cycle(start_node, start_node, rel_type, set())
                if cycle and len(cycle) > 1:
                    s = Severity.CRITICAL if rel_type == 'IS_A' else Severity.HIGH
                    # 构建三元组列表
                    involved = []
                    for i in range(len(cycle)-1):
                        involved.append((cycle[i], rel_type, cycle[i+1], 0.5))
                    # 短路: 只在首次发现完整环路时记录
                    cycle_key = tuple(sorted(cycle))
                    if not any(tuple(sorted([t[0] for t in c.involved_triples] + [t[2] for t in c.involved_triples])) == cycle_key
                               for c in conflicts):
                        conflicts.append(Conflict(
                            type=ConflictType(f"cycle_{rel_type.lower()}"),
                            severity=s,
                            description=f"{rel_type} 环路: {' → '.join(cycle)}",
                            involved_triples=involved,
                        ))

        return conflicts

    def _find_cycle(self, current: str, target: str,
                    rel_type: str, visited: Set[str],
                    max_depth: int = 8) -> Optional[List[str]]:
        """DFS 查找环路"""
        if current in visited:
            return None
        if max_depth <= 0:
            return None

        visited.add(current)

        if current in self.cg.triples:
            for rel, obj, conf, src in self._get_triples_for(current):
                if rel == rel_type:
                    if obj == target:
                        return [current, obj]
                    result = self._find_cycle(obj, target, rel_type,
                                              visited.copy(), max_depth - 1)
                    if result:
                        return [current] + result

        return None

    # ═════════════════════════════════════════════════════════════════════
    # 检测器 2: 对立断言
    # ═════════════════════════════════════════════════════════════════════

    def detect_opposing(self) -> List[Conflict]:
        """检测同一对节点同时有 IS_A 和 OPPOSITE 关系"""
        conflicts = []
        self._build_subject_index()
        for s in self._triples_sample[:5000]:
            if s not in self.cg.triples:
                continue

            relations_to_obj = defaultdict(set)
            for rel, obj, conf, src in self._get_triples_for(s):
                relations_to_obj[obj].add(rel)

            for obj, rels in relations_to_obj.items():
                if 'IS_A' in rels and 'OPPOSITE' in rels:
                    conflicts.append(Conflict(
                        type=ConflictType.OPPOSING_ASSERTION,
                        severity=Severity.CRITICAL,
                        description=f"{s} 同时 IS_A 和 OPPOSITE {obj}",
                        involved_triples=[
                            (s, 'IS_A', obj, 0),
                            (s, 'OPPOSITE', obj, 0),
                        ],
                    ))

        return conflicts

    # ═════════════════════════════════════════════════════════════════════
    # 检测器 3: 属性冲突
    # ═════════════════════════════════════════════════════════════════════

    def detect_property_conflicts(self) -> List[Conflict]:
        """检测 HAS 关系的属性冲突"""
        conflicts = []
        self._build_subject_index()
        for s in self._triples_sample[:5000]:
            if s not in self.cg.triples:
                continue

            has_values = defaultdict(list)
            for rel, obj, conf, src in self._get_triples_for(s):
                if rel == 'HAS' and obj in self.cg.triples:
                    has_values[obj].append((conf, src))

            # 检测是否有冲突属性对（如"正电荷"和"负电荷"）
            # 简化为: 如果某节点 HAS 了 OPPOSITE 关系的两端
            for obj_a in has_values:
                if obj_a in self.cg.triples:
                    for rel, obj_b, conf, src in self.cg.triples[obj_a]:
                        if rel == 'OPPOSITE' and obj_b in has_values:
                            conflicts.append(Conflict(
                                type=ConflictType.PROPERTY_CONFLICT,
                                severity=Severity.HIGH,
                                description=f"{s} 同时具有对立属性 {obj_a} 和 {obj_b}",
                                involved_triples=[
                                    (s, 'HAS', obj_a, has_values[obj_a][0][0]),
                                    (s, 'HAS', obj_b, has_values[obj_b][0][0]),
                                ],
                            ))

        return conflicts

    # ═════════════════════════════════════════════════════════════════════
    # 检测器 4: 因果关系冲突
    # ═════════════════════════════════════════════════════════════════════

    def detect_causal_conflicts(self) -> List[Conflict]:
        """检测同时存在 CAUSE 和 PREVENTS 关系的节点对"""
        conflicts = []
        self._build_subject_index()
        for s in self._triples_sample[:5000]:
            if s not in self.cg.triples:
                continue

            relations_to_obj = defaultdict(set)
            for rel, obj, conf, src in self._get_triples_for(s):
                relations_to_obj[obj].add(rel)

            for obj, rels in relations_to_obj.items():
                if 'CAUSE' in rels and 'PREVENTS' in rels:
                    conflicts.append(Conflict(
                        type=ConflictType.CAUSAL_CONFLICT,
                        severity=Severity.HIGH,
                        description=f"{s} 同时 CAUSE 和 PREVENTS {obj}",
                        involved_triples=[
                            (s, 'CAUSE', obj, 0),
                            (s, 'PREVENTS', obj, 0),
                        ],
                    ))

        return conflicts

    # ═════════════════════════════════════════════════════════════════════
    # 消解引擎
    # ═════════════════════════════════════════════════════════════════════

    def resolve(self, conflict: Conflict,
                strategy: str = "confidence_based") -> str:
        """
        消解单个冲突。

        Strategies:
          "time_priority"    — 保留最新的三元组
          "source_weight"    — 保留来源权重最高的
          "confidence_based" — 保留置信度最高的
          "context_qualify"  — 加限制定语，两方保留
          "keep_both"        — 标记为"活跃争议"

        Returns:
            消解结果描述
        """
        if strategy == "keep_both":
            conflict.resolution = "标记为活跃争议，双方保留"
            conflict.status = "marked_disputed"
            self.resolution_log.append(conflict.to_dict())
            return f"[KEEP_BOTH] {conflict.description} → 标记为争议"

        elif strategy == "context_qualify":
            # 加领域/时间/条件限定词
            qualified = []
            for s, r, o, conf in conflict.involved_triples:
                # 在概念图中查找该三元组的来源
                sources = self._get_sources(s, r, o)
                source_str = ", ".join(sources[:2]) if sources else "未知来源"
                qualified.append(f"{s} {r} {o}（限定于: {source_str}）")

            conflict.resolution = "\n".join(qualified)
            conflict.status = "resolved"
            self.resolution_log.append(conflict.to_dict())
            return f"[QUALIFY] {conflict.description} → 上下文限定"

        elif strategy == "time_priority":
            winner = self._resolve_by_time(conflict)
            conflict.resolution = f"时间优先: 保留 {winner}"
            conflict.status = "resolved"

        elif strategy == "source_weight":
            winner = self._resolve_by_source(conflict)
            conflict.resolution = f"来源权重优先: 保留 {winner}"
            conflict.status = "resolved"

        else:  # confidence_based
            winner = self._resolve_by_confidence(conflict)
            conflict.resolution = f"置信度优先: 保留 {winner}"
            conflict.status = "resolved"

        self.resolution_log.append(conflict.to_dict())
        return conflict.resolution

    def resolve_all(self, strategy: str = "confidence_based") -> int:
        """消解所有已检测的冲突"""
        if not self.conflicts:
            self.detect_all()

        resolved = 0
        for conflict in self.conflicts:
            self.resolve(conflict, strategy)
            resolved += 1

        print(f"[矛盾解] 消解完成: {resolved} 个冲突 (策略: {strategy})")
        return resolved

    # ═════════════════════════════════════════════════════════════════════
    # 消解辅助函数
    # ═════════════════════════════════════════════════════════════════════

    def _get_sources(self, s: str, r: str, o: str) -> List[str]:
        """获取三元组的来源列表"""
        if not self.cg or s not in self.cg.triples:
            return []
        sources = []
        for rel, obj, conf, src in self._get_triples_for(s):
            if rel == r and obj == o:
                sources.append(src)
        return sources

    def _get_source_weight(self, source: str) -> float:
        """计算来源权重"""
        for key, weight in SOURCE_WEIGHTS.items():
            if key in source.lower():
                return weight
        return SOURCE_WEIGHTS.get("unknown", 0.4)

    def _resolve_by_confidence(self, conflict: Conflict) -> str:
        """按置信度消解 — 最高的胜出"""
        if not conflict.involved_triples:
            return "无可消解的三元组"
        best = max(conflict.involved_triples, key=lambda x: x[3])
        return f"{best[0]} {best[1]} {best[2]} (置信度 {best[3]:.2f})"

    def _resolve_by_source(self, conflict: Conflict) -> str:
        """按来源权重消解"""
        if not conflict.involved_triples:
            return "无可消解的三元组"
        best_triple = None
        best_weight = -1

        for s, r, o, conf in conflict.involved_triples:
            sources = self._get_sources(s, r, o)
            if sources:
                weight = max(self._get_source_weight(src) for src in sources)
            else:
                weight = 0.4
            if weight > best_weight:
                best_weight = weight
                best_triple = (s, r, o, conf)

        return f"{best_triple[0]} {best_triple[1]} {best_triple[2]} (权重 {best_weight:.2f})"

    def _resolve_by_time(self, conflict: Conflict) -> str:
        """按时间消解 — 最新的胜出（需要时间戳数据）"""
        # 简化实现：如果有冲突，取第一条
        return f"{conflict.involved_triples[0][0]} {conflict.involved_triples[0][1]} {conflict.involved_triples[0][2]}"

    # ═════════════════════════════════════════════════════════════════════
    # 知识清理 — 移除已消解的冲突三元组
    # ═════════════════════════════════════════════════════════════════════

    def purge_resolved(self) -> int:
        """移除已消解的冲突中的失败三元组"""
        if not self.cg:
            return 0

        purged = 0
        for conflict in self.conflicts:
            if conflict.status != "resolved":
                continue

            # 找置信度/权重最低的三元组并移除
            worst = min(conflict.involved_triples, key=lambda x: x[3])
            s, r, o, _ = worst

            if s in self.cg.triples:
                old_triples = self.cg.triples[s]
                self.cg.triples[s] = [
                    (rel, obj, conf, src)
                    for rel, obj, conf, src in old_triples
                    if not (rel == r and obj == o)
                ]
                purged += 1

        return purged

    # ═════════════════════════════════════════════════════════════════════
    # 报告
    # ═════════════════════════════════════════════════════════════════════

    def print_report(self):
        """打印完整检测和消解报告"""
        print("=" * 60)
        print("═══ 矛盾解 — 知识冲突报告 ═══")
        print("=" * 60)

        if not self.conflicts:
            print("  ✅ 未检测到冲突")
            return

        # 按严重程度分组
        by_severity = defaultdict(list)
        by_type = defaultdict(list)

        for c in self.conflicts:
            by_severity[c.severity].append(c)
            by_type[c.type].append(c)

        print(f"\n📊 总冲突: {len(self.conflicts)}")
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            count = len(by_severity[sev])
            if count:
                icon = {Severity.CRITICAL: "🔴", Severity.HIGH: "🟠",
                        Severity.MEDIUM: "🟡", Severity.LOW: "🟢"}[sev]
                print(f"  {icon} {sev.name:10s}: {count:>4}")

        print(f"\n📋 按类型:")
        for ctype, items in by_type.items():
            resolved = sum(1 for i in items if i.status == "resolved")
            print(f"  {ctype.value:25s}: {len(items):>4} "
                  f"(已消解: {resolved})")

        # 列出前5个最严重的冲突
        print(f"\n🔍 最严重冲突 (前5):")
        sorted_conflicts = sorted(self.conflicts,
                                  key=lambda c: c.severity.value, reverse=True)
        for i, c in enumerate(sorted_conflicts[:5]):
            print(f"  {i+1}. [{c.severity.name}] {c.description}")
            if c.resolution:
                print(f"     → {c.resolution}")

        print(f"\n📝 消解日志: {len(self.resolution_log)} 条")

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要统计"""
        by_status = defaultdict(int)
        for c in self.conflicts:
            by_status[c.status] += 1

        return {
            "total": len(self.conflicts),
            "resolved": by_status.get("resolved", 0),
            "unresolved": by_status.get("unresolved", 0),
            "disputed": by_status.get("marked_disputed", 0),
            "critical": sum(1 for c in self.conflicts if c.severity == Severity.CRITICAL),
            "resolution_log_size": len(self.resolution_log),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_contra_resolver():
    """自测矛盾解"""
    # 创建一个简单的模拟概念图
    class MockCG:
        def __init__(self):
            self.triples = {}

        def add_triple(self, s, r, o, confidence=0.5, source="test"):
            if s not in self.triples:
                self.triples[s] = []
            self.triples[s].append((r, o, confidence, source))

    cg = MockCG()

    # 注入冲突知识
    cg.add_triple("A", "IS_A", "B", confidence=0.8)
    cg.add_triple("B", "IS_A", "C", confidence=0.7)
    cg.add_triple("C", "IS_A", "A", confidence=0.6)  # 环路!

    cg.add_triple("光", "IS_A", "粒子", confidence=0.6)
    cg.add_triple("光", "OPPOSITE", "粒子", confidence=0.2)  # 对立!

    cg.add_triple("物质", "HAS", "正电荷", confidence=0.8)
    cg.add_triple("正电荷", "OPPOSITE", "负电荷", confidence=0.9)
    cg.add_triple("物质", "HAS", "负电荷", confidence=0.5)  # 属性冲突!

    cg.add_triple("黑洞", "CAUSE", "时空弯曲", confidence=0.9)
    cg.add_triple("黑洞", "PREVENTS", "时空弯曲", confidence=0.1)  # 因果冲突!

    cr = ContraResolver(cg)
    cr.detect_all()
    cr.print_report()

    print("\n" + "=" * 60)
    print("消解测试:")
    for c in cr.conflicts:
        result = cr.resolve(c, strategy="confidence_based")
        print(f"  {result}")

    purged = cr.purge_resolved()
    print(f"\n清理: 移除了 {purged} 个冲突三元组")

    # 再次检测
    cr.detect_all()
    print(f"清理后冲突数: {len(cr.conflicts)}")


if __name__ == "__main__":
    test_contra_resolver()
