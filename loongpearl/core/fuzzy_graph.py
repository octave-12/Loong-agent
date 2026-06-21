#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠模糊格 (FuzzyGraph) — 显式概率推理，基于 Dempster-Shafer 证据理论
════════════════════════════════════════════════════════════════════════════

LLM 的概率藏在 softmax 里，不可审计。模糊格用 D-S 证据理论实现显式概率：
  - belief_mass m(A):  精确支持命题 A 的证据量
  - plausibility Pl(A): 不反对命题 A 的证据量
  - [Bel(A), Pl(A)]:    置信区间
  - Dempster 组合规则:   两条独立证据自动融合

优势: 全程可审计——每条结论都有证据链和溯源

════════════════════════════════════════════════════════════════════════════
核心概念
════════════════════════════════════════════════════════════════════════════

  基本概率分配 (BPA):  m: 2^Ω → [0, 1], 其中 ∑m(A) = 1, m(∅) = 0
  信念函数 Bel(A):      Bel(A) = ∑_{B⊆A} m(B)
  似然函数 Pl(A):       Pl(A) = ∑_{B∩A≠∅} m(B)
  Dempster 组合:        m₁⊕m₂(A) = ∑_{B∩C=A} m₁(B)·m₂(C) / (1-K)
                        其中 K = ∑_{B∩C=∅} m₁(B)·m₂(C)

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.fuzzy_graph import FuzzyGraph, BPA

    fg = FuzzyGraph(concept_graph)

    # 为一条三元组添加证据
    fg.add_evidence("电子", "PART_OF", "原子",
        source="量子力学教材", mass=0.85)

    fg.add_evidence("电子", "PART_OF", "原子",
        source="化学教材", mass=0.92)

    # 查询融合后的置信度
    bel = fg.belief("电子", "PART_OF", "原子")
    print(f"Belief: {bel:.2%}")  # D-S 组合后的信念质量
"""
import math
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ═══════════════════════════════════════════════════════════════════════════
# 来源权重表 — 不同来源可信度不同
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_WEIGHTS = {
    "wikipedia_dump": 0.7,
    "concept_graph": 0.5,
    "user_input": 0.6,
    "cedict": 0.4,
    "unihan": 0.4,
    "perturbation": 0.2,
}


def resolve_source_weight(source: str) -> float:
    """根据来源名解析权重。精确匹配 → 前缀匹配 → 默认1.0"""
    if source in SOURCE_WEIGHTS:
        return SOURCE_WEIGHTS[source]
    for prefix, weight in SOURCE_WEIGHTS.items():
        if source.startswith(prefix):
            return weight
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════
# D-S 证据理论核心
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Evidence:
    """一条证据"""
    source: str                   # 来源
    mass: float                   # 基本概率分配 (0~1)
    description: str = ""         # 证据描述
    timestamp: float = 0.0        # 时间戳
    source_weight: float = 1.0    # 来源权重 (0~1)

    def __post_init__(self):
        if self.mass < 0 or self.mass > 1:
            raise ValueError(f"mass 必须在 [0,1] 范围内，得到 {self.mass}")
        if self.source_weight < 0 or self.source_weight > 1:
            raise ValueError(f"source_weight 必须在 [0,1] 范围内，得到 {self.source_weight}")

    @property
    def effective_mass(self) -> float:
        """来源加权后的有效质量"""
        return self.mass * self.source_weight


@dataclass
class BPA:
    """基本概率分配 — 一个命题及其证据组合"""
    proposition: str              # 命题描述
    evidences: List[Evidence] = field(default_factory=list)
    combined_mass: float = 0.0    # D-S 组合后的质量
    conflict_with: List[str] = field(default_factory=list)  # 与此冲突的其他命题

    def combine(self) -> float:
        """
        Dempster 组合规则: 将所有证据融合为一个信念质量。
        m₁⊕m₂(A) = ∑_{B∩C=A} m₁(B)·m₂(C) / (1 - K)
        """
        if not self.evidences:
            return 0.0

        if len(self.evidences) == 1:
            self.combined_mass = self.evidences[0].mass
            return self.combined_mass

        # 迭代组合所有证据
        combined = {self.proposition: self.evidences[0].mass,
                    "Ω": 1 - self.evidences[0].mass}  # Ω = 全集

        for ev in self.evidences[1:]:
            new_combined = {}
            K = 0.0  # 冲突度量

            m2 = {self.proposition: ev.mass, "Ω": 1 - ev.mass}

            for A, m1_A in combined.items():
                for B, m2_B in m2.items():
                    intersection = self._intersect(A, B)
                    if intersection is None:
                        K += m1_A * m2_B
                    else:
                        new_combined[intersection] = new_combined.get(intersection, 0) + m1_A * m2_B

            # 归一化
            if K < 1:
                norm = 1 - K
                for key in new_combined:
                    new_combined[key] /= norm
            else:
                # 完全冲突 → 等权重
                new_combined = combined

            combined = new_combined

        self.combined_mass = combined.get(self.proposition, 0.0)
        return self.combined_mass

    def _intersect(self, A: str, B: str) -> Optional[str]:
        """计算两个命题的交集"""
        if A == B:
            return A
        if A == "Ω":
            return B
        if B == "Ω":
            return A
        return None  # 冲突

    def belief(self) -> float:
        """信任函数 Bel(A)"""
        return self.combined_mass

    def plausibility(self) -> float:
        """似然函数 Pl(A) = 1 - Bel(¬A)"""
        # 简化: 在此模型中，¬A 的支持来自冲突质量
        return min(1.0, self.combined_mass + (1 - self.combined_mass) * 0.3)

    def uncertainty_interval(self) -> Tuple[float, float]:
        """置信区间 [Bel, Pl]"""
        return (self.belief(), self.plausibility())


class DempsterShafer:
    """D-S 证据理论引擎"""

    @staticmethod
    def combine_evidences(evidences: List[Evidence]) -> float:
        """组合多条证据，返回融合后的信念质量"""
        if not evidences:
            return 0.0

        bpa = BPA(proposition="temp", evidences=evidences)
        return bpa.combine()

    @staticmethod
    def conflict(mass1: float, mass2: float) -> float:
        """计算两条证据的冲突量 K"""
        return mass1 * mass2


# ═══════════════════════════════════════════════════════════════════════════
# 模糊格主类
# ═══════════════════════════════════════════════════════════════════════════

class FuzzyGraph:
    """
    模糊格 — 在概念图上增加 D-S 证据理论层，实现显式概率推理。

    设计:
      - 每条三元组可携带多条独立证据
      - D-S 组合自动融合证据 → 单一置信度
      - 支持查询: Bel(命题), Pl(命题), 置信区间
      - 冲突检测: 发现互相矛盾的证据
    """

    def __init__(self, concept_graph=None):
        self.cg = concept_graph
        # 证据库: (subject, relation, object) → BPA
        self._bpas: Dict[Tuple[str, str, str], BPA] = {}
        # 冲突记录
        self._conflicts: List[Dict[str, Any]] = []

    # ═════════════════════════════════════════════════════════════════════
    # 证据管理
    # ═════════════════════════════════════════════════════════════════════

    def add_evidence(self, subject: str, relation: str, obj: str,
                     source: str, mass: float = 0.5,
                     description: str = "",
                     timestamp: float = None,
                     source_weight: float = None) -> BPA:
        """
        为一条三元组添加证据。

        Args:
            subject:       主体
            relation:      关系
            obj:           客体
            source:        证据来源
            mass:          基本概率分配 (0~1)
            description:   证据描述
            source_weight: 来源权重 (0~1)，None 则从 SOURCE_WEIGHTS 自动解析

        Returns:
            更新后的 BPA
        """
        import time
        if timestamp is None:
            timestamp = time.time()

        if source_weight is None:
            source_weight = resolve_source_weight(source)

        ev = Evidence(
            source=source,
            mass=mass,
            description=description,
            timestamp=timestamp,
            source_weight=source_weight,
        )

        key = (subject, relation, obj)
        if key not in self._bpas:
            self._bpas[key] = BPA(proposition=f"{subject} {relation} {obj}")

        self._bpas[key].evidences.append(ev)
        self._bpas[key].combine()  # 重新组合

        # 同步到概念图
        if self.cg:
            self.cg.add_triple(
                subject, relation, obj,
                confidence=self._bpas[key].combined_mass
            )

        return self._bpas[key]

    def add_evidence_batch(self, triples: List[Tuple[str, str, str,
                                                      str, float]]) -> int:
        """批量添加证据。每项: (s, r, o, source, mass)"""
        count = 0
        for s, r, o, source, mass in triples:
            self.add_evidence(s, r, o, source, mass)
            count += 1
        return count

    # ═════════════════════════════════════════════════════════════════════
    # 查询
    # ═════════════════════════════════════════════════════════════════════

    def belief(self, subject: str, relation: str, obj: str) -> float:
        """查询信念质量 Bel(命题)"""
        key = (subject, relation, obj)
        bpa = self._bpas.get(key)
        return bpa.belief() if bpa else 0.0

    def plausibility(self, subject: str, relation: str, obj: str) -> float:
        """查询似然性 Pl(命题)"""
        key = (subject, relation, obj)
        bpa = self._bpas.get(key)
        return bpa.plausibility() if bpa else 0.0

    def uncertainty(self, subject: str, relation: str, obj: str) -> Tuple[float, float]:
        """查询置信区间 [Bel, Pl]"""
        key = (subject, relation, obj)
        bpa = self._bpas.get(key)
        return bpa.uncertainty_interval() if bpa else (0.0, 0.0)

    def get_evidences(self, subject: str, relation: str,
                      obj: str) -> List[Evidence]:
        """获取命题的所有证据"""
        key = (subject, relation, obj)
        bpa = self._bpas.get(key)
        return bpa.evidences if bpa else []

    # ═════════════════════════════════════════════════════════════════════
    # 时间衰减 + 多源加权融合 (L5 升级)
    # ═════════════════════════════════════════════════════════════════════

    def combine_with_decay(self, subject: str, relation: str, obj: str,
                           memory_timeline=None,
                           half_life_days: float = 90.0,
                           decay_factor: float = None) -> float:
        """
        D-S 融合时应用时间衰减。

        对命题的每条证据:
          mass_effective = mass * source_weight * decay_factor
        再用 Dempster 组合规则融合。

        Args:
            subject, relation, obj: 三元组
            memory_timeline:        MemoryTimeline 实例，用于查询时间衰减
            half_life_days:         半衰期(天)，默认90
            decay_factor:           直接指定衰减因子，优先级高于 memory_timeline

        Returns:
            时间衰减后的 D-S 融合信念质量
        """
        evidences = self.get_evidences(subject, relation, obj)
        if not evidences:
            return 0.0

        # 确定衰减因子
        if decay_factor is None:
            if memory_timeline is not None:
                decay_factor = memory_timeline.time_decay_mass(
                    subject, half_life_days=half_life_days
                )
            else:
                decay_factor = 1.0

        # 应用衰减 → 生成临时 Evidence 列表
        decayed = []
        for ev in evidences:
            eff = ev.mass * ev.source_weight * decay_factor
            eff = min(1.0, max(0.0, eff))
            decayed.append(Evidence(
                source=f"{ev.source}[decayed]",
                mass=eff,
                description=ev.description,
                timestamp=ev.timestamp,
                source_weight=ev.source_weight,
            ))

        return DempsterShafer.combine_evidences(decayed)

    def multi_source_fuse(self, subject: str, relation: str, obj: str,
                          memory_timeline=None,
                          half_life_days: float = 90.0,
                          decay_factor: float = None) -> Dict[str, Any]:
        """
        多源加权融合：查询所有来源证据，应用来源权重+时间衰减，D-S 融合。

        这是完整的多源证据融合管道，输出最终置信度及置信区间。

        Args:
            subject, relation, obj: 三元组
            memory_timeline:        MemoryTimeline 实例（可选）
            half_life_days:         半衰期(天)，默认90
            decay_factor:           直接指定衰减因子，优先级高于 memory_timeline

        Returns:
            {
                "proposition": str,
                "belief": float,           # Bel: 信念质量
                "plausibility": float,     # Pl: 似然性
                "interval": [Bel, Pl],     # 置信区间
                "combined_mass": float,    # D-S 组合后的质量
                "evidence_count": int,     # 证据条数
                "sources": [str, ...],     # 来源列表
                "decay_factor": float,     # 时间衰减因子
                "source_weights_used": {source: weight},
            }
        """
        evidences = self.get_evidences(subject, relation, obj)

        if not evidences:
            return {
                "proposition": f"{subject} {relation} {obj}",
                "belief": 0.0,
                "plausibility": 0.0,
                "interval": [0.0, 0.0],
                "combined_mass": 0.0,
                "evidence_count": 0,
                "sources": [],
                "decay_factor": 1.0,
                "source_weights_used": {},
            }

        # 时间衰减因子
        if decay_factor is None:
            if memory_timeline is not None:
                decay_factor = memory_timeline.time_decay_mass(
                    subject, half_life_days=half_life_days
                )
            else:
                decay_factor = 1.0

        # 应用来源权重 + 时间衰减
        decayed = []
        sources_seen = []
        weights_used = {}

        for ev in evidences:
            eff = ev.mass * ev.source_weight * decay_factor
            eff = min(1.0, max(0.0, eff))
            decayed.append(Evidence(
                source=ev.source,
                mass=eff,
                description=ev.description,
                timestamp=ev.timestamp,
                source_weight=ev.source_weight,
            ))
            if ev.source not in sources_seen:
                sources_seen.append(ev.source)
            weights_used[ev.source] = ev.source_weight

        # D-S 融合
        combined = DempsterShafer.combine_evidences(decayed)

        # 计算 Pl
        bpa = BPA(
            proposition=f"{subject} {relation} {obj}",
            evidences=decayed,
            combined_mass=combined,
        )

        return {
            "proposition": bpa.proposition,
            "belief": bpa.belief(),
            "plausibility": bpa.plausibility(),
            "interval": list(bpa.uncertainty_interval()),
            "combined_mass": combined,
            "evidence_count": len(evidences),
            "sources": sources_seen,
            "decay_factor": decay_factor,
            "source_weights_used": weights_used,
        }

    def total_evidence_count(self) -> int:
        """所有证据总数"""
        return sum(len(bpa.evidences) for bpa in self._bpas.values())

    # ═════════════════════════════════════════════════════════════════════
    # 概率比较 — "A 比 B 更可能"
    # ═════════════════════════════════════════════════════════════════════

    def compare_propositions(self,
                             prop_a: Tuple[str, str, str],
                             prop_b: Tuple[str, str, str]) -> Dict[str, Any]:
        """
        比较两个命题的相对可信度。

        Returns:
            {"winner": "A"|"B"|"tie", "bel_a": ..., "bel_b": ..., "margin": ...}
        """
        bel_a = self.belief(*prop_a)
        bel_b = self.belief(*prop_b)

        if bel_a > bel_b:
            return {"winner": "A", "bel_a": bel_a, "bel_b": bel_b,
                    "margin": bel_a - bel_b}
        elif bel_b > bel_a:
            return {"winner": "B", "bel_a": bel_a, "bel_b": bel_b,
                    "margin": bel_b - bel_a}
        else:
            return {"winner": "tie", "bel_a": bel_a, "bel_b": bel_b,
                    "margin": 0.0}

    # ═════════════════════════════════════════════════════════════════════
    # 冲突检测
    # ═════════════════════════════════════════════════════════════════════

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        检测相互矛盾的证据。
        例如: "光是粒子"(mass=0.6) vs "光是波"(mass=0.7)
        """
        conflicts = []
        keys = list(self._bpas.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s1, r1, o1 = keys[i]
                s2, r2, o2 = keys[j]

                # 相同主体 + 相同关系 + 不同客体 → 潜在矛盾
                if s1 == s2 and r1 == r2 and o1 != o2:
                    mass1 = self._bpas[keys[i]].combined_mass
                    mass2 = self._bpas[keys[j]].combined_mass
                    K = mass1 * mass2  # 冲突度量

                    conflicts.append({
                        "type": "same_subject_relation",
                        "proposition_a": f"{s1} {r1} {o1}",
                        "proposition_b": f"{s2} {r2} {o2}",
                        "mass_a": mass1,
                        "mass_b": mass2,
                        "conflict_K": K,
                        "evidence_count_a": len(self._bpas[keys[i]].evidences),
                        "evidence_count_b": len(self._bpas[keys[j]].evidences),
                    })

        self._conflicts = conflicts
        return conflicts

    def resolve_conflict(self, conflict: Dict[str, Any],
                         strategy: str = "source_weight") -> str:
        """
        消解冲突。

        Strategies:
          - "source_weight": 证据来源多的一方胜
          - "higher_mass": 质量高的一方胜
          - "time_priority": 新证据覆盖旧证据（需时间戳）
          - "keep_both": 保留两个命题，标注为"活跃争议"
        """
        if strategy == "source_weight":
            a_count = conflict["evidence_count_a"]
            b_count = conflict["evidence_count_b"]
            return "A" if a_count > b_count else "B" if b_count > a_count else "tie"
        elif strategy == "higher_mass":
            return "A" if conflict["mass_a"] > conflict["mass_b"] else "B"
        elif strategy == "keep_both":
            return "keep_both"
        else:
            return "keep_both"

    # ═════════════════════════════════════════════════════════════════════
    # 决策 — 回答 YES/NO/CERTAIN 问题
    # ═════════════════════════════════════════════════════════════════════

    def decide(self, subject: str, relation: str, obj: str,
               threshold: float = 0.6) -> Dict[str, Any]:
        """
        基于证据做出决策判断。

        Returns:
            decision: "YES"/"NO"/"UNCERTAIN"
            belief_support: Bel 值
            evidence_count: 证据条数
            quality: "strong"/"moderate"/"weak"/"none"
        """
        bel = self.belief(subject, relation, obj)
        evidence_count = len(self.get_evidences(subject, relation, obj))

        if evidence_count == 0:
            quality = "none"
            decision = "UNCERTAIN"
        elif bel >= 0.9:
            quality = "strong"
            decision = "YES" if bel >= threshold else "UNCERTAIN"
        elif bel >= 0.6:
            quality = "moderate"
            decision = "YES" if bel >= threshold else "UNCERTAIN"
        elif bel >= 0.3:
            quality = "weak"
            decision = "UNCERTAIN"
        else:
            quality = "weak"
            decision = "NO"

        return {
            "decision": decision,
            "belief_support": bel,
            "evidence_count": evidence_count,
            "quality": quality,
            "proposition": f"{subject} {relation} {obj}",
        }

    # ═════════════════════════════════════════════════════════════════════
    # 持久化
    # ═════════════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存模糊格"""
        data = {}
        for (s, r, o), bpa in self._bpas.items():
            key = f"{s}|||{r}|||{o}"
            data[key] = {
                "proposition": bpa.proposition,
                "combined_mass": bpa.combined_mass,
                "evidences": [{
                    "source": e.source,
                    "mass": e.mass,
                    "description": e.description,
                    "timestamp": e.timestamp,
                    "source_weight": e.source_weight,
                } for e in bpa.evidences],
                "conflict_with": bpa.conflict_with,
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载模糊格"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key_str, bpa_data in data.items():
            parts = key_str.split("|||")
            if len(parts) == 3:
                s, r, o = parts
                bpa = BPA(
                    proposition=bpa_data["proposition"],
                    combined_mass=bpa_data["combined_mass"],
                    conflict_with=bpa_data.get("conflict_with", []),
                )
                for ev_data in bpa_data["evidences"]:
                    bpa.evidences.append(Evidence(
                        source=ev_data["source"],
                        mass=ev_data["mass"],
                        description=ev_data.get("description", ""),
                        timestamp=ev_data.get("timestamp", 0.0),
                        source_weight=ev_data.get("source_weight", 1.0),
                    ))
                self._bpas[(s, r, o)] = bpa

    # ═════════════════════════════════════════════════════════════════════
    # 统计
    # ═════════════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        total_ev = self.total_evidence_count()
        high_conf = sum(1 for bpa in self._bpas.values() if bpa.combined_mass >= 0.8)
        low_conf = sum(1 for bpa in self._bpas.values() if bpa.combined_mass < 0.3)
        return {
            "propositions": len(self._bpas),
            "total_evidences": total_ev,
            "avg_evidences_per_prop": total_ev / max(1, len(self._bpas)),
            "high_confidence_props": high_conf,
            "low_confidence_props": low_conf,
            "conflicts_detected": len(self._conflicts),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_fuzzy_graph():
    """自测模糊格"""
    fg = FuzzyGraph()

    print("=" * 60)
    print("1. 基本证据添加与 D-S 组合")
    # 为"电子是原子的组成部分"添加多条独立证据
    bpa = fg.add_evidence("电子", "PART_OF", "原子",
                          source="量子力学教材第3章", mass=0.85)
    print(f"  单条证据: Bel={bpa.belief():.2%}")

    fg.add_evidence("电子", "PART_OF", "原子",
                    source="化学基础教材", mass=0.92)
    print(f"  两条组合: Bel={fg.belief('电子', 'PART_OF', '原子'):.2%}")

    fg.add_evidence("电子", "PART_OF", "原子",
                    source="物理百科", mass=0.88)
    bel = fg.belief("电子", "PART_OF", "原子")
    pl = fg.plausibility("电子", "PART_OF", "原子")
    print(f"  三条组合: Bel={bel:.2%}, Pl={pl:.2%}")
    print(f"  置信区间: {fg.uncertainty('电子', 'PART_OF', '原子')}")

    print("\n2. 添加矛盾证据")
    fg.add_evidence("光", "IS_A", "粒子", source="牛顿光学", mass=0.6)
    fg.add_evidence("光", "IS_A", "波", source="惠更斯原理", mass=0.7)
    conflicts = fg.detect_conflicts()
    for c in conflicts:
        print(f"  冲突: {c['proposition_a']} (mass={c['mass_a']})")
        print(f"      vs {c['proposition_b']} (mass={c['mass_b']})")
        print(f"      K={c['conflict_K']:.2f}")

    print("\n3. 决策判断")
    decision = fg.decide("电子", "PART_OF", "原子")
    print(f"  电子 PART_OF 原子: {decision}")

    decision = fg.decide("火星", "HAS", "生命")
    print(f"  火星 HAS 生命: {decision}")

    print("\n4. 命题比较")
    result = fg.compare_propositions(
        ("光", "IS_A", "粒子"),
        ("光", "IS_A", "波")
    )
    print(f"  光 IS_A 粒子 vs 光 IS_A 波: {result}")

    print("\n5. 统计")
    s = fg.stats()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # 持久化测试
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "fuzzy_test.json")
    fg.save(tmp)
    fg2 = FuzzyGraph()
    fg2.load(tmp)
    print(f"\n6. 持久化: 保存={s['propositions']}命题 → 加载={fg2.stats()['propositions']}命题")
    os.remove(tmp)


if __name__ == "__main__":
    test_fuzzy_graph()
