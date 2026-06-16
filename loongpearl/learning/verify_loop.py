#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠闭环验证引擎 (verify_loop.py) — 推断→搜索→修正
═══════════════════════════════════════════════════════

龙珠超越 LLM 的核心优势之二：可验证性。

LLM 的推断是黑箱概率，无法追溯"为什么这样推断"，也无法系统化验证。
龙珠的每个推断都可以：
  1. 追溯到具体的归纳推理步骤（transitive closure）
  2. 用搜索/能量评估独立验证
  3. 根据验证结果修正置信度
  4. 记录完整的验证链

══════════════════════════════════════════════════════════════════
闭环流程
══════════════════════════════════════════════════════════════════

  概念图推断三元组 (低置信度)
      │
      ▼
  构造验证查询 ("X是Y的组成部分吗")
      │
      ▼
  多源并发搜索 (Bing/DuckDuckGo/百科)
      │
      ▼
  证据提取 (正则匹配 + 关键词命中)
      │
      ├─ 确认 → 提升置信度 (×1.5)
      ├─ 否定 → 降低置信度 (×0.3) / 标记矛盾
      └─ 无证据 → 保持，标记待验证
      │
      ▼
  记录验证结果 → 概念图更新

══════════════════════════════════════════════════════════════════
用法
══════════════════════════════════════════════════════════════════

    vf = VerifyLoop(concept_graph, searcher)
    
    # 验证单条推断
    result = vf.verify_triple("电子", "PART_OF", "物质")
    # → {'verdict': 'confirmed', 'confidence_delta': +0.2, 'evidence': [...]}
    
    # 批量验证所有推断
    report = vf.verify_all_inferred()
    # → {'verified': 15, 'contradicted': 2, 'uncertain': 8}
    
    # 持续验证循环
    vf.run_loop(max_rounds=10)  # 每轮验证10条
"""

import re
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    CONFIRMED = "confirmed"       # 搜索证据支持
    CONTRADICTED = "contradicted" # 搜索证据反对
    UNCERTAIN = "uncertain"       # 无明确证据
    ERROR = "error"               # 搜索失败


@dataclass
class VerificationResult:
    """一次验证的结果"""
    subject: str
    relation: str
    object: str
    verdict: Verdict
    confidence_before: float
    confidence_after: float
    evidence: List[str] = field(default_factory=list)
    query: str = ""
    sources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class VerifyLoop:
    """
    闭环验证引擎。

    原理:
      - 对每个推断出的低置信度三元组，构造自然语言查询
      - 用 WebSearcher 多源并发搜索
      - 从搜索结果中提取正面/负面证据
      - 根据证据调整置信度

    不依赖 LLM。证据提取用正则 + 关键词命中。
    """

    def __init__(self, concept_graph, searcher=None):
        self.cg = concept_graph
        self._searcher = searcher
        self.verify_log: List[VerificationResult] = []
        self.stats = {
            'total_verified': 0,
            'confirmed': 0,
            'contradicted': 0,
            'uncertain': 0,
        }

    @property
    def searcher(self):
        if self._searcher is None:
            from loongpearl.web.searcher import WebSearcher
            self._searcher = WebSearcher(timeout=10, cache_enabled=True)
        return self._searcher

    # ═══════════════════════════════════════════════════════════════
    # 查询构造
    # ═══════════════════════════════════════════════════════════════

    def _build_query(self, subject: str, relation: str, obj: str) -> str:
        """
        将三元组转化为自然语言验证查询。

        例:
          ("电子", "PART_OF", "原子") → "电子是原子的组成部分吗"
          ("猫", "IS_A", "动物")      → "猫是一种动物吗"
          ("细胞", "HAS", "细胞核")   → "细胞包含细胞核吗"
        """
        templates = {
            "PART_OF":  f"{subject}是{obj}的组成部分 组成",
            "IS_A":     f"{subject}是{obj}的一种 属于",
            "HAS":      f"{subject}包含{obj} 具有",
            "CAUSE":    f"{subject}导致{obj} 引起",
            "OPPOSITE": f"{subject}和{obj} 对立 相反",
            "RELATED":  f"{subject}和{obj} 关系 关联",
        }
        return templates.get(relation, f"{subject} {obj} 关系 关联")

    # ═══════════════════════════════════════════════════════════════
    # 证据提取
    # ═══════════════════════════════════════════════════════════════

    def _extract_evidence(
        self,
        search_text: str,
        subject: str,
        relation: str,
        obj: str,
    ) -> Tuple[float, List[str]]:
        """
        从搜索文本中提取证据。

        Returns:
            (evidence_score, evidence_snippets)
            evidence_score: -1.0(强烈反对) ~ +1.0(强烈支持), 0=无证据
        """
        score = 0.0
        snippets = []

        # 正面证据模式
        positive_patterns = {
            "PART_OF": [
                rf'{re.escape(subject)}.*(?:组成|构成|属于).*{re.escape(obj)}',
                rf'{re.escape(obj)}.*(?:由|包含).*{re.escape(subject)}.*(?:组成|构成)',
                rf'{re.escape(subject)}.*是.*{re.escape(obj)}.*(?:一部分|组成部分)',
            ],
            "IS_A": [
                rf'{re.escape(subject)}.*(?:是|属于).*{re.escape(obj)}',
                rf'{re.escape(subject)}.*一种.*{re.escape(obj)}',
            ],
            "HAS": [
                rf'{re.escape(subject)}.*(?:包含|拥有|具有).*{re.escape(obj)}',
                rf'{re.escape(obj)}.*是.*{re.escape(subject)}.*(?:的|拥有)',
            ],
            "CAUSE": [
                rf'{re.escape(subject)}.*(?:导致|引起|产生).*{re.escape(obj)}',
                rf'{re.escape(obj)}.*(?:由|因为).*{re.escape(subject)}.*(?:导致|引起)',
            ],
        }

        patterns = positive_patterns.get(relation, [])
        if not patterns:
            # 兜底：要求两个概念出现在同一句话中(不超过30字)，而非全文任意位置
            patterns = [rf'{re.escape(subject)}.{{0,30}}{re.escape(obj)}']

        # 负面证据模式
        negative_patterns = [
            rf'{re.escape(subject)}.*(?:不是|不属于|并非).*{re.escape(obj)}',
            rf'{re.escape(subject)}.*与.*{re.escape(obj)}.*(?:无关|没有关系)',
        ]

        for pat in patterns:
            matches = re.findall(pat, search_text, re.IGNORECASE)
            if matches:
                score += 0.25 * len(matches)
                # 提取匹配片段
                for m in re.finditer(pat, search_text, re.IGNORECASE):
                    start = max(0, m.start() - 20)
                    end = min(len(search_text), m.end() + 20)
                    snippet = search_text[start:end].strip()
                    if snippet not in snippets:
                        snippets.append(snippet)

        for pat in negative_patterns:
            matches = re.findall(pat, search_text, re.IGNORECASE)
            if matches:
                score -= 0.5 * len(matches)

        # 简单关键词共现
        if subject in search_text and obj in search_text:
            # 两者都出现但没匹配到明确模式 → 弱正面信号
            if score == 0.0:
                score = 0.1

        return max(-1.0, min(1.0, score)), snippets[:5]

    # ═══════════════════════════════════════════════════════════════
    # 单条验证
    # ═══════════════════════════════════════════════════════════════

    def verify_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> VerificationResult:
        """
        验证一条三元组。

        步骤:
          1. 构造自然语言查询
          2. 多源并发搜索
          3. 提取证据
          4. 判定判决
          5. 更新置信度
        """
        # 获取当前置信度
        key = f"{subject}|{relation}|{obj}"
        triple = self.cg.triples.get(key)
        conf_before = triple.confidence if triple else 0.5

        # 构造查询
        query = self._build_query(subject, relation, obj)

        # 搜索
        evidence_snippets = []
        sources = []
        all_text = ""

        try:
            results = self.searcher.search(query)
            if results and hasattr(results, 'results'):
                for r in results.results[:3]:
                    text = ""
                    if hasattr(r, 'snippet'):
                        text = r.snippet
                    elif hasattr(r, 'text'):
                        text = r.text
                    if text:
                        all_text += " " + text
                    if hasattr(r, 'url'):
                        sources.append(r.url)
        except Exception as e:
            return VerificationResult(
                subject=subject, relation=relation, object=obj,
                verdict=Verdict.ERROR,
                confidence_before=conf_before,
                confidence_after=conf_before,
                query=query,
            )

        if not all_text.strip():
            return VerificationResult(
                subject=subject, relation=relation, object=obj,
                verdict=Verdict.UNCERTAIN,
                confidence_before=conf_before,
                confidence_after=conf_before,
                query=query,
            )

        # 提取证据
        evidence_score, evidence_snippets = self._extract_evidence(
            all_text, subject, relation, obj
        )

        # 判定
        if evidence_score > 0.3:
            verdict = Verdict.CONFIRMED
            conf_multiplier = 1.0 + evidence_score * 0.5  # 最高提升50%
        elif evidence_score < -0.2:
            verdict = Verdict.CONTRADICTED
            conf_multiplier = 0.3  # 降低到30%
        else:
            verdict = Verdict.UNCERTAIN
            conf_multiplier = 1.0  # 不变

        conf_after = min(1.0, conf_before * conf_multiplier)

        # 更新概念图中的置信度
        if triple:
            triple.confidence = conf_after

        result = VerificationResult(
            subject=subject,
            relation=relation,
            object=obj,
            verdict=verdict,
            confidence_before=conf_before,
            confidence_after=conf_after,
            evidence=evidence_snippets,
            query=query,
            sources=sources,
        )

        self.verify_log.append(result)
        self.stats['total_verified'] += 1
        self.stats[verdict.value] += 1

        return result

    # ═══════════════════════════════════════════════════════════════
    # 批量验证
    # ═══════════════════════════════════════════════════════════════

    def verify_all_inferred(
        self,
        max_verify: int = 20,
        max_confidence: float = 0.5,
        min_confidence: float = 0.1,
    ) -> Dict[str, Any]:
        """
        验证概念图中所有低置信度推断三元组。

        Args:
            max_verify: 最多验证条数
            max_confidence: 只验证置信度低于此值的
            min_confidence: 最低置信度阈值（太低的不值得验证）
        """
        # 找出所有推断来源的低置信度三元组
        candidates = []
        for key, triple in self.cg.triples.items():
            if triple.source != "infer":
                continue
            if triple.confidence < min_confidence or triple.confidence > max_confidence:
                continue
            candidates.append((triple.confidence, triple))

        # 按置信度从低到高排序（优先验证最不确定的）
        candidates.sort(key=lambda x: x[0])

        results = []
        for _, triple in candidates[:max_verify]:
            result = self.verify_triple(triple.subject, triple.relation, triple.object)
            results.append(result)
            time.sleep(0.5)  # 避免搜索频率过高

        # 统计
        confirmed = sum(1 for r in results if r.verdict == Verdict.CONFIRMED)
        contradicted = sum(1 for r in results if r.verdict == Verdict.CONTRADICTED)
        uncertain = sum(1 for r in results if r.verdict == Verdict.UNCERTAIN)

        return {
            'total': len(results),
            'confirmed': confirmed,
            'contradicted': contradicted,
            'uncertain': uncertain,
            'results': [
                {
                    'triple': f"{r.subject} {r.relation} {r.object}",
                    'verdict': r.verdict.value,
                    'confidence': f"{r.confidence_before:.2f} → {r.confidence_after:.2f}",
                    'query': r.query,
                }
                for r in results
            ],
        }

    def verify_lowest_confidence(
        self,
        n: int = 10,
    ) -> Dict[str, Any]:
        """验证置信度最低的N条推断"""
        # 找最低置信度的推断三元组
        inferred = [
            (t.confidence, t)
            for t in self.cg.triples.values()
            if t.source == "infer" and t.confidence < 0.5
        ]
        inferred.sort(key=lambda x: x[0])

        results = []
        for _, triple in inferred[:n]:
            result = self.verify_triple(triple.subject, triple.relation, triple.object)
            results.append(result)
            time.sleep(0.3)

        confirmed = sum(1 for r in results if r.verdict == Verdict.CONFIRMED)
        contradicted = sum(1 for r in results if r.verdict == Verdict.CONTRADICTED)

        # 如果有矛盾的，降低置信度并标记
        for r in results:
            if r.verdict == Verdict.CONTRADICTED:
                key = f"{r.subject}|{r.relation}|{r.object}"
                if key in self.cg.triples:
                    self.cg.triples[key].confidence = max(0.05, r.confidence_after)

        return {
            'total': len(results),
            'confirmed': confirmed,
            'contradicted': contradicted,
            'uncertain': sum(1 for r in results if r.verdict == Verdict.UNCERTAIN),
        }

    # ═══════════════════════════════════════════════════════════════
    # 持续验证循环
    # ═══════════════════════════════════════════════════════════════

    def run_loop(
        self,
        max_rounds: int = 5,
        per_round: int = 5,
        interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        持续验证循环。

        每轮: 取最低置信度的 per_round 条推断 → 搜索验证 → 修正置信度
        """
        total_confirmed = 0
        total_contradicted = 0

        for rnd in range(max_rounds):
            report = self.verify_lowest_confidence(n=per_round)
            total_confirmed += report['confirmed']
            total_contradicted += report['contradicted']

            if rnd < max_rounds - 1:
                time.sleep(interval)

        return {
            'rounds': max_rounds,
            'total_verified': max_rounds * per_round,
            'total_confirmed': total_confirmed,
            'total_contradicted': total_contradicted,
        }

    # ═══════════════════════════════════════════════════════════════
    # 报告
    # ═══════════════════════════════════════════════════════════════

    def report(self) -> Dict[str, Any]:
        """生成验证统计报告"""
        return {
            'stats': dict(self.stats),
            'recent': [
                {
                    'triple': f"{r.subject} {r.relation} {r.object}",
                    'verdict': r.verdict.value,
                    'confidence': f"{r.confidence_before:.2f}→{r.confidence_after:.2f}",
                    'query': r.query,
                    'evidence': r.evidence[:2],
                }
                for r in self.verify_log[-10:]
            ],
        }


# ============================================================================
# 演示
# ============================================================================

def demo_verify_loop(concept_graph):
    """演示闭环验证"""
    vf = VerifyLoop(concept_graph)

    print("=" * 60)
    print("🔄 龙珠闭环验证引擎 — 演示")
    print("=" * 60)

    inferred_count = sum(1 for t in concept_graph.triples.values() if t.source == "infer")
    print(f"   推断三元组: {inferred_count} 条")

    # 找最低置信度的推断
    inferred = [
        (t.confidence, t)
        for t in concept_graph.triples.values()
        if t.source == "infer"
    ]
    inferred.sort(key=lambda x: x[0])

    print(f"\n📋 最低置信度推断 (前10):")
    for conf, t in inferred[:10]:
        print(f"   {t.subject} {t.relation} {t.object} (信:{conf:.2f})")

    # 验证前3条（不实际搜索，演示流程）
    print(f"\n🔍 验证流程演示 (不联网):")
    for conf, t in inferred[:3]:
        query = vf._build_query(t.subject, t.relation, t.object)
        print(f"   {t.subject} {t.relation} {t.object}")
        print(f"   → 查询: '{query}'")
        result = vf.verify_triple(t.subject, t.relation, t.object)
        print(f"   → 判决: {result.verdict.value} "
              f"(信:{result.confidence_before:.2f}→{result.confidence_after:.2f})")

    print(f"\n📊 统计: {vf.stats}")

    return vf


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from loongpearl.core.zichang import HanziAnchorField
    from loongpearl.core.concept_graph import ConceptGraph

    PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    field_path = os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt')
    cg_path = os.path.join(PROJECT, 'data/models/concept_graph')

    field = HanziAnchorField.load(field_path, freeze=True)

    cg = ConceptGraph(field)
    if os.path.exists(cg_path + '.json'):
        cg.load(cg_path)

    demo_verify_loop(cg)
