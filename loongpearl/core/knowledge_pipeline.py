#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠通用知识获取管线 (KnowledgePipeline) — 自主决定学什么→从哪学→怎么消化
════════════════════════════════════════════════════════════════════════════

替代 auto_learn.py 中单薄的 learn_one()。统一管理：
  需求检测 → 多源采集 → 知识精炼 → 能量景观/概念图/万象格/模糊格更新

原则:
  - 不区分数据类型——统一的知识获取接口
  - 所有本地词典(IDIOM/UNIHAN/CEDICT/唐诗)作为一等采集源
  - 所有精炼结果自动汇入概念图和能量景观
  - 闭环保存: 只在新知识确认为真时持久化

════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
import re
import logging
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

log = logging.getLogger('knowledge_pipeline')


# ═══════════════════════════════════════════════════════════════════════════
# 需求类型
# ═══════════════════════════════════════════════════════════════════════════

class DemandType(Enum):
    BLIND_SPOT = auto()           # 能量景观盲区
    LOW_CONFIDENCE = auto()       # 低置信度三元组
    SPARSE_NODE = auto()          # 度<3的孤立节点
    USER_CONCEPT = auto()         # 用户查询中的新概念
    CONTRADICTION = auto()        # 矛盾标记的三元组
    STALE_KNOWLEDGE = auto()      # 长期未更新的知识


@dataclass
class KnowledgeDemand:
    """一条知识需求"""
    type: DemandType
    target: str                   # 目标概念/字符/三元组
    context: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5         # 优先级 0~1
    created_at: float = field(default_factory=time.time)

    def __hash__(self):
        return hash((self.type, self.target))


@dataclass
class AcquisitionResult:
    """一次采集的结果"""
    source: str                   # 来源 (wikipedia / web_search / local_dict / user)
    query: str                    # 采集查询
    raw_text: str = ""            # 原始文本
    triples: List[Tuple[str, str, str, float]] = field(default_factory=list)
    processes: List[Dict] = field(default_factory=list)
    conditionals: List[Dict] = field(default_factory=list)
    bigrams: Dict[Tuple[str, str], int] = field(default_factory=dict)
    new_concepts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 通用知识获取管线
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgePipeline:
    """
    通用知识获取管线。

    替换 auto_learn.py 的 learn_one()，统一管理：
      - 需求检测 (盲区/低信/稀疏/用户/矛盾)
      - 多源路由 (Web搜索/Wikipedia/本地词典/用户对话)
      - 知识精炼 (三元组/过程/条件/字对提取)
      - 汇入更新 (概念图/能量景观/万象格/模糊格)
    """

    def __init__(self, field=None, landscape=None, concept_graph=None,
                 orchestrator=None, learner=None):
        self.field = field
        self.landscape = landscape
        self.cg = concept_graph
        self.orch = orchestrator
        self.learner = learner

        # 需求队列
        self.demand_queue: List[KnowledgeDemand] = []
        self._processed_demands: Set[str] = set()

        # 本地词典 (惰性加载)
        self._idiom_dict = None
        self._unihan_dict = None
        self._cedict = None
        self._tang_ngrams = None

        # 采集统计
        self.stats = {
            'total_demands': 0,
            'total_acquisitions': 0,
            'total_triples_added': 0,
            'total_bigrams_added': 0,
            'total_energy_updates': 0,
            'errors': 0,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 第1层: 需求检测
    # ═══════════════════════════════════════════════════════════════════

    def detect_demands(self, max_demands: int = 10) -> List[KnowledgeDemand]:
        """多维度检测知识需求"""
        demands = []

        # 1. 盲区检测 (从现有的 MultiFactorDetector)
        demands.extend(self._detect_blind_spots(max_demands // 4))

        # 2. 低置信度区域
        demands.extend(self._detect_low_confidence(max_demands // 4))

        # 3. 稀疏节点
        demands.extend(self._detect_sparse_nodes(max_demands // 4))

        # 4. 待处理的矛盾
        demands.extend(self._detect_contradictions(max_demands // 4))

        # 排序去重
        unique = {}
        for d in demands:
            key = f"{d.type.name}:{d.target}"
            if key not in unique or d.priority > unique[key].priority:
                unique[key] = d

        self.demand_queue = sorted(unique.values(), key=lambda x: -x.priority)
        self.stats['total_demands'] += len(self.demand_queue)

        return self.demand_queue[:max_demands]

    def feed_user_concept(self, concept: str, context: str = ""):
        """用户对话中出现了新概念 → 加入需求队列"""
        if self._is_trivial(concept):
            return
        if concept in self._processed_demands:
            return

        demand = KnowledgeDemand(
            type=DemandType.USER_CONCEPT,
            target=concept,
            context={"query": context},
            priority=0.9,  # 用户关心的概念最高优先级
        )
        self.demand_queue.append(demand)

    # ── 检测器实现 ──

    def _detect_blind_spots(self, n: int) -> List[KnowledgeDemand]:
        """从能量景观盲区中提取需求"""
        demands = []
        if not self.landscape or not self.field:
            return demands

        # 简化: 随机采样盲区字符
        try:
            import torch
            if hasattr(self.landscape, 'energy') and hasattr(self.field, 'hanzi_list'):
                # 采样能量最高的区域(盲区)
                energy = self.landscape.energy.detach().cpu()
                n_samples = min(100, energy.shape[0])
                top_idx = torch.topk(energy.view(-1), n_samples).indices.tolist()

                count = 0
                for idx in top_idx:
                    if count >= n:
                        break
                    char = self.field.hanzi_list[idx]
                    if not self._is_trivial(char) and char not in self._processed_demands:
                        demands.append(KnowledgeDemand(
                            type=DemandType.BLIND_SPOT,
                            target=char,
                            context={"energy": energy.view(-1)[idx].item()},
                            priority=0.7,
                        ))
                        count += 1
        except Exception:
            pass
        return demands

    def _detect_low_confidence(self, n: int) -> List[KnowledgeDemand]:
        """找到置信度低的三元组"""
        demands = []
        if not self.cg or not hasattr(self.cg, 'triples'):
            return demands

        candidates = []
        for key, triple in list(self.cg.triples.items())[:30000]:
            # triple 是 Triple 对象: subject, relation, object, confidence
            conf = getattr(triple, 'confidence', getattr(triple, 'c', 0.5))
            if 0.2 < conf < 0.5:
                candidates.append((triple.subject, triple.relation, triple.object, conf))

        candidates.sort(key=lambda x: x[3])
        for s, r, o, conf in candidates[:n]:
            target = f"{s} {r} {o}"
            if target not in self._processed_demands:
                demands.append(KnowledgeDemand(
                    type=DemandType.LOW_CONFIDENCE,
                    target=target,
                    context={"subject": s, "relation": r, "object": o,
                            "confidence": conf},
                    priority=0.6,
                ))

        return demands

    def _detect_sparse_nodes(self, n: int) -> List[KnowledgeDemand]:
        """找到度很低但有潜力的节点"""
        demands = []
        if not self.cg:
            return demands

        # 统计每个subject的出现次数
        deg = defaultdict(int)
        for key, triple in list(self.cg.triples.items())[:50000]:
            deg[triple.subject] += 1

        candidates = [(s, d) for s, d in deg.items() 
                     if 1 <= d <= 3 and len(s) >= 3]
        candidates.sort(key=lambda x: x[1])
        for s, d in candidates[:n]:
            if s not in self._processed_demands:
                demands.append(KnowledgeDemand(
                    type=DemandType.SPARSE_NODE,
                    target=s,
                    context={"degree": d},
                    priority=0.5,
                ))

        return demands

    def _detect_contradictions(self, n: int) -> List[KnowledgeDemand]:
        """检测标记为矛盾的节点"""
        demands = []
        if not self.orch or not self.orch._contra:
            return demands

        try:
            conflicts = self.orch.contra.detect_all()
            for c in conflicts[:n]:
                for s, r, o, conf in c.involved_triples:
                    target = f"{s} {r} {o}"
                    if target not in self._processed_demands:
                        demands.append(KnowledgeDemand(
                            type=DemandType.CONTRADICTION,
                            target=target,
                            context={"conflict_type": c.type.value},
                            priority=0.8,
                        ))
        except Exception:
            pass

        return demands

    # ═══════════════════════════════════════════════════════════════════
    # 第2层: 多源采集路由
    # ═══════════════════════════════════════════════════════════════════

    def acquire(self, demand: KnowledgeDemand) -> AcquisitionResult:
        """
        根据需求类型，路由到最佳采集源。

        路由规则:
          BLIND_SPOT      → 本地词典(idiom/unihan) + Web搜索
          LOW_CONFIDENCE  → Wikipedia + Web搜索
          SPARSE_NODE     → Wikipedia + 本地词典(cedict)
          USER_CONCEPT    → Wikipedia + Web搜索 (用户关心的高优先级)
          CONTRADICTION   → Web搜索 (找权威来源裁决)
        """
        result = AcquisitionResult(source="none", query=demand.target)
        self.stats['total_acquisitions'] += 1

        # ── 所有需求先从本地词典查 ──
        local = self._acquire_from_local_dicts(demand)
        if local:
            result = self._merge_results(result, local)

        # ── 根据类型选择在线源 ──
        if demand.type == DemandType.BLIND_SPOT:
            # 盲区 → 先从本地词典补，不够再Web搜
            if not result.triples:
                web = self._acquire_from_web(demand)
                result = self._merge_results(result, web)

        elif demand.type == DemandType.LOW_CONFIDENCE:
            # 低信 → Wikipedia+Web双路验证
            wiki = self._acquire_from_wikipedia(demand)
            web = self._acquire_from_web(demand)
            result = self._merge_results(result, wiki)
            result = self._merge_results(result, web)

        elif demand.type == DemandType.SPARSE_NODE:
            # 稀疏 → Wikipedia为主
            wiki = self._acquire_from_wikipedia(demand)
            result = self._merge_results(result, wiki)

        elif demand.type in (DemandType.USER_CONCEPT, DemandType.CONTRADICTION):
            # 高优先级 → 双路
            wiki = self._acquire_from_wikipedia(demand)
            web = self._acquire_from_web(demand)
            result = self._merge_results(result, wiki)
            result = self._merge_results(result, web)

        self._processed_demands.add(demand.target)
        return result

    def _acquire_from_local_dicts(self, demand: KnowledgeDemand) -> AcquisitionResult:
        """
        从本地知识源采集。优先查概念图（已消化），回退文件。

        消化完成后概念图已包含全部本地知识，文件只作备份。
        """
        result = AcquisitionResult(source="local", query=demand.target)

        target = demand.target

        # 单字: 查概念图中的成语 COOCCURS_IN + Unihan HAS_PINYIN
        if len(target) == 1 and '\u4e00' <= target <= '\u9fff':
            result = self._merge_results(result, self._cg_lookup_idioms(target))
            result = self._merge_results(result, self._cg_lookup_unihan(target))

        # 多字: 查概念图中的 DEFINED_AS + COOCCURS_WITH
        if len(target) >= 2:
            result = self._merge_results(result, self._cg_lookup_term(target))
            # 字对: POETIC_WITH
            if len(target) == 2:
                result = self._merge_results(result, self._cg_lookup_tang(target))

        return result

    def _cg_lookup_idioms(self, char: str) -> AcquisitionResult:
        """从概念图查包含某字的成语 (已消化自 idioms.json)"""
        result = AcquisitionResult(source="cg:idioms", query=char)
        if not self.cg:
            return result
        for key, t in list(self.cg.triples.items())[:200000]:
            if getattr(t, 'relation', '') == 'COOCCURS_IN' and t.subject == char:
                idiom = t.object
                if len(idiom) == 4:
                    result.new_concepts.append(idiom)
                    for i in range(len(idiom)-1):
                        a, b = idiom[i], idiom[i+1]
                        if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                            result.bigrams[(a,b)] = result.bigrams.get((a,b), 0) + 1
                            result.triples.append((a, "COOCCURS_WITH", b, 0.7))
                if len(result.new_concepts) >= 20:
                    break
        return result

    def _cg_lookup_unihan(self, char: str) -> AcquisitionResult:
        """从概念图查汉字属性 (已消化自 Unihan)"""
        result = AcquisitionResult(source="cg:unihan", query=char)
        if not self.cg:
            return result
        for key, t in list(self.cg.triples.items())[:100000]:
            if t.subject == char and t.relation in ('HAS_PINYIN', 'DEFINED_AS', 'HAS'):
                result.triples.append((char, t.relation, t.object, t.confidence))
        return result

    def _cg_lookup_term(self, term: str) -> AcquisitionResult:
        """从概念图查词条定义+共现 (已消化自 CEDICT)"""
        result = AcquisitionResult(source="cg:cedict", query=term)
        if not self.cg:
            return result
        for key, t in list(self.cg.triples.items())[:200000]:
            if t.subject == term and t.relation in ('DEFINED_AS', 'HAS_PINYIN', 'IS_A'):
                result.triples.append((term, t.relation, t.object, t.confidence))
            # 字间共现
            if t.relation == 'COOCCURS_WITH' and term[0] == t.subject and len(term) >= 2 and term[1] == t.object:
                result.bigrams[(t.subject, t.object)] = result.bigrams.get((t.subject, t.object), 0) + 1
        return result

    def _cg_lookup_tang(self, pair: str) -> AcquisitionResult:
        """从概念图查唐诗字对 (已消化自 tang_poetry_ngrams)"""
        result = AcquisitionResult(source="cg:tang", query=pair)
        if not self.cg or len(pair) != 2:
            return result
        for key, t in list(self.cg.triples.items())[:100000]:
            if t.relation == 'POETIC_WITH' and t.subject == pair[0] and t.object == pair[1]:
                result.triples.append((pair[0], "POETIC_WITH", pair[1], t.confidence))
                result.bigrams[(pair[0], pair[1])] = int(t.confidence * 20)
                break
        return result

    def _acquire_from_web(self, demand: KnowledgeDemand) -> AcquisitionResult:
        """Web搜索"""
        result = AcquisitionResult(source="web_search", query=demand.target)

        if not self.orch or not hasattr(self.orch, 'autonomous'):
            return result

        try:
            # 使用 autonomous learner 的搜索引擎
            autonomous = self.orch.autonomous if hasattr(self.orch, 'autonomous') else None
            if not autonomous:
                # fallback: check if learner has search
                if self.learner and hasattr(self.learner, 'autonomous_learner'):
                    autonomous = self.learner.autonomous_learner

            if autonomous and hasattr(autonomous, 'learn_if_unknown'):
                search_result = autonomous.learn_if_unknown(
                    query_text=f"{demand.target} 知识 解释",
                    query_vec=None,
                    auto_search=True,
                )
                if search_result:
                    result.raw_text = search_result.get('raw_text', '')
                    result.triples.extend(
                        search_result.get('triples', [])
                    )
        except Exception as e:
            result.errors.append(f"web_search: {e}")

        return result

    def _acquire_from_wikipedia(self, demand: KnowledgeDemand) -> AcquisitionResult:
        """Wikipedia采集"""
        result = AcquisitionResult(source="wikipedia", query=demand.target)

        if not self.orch or not self.orch._harvester:
            return result

        try:
            harvester = self.orch.harvester
            # 用概念名作为Wikipedia标题
            title = demand.target
            if len(title) >= 2:
                added = harvester.harvest_wikipedia(
                    titles=[title],
                    max_per_page=20,
                )
                if added:
                    result.triples.append(
                        (demand.target, "RELATED", f"wikipedia:{title}", 0.5)
                    )
        except Exception as e:
            result.errors.append(f"wikipedia: {e}")

        return result

    # ── 本地词典查询 ──

    def _load_idioms(self):
        if self._idiom_dict is not None:
            return
        try:
            path = os.path.join(PROJECT_ROOT, 'data', 'dicts', 'idioms.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._idiom_dict = json.load(f)
        except Exception:
            self._idiom_dict = []

    def _lookup_idioms_for_char(self, char: str) -> Optional[AcquisitionResult]:
        """查找包含某字的所有成语"""
        self._load_idioms()
        if not self._idiom_dict:
            return None

        result = AcquisitionResult(source="idiom_dict", query=char)
        found = [i for i in self._idiom_dict if char in i]

        for idiom in found[:20]:
            # 成语中的每对相邻字 → 共现边
            for i in range(len(idiom) - 1):
                a, b = idiom[i], idiom[i+1]
                if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                    result.bigrams[(a, b)] = result.bigrams.get((a, b), 0) + 1
                    result.triples.append((a, "COOCCURS", b, 0.7))
            # 成语本身作为概念
            result.new_concepts.append(idiom)

        return result

    def _load_unihan(self):
        if self._unihan_dict is not None:
            return
        try:
            path = os.path.join(PROJECT_ROOT, 'data', 'dicts', 'dict_unihan.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._unihan_dict = json.load(f)
        except Exception:
            self._unihan_dict = {}

    def _lookup_unihan(self, char: str) -> Optional[AcquisitionResult]:
        """查询汉字部首/笔画等属性"""
        self._load_unihan()
        if not self._unihan_dict:
            return None

        entry = self._unihan_dict.get(char, {})
        if not entry:
            return None

        result = AcquisitionResult(source="unihan_dict", query=char)

        # 部首
        radical = entry.get('radical', '')
        if radical and len(radical) == 1:
            result.triples.append((char, "HAS_RADICAL", radical, 0.9))

        # 笔画
        strokes = entry.get('stroke_count', 0)
        if strokes:
            result.triples.append((char, "HAS", f"{strokes}画", 0.8))

        # 异体字
        variants = entry.get('variants', [])
        for v in variants[:5]:
            if len(v) == 1:
                result.triples.append((char, "RELATED", v, 0.6))

        return result

    def _load_cedict(self):
        if self._cedict is not None:
            return
        try:
            path = os.path.join(PROJECT_ROOT, 'data', 'dicts', 'cedict_parsed.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._cedict = json.load(f)
        except Exception:
            self._cedict = {}

    def _lookup_cedict(self, term: str) -> Optional[AcquisitionResult]:
        """查询CC-CEDICT词条"""
        self._load_cedict()
        if not self._cedict:
            return None

        result = AcquisitionResult(source="cedict", query=term)

        # 精确匹配
        if term in self._cedict:
            entry = self._cedict[term]
            if isinstance(entry, dict):
                definition = entry.get('definition', '')
                if definition:
                    result.triples.append((term, "DEFINED_AS", definition[:50], 0.7))

            # 字间共现
            for i in range(len(term) - 1):
                a, b = term[i], term[i+1]
                if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                    result.bigrams[(a, b)] = result.bigrams.get((a, b), 0) + 2

        # 模糊匹配
        for key in list(self._cedict.keys())[:50000]:
            if term in key and key != term:
                result.new_concepts.append(key)
                if len(result.new_concepts) >= 10:
                    break

        return result

    def _load_tang_ngrams(self):
        if self._tang_ngrams is not None:
            return
        try:
            path = os.path.join(PROJECT_ROOT, 'data', 'dicts', 'tang_poetry_ngrams.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._tang_ngrams = json.load(f)
        except Exception:
            self._tang_ngrams = {}

    def _lookup_tang_bigram(self, pair: str) -> Optional[AcquisitionResult]:
        """查询唐诗字对共现"""
        self._load_tang_ngrams()
        if not self._tang_ngrams:
            return None

        result = AcquisitionResult(source="tang_poetry", query=pair)

        bigrams = self._tang_ngrams.get('bigrams', {})
        key = f"{pair[0]}|{pair[1]}"
        if key in bigrams:
            freq = bigrams[key]
            result.bigrams[(pair[0], pair[1])] = freq
            result.triples.append((pair[0], "POETIC_WITH", pair[1], min(0.9, freq / 20)))

        return result

    # ═══════════════════════════════════════════════════════════════════
    # 第3层: 知识精炼与汇入
    # ═══════════════════════════════════════════════════════════════════

    def refine_and_inject(self, result: AcquisitionResult):
        """
        将采集结果精炼并注入到所有知识库中。

        注入目标:
          1. 概念图 → 三元组
          2. 能量景观 → Hebbian更新 (如果可用)
          3. 万象格 → 过程/条件
          4. 模糊格 → D-S证据
        """
        total_added = 0

        # ── 1. 概念图: 添加三元组 ──
        if self.cg:
            for s, r, o, conf in result.triples:
                if self._is_valid_triple(s, r, o):
                    try:
                        self.cg.add_triple(s, r, o, confidence=conf, source=result.source)
                        total_added += 1
                    except Exception:
                        pass

            # 添加新概念节点
            for concept in result.new_concepts:
                if len(concept) >= 2:
                    key = f"{concept}|RELATED|{result.query[:20]}"
                    if key not in self.cg.triples:
                        try:
                            self.cg.add_triple(
                                concept, "RELATED", result.query[:20],
                                confidence=0.3, source=result.source
                            )
                            total_added += 1
                        except Exception:
                            pass

        self.stats['total_triples_added'] += total_added

        # ── 2. 能量景观: Hebbian更新字对共现 ──
        if result.bigrams and self.learner and hasattr(self.learner, 'inject_pairs'):
            try:
                pair_list = [(a, b) for (a, b), freq in result.bigrams.items()
                            if freq >= 2]
                if pair_list:
                    n = self.learner.inject_pairs(pair_list)
                    self.stats['total_energy_updates'] += n
                    self.stats['total_bigrams_added'] += len(pair_list)
            except Exception:
                pass

        # ── 3. 万象格: 注入过程和条件 ──
        if result.processes and self.orch and self.orch._mkg:
            for proc in result.processes:
                try:
                    self.orch.mkg.add_process(
                        name=proc.get('name', result.query),
                        steps=proc.get('steps', []),
                        domain=proc.get('domain', ''),
                    )
                except Exception:
                    pass

        if result.conditionals and self.orch and self.orch._mkg:
            for cond in result.conditionals:
                try:
                    self.orch.mkg.add_conditional(
                        condition=cond.get('condition', {}),
                        consequent=cond.get('consequent', ('', '', '')),
                        confidence=cond.get('confidence', 0.7),
                        domain=cond.get('domain', ''),
                    )
                except Exception:
                    pass

        # ── 4. 模糊格: 添加D-S证据 ──
        if self.orch and self.orch._fuzzy and result.triples:
            for s, r, o, conf in result.triples[:20]:
                try:
                    self.orch.fuzzy.add_evidence(
                        s, r, o,
                        source=result.source,
                        mass=conf,
                    )
                except Exception:
                    pass

        return total_added

    # ═══════════════════════════════════════════════════════════════════
    # 主循环: 一轮完整获取
    # ═══════════════════════════════════════════════════════════════════

    def tick(self, max_demands: int = 5, max_acquire: int = 3) -> Dict[str, Any]:
        """
        执行一轮知识获取。

        Returns:
            {"demands_found": N, "acquired": N, "triples_added": N, ...}
        """
        # 1. 检测需求
        demands = self.detect_demands(max_demands=max_demands)
        if not demands:
            return {"demands_found": 0, "acquired": 0, "triples_added": 0,
                    "total_bigrams": 0, "total_energy_updates": 0}

        # 2. 逐条采集+精炼
        total_added = 0
        acquired = 0
        for demand in demands[:max_acquire]:
            try:
                result = self.acquire(demand)
                if result.triples or result.bigrams or result.new_concepts:
                    n = self.refine_and_inject(result)
                    total_added += n
                    acquired += 1
                    log.debug(f"  📖 {demand.type.name}: '{demand.target}' "
                             f"→ {len(result.triples)}三元组 "
                             f"{len(result.bigrams)}字对 "
                             f"[{result.source}]")
            except Exception as e:
                self.stats['errors'] += 1
                log.debug(f"  获取失败: {demand.target} - {e}")

        return {
            "demands_found": len(demands),
            "acquired": acquired,
            "triples_added": total_added,
            "total_bigrams": self.stats['total_bigrams_added'],
            "total_energy_updates": self.stats['total_energy_updates'],
        }

    # ═══════════════════════════════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════════════════════════════

    def _is_trivial(self, text: str) -> bool:
        if not text or len(text) < 1:
            return True
        if not any('\u4e00' <= c <= '\u9fff' for c in text):
            return True
        return False

    def _is_valid_triple(self, s: str, r: str, o: str) -> bool:
        if self._is_trivial(s) or self._is_trivial(o):
            return False
        if len(s) > 50 or len(o) > 100:
            return False
        if s == o:
            return False
        return True

    def _merge_results(self, a: AcquisitionResult,
                        b: AcquisitionResult) -> AcquisitionResult:
        """合并两个采集结果"""
        a.triples.extend(b.triples)
        a.processes.extend(b.processes)
        a.conditionals.extend(b.conditionals)
        for k, v in b.bigrams.items():
            a.bigrams[k] = a.bigrams.get(k, 0) + v
        a.new_concepts.extend(b.new_concepts)
        a.errors.extend(b.errors)
        if b.raw_text:
            a.raw_text += "\n" + b.raw_text
        return a

    def status_report(self) -> str:
        return (
            f"═══ 知识获取管线 ═══\n"
            f"  需求检测: {self.stats['total_demands']}\n"
            f"  采集次数: {self.stats['total_acquisitions']}\n"
            f"  三元组注入: {self.stats['total_triples_added']}\n"
            f"  字对注入: {self.stats['total_bigrams_added']}\n"
            f"  能量更新: {self.stats['total_energy_updates']}\n"
            f"  错误: {self.stats['errors']}\n"
            f"  已处理需求: {len(self._processed_demands)}"
        )
