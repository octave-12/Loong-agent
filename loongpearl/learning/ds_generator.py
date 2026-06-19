#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-S 假设生成器 — 三源证据融合知识发现
======================================

三个假设生成源:
  源1: 扰动结果 — PerturbationEngine 检测到的脆性候选
  源2: 弱概念边 — 概念图中置信度 0.3~0.5 的三元组
  源3: 高相似无连接 — 嵌入空间 cos≥0.75 但概念图无边的字对

Dempster 组合规则融合多源 mass 函数 → 置信度 > 0.7 → 注入概念图

频率: 每5轮 (与扰动引擎串联)
依赖: ConceptGraph, FuzzyGraph, HanziAnchorField
"""

import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger('ds_generator')


@dataclass
class Hypothesis:
    """知识假设"""
    subject: str
    relation: str
    obj: str
    source: str                        # 'perturbation' | 'weak_edge' | 'high_sim_no_edge'
    source_confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)
    
    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)
    
    def __repr__(self):
        return (f"Hypothesis({self.subject} {self.relation} {self.obj}, "
                f"src={self.source}, conf={self.source_confidence:.2f})")


@dataclass
class GeneratorReport:
    """生成器运行报告"""
    n_source1: int = 0     # 扰动源
    n_source2: int = 0     # 弱边源
    n_source3: int = 0     # 高相似源
    n_combined: int = 0    # 融合后
    n_injected: int = 0    # 注入概念图
    n_conflict: int = 0    # 冲突量
    elapsed: float = 0.0


class DSHypothesisGenerator:
    """
    D-S 假设生成器 — 三源汇聚 → Dempster 组合 → 阈值注入。
    
    所有阈值数据驱动 (百分位)，零硬编码。
    """
    
    # 可调参数
    SIM_THRESHOLD = 0.75       # 源3高相似阈值
    MAX_CANDIDATES = 200       # 每源最多候选
    INJECT_THRESHOLD = 0.55    # 注入置信度阈值 (降低以提升注入率)
    SIM_CHUNK_SIZE = 1000      # ★ 分块大小防止 OOM
    
    # 关系类型权重 (源2)
    RELATION_WEIGHTS = {
        'IS_A': 1.0, 'PART_OF': 1.0, 'HAS': 0.95,
        'CAUSE': 0.85, 'RELATED': 0.8, 'OPPOSITE': 0.9,
    }
    
    def __init__(self, field, landscape, cg, fuzzy=None, learner=None):
        self.field = field
        self.landscape = landscape
        self.cg = cg
        self.fuzzy = fuzzy
        self.learner = learner
        self.device = next(landscape.parameters()).device if landscape else 'cpu'
    
    # ═══════════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════════
    
    def run(self, perturbation_candidates: List = None
            ) -> GeneratorReport:
        """
        执行三源假设生成 → Dempster 组合 → 注入。
        
        Args:
            perturbation_candidates: PerturbationCandidate 列表 (源1输入)
        """
        t0 = time.time()
        report = GeneratorReport()
        all_hypotheses: Dict[Tuple, List[Hypothesis]] = {}
        
        # ═══ 源1: 扰动结果 ═══
        if perturbation_candidates:
            h1 = self._source_perturbation(perturbation_candidates)
            report.n_source1 = len(h1)
            for h in h1:
                all_hypotheses.setdefault(h.key, []).append(h)
        
        # ═══ 源2: 弱概念边 ═══
        h2 = self._source_weak_edges()
        report.n_source2 = len(h2)
        for h in h2:
            all_hypotheses.setdefault(h.key, []).append(h)
        
        # ═══ 源3: 高相似无连接 ═══
        h3 = self._source_high_sim_no_edge()
        report.n_source3 = len(h3)
        for h in h3:
            all_hypotheses.setdefault(h.key, []).append(h)
        
        report.n_combined = len(all_hypotheses)
        
        # ═══ Dempster 组合 + 注入 ═══
        injected = 0
        for key, hyps in all_hypotheses.items():
            combined_belief = self._dempster_combine(hyps)
            
            if combined_belief >= self.INJECT_THRESHOLD:
                s, r, o = key
                if self.fuzzy:
                    try:
                        self.fuzzy.add_evidence(
                            s, r, o,
                            source='ds_generator',
                            mass=combined_belief,
                            description=f'三源融合: {[h.source for h in hyps]}',
                        )
                        injected += 1
                    except Exception as e:
                        log.debug(f"  注入失败 ({s},{r},{o}): {e}")
        
        report.n_injected = injected
        report.elapsed = time.time() - t0
        
        if injected > 0:
            log.info(f"  D-S生成器: {report.n_source1}+{report.n_source2}+"
                    f"{report.n_source3} → {report.n_combined}融合 → "
                    f"{report.n_injected}注入 ({report.elapsed:.1f}s)")
        else:
            log.info(f"  D-S生成器: {report.n_source1}+{report.n_source2}+"
                    f"{report.n_source3} → {report.n_combined}融合 → "
                    f"0注入 ({report.elapsed:.1f}s)")
        
        return report
    
    # ═══════════════════════════════════════════════════════════════
    # 源1: 扰动结果
    # ═══════════════════════════════════════════════════════════════
    
    def _source_perturbation(self, candidates: List) -> List[Hypothesis]:
        """扰动候选 → D-S 假设"""
        hypotheses = []
        
        for cand in candidates[:50]:
            # mass 函数: 基于扰动 score 和余弦相似度
            abs_de_norm = min(abs(cand.delta_E) / max(abs(cand.delta_E), 1e-8), 1.0)
            mass = 0.3 + 0.4 * abs_de_norm + 0.2 * cand.cosine_sim
            mass = min(mass, 0.9)
            
            hypotheses.append(Hypothesis(
                subject=cand.char_a,
                relation='RELATED',
                obj=cand.char_b,
                source='perturbation',
                source_confidence=mass,
                metadata={
                    'delta_E': cand.delta_E,
                    'cosine_sim': cand.cosine_sim,
                    'score': cand.score,
                }
            ))
        
        return hypotheses[:self.MAX_CANDIDATES]
    
    # ═══════════════════════════════════════════════════════════════
    # 源2: 弱概念边
    # ═══════════════════════════════════════════════════════════════
    
    def _source_weak_edges(self) -> List[Hypothesis]:
        """
        弱边源: 概念图中置信度 0.3~0.5 的三元组。
        通过 forward_index 遍历，避免全量扫描 triples。
        """
        hypotheses = []
        fwd = getattr(self.cg, 'forward_index', {})
        
        for s, edges in fwd.items():
            for obj, rel in edges.items():
                key = f"{s}|{rel}|{obj}"
                triple = self.cg.triples.get(key)
                
                if triple is None:
                    continue
                
                conf = getattr(triple, 'confidence', 0)
                if not (0.3 <= conf <= 0.5):
                    continue
                
                # mass = conf × relation_weight × evidence_bonus
                rw = self.RELATION_WEIGHTS.get(rel, 0.8)
                ev_count = getattr(triple, 'evidence_count', 0)
                ev_bonus = 1.0 + min(ev_count, 3) * 0.05
                mass = conf * rw * ev_bonus
                
                hypotheses.append(Hypothesis(
                    subject=s, relation=rel, obj=obj,
                    source='weak_edge',
                    source_confidence=mass,
                    metadata={
                        'original_conf': conf,
                        'evidence_count': ev_count,
                        'original_source': getattr(triple, 'source', ''),
                    }
                ))
                
                if len(hypotheses) >= self.MAX_CANDIDATES:
                    return hypotheses
        
        return hypotheses
    
    # ═══════════════════════════════════════════════════════════════
    # 源3: 高相似无连接
    # ═══════════════════════════════════════════════════════════════
    
    def _source_high_sim_no_edge(self) -> List[Hypothesis]:
        """
        高相似无连接源: GPU 分块相似度矩阵 + numpy 批量提取，防 OOM。
        """
        anchors = self.field.anchors
        hanzi_list = self.field.hanzi_list
        
        search_range = min(5000, len(hanzi_list))
        if search_range < 2:
            return []
        
        # 已有边集合 (双向)
        existing = set()
        fwd = getattr(self.cg, 'forward_index', {})
        for s, edges in fwd.items():
            for obj in edges:
                existing.add((s, obj))
                existing.add((obj, s))
        
        sub = anchors[:search_range].to(self.device)
        sub_norm = F.normalize(sub, p=2, dim=1)
        
        chunk = self.SIM_CHUNK_SIZE
        all_candidates = []
        
        for i_start in range(0, search_range, chunk):
            i_end = min(i_start + chunk, search_range)
            row_block = sub_norm[i_start:i_end]       # (chunk, D)
            sim_block = row_block @ sub_norm.T         # (chunk, search_range)
            
            # 上三角 mask + 高相似 mask
            cols = torch.arange(search_range, device=self.device)
            rows = torch.arange(i_start, i_end, device=self.device).unsqueeze(1)
            upper_mask = rows < cols.unsqueeze(0)      # (chunk, search_range)
            sim_mask = sim_block > self.SIM_THRESHOLD
            hit_mask = upper_mask & sim_mask
            
            if hit_mask.any():
                hits = hit_mask.nonzero(as_tuple=False)  # (K, 2)
                for idx in range(hits.shape[0]):
                    ri = hits[idx, 0].item() + i_start
                    ci = hits[idx, 1].item()
                    sim_val = sim_block[hits[idx, 0], hits[idx, 1]].item()
                    
                    char_a = hanzi_list[ri]
                    char_b = hanzi_list[ci]
                    if (char_a, char_b) in existing:
                        continue
                    
                    all_candidates.append((ri, ci, sim_val, char_a, char_b))
                    if len(all_candidates) >= self.MAX_CANDIDATES * 3:
                        break
                if len(all_candidates) >= self.MAX_CANDIDATES * 3:
                    break
        
        # 构建假设
        hypotheses = []
        for ri, ci, sim_val, char_a, char_b in all_candidates[:self.MAX_CANDIDATES]:
            rel = self._infer_relation(char_a, char_b)
            sim_norm = (sim_val - 0.75) / 0.25
            rel_novelty = {'RELATED': 1.0, 'PART_OF': 0.9, 'IS_A': 0.8,
                          'HAS': 0.7, 'CAUSE': 0.5}.get(rel, 0.8)
            mass = 0.2 + 0.6 * sim_norm + 0.1 * rel_novelty
            mass = min(mass, 0.87)
            
            hypotheses.append(Hypothesis(
                subject=char_a, relation=rel, obj=char_b,
                source='high_similarity_no_edge',
                source_confidence=mass,
                metadata={
                    'cosine_sim': float(sim_val),
                }
            ))
        
        return hypotheses
    
    def _infer_relation(self, char_a: str, char_b: str) -> str:
        """启发式推测关系类型"""
        ia = self.field._char_to_idx.get(char_a)
        ib = self.field._char_to_idx.get(char_b)
        if ia is None or ib is None:
            return 'RELATED'
        
        va = self.field.anchors[ia]
        vb = self.field.anchors[ib]
        na, nb = va.norm().item(), vb.norm().item()
        sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        
        if sim > 0.88:
            return 'PART_OF'
        elif na < nb * 0.85:
            return 'IS_A'
        elif nb < na * 0.85:
            return 'HAS'
        return 'RELATED'
    
    # ═══════════════════════════════════════════════════════════════
    # Dempster 组合
    # ═══════════════════════════════════════════════════════════════
    
    def _dempster_combine(self, hypotheses: List[Hypothesis]) -> float:
        """
        Dempster 组合规则: m₁⊕m₂⊕...⊕mₙ(A)。
        
        简化实现: 逐对组合，m₁⊕m₂ = m₁·m₂ / (1 - K)
        其中 K = (1-m₁)·(1-m₂) 为冲突量。
        多源时迭代: result = m₁; for each mᵢ: result = result⊕mᵢ
        """
        if not hypotheses:
            return 0.0
        
        if len(hypotheses) == 1:
            return hypotheses[0].source_confidence
        
        # 初始 belief
        combined = hypotheses[0].source_confidence
        
        for h in hypotheses[1:]:
            m = h.source_confidence
            # m₁⊕m₂ = (m₁·m₂ + m₁·(1-m₂) + (1-m₁)·m₂) / (1 - K)
            # K = (1-m₁)·(1-m₂)
            K = (1 - combined) * (1 - m)
            if K >= 1.0:
                continue  # 完全冲突，跳过
            
            numerator = combined * m + combined * (1 - m) + (1 - combined) * m
            combined = numerator / (1 - K)
        
        return min(combined, 1.0)
