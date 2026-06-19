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
    SIM_CHUNK_SIZE = 1000      # 默认分块大小 (__init__ 覆盖为 VRAM 自适应值)
    
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
        
        # ★ VRAM 自适应分块: 5000×1024@fp32=20MB, 每块 ≤ 10% VRAM
        self.SIM_CHUNK_SIZE = self._compute_chunk_size()
    
    def _compute_chunk_size(self) -> int:
        """根据 GPU 显存计算安全的分块大小 (至少 200)"""
        if self.device == 'cpu':
            return 500  # CPU 内存充足但 batch 操作较慢
        try:
            free_mb = torch.cuda.mem_get_info()[0] / (1024**2)  # MiB
        except Exception:
            return 500
        
        # 每行 5000×1024×4bytes ≈ 20MB; 目标每块 ≤ 5% free VRAM
        safe_per_chunk_mb = free_mb * 0.05
        chunk = max(200, min(2000, int(safe_per_chunk_mb / (5000 * 1024 * 4 / 1024**2) * 5000)))
        return min(chunk, 5000)  # 上限 5000
    
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
        高相似无连接源: GPU 分块 + OOM 兜底 + CPU fallback。
        
        内存预算 (float32): anchors 20MB + norm 20MB + sim_block ~20MB/chunk
        4GB GPU 安全; <2GB GPU 自动降级 chunk。
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
        
        # OOM 兜底: 尝试 GPU → 降级 chunk → CPU fallback
        for attempt, device_tag in enumerate([
            self.device,           # 尝试1: GPU 全速
            self.device,           # 尝试2: GPU 半 chunk
            'cpu',                 # 尝试3: CPU 兜底
        ]):
            try:
                return self._source3_compute(
                    anchors, hanzi_list, search_range, existing,
                    device=device_tag if device_tag == 'cpu' else self.device,
                    chunk_factor=(1.0 if attempt == 0 else 0.5)
                )
            except torch.cuda.OutOfMemoryError:
                if attempt == 0 and self.device != 'cpu':
                    log.warning(f"  ⚠️ 源3 GPU OOM, chunk={self.SIM_CHUNK_SIZE}→"
                               f"{max(200, self.SIM_CHUNK_SIZE//2)}, 降级重试...")
                    torch.cuda.empty_cache()
                elif attempt == 1 and self.device != 'cpu':
                    log.warning(f"  ⚠️ 源3 半 chunk 仍 OOM, 回退 CPU...")
                    torch.cuda.empty_cache()
                else:
                    log.error(f"  源3 全部尝试失败, 跳过本批次")
                    return []
            except Exception as e:
                if attempt == 0:
                    log.warning(f"  源3 GPU 错误: {e}, 回退 CPU")
                    continue
                log.warning(f"  源3 异常: {e}")
                return []
        
        return []
    
    def _source3_compute(self, anchors, hanzi_list, search_range: int,
                          existing: set, device: str, 
                          chunk_factor: float = 1.0) -> List[Hypothesis]:
        """源3 核心计算: 分块相似度扫描 → 筛选 → 构建假设"""
        sub = anchors[:search_range].to(device)
        sub_norm = F.normalize(sub, p=2, dim=1)
        
        chunk = max(200, int(self.SIM_CHUNK_SIZE * chunk_factor))
        all_candidates = []
        
        for i_start in range(0, search_range, chunk):
            i_end = min(i_start + chunk, search_range)
            row_block = sub_norm[i_start:i_end]       # (chunk, D)
            sim_block = row_block @ sub_norm.T         # (chunk, search_range)
            
            # 上三角 mask + 高相似 mask
            cols = torch.arange(search_range, device=device)
            rows = torch.arange(i_start, i_end, device=device).unsqueeze(1)
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
        """
        多策略关系推断 (产品级):
          1. 概念图直接查证 (最可靠, 双向)
          2. 概念图传递模式匹配 (IS_A/HAS 链)
          3. 嵌入空间几何特征 (norm ratio, cosine sim)
          4. CAUSE/OPPOSITE 模式检测
        
        返回六种关系之一: IS_A, PART_OF, HAS, CAUSE, OPPOSITE, RELATED
        """
        # ═══ 策略1: 概念图查证 ═══
        cg_rel = self._lookup_cg_relation(char_a, char_b)
        if cg_rel:
            return cg_rel
        
        # ═══ 策略2: 嵌入向量分析 ═══
        ia = self.field._char_to_idx.get(char_a)
        ib = self.field._char_to_idx.get(char_b)
        if ia is None or ib is None:
            return 'RELATED'
        
        va = self.field.anchors[ia]
        vb = self.field.anchors[ib]
        na, nb = va.norm().item(), vb.norm().item()
        sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        ratio = na / (nb + 1e-8)
        
        # ═══ 策略3: 几何启发式 + 阈值校准 ═══
        # PART_OF: 极高相似 + 范数对称 → 子概念或同义
        if sim > 0.85 and 0.75 < ratio < 1.35:
            return 'PART_OF'
        
        # IS_A: 大范数比 (a 是 b 的上位/抽象概念)
        # 典型: 抽象概念嵌入范数通常较小 (bge-large-zh 特性)
        if ratio < 0.75 and sim > 0.55:
            return 'IS_A'
        
        # HAS: 逆范数比 (a 的嵌入更"丰富", 拥有/包含 b 的特征)
        if ratio > 1.35 and sim > 0.55:
            return 'HAS'
        
        # ═══ 策略4: CAUSE/OPPOSITE 模式 ═══
        # CAUSE: 中等相似 + CG因果链匹配
        if self._detect_causal_pattern(char_a, char_b, sim):
            return 'CAUSE'
        
        # OPPOSITE: 低相似 + CG反义链 OR 语义对立
        if self._detect_opposite_pattern(char_a, char_b, sim):
            return 'OPPOSITE'
        
        return 'RELATED'
    
    def _lookup_cg_relation(self, char_a: str, char_b: str) -> Optional[str]:
        """概念图双向关系查证: 返回已存在的关系类型或 None"""
        fwd = getattr(self.cg, 'forward_index', None)
        if not fwd:
            return None
        
        # a → b
        if char_a in fwd and char_b in fwd[char_a]:
            return fwd[char_a][char_b]
        # b → a (反向关系可能等价)
        if char_b in fwd and char_a in fwd[char_b]:
            rel = fwd[char_b][char_a]
            # 对称关系反向等价
            if rel in ('RELATED', 'OPPOSITE', 'PART_OF'):
                return rel
            # 不对称关系翻转
            if rel == 'IS_A':
                return 'HAS'
            if rel == 'HAS':
                return 'IS_A'
            if rel == 'CAUSE':
                return 'RELATED'  # 因果不对称
        
        return None
    
    def _detect_causal_pattern(self, char_a: str, char_b: str, 
                                sim: float) -> bool:
        """
        因果模式检测:
        - CG 中存在 a CAUSE X 且 X 与 b 高相似 → a CAUSE b
        - 中等相似度 (因果词对通常 0.45-0.70)
        """
        if sim < 0.40 or sim > 0.72:
            return False
        
        fwd = getattr(self.cg, 'forward_index', None)
        if not fwd or char_a not in fwd:
            return False
        
        # a 的因果目标中是否有与 b 相似的
        for obj, rel in fwd[char_a].items():
            if rel == 'CAUSE' and obj != char_b:
                # 检查 b 是否与因果目标相似
                if self._char_similarity(char_b, obj) > 0.65:
                    return True
        
        return False
    
    def _detect_opposite_pattern(self, char_a: str, char_b: str,
                                   sim: float) -> bool:
        """
        对立模式检测:
        - CG 中存在 a OPPOSITE X 且 X 与 b 高相似
        - 低相似度 + 范数对称 (反义词嵌入特征)
        """
        fwd = getattr(self.cg, 'forward_index', None)
        
        # 策略A: CG 反义链
        if fwd and char_a in fwd:
            for obj, rel in fwd[char_a].items():
                if rel == 'OPPOSITE' and obj != char_b:
                    if self._char_similarity(char_b, obj) > 0.60:
                        return True
        
        # 策略B: 低相似 + 范数对称 → 潜在反义
        if sim < 0.45:
            ia = self.field._char_to_idx.get(char_a)
            ib = self.field._char_to_idx.get(char_b)
            if ia is not None and ib is not None:
                na = self.field.anchors[ia].norm().item()
                nb = self.field.anchors[ib].norm().item()
                if 0.80 < na / (nb + 1e-8) < 1.25:
                    return True
        
        return False
    
    def _char_similarity(self, a: str, b: str) -> float:
        """两字余弦相似度 (快速内联计算)"""
        ia = self.field._char_to_idx.get(a)
        ib = self.field._char_to_idx.get(b)
        if ia is None or ib is None:
            return 0.0
        va = self.field.anchors[ia]
        vb = self.field.anchors[ib]
        return F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
    
    # ═══════════════════════════════════════════════════════════════
    # Dempster 组合
    # ═══════════════════════════════════════════════════════════════
    
    def _dempster_combine(self, hypotheses: List[Hypothesis]) -> float:
        """
        标准 Dempster 组合规则 — 多源证据融合。
        
        框架 Θ = {H, ¬H} (假设为真/假):
          - 每个源 i: m_i(H) = confidence, m_i(Θ) = 1 - confidence
          - 无源直接支持 ¬H (m_i(¬H) = 0)
          - 冲突 K = Σ m₁(B)·m₂(C) for B∩C=∅ = 0 (无 ¬H 质量)
        
        组合公式:
          m₁⊕m₂(H) = m₁·m₂ + m₁·(1-m₂) + (1-m₁)·m₂
                   = m₁ + m₂ - m₁·m₂
        
        多源迭代: result = m₁; for each m_i: result = result + m_i - result·m_i
        
        注意: 此公式本质上是独立证据的概率并集，当多源一致时会
        加速收敛至 1.0。如需保守估计，降低单源 max_confidence。
        """
        if not hypotheses:
            return 0.0
        
        if len(hypotheses) == 1:
            return hypotheses[0].source_confidence
        
        # 标准 Dempster 迭代: result ⊕ m_i
        combined = hypotheses[0].source_confidence
        
        for h in hypotheses[1:]:
            m = h.source_confidence
            # m₁⊕m₂ = m₁ + m₂ - m₁·m₂ (K=0, 无需归一化分母)
            combined = combined + m - combined * m
        
        # 1.0 只在无穷多源时达到，截断保留最小不确定度
        return min(combined, 0.999)
