#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗扰动引擎 — 自对抗鲁棒性检测
================================

核心思想: "以扰动测脆性，以脆性定位虚假关联"

流程:
  1. 从94117锚点中采样2000子集→计算余弦相似度→取最低P10的远距对
  2. 向景观参数注入高斯噪声(per-param scaled) 
  3. 测量远距对中点能量在扰动前后的变化
  4. 数据驱动阈值过滤: P10能量异常低 + P5能量异常下降
  5. 提交D-S模糊格验证
  6. 虚假关联→负Hebbian修正(unlearn_chars)

插入点: daemon_tick_v2 步骤2.5 (学习注入后, 衰减前)
频率: 每5轮

依赖: FreqEnergyLandscape, HanziAnchorField, FuzzyGraph, Learner
"""

import time
import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger('perturbation')


@dataclass
class PerturbationCandidate:
    """扰动检测到的可疑候选"""
    idx_a: int
    idx_b: int
    char_a: str = ""
    char_b: str = ""
    score: float = 0.0        # 综合可疑评分 (越高越可疑)
    energy_before: float = 0.0
    energy_after: float = 0.0
    delta_E: float = 0.0
    cosine_sim: float = 0.0
    
    def __repr__(self):
        return (f"Candidate({self.char_a}-{self.char_b}, "
                f"score={self.score:.2f}, ΔE={self.delta_E:.2f})")


@dataclass 
class PerturbationReport:
    """扰动引擎运行报告"""
    n_distant_pairs: int = 0
    n_candidates: int = 0
    n_corrected: int = 0         # 负Hebbian修正数
    n_verified: int = 0          # D-S验证通过数
    fragility_score: float = 0.0  # 整体脆性指标
    energy_low_thresh: float = 0.0
    energy_drop_thresh: float = 0.0
    elapsed: float = 0.0
    candidates: list = field(default_factory=list)  # ★ 候选列表供 D-S 生成器消费
    
    def __repr__(self):
        return (f"PerturbReport({self.n_candidates} candidates, "
                f"{self.n_corrected} corrected, "
                f"fragility={self.fragility_score:.3f})")


class PerturbationEngine:
    """
    对抗扰动引擎。
    
    检测能量景观的脆性区域——远距字对在中点异常低能或扰动后异常下降。
    """
    
    # 可调参数 (数据驱动的百分位基准)
    ENERGY_PCTL = 10.0       # 能量异常低的百分位 (P10)
    ENERGY_DROP_PCTL = 5.0   # 能量异常下降的百分位 (P5)
    PERTURB_STD = 0.01       # 噪声标准差 (相对于参数范数)
    N_SUBSET = 2000          # 锚点子集大小
    N_DISTANT_PAIRS = 200    # 远距对数量
    MAX_CANDIDATES = 30      # 最多候选数
    
    def __init__(self, field, landscape, learner=None, fuzzy=None, cg=None):
        self.field = field
        self.landscape = landscape
        self.learner = learner
        self.fuzzy = fuzzy
        self.cg = cg
        self.device = next(landscape.parameters()).device
    
    # ═══════════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════════
    
    def run(self) -> PerturbationReport:
        """
        执行一次完整的扰动检测→D-S验证→修正循环。
        """
        t0 = time.time()
        report = PerturbationReport()
        
        # 1. 采样远距字对
        pair_indices, midpoints, similarities = self._sample_distant_pairs()
        n = len(pair_indices)
        report.n_distant_pairs = n
        if n == 0:
            report.elapsed = time.time() - t0
            return report
        
        # 2. 扰动前的能量基线
        E_before = self._compute_midpoint_energies(midpoints)
        
        # 3. 注入对抗扰动
        param_backup = self._perturb_params()
        
        # 4. 扰动后的能量
        E_after = self._compute_midpoint_energies(midpoints)
        
        # 5. 恢复参数
        self._restore_params(param_backup)
        
        # 6. 数据驱动阈值
        delta_E = E_after - E_before  # 负值=下降
        
        en = E_before.cpu().numpy()
        de = delta_E.cpu().numpy()
        
        energy_low_thresh = float(np.percentile(en, self.ENERGY_PCTL))
        energy_drop_thresh = float(np.percentile(de, self.ENERGY_DROP_PCTL))
        report.energy_low_thresh = energy_low_thresh
        report.energy_drop_thresh = energy_drop_thresh
        
        # 7. 筛选候选
        candidates = []
        for i in range(n):
            eb = en[i]
            ded = de[i]
            
            # 条件A: 能量异常低 (远距对已有低能盆地)
            cond_a = eb < energy_low_thresh
            # 条件B: 扰动后能量异常下降 (脆性)
            cond_b = ded < energy_drop_thresh
            
            if cond_a or cond_b:
                score = -eb - ded * 2.0  # 越低能+越大降幅→越可疑
                
                ia, ib = pair_indices[i][0].item(), pair_indices[i][1].item()
                candidates.append(PerturbationCandidate(
                    idx_a=ia, idx_b=ib,
                    char_a=self.field.hanzi_list[ia],
                    char_b=self.field.hanzi_list[ib],
                    score=score,
                    energy_before=eb, energy_after=E_after[i].item(),
                    delta_E=ded, cosine_sim=similarities[i].item(),
                ))
        
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:self.MAX_CANDIDATES]
        report.n_candidates = len(candidates)
        report.candidates = candidates  # ★ 暴露给 D-S 生成器
        
        if not candidates:
            report.elapsed = time.time() - t0
            return report
        
        # 8. D-S验证 + 修正
        report.n_corrected, report.n_verified = self._verify_and_correct(candidates)
        
        # 9. 脆性评分
        report.fragility_score = float(np.mean(np.abs(de)) / max(np.std(de), 1e-8))
        
        report.elapsed = time.time() - t0
        log.info(f"  扰动引擎: {report.n_distant_pairs}远距对 → "
                f"{report.n_candidates}候选 → {report.n_corrected}修正 "
                f"({report.elapsed:.1f}s)")
        return report
    
    # ═══════════════════════════════════════════════════════════════
    # 远距对采样
    # ═══════════════════════════════════════════════════════════════
    
    def _sample_distant_pairs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从锚点中采样语义距离远的字对 (低余弦相似度)。
        
        Returns:
            pair_indices: (N, 2) 全局锚点索引
            midpoints: (N, 1024) L2归一化中点
            similarities: (N,) 余弦相似度
        """
        anchors = self.field.anchors.to(self.device)
        n_total = anchors.shape[0]
        
        # 随机子集
        subset_idx = torch.randperm(n_total)[:self.N_SUBSET]
        subset = anchors[subset_idx]
        subset_norm = F.normalize(subset, p=2, dim=1)
        
        # 相似度矩阵 (2000×2000, ~4M entries, GPU ~5ms)
        sim_matrix = subset_norm @ subset_norm.T
        
        # 上三角 (排除自对) — 索引必须在同一设备
        triu_idx = torch.triu_indices(
            self.N_SUBSET, self.N_SUBSET, offset=1,
            device=self.device
        )
        triu_sims = sim_matrix[triu_idx[0], triu_idx[1]]
        
        # 取最低相似度的百分位 → 远距对
        threshold = torch.quantile(triu_sims, self.ENERGY_PCTL / 100.0)
        distant_mask = triu_sims <= threshold
        distant_sims = triu_sims[distant_mask]
        distant_pairs = triu_idx[:, distant_mask]
        
        if distant_pairs.shape[1] == 0:
            return (
                torch.empty(0, 2, dtype=torch.long),
                torch.empty(0, 1024),
                torch.empty(0),
            )
        
        # 随机采样 n_pairs
        n_available = distant_pairs.shape[1]
        n_sample = min(self.N_DISTANT_PAIRS, n_available)
        sample_idx = torch.randperm(n_available, device=self.device)[:n_sample]
        distant_pairs = distant_pairs[:, sample_idx]
        distant_sims = distant_sims[sample_idx]
        
        # 映射回全局索引 → 移到 CPU 索引 CPU 的 subset_idx
        dp_cpu = distant_pairs.cpu()
        global_a = subset_idx[dp_cpu[0]]
        global_b = subset_idx[dp_cpu[1]]
        pair_indices = torch.stack([global_a, global_b], dim=1)
        
        # 中点向量 — 索引移到 GPU 再索引锚点
        midpoints = F.normalize(
            (anchors[global_a.to(self.device)] + 
             anchors[global_b.to(self.device)]) / 2,
            p=2, dim=1,
        )
        
        return pair_indices, midpoints, distant_sims
    
    # ═══════════════════════════════════════════════════════════════
    # 扰动操作
    # ═══════════════════════════════════════════════════════════════
    
    def _perturb_params(self) -> Dict:
        """注入对抗扰动到能量景观参数，返回备份"""
        backup = {}
        with torch.no_grad():
            for name, param in self.landscape.named_parameters():
                backup[name] = param.data.clone()
                # Per-param scaled noise: σ * ||param|| / sqrt(numel)
                scale = self.PERTURB_STD * param.data.norm() / max(
                    param.data.numel() ** 0.5, 1.0
                )
                noise = torch.randn_like(param.data) * scale
                param.data.add_(noise)
        return backup
    
    def _restore_params(self, backup: Dict):
        """恢复原始参数"""
        with torch.no_grad():
            for name, param in self.landscape.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    
    # ═══════════════════════════════════════════════════════════════
    # 能量计算
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_midpoint_energies(self, midpoints: torch.Tensor) -> torch.Tensor:
        """批量计算中点能量"""
        if midpoints.shape[0] == 0:
            return torch.empty(0)
        
        with torch.no_grad():
            return self.landscape.energy(midpoints.to(self.device)).cpu()
    
    # ═══════════════════════════════════════════════════════════════
    # D-S验证 + 修正
    # ═══════════════════════════════════════════════════════════════
    
    def _verify_and_correct(self, candidates: List[PerturbationCandidate]
                            ) -> Tuple[int, int]:
        """
        D-S模糊格验证候选。虚假关联→负Hebbian修正。
        
        Returns: (corrected, verified)
        """
        corrected = 0
        verified = 0
        
        for cand in candidates:
            # 查询概念图: 这对字是否已有边?
            has_edge = self._check_edge(cand.char_a, cand.char_b)
            
            if has_edge:
                # 已有边 → 可能是合法知识, 标记验证
                verified += 1
                continue
            
            # 无已知边 + 远距 → 虚假关联 → 负Hebbian修正
            if self.learner and hasattr(self.learner, 'unlearn_chars'):
                try:
                    self.learner.unlearn_chars(
                        cand.char_a, cand.char_b, strength=0.3
                    )
                    corrected += 1
                except Exception as e:
                    log.debug(f"  unlearn 失败 ({cand.char_a},{cand.char_b}): {e}")
        
        return corrected, verified
    
    def _check_edge(self, char_a: str, char_b: str) -> bool:
        """检查概念图中是否存在边 (双向)"""
        # 优先使用直接 cg 引用 (PerturbationEngine 构造时传入)
        cg = self.cg
        if cg is None:
            # 回退: 从 fuzzy 间接获取 (兼容旧构造)
            cg = getattr(self.fuzzy, 'cg', None) if self.fuzzy else None
        if cg is None:
            return False
        
        fwd = getattr(cg, 'forward_index', {})
        return (
            char_a in fwd.get(char_b, {}) or
            char_b in fwd.get(char_a, {})
        )
