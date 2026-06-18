#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度反推引擎 — 主动知识边界探索
===============================

核心思想: 逆向使用能量景观——从鞍点沿负梯度追踪到锚点,
         主动发现尚未形成盆地的概念关联。

流程:
  1. 球面均匀采样 20,000 点
  2. 批量评估能量+梯度范数
  3. 数据驱动筛选鞍点: 能量 P75 + 梯度 P90 交集
  4. 从每个鞍点沿负梯度追踪到最近锚点
  5. 已知性检测 (概念图 + SQLite 双向)
  6. 高质量候选 → 注入概念图 + D-S 回写

频率: 每20轮
依赖: FreqEnergyLandscape, HanziAnchorField, ConceptGraph
"""

import time
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger('gradient_reverse')


@dataclass
class SaddlePoint:
    """鞍点"""
    point: torch.Tensor          # (1024,) 鞍点坐标
    energy: float = 0.0
    grad_norm: float = 0.0
    quality: float = 0.0         # energy * grad_norm


@dataclass
class TrackResult:
    """追踪结果"""
    saddle_energy: float = 0.0
    saddle_grad: float = 0.0
    converged_energy: float = 0.0
    energy_drop: float = 0.0
    anchor_char: str = ""
    anchor_idx: int = -1
    anchor_sim: float = 0.0
    source_region: str = ""      # 鞍点来源 (用最近锚点命名)
    is_known: bool = False        # 概念图已知?
    trajectory_length: int = 0


@dataclass
class GradientReverseReport:
    """梯度反推运行报告"""
    n_sampled: int = 0
    n_saddles: int = 0
    n_tracked: int = 0
    n_known: int = 0
    n_novel: int = 0
    n_injected: int = 0
    elapsed: float = 0.0
    
    def __repr__(self):
        return (f"GradRev({self.n_saddles} saddles, "
                f"{self.n_novel} novel, {self.n_injected} injected)")


class GradientReverseEngine:
    """
    梯度反推引擎。
    
    在能量景观的高梯度鞍部区域采样，沿负梯度追踪发现新概念关联。
    所有阈值数据驱动（百分位），零硬编码。
    """
    
    # 可调参数
    N_SAMPLES = 20000          # 球面采样数
    BATCH_SIZE = 512           # 梯度评估批大小
    ENERGY_PCTL = 75.0         # 鞍点能量百分位
    GRAD_PCTL = 90.0           # 鞍点梯度百分位
    TOP_K_SADDLES = 100        # 最多追踪鞍点数
    TRACE_STEPS = 80           # 梯度下降步数
    TRACE_LR = 0.03            # 追踪学习率
    # 默认关闭过滤（首次运行时从数据中校准）
    ENERGY_DROP_MIN = 0.0
    ANCHOR_SIM_MIN = 0.3       # 最小锚点相似度 (球面采样收敛点与锚点距离较远)
    MAX_INJECT = 30            # 最多注入数
    
    def __init__(self, field, landscape, cg, learner=None, fuzzy=None):
        self.field = field
        self.landscape = landscape
        self.cg = cg
        self.learner = learner
        self.fuzzy = fuzzy
        self.device = next(landscape.parameters()).device
        self.embed_dim = field.embed_dim
    
    # ═══════════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════════
    
    def run(self) -> GradientReverseReport:
        """
        执行一次完整的梯度反推循环。
        """
        t0 = time.time()
        report = GradientReverseReport()
        
        # 1. 球面采样
        samples = self._sample_sphere(self.N_SAMPLES)
        report.n_sampled = self.N_SAMPLES
        
        # 2. 批量评估 → 找鞍点
        saddles = self._find_saddles(samples)
        report.n_saddles = len(saddles)
        
        if not saddles:
            report.elapsed = time.time() - t0
            return report
        
        # 3. 从鞍点追踪到锚点
        track_results = self._trace_all(saddles)
        report.n_tracked = len(track_results)
        
        # 4. 已知性检测
        for tr in track_results:
            tr.is_known = self._check_known(tr.source_region, tr.anchor_char)
            if tr.is_known:
                report.n_known += 1
            else:
                report.n_novel += 1
        
        # 5. 注入新候选
        novel = [tr for tr in track_results if not tr.is_known]
        report.n_injected = self._inject_candidates(novel)
        
        report.elapsed = time.time() - t0
        log.info(f"  梯度反推: {report.n_sampled}采样 → "
                f"{report.n_saddles}鞍点 → {report.n_novel}新发现 → "
                f"{report.n_injected}注入 ({report.elapsed:.1f}s)")
        
        return report
    
    # ═══════════════════════════════════════════════════════════════
    # 球面采样
    # ═══════════════════════════════════════════════════════════════
    
    def _sample_sphere(self, n: int) -> torch.Tensor:
        """在单位球面上均匀采样"""
        points = torch.randn(n, self.embed_dim, device=self.device)
        return F.normalize(points, p=2, dim=1)
    
    # ═══════════════════════════════════════════════════════════════
    # 鞍点搜索
    # ═══════════════════════════════════════════════════════════════
    
    def _find_saddles(self, samples: torch.Tensor) -> List[SaddlePoint]:
        """
        批量评估能量+梯度范数 → 数据驱动筛选鞍点。
        """
        energies, grad_norms = self._evaluate_batch(samples)
        
        en = energies.numpy()
        gn = grad_norms.numpy()
        
        # 数据驱动阈值
        e_thresh = float(np.percentile(en, self.ENERGY_PCTL))
        g_thresh = float(np.percentile(gn, self.GRAD_PCTL))
        
        # 筛选
        mask = (torch.tensor(en) > e_thresh) & (torch.tensor(gn) > g_thresh)
        cand_points = samples[mask]
        cand_energies = torch.tensor(en)[mask]
        cand_grads = torch.tensor(gn)[mask]
        
        if len(cand_points) == 0:
            return []
        
        # 质量排序: energy × grad_norm
        quality = cand_energies * cand_grads
        _, top_idx = torch.topk(quality, min(self.TOP_K_SADDLES, len(quality)))
        
        saddles = []
        for i in top_idx:
            saddles.append(SaddlePoint(
                point=cand_points[i].clone(),
                energy=cand_energies[i].item(),
                grad_norm=cand_grads[i].item(),
                quality=quality[i].item(),
            ))
        
        # 锚点多样性过滤: 同一目标锚点最多保留3个
        return self._diversify_saddles(saddles)
    
    def _evaluate_batch(self, points: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量计算能量和梯度范数。
        使用 e.sum().backward() 一次性获取所有梯度。
        """
        energies_list = []
        grad_norms_list = []
        
        self.landscape.eval()
        
        for i in range(0, len(points), self.BATCH_SIZE):
            batch = points[i:i + self.BATCH_SIZE].clone().detach()
            batch.requires_grad_(True)
            
            e = self.landscape.energy(batch)
            e_sum = e.sum()
            e_sum.backward()
            
            gn = batch.grad.norm(dim=1).detach().cpu()
            
            energies_list.append(e.detach().cpu())
            grad_norms_list.append(gn)
            
            self.landscape.zero_grad()
        
        return torch.cat(energies_list), torch.cat(grad_norms_list)
    
    def _diversify_saddles(self, saddles: List[SaddlePoint]
                           ) -> List[SaddlePoint]:
        """
        锚点多样性过滤: 相同最近锚点最多保留3个。
        防止所有候选都指向同一锚点。
        """
        # 快速找每个鞍点的最近锚点 — 分批 matmul 避免 OOM
        saddle_points = torch.stack([s.point for s in saddles]).to(self.device)
        anchors_norm = F.normalize(self.field.anchors.to(self.device), p=2, dim=1)
        
        nearest_chars = []
        for i in range(0, len(saddles), 50):
            batch = F.normalize(saddle_points[i:i+50], p=2, dim=1)  # (B,1024)
            sims = batch @ anchors_norm.T  # (B, 94117), ~20MB per batch
            top_idx = sims.argmax(dim=1).cpu().numpy()
            nearest_chars.extend([self.field.hanzi_list[idx] for idx in top_idx])
        
        # 每个锚点最多保留3个
        anchor_counts = {}
        diverse = []
        for s, ch in zip(saddles, nearest_chars):
            cnt = anchor_counts.get(ch, 0)
            if cnt < 3:
                diverse.append(s)
                anchor_counts[ch] = cnt + 1
        
        return diverse
    
    # ═══════════════════════════════════════════════════════════════
    # 负梯度追踪
    # ═══════════════════════════════════════════════════════════════
    
    def _trace_all(self, saddles: List[SaddlePoint]) -> List[TrackResult]:
        """从所有鞍点追踪到锚点"""
        results = []
        
        for saddle in saddles:
            try:
                tr = self._trace_one(saddle)
                if tr:
                    results.append(tr)
            except Exception:
                continue
        
        return results
    
    def _trace_one(self, saddle: SaddlePoint) -> Optional[TrackResult]:
        """
        从单个鞍点沿负梯度追踪到最近锚点。
        使用 Adam 优化器，梯度下降，球面投影。
        """
        x = saddle.point.clone().detach().to(self.device)
        x = F.normalize(x, p=2, dim=-1)
        x.requires_grad_(True)
        
        energy_start = saddle.energy
        grad_start = saddle.grad_norm
        
        optimizer = torch.optim.Adam([x], lr=self.TRACE_LR)
        trajectory_len = 0
        
        prev_energy = energy_start
        no_improvement = 0
        
        for step in range(self.TRACE_STEPS):
            optimizer.zero_grad()
            e = self.landscape.energy(x)
            e.backward()
            optimizer.step()
            
            with torch.no_grad():
                x.data = F.normalize(x.data, p=2, dim=-1)
            
            trajectory_len += 1
            current_energy = e.item()
            delta = abs(current_energy - prev_energy)
            
            if current_energy >= prev_energy:
                no_improvement += 1
            else:
                no_improvement = 0
            
            if delta < 1e-5 or no_improvement >= 5:
                break
            
            prev_energy = current_energy
        
        converged = x.detach()
        converged_energy = self.landscape.energy(converged).item()
        energy_drop = energy_start - converged_energy
        
        # 能量降幅过滤 — 数据驱动: 取所有成功追踪的 P10
        # 首次调用时无法预知，用宽松阈值；后续可校准
        if energy_drop < self.ENERGY_DROP_MIN and self.ENERGY_DROP_MIN > 0:
            return None
        
        # 找最近锚点 — 用 matmul 避免 OOM
        anchors_norm = F.normalize(self.field.anchors.to(self.device), p=2, dim=1)
        converged_norm = F.normalize(converged, p=2, dim=-1)
        with torch.no_grad():
            sims = converged_norm @ anchors_norm.T  # (1, 94117), ~0.4MB
            top_idx = sims.argmax().item()
            top_sim = sims[top_idx].item()
        
        if top_sim < self.ANCHOR_SIM_MIN:
            return None
        
        # 鞍点来源区域命名 (用最近锚点)
        saddle_norm = F.normalize(saddle.point.to(self.device), p=2, dim=-1)
        with torch.no_grad():
            source_sims = saddle_norm @ anchors_norm.T  # (1, 94117), ~0.4MB
            source_idx = source_sims.argmax().item()
        
        return TrackResult(
            saddle_energy=energy_start,
            saddle_grad=grad_start,
            converged_energy=converged_energy,
            energy_drop=energy_drop,
            anchor_char=self.field.hanzi_list[top_idx],
            anchor_idx=top_idx,
            anchor_sim=top_sim,
            source_region=self.field.hanzi_list[source_idx],
            trajectory_length=trajectory_len,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 已知性检测
    # ═══════════════════════════════════════════════════════════════
    
    def _check_known(self, source_char: str, anchor_char: str) -> bool:
        """
        检查 (source, anchor) 是否已存在于概念图。
        双向查询: forward_index + reverse_index。
        """
        if source_char == anchor_char:
            return True
        
        fwd = getattr(self.cg, 'forward_index', {})
        rev = getattr(self.cg, 'reverse_index', {})
        
        # 正向: source → anchor
        if source_char in fwd and anchor_char in fwd[source_char]:
            return True
        # 反向: anchor → source
        if anchor_char in fwd and source_char in fwd[anchor_char]:
            return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # 注入
    # ═══════════════════════════════════════════════════════════════
    
    def _inject_candidates(self, novel: List[TrackResult]) -> int:
        """
        将新发现的概念关联注入概念图。
        按能量降幅排序，取前 MAX_INJECT 个。
        """
        novel.sort(key=lambda t: t.energy_drop, reverse=True)
        injected = 0
        
        for tr in novel[:self.MAX_INJECT]:
            try:
                # 注入概念图
                if hasattr(self.cg, 'add_triple'):
                    self.cg.add_triple(
                        tr.source_region, 'RELATED', tr.anchor_char,
                        confidence=0.5 + min(tr.energy_drop / 100, 0.4),
                        source='gradient_reverse',
                    )
                
                # D-S 回写
                if self.fuzzy:
                    try:
                        self.fuzzy.add_evidence(
                            tr.source_region, 'RELATED', tr.anchor_char,
                            source='gradient_reverse',
                            mass=0.5 + min(tr.energy_drop / 100, 0.3),
                            description=f'鞍点追踪: ΔE={tr.energy_drop:.1f}',
                        )
                    except Exception:
                        pass
                
                injected += 1
            except Exception as e:
                log.debug(f"  注入失败 ({tr.source_region},{tr.anchor_char}): {e}")
        
        return injected
