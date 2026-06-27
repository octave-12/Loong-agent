#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 好奇心引擎 — 内在驱动的探索系统
═══════════════════════════════════════════════════════

替代旧的盲区扫描 (CognitiveTerrain)，不是被动等盲区信号，
而是主动检测「场中波动最剧烈但盆地最浅的区域」→ 未知的未知。

核心指标:
  知识好奇心 = 高激活方差 + 低盆地深度 + 高频激活
    - 高方差: 对这个区域的认知不稳定 (每次想都不一样)
    - 浅盆地: 还没形成稳固的概念 (梯度平坦)
    - 高频激活: 经常被问到/涉及 (用户关心的)

探索动作:
  1. 好奇心评分 → 选出 top-N 待探索锚点
  2. 矛盾解主动制造对抗扰动 (contra + perturbation)
  3. D-S 裁决评估是否形成新盆地
  4. 稳定盆地 → 生成假设三元组 → 标记「待外部验证」
"""

import torch
import math
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field as dc_field

from .dragon_field import DragonField, FieldResult

log = logging.getLogger(__name__)


@dataclass
class CuriositySignal:
    """好奇心信号 — 指向值得探索的区域"""
    anchor_idx: int           # 锚点索引 (字场中的位置)
    anchor_char: str          # 对应的汉字
    curiosity_score: float    # 综合得分 (越高越值得探索)
    fluctuation: float        # MC方差 (认知不稳定度)
    basin_shallowness: float  # 盆地浅度 (1 - depth_norm)
    activation_count: int     # 最近激活次数

    # 探索建议
    suggested_action: str     # 'perturb' | 'verify' | 'learn'
    related_triple_ids: List[int] = dc_field(default_factory=list)


class CuriosityEngine:
    """
    内在驱动的好奇心引擎。

    不是「扫描盲区」而是「感知场的波动」。
    场中不稳定但频繁被触达的区域 → 认知食欲的指向。
    """

    def __init__(
        self,
        field: DragonField,
        hanzi_list: Optional[List[str]] = None,
        device: str = 'cuda',
    ):
        self.field = field
        self.hanzi_list = hanzi_list or []
        self.device = device

        # 每个模式的探索状态追踪
        self.register_exploration_state()

        # 参数
        self.curiosity_decay = 0.95        # 探索后衰减
        self.min_fluctuation = 0.005       # 最小方差阈值
        self.mc_samples = 8                # MC采样次数
        self.proactive_perturb_prob = 0.3  # 主动扰动概率

        # 统计
        self._total_explorations = 0
        self._discoveries = 0

    def register_exploration_state(self):
        """注册探索状态追踪张量"""
        N = self.field.num_patterns
        field = self.field

        if not hasattr(field, '_curiosity_score'):
            field.register_buffer(
                '_curiosity_score', torch.zeros(N)
            )
        if not hasattr(field, '_last_explored_round'):
            field.register_buffer(
                '_last_explored_round', torch.zeros(N, dtype=torch.int32)
            )
        if not hasattr(field, '_fluctuation_history'):
            field.register_buffer(
                '_fluctuation_history', torch.zeros(N)
            )

    # ── 好奇心评分 ────────────────────────────────────────────

    def score_curiosity(
        self,
        n_samples: int = 50,
        round_number: int = 0,
    ) -> List[CuriositySignal]:
        """
        评估场的「认知食欲」— 找最值得探索的区域。

        采样随机锚点 → MC Dropout 测波动 → 综合评分。

        Args:
            n_samples: 采样锚点数
            round_number: 当前轮次 (用于衰减)

        Returns:
            按好奇心得分降序的探索信号列表
        """
        N = self.field.num_patterns
        if N == 0:
            return []

        # 采样 (不全量扫, 省计算)
        sample_n = min(n_samples * 3, N)
        indices = torch.randperm(N)[:sample_n]

        signals = []
        patterns = self.field.patterns.to(dtype=torch.float32)

        for idx in indices.tolist():
            anchor_vec = patterns[idx].to(device=self.device)

            # MC Dropout 测波动
            vecs = []
            for _ in range(self.mc_samples):
                noise = torch.randn_like(anchor_vec) * 0.03
                noisy = anchor_vec + noise
                noisy = torch.nn.functional.normalize(noisy, dim=-1)

                result = self.field.converge(
                    noisy, patterns=patterns.to(self.device),
                    max_steps=10, convergence_threshold=1e-3,
                )
                vecs.append(result.convergent_vector)

            # 波动 = 多次收敛结果的方差
            stacked = torch.stack(vecs)  # (S, D)
            fluctuation = ((stacked - stacked.mean(0)) ** 2).mean().item()

            if fluctuation < self.min_fluctuation:
                continue

            # 盆地浅度 = 平均能量的倒数
            avg_energy = 0.0
            for v in vecs:
                e = self.field.energy(v.to(self.device), patterns.to(self.device))
                avg_energy += e.item()
            avg_energy /= self.mc_samples
            basin_shallowness = float(
                1.0 / (1.0 + math.exp(-avg_energy / 10.0))
            )

            # 激活次数 (归一化)
            act_count = self.field._activation_count[idx].item()
            act_norm = min(act_count / 10.0, 1.0)

            # 综合得分: 波动 * 盆地浅度 * 激活热度
            score = fluctuation * basin_shallowness * (0.5 + 0.5 * act_norm)

            # 衰减: 最近探索过就抑制
            if self.field._last_explored_round[idx] > 0:
                rounds_since = round_number - self.field._last_explored_round[idx].item()
                decay = self.curiosity_decay ** max(rounds_since, 0)
                score *= decay

            if score > 0.001:
                char = (
                    self.hanzi_list[idx] if idx < len(self.hanzi_list)
                    else f"#{idx}"
                )
                signals.append(CuriositySignal(
                    anchor_idx=idx,
                    anchor_char=char,
                    curiosity_score=score,
                    fluctuation=fluctuation,
                    basin_shallowness=basin_shallowness,
                    activation_count=act_count,
                    suggested_action=self._suggest_action(
                        fluctuation, basin_shallowness
                    ),
                ))

        # 排序: 最值得探索的在前
        signals.sort(key=lambda s: s.curiosity_score, reverse=True)
        return signals[:n_samples]

    def _suggest_action(
        self, fluctuation: float, shallowness: float
    ) -> str:
        """根据波动特征建议探索动作"""
        if fluctuation > 0.05 and shallowness < 0.3:
            return 'learn'       # 高波动+浅盆地 → 需要学习新知识
        elif fluctuation > 0.02:
            return 'perturb'     # 中等波动 → 制造扰动看能否稳定
        else:
            return 'verify'      # 低波动 → 已稳定, 验证是否正确

    # ── 主动探索 ────────────────────────────────────────────

    def explore(
        self,
        signal: CuriositySignal,
        round_number: int = 0,
    ) -> Dict[str, Any]:
        """
        对好奇心信号执行主动探索。

        返回探索结果: 是否发现新盆地, 候选三元组等。
        """
        self._total_explorations += 1
        idx = signal.anchor_idx
        patterns = self.field.patterns.to(dtype=torch.float32)
        anchor = patterns[idx].to(device=self.device)

        result = {'action': signal.suggested_action, 'discovery': False}

        if signal.suggested_action == 'perturb':
            # 主动对抗扰动 — 沿梯度反方向推
            perturbed = self._proactive_perturb(anchor, patterns)
            if perturbed is not None:
                result['perturbed_vector'] = perturbed.cpu()
                result['discovery'] = True

        elif signal.suggested_action == 'learn':
            # 标记为待学习 → 交给知识管线
            result['needs_learning'] = True
            result['anchor_char'] = signal.anchor_char
            result['discovery'] = True

        elif signal.suggested_action == 'verify':
            # 验证已有盆地的稳定性
            mc_result = self.field.mc_uncertainty(
                anchor, n_samples=8, dropout_rate=0.15,
                patterns=patterns.to(self.device),
            )
            result['stability'] = mc_result['is_stable']
            result['variance'] = mc_result['variance']

        # 更新探索状态
        self.field._curiosity_score[idx] = signal.curiosity_score
        self.field._last_explored_round[idx] = round_number
        self.field._fluctuation_history[idx] = signal.fluctuation

        if result.get('discovery'):
            self._discoveries += 1

        return result

    def _proactive_perturb(
        self,
        anchor: torch.Tensor,
        patterns: torch.Tensor,
        strength: float = 0.1,
        n_directions: int = 5,
    ) -> Optional[torch.Tensor]:
        """
        主动对抗扰动: 向多个方向微推 → 检查是否形成新盆地。
        """
        device = anchor.device
        pat_gpu = patterns.to(device)

        base_result = self.field.converge(
            anchor, patterns=pat_gpu, max_steps=10,
        )

        for _ in range(n_directions):
            # 随机扰动方向
            direction = torch.randn_like(anchor)
            direction = torch.nn.functional.normalize(direction, dim=-1)
            perturbed = anchor + direction * strength

            result = self.field.converge(
                perturbed, patterns=pat_gpu, max_steps=10,
            )

            # 如果扰动后收敛到不同的盆地 (距离 > 0.3)
            dist = (
                1.0 - torch.nn.functional.cosine_similarity(
                    base_result.convergent_vector.unsqueeze(0),
                    result.convergent_vector.unsqueeze(0),
                )
            ).item()

            if dist > 0.3 and result.basin_depth > base_result.basin_depth:
                # 发现了更深的新盆地!
                return result.convergent_vector

        return None

    # ── 场驱动调度 ────────────────────────────────────────────

    def should_tick(self) -> float:
        """
        场状态驱动调度: 返回 0~1 的「活跃度」。
        替代定时器 — 场活跃时频繁执行，场稳定时降低频率。

        Returns:
            0.0 (场完全稳定, 不需要做任何事)
            ~
            1.0 (场非常活跃, 需要立即处理)
        """
        if self.field.num_patterns == 0:
            return 0.0

        # 计算场中高波动模式的占比
        N = self.field.num_patterns
        curiosity = self.field._curiosity_score
        if curiosity.numel() == 0:
            return 0.1  # 冷启动, 稍微活跃

        # 好奇心得分 > 0.01 的模式占比
        active_ratio = (curiosity > 0.01).float().mean().item()

        # 最近好奇心得分的平均值
        mean_curiosity = curiosity.mean().item()

        # 综合活跃度
        activity = 0.3 * active_ratio + 0.7 * min(mean_curiosity * 10, 1.0)
        return max(min(activity, 1.0), 0.01)

    def recommended_interval(self) -> float:
        """
        根据场活跃度推荐下次守护循环间隔 (秒)。
        活跃度高 → 间隔短; 活跃度低 → 间隔长。
        """
        activity = self.should_tick()
        if activity > 0.5:
            return 15.0   # 高频: 15秒
        elif activity > 0.2:
            return 60.0   # 中频: 1分钟
        elif activity > 0.05:
            return 300.0  # 低频: 5分钟
        else:
            return 600.0  # 几乎休眠: 10分钟

    # ── 统计 ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            'total_explorations': self._total_explorations,
            'discoveries': self._discoveries,
            'activity': self.should_tick(),
            'recommended_interval_s': self.recommended_interval(),
        }
