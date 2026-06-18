#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠学习机制（loongpearl_learner.py）—— 局部Hebbian学习与自知无知判断
========================================================================
在字场和能量景观之上，实现三大核心机制：

  1. HebbianLearner  —— 局部Hebbian学习（用进），强化被使用的知识通路
  2. SelfIgnoranceDetector —— 自知无知判断，检测系统是否"知道"某问题
  3. WeightDecayScheduler —— 全局遗忘（废退），未被强化的通路逐渐衰减

核心原理:
  - 字场是静态知识基底（永久冻结），不会被学习修改
  - 能量景观是可塑的（通过局部微调实现学习）
  - Hebbian规则: "同时激活的神经元之间的连接被强化"
    在龙珠中表现为: 查询锚点→答案锚点路径上的能量被降低

依赖: torch, numpy, zichang, energy_landscape

作者: Hermes + 李泽坤
版本: 1.0.0 (初代龙珠)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# 第一部分：激活追踪器 —— 记录每个锚点的使用频率
# ============================================================================

class ActivationTracker:
    """
    锚点激活追踪器。
    
    记录每个汉字锚点被查询/使用的频率和时间。
    这是实现"用进废退"的数据基础：
      - 高频使用的锚点 → 周围能量降低（盆地加深）
      - 长期未用的锚点 → 周围能量衰减（盆地变浅）
    """
    
    def __init__(self, num_anchors: int):
        self.num_anchors = num_anchors
        
        # 激活计数（累计使用次数）
        self.activation_counts = torch.zeros(num_anchors, dtype=torch.float32)
        
        # 最后激活时间戳
        self.last_activated = torch.zeros(num_anchors, dtype=torch.float32)
        
        # 全局时间步（每次 update 递增）
        self.global_step = 0
        
        # 衰减因子：每步衰减多少（模拟遗忘曲线）
        self.decay_rate = 0.999
    
    def record(self, anchor_indices: torch.Tensor):
        """
        记录锚点被激活。
        
        Args:
            anchor_indices: 被激活的锚点索引 (N,) 或标量
        """
        if anchor_indices.dim() == 0:
            anchor_indices = anchor_indices.unsqueeze(0)
        
        self.global_step += 1
        
        for idx in anchor_indices:
            i = idx.item()
            # 指数移动平均更新：新激活 + 旧值的衰减
            self.activation_counts[i] = 1.0 + self.activation_counts[i] * self.decay_rate
            self.last_activated[i] = float(self.global_step)
    
    def get_activity(self, anchor_indices: torch.Tensor) -> torch.Tensor:
        """获取指定锚点的当前活跃度"""
        return self.activation_counts[anchor_indices]
    
    def get_idle_time(self, anchor_indices: torch.Tensor) -> torch.Tensor:
        """获取指定锚点距离上次激活的步数"""
        return self.global_step - self.last_activated[anchor_indices]
    
    def most_active(self, k: int = 10) -> List[Tuple[int, float]]:
        """返回最活跃的k个锚点"""
        values, indices = torch.topk(self.activation_counts, min(k, self.num_anchors))
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
    
    def most_idle(self, k: int = 10) -> List[Tuple[int, float]]:
        """返回最少使用的k个锚点"""
        idle = self.global_step - self.last_activated
        values, indices = torch.topk(idle, min(k, self.num_anchors))
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]


# ============================================================================
# 第二部分：Hebbian 学习器 —— 用进
# ============================================================================

class HebbianLearner:
    """
    局部 Hebbian 学习器 —— 实现"用进"。
    
    核心思想:
      当用户确认某次查询-答案关联正确时（正反馈），在能量景观中
      降低查询锚点和答案锚点之间路径的能量，使未来的类似查询
      更容易收敛到正确答案。
    
    学习策略:
      1. 找到查询向量和答案向量在字场中的最近锚点
      2. 在两个锚点之间生成插值点（路径采样）
      3. 对这些插值点运行梯度下降，局部降低能量
      4. 只在锚点附近区域更新，不破坏全局结构
    
    这种局部更新的好处:
      - 不影响远距离锚点的能量盆地
      - 逐次累积，路径逐渐变成"高速通道"
      - 符合 Hebbian 规则（同时激活→强化连接）
    """
    
    def __init__(
        self,
        landscape: 'EnergyLandscape',      # type: ignore
        anchor_field: 'HanziAnchorField',  # type: ignore
        learning_rate: float = 0.001,
        n_interpolation_points: int = 8,
        local_radius: float = 0.05,
        max_update_steps: int = 5,
        device: str = "cpu",
    ):
        """
        初始化 Hebbian 学习器。
        
        Args:
            landscape: 能量景观实例
            anchor_field: 字场实例
            learning_rate: 学习率
            n_interpolation_points: 锚点间插值点数量
            local_radius: 局部更新半径（控制更新范围）
            max_update_steps: 每次更新最多迭代步数
            device: 计算设备
        """
        self.landscape = landscape
        self.anchors = anchor_field
        self.lr = learning_rate
        self.n_points = n_interpolation_points
        self.local_radius = local_radius
        self.max_steps = max_update_steps
        self.device = device
        
        # 激活追踪器
        self.tracker = ActivationTracker(anchor_field.num_hanzi)
        
        # 学习历史
        self.history: List[Dict] = []
        
        self.landscape.to(device)
    
    def update(
        self,
        query_vec: torch.Tensor,
        converged_vec: torch.Tensor,
        feedback: float = 1.0,
    ) -> Dict:
        """
        局部 Hebbian 更新。
        
        正向反馈（feedback > 0）: 降低路径能量 → 强化关联
        负向反馈（feedback < 0）: 提高路径能量 → 削弱关联
        
        Args:
            query_vec: 查询向量 (embed_dim,)
            converged_vec: 收敛后的答案向量 (embed_dim,)
            feedback: 反馈强度，正=强化，负=削弱
        
        Returns:
            dict: 更新统计信息
        """
        if abs(feedback) < 1e-6:
            return {'status': 'skipped', 'reason': 'feedback too small'}
        
        # 找到查询和答案的最近锚点
        q_vec = query_vec.unsqueeze(0).to(self.device)
        c_vec = converged_vec.unsqueeze(0).to(self.device)
        
        q_indices, _, _ = self.anchors.find_nearest(q_vec.cpu(), k=1)
        c_indices, _, _ = self.anchors.find_nearest(c_vec.cpu(), k=1)
        
        q_idx = q_indices[0].item()
        c_idx = c_indices[0].item()
        
        # 记录激活
        self.tracker.record(torch.tensor([q_idx, c_idx]))
        
        if q_idx == c_idx:
            return {
                'status': 'skipped',
                'reason': 'same anchor',
                'anchor': self.anchors.hanzi_list[q_idx],
            }
        
        # 获取锚点向量
        q_anchor = self.anchors.anchors[q_idx].to(self.device)
        c_anchor = self.anchors.anchors[c_idx].to(self.device)
        
        # 在查询锚点和答案锚点之间生成插值点
        alphas = torch.linspace(0.1, 0.9, self.n_points, device=self.device)
        interpolated = q_anchor.unsqueeze(0) * (1 - alphas).unsqueeze(1) + \
                       c_anchor.unsqueeze(0) * alphas.unsqueeze(1)
        interpolated = F.normalize(interpolated, p=2, dim=1)
        
        # 添加微小的局部噪声（防止过拟合到精确的插值线）
        noise = torch.randn_like(interpolated) * self.local_radius
        targets = F.normalize(interpolated + noise, p=2, dim=1)
        
        # 记录更新前的能量
        with torch.no_grad():
            energy_before = self.landscape.energy(interpolated).mean().item()
        
        # 局部梯度下降：降低（或升高）插值路径上的能量
        self.landscape.train()
        optimizer = torch.optim.SGD(self.landscape.parameters(), lr=self.lr * abs(feedback))
        
        # 目标能量：当前能量减去 feedback（正反馈=降低能量，负反馈=升高能量）
        target_shift = -feedback * 0.5  # 缩放因子，避免单次更新过大
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            current_energy = self.landscape.energy(targets)
            loss = F.mse_loss(
                current_energy,
                current_energy.detach() + target_shift
            )
            loss.backward()
            
            # 梯度裁剪，确保局部性
            torch.nn.utils.clip_grad_norm_(self.landscape.parameters(), 0.1)
            optimizer.step()
        
        # 记录更新后的能量
        with torch.no_grad():
            energy_after = self.landscape.energy(interpolated).mean().item()
        
        # 记录历史
        record = {
            'timestamp': time.time(),
            'query_char': self.anchors.hanzi_list[q_idx],
            'answer_char': self.anchors.hanzi_list[c_idx],
            'feedback': feedback,
            'energy_before': energy_before,
            'energy_after': energy_after,
            'energy_delta': energy_after - energy_before,
            'steps': self.max_steps,
        }
        self.history.append(record)
        
        return {
            'status': 'updated',
            **record,
        }
    
    def reinforce_path(
        self,
        char_a: str,
        char_b: str,
        strength: float = 1.0,
    ) -> Dict:
        """
        直接强化两个汉字之间的关联路径。
        
        这是 Hebbian 学习的直接形式：给定两个汉字，在它们之间
        建立（或强化）低能量路径。
        
        Args:
            char_a: 第一个汉字
            char_b: 第二个汉字
            strength: 强化强度
        
        Returns:
            更新结果字典
        """
        a_idx = self.anchors._char_to_idx.get(char_a)
        b_idx = self.anchors._char_to_idx.get(char_b)
        
        if a_idx is None or b_idx is None:
            missing = [c for c, i in [(char_a, a_idx), (char_b, b_idx)] if i is None]
            return {'status': 'error', 'reason': f'汉字不存在: {missing}'}
        
        vec_a = self.anchors.anchors[a_idx]
        vec_b = self.anchors.anchors[b_idx]
        
        return self.update(vec_a, vec_b, feedback=strength)
    
    def weaken_path(
        self,
        char_a: str,
        char_b: str,
        strength: float = 1.0,
    ) -> Dict:
        """削弱两个汉字之间的关联（负向学习）"""
        return self.reinforce_path(char_a, char_b, strength=-strength)
    
    def get_stats(self) -> Dict:
        """获取学习统计"""
        if not self.history:
            return {'total_updates': 0}
        
        recent = self.history[-100:]
        avg_delta = np.mean([h['energy_delta'] for h in recent])
        
        return {
            'total_updates': len(self.history),
            'recent_avg_delta': avg_delta,
            'top_active': self.tracker.most_active(5),
            'top_idle': self.tracker.most_idle(5),
        }


# ============================================================================
# 第三部分：自知无知检测器
# ============================================================================

class SelfIgnoranceDetector:
    """
    自知无知检测器 —— 判断系统是否"知道"某个问题。
    
    核心原理:
      训练后的能量景观中，已知知识区域（锚点附近）具有陡峭的能量梯度，
      而未知区域（远离任何锚点）能量相对平坦。
      
      通过计算查询点附近的能量梯度模长来判断:
        - 梯度大 → 附近有吸引子盆地 → 系统"知道"
        - 梯度小 → 附近无吸引子 → 系统"不知道"
      
      辅助信号:
        - 能量值: 极低能量 = 深盆地 = 高度已知
        - 最近锚点距离: 距离近 = 接近已知知识
      
      综合三个信号给出 0~1 的置信度评分。
    
    使用方式:
      detector = SelfIgnoranceDetector(landscape, anchor_field)
      known, confidence = detector.check(query_vector)
      if confidence < 0.3:
          print("我不太确定这个问题的答案...")
    """
    
    def __init__(
        self,
        landscape: 'EnergyLandscape',      # type: ignore
        anchor_field: 'HanziAnchorField',  # type: ignore
        gradient_threshold: float = 0.05,
        energy_low_threshold: float = -14.0,   # 只在深盆地(-14以下)才拿满分
        energy_high_threshold: float = -8.0,   # -8以上视为高能未知区
        device: str = "cpu",
    ):
        """
        初始化自知无知检测器。
        
        Args:
            landscape: 能量景观实例
            anchor_field: 字场实例
            gradient_threshold: 梯度模长阈值（高于此值视为"已知"）
            energy_low_threshold: 低能量阈值（低于此值增加置信度）
            energy_high_threshold: 高能量阈值（高于此值降低置信度）
            device: 计算设备
        """
        self.landscape = landscape
        self.anchors = anchor_field
        self.gradient_threshold = gradient_threshold
        self.energy_low = energy_low_threshold
        self.energy_high = energy_high_threshold
        self.device = device
        
        self.landscape.to(device)
        
        # 统计信息（用于自适应阈值调整）
        self.reference_gradients: List[float] = []
        self.reference_energies: List[float] = []
        self.is_calibrated = False
    
    def calibrate(self, n_samples: int = 1000):
        """
        校准检测器：在已知锚点上采样，建立参考分布。
        
        在字场锚点上计算梯度模长和能量值，作为"已知"的参考基线。
        之后可以将查询点的值与参考分布比较，判断是否已知。
        """
        print(f"校准自知无知检测器 ({n_samples} 样本)...")
        
        # 校准: 在锚点 AND 随机点上采样，建立两个参考分布
        # 锚点分布 → 窄而低梯度（已知区域）
        # 随机分布 → 宽而高梯度（未知区域）
        indices = torch.randperm(self.anchors.num_hanzi)[:n_samples]
        sample_anchors = self.anchors.anchors[indices]
        
        grads = []
        energies = []
        
        self.landscape.train()
        
        for anchor in sample_anchors:
            grad_norm = self._compute_gradient_norm(anchor.to(self.device))
            grads.append(grad_norm)
            with torch.no_grad():
                e = self.landscape.energy(anchor.unsqueeze(0).to(self.device)).item()
                energies.append(e)
        
        grad_array = np.array(grads)
        self.anchor_grad_mean = float(grad_array.mean())
        self.anchor_grad_std = float(grad_array.std())
        
        energy_array = np.array(energies)
        self.anchor_energy_mean = float(energy_array.mean())
        self.anchor_energy_std = float(energy_array.std())
        
        self.reference_gradients = grads
        self.reference_energies = energies
        self.is_calibrated = True
        
        # 阈值: 锚点能量的均值 - 2*std = 已知盆地的能量上限
        self.gradient_threshold = self.anchor_grad_mean + 3 * self.anchor_grad_std
        
        print(f"  梯度模长: mean={self.anchor_grad_mean:.4f}, "
              f"std={self.anchor_grad_std:.4f}, "
              f"min={grad_array.min():.4f}, max={grad_array.max():.4f}")
        print(f"  自动阈值: {self.gradient_threshold:.4f} (mean+3σ)")
        print(f"  能量: mean={self.anchor_energy_mean:.2f}, "
              f"std={self.anchor_energy_std:.2f}, "
              f"range=[{energy_array.min():.2f}, {energy_array.max():.2f}]")
    
    def _compute_gradient_norm(self, x: torch.Tensor) -> float:
        """计算能量函数在点x处的梯度模长"""
        x = x.clone().detach().requires_grad_(True)
        energy = self.landscape.energy(x.unsqueeze(0))
        energy.backward()
        return x.grad.norm().item()
    
    def check(
        self,
        query_vec: torch.Tensor,
        return_details: bool = False,
    ) -> Dict:
        """
        判断查询点是否在已知知识范围内。
        
        Args:
            query_vec: 查询向量 (embed_dim,)
            return_details: 是否返回详细分析
        
        Returns:
            dict: {
                'is_known': 是否已知 (bool)
                'confidence': 置信度 0~1 (float)
                'gradient_norm': 梯度模长 (float)
                'energy': 能量值 (float)
                'anchor_distance': 最近锚点距离 (float)
                'nearest_chars': 最近汉字列表
                'diagnosis': 诊断结论 (str)
            }
        """
        x = query_vec.detach().to(self.device)
        
        # 信号1: 梯度模长 —— z-score 法
        # 在锚点附近梯度窄(mean±std), 偏离锚点的点梯度异常大
        # z-score < 2 → 接近锚点分布 → 已知
        self.landscape.train()
        grad_norm = self._compute_gradient_norm(x)
        
        if hasattr(self, 'anchor_grad_std') and self.anchor_grad_std > 0:
            z_grad = abs(grad_norm - self.anchor_grad_mean) / self.anchor_grad_std
            # 句子嵌入天然偏移锚点5-7σ，用宽容差(÷10)
            grad_score = max(0.0, 1.0 - z_grad / 10.0)
        else:
            grad_score = min(grad_norm / max(self.gradient_threshold, 1e-6), 1.0)
        
        # 信号2: 能量值 —— 在锚点能量分布内 → 已知
        with torch.no_grad():
            energy = self.landscape.energy(x.unsqueeze(0)).item()
        
        if hasattr(self, 'anchor_energy_std') and self.anchor_energy_std > 0:
            z_energy = abs(energy - self.anchor_energy_mean) / self.anchor_energy_std
            # 句子嵌入能量偏移锚点3-7σ，用宽容差(÷8)
            energy_score = max(0.0, 1.0 - z_energy / 8.0)
        else:
            if energy <= self.energy_low:
                energy_score = 1.0
            elif energy >= self.energy_high:
                energy_score = 0.0
            else:
                energy_score = 1.0 - (energy - self.energy_low) / (self.energy_high - self.energy_low)
        
        # 信号3: 最近锚点距离
        _, chars, sims = self.anchors.find_nearest(
            x.cpu().unsqueeze(0), k=3
        )
        anchor_dist = 1.0 - sims[0].item()  # 余弦距离 = 1 - 余弦相似度
        
        # 距离置信度（距离近→高置信）
        dist_score = max(1.0 - anchor_dist / 0.5, 0.0)  # 距离0.5以上→0分
        
        # 综合置信度（加权平均）
        # 能量权重提升: BGE编码下所有查询梯度相似，能量是最有区分度的信号
        confidence = 0.2 * grad_score + 0.5 * energy_score + 0.3 * dist_score
        confidence = max(0.0, min(1.0, confidence))
        
        is_known = confidence > 0.5
        
        # 诊断
        if confidence > 0.8:
            diagnosis = "高度确定——查询在深度已知区域内"
        elif confidence > 0.5:
            diagnosis = "基本确定——查询在已知区域内"
        elif confidence > 0.3:
            diagnosis = "不太确定——查询处于已知与未知交界"
        elif confidence > 0.1:
            diagnosis = "非常不确定——查询基本在未知区域"
        else:
            diagnosis = "完全未知——查询远离任何已知锚点"
        
        result = {
            'is_known': is_known,
            'confidence': confidence,
            'gradient_norm': grad_norm,
            'energy': energy,
            'anchor_distance': anchor_dist,
            'nearest_chars': chars,
            'diagnosis': diagnosis,
        }
        
        if return_details:
            result.update({
                'grad_score': grad_score,
                'energy_score': energy_score,
                'dist_score': dist_score,
            })
        
        return result
    
    def check_text(
        self,
        text: str,
        reasoner: 'DragonBallReasoner' = None,  # type: ignore
    ) -> Dict:
        """
        对文本查询进行自知无知判断。
        
        便捷方法：将文本编码为嵌入向量后调用 check()。
        
        Args:
            text: 输入文本
            reasoner: DragonBallReasoner 实例（可选，用于编码）
        
        Returns:
            同 check()
        """
        # 文本→嵌入
        if reasoner is not None:
            query_vec = reasoner.zichang.encode_text(text)
            if query_vec.shape[0] == 0:
                return {'is_known': False, 'confidence': 0.0, 
                        'diagnosis': f"输入'{text}'中无有效汉字"}
            query_vec = query_vec.mean(dim=0)
        else:
            query_vec = self.anchors.encode_text(text)
            if query_vec.shape[0] == 0:
                return {'is_known': False, 'confidence': 0.0,
                        'diagnosis': f"输入'{text}'中无有效汉字"}
            query_vec = query_vec.mean(dim=0)
        
        result = self.check(query_vec)
        result['input_text'] = text
        return result


# ============================================================================
# 第四部分：能量衰减调度器 —— 废退
# ============================================================================

class WeightDecayScheduler:
    """
    能量衰减调度器 —— 实现"废退"。
    
    Hebbian学习的另一面：长期不被使用的知识通路会慢慢被遗忘。
    这模拟了生物神经系统中的突触修剪机制。
    
    工作机制:
      1. 周期性地对所有能量景观参数施加微小衰减
      2. 衰减强度与全局激活水平成反比（活跃系统衰减慢）
      3. 可选：对长期未激活锚点附近的权重额外衰减
    
    注意: 衰减应非常缓慢（典型 decay_rate = 0.9999），
          过快会破坏已学到的知识结构。
    """
    
    def __init__(
        self,
        landscape: 'EnergyLandscape',  # type: ignore
        tracker: ActivationTracker,
        base_decay_rate: float = 0.9999,
        idle_penalty: float = 0.9995,
        decay_interval: int = 100,
        device: str = "cpu",
    ):
        """
        初始化衰减调度器。
        
        Args:
            landscape: 能量景观实例
            tracker: 激活追踪器
            base_decay_rate: 基础衰减率（每步乘以此系数）
            idle_penalty: 长期未用锚点的额外衰减率
            decay_interval: 衰减间隔（多少步执行一次）
            device: 计算设备
        """
        self.landscape = landscape
        self.tracker = tracker
        self.base_decay_rate = base_decay_rate
        self.idle_penalty = idle_penalty
        self.decay_interval = decay_interval
        self.device = device
        
        self.step_counter = 0
        self.decay_history: List[Dict] = []
        
        self.landscape.to(device)
    
    def step(self) -> Dict:
        """
        执行一步衰减（如果到达间隔）。
        
        Returns:
            dict: 衰减统计（如果执行了衰减）或 {'status': 'skipped'}
        """
        self.step_counter += 1
        
        if self.step_counter % self.decay_interval != 0:
            return {'status': 'skipped', 'step': self.step_counter}
        
        return self._apply_decay()
    
    def _apply_decay(self) -> Dict:
        """应用全局衰减"""
        # 计算活跃度调整因子
        total_active = self.tracker.activation_counts.sum().item()
        avg_activity = total_active / max(self.tracker.num_anchors, 1)
        
        # 活跃度高时减缓衰减（保护活跃知识）
        activity_factor = min(avg_activity / 10.0, 1.0)
        effective_rate = 1.0 - (1.0 - self.base_decay_rate) * (1.0 - activity_factor * 0.5)
        
        with torch.no_grad():
            param_norms_before = sum(p.norm().item() for p in self.landscape.parameters())
            
            for name, param in self.landscape.named_parameters():
                if param.requires_grad:
                    param.data *= effective_rate
            
            param_norms_after = sum(p.norm().item() for p in self.landscape.parameters())
        
        record = {
            'step': self.step_counter,
            'effective_rate': effective_rate,
            'avg_activity': avg_activity,
            'norm_before': param_norms_before,
            'norm_after': param_norms_after,
            'norm_delta': param_norms_after - param_norms_before,
        }
        self.decay_history.append(record)
        
        return {'status': 'decayed', **record}
    
    def targeted_decay(self, anchor_indices: torch.Tensor, extra_decay: float = 0.999):
        """
        对特定锚点附近应用额外衰减。
        
        用于"定向遗忘"：当某些知识被确认错误后，
        对相关锚点附近的能量进行更强的衰减。
        
        Args:
            anchor_indices: 目标锚点索引
            extra_decay: 额外衰减率
        
        Returns:
            dict: 衰减统计
        """
        # 找到锚点向量，对能量景观进行针对性衰减
        # 通过在锚点附近的高能量区域增加梯度上升来实现
        targets = self.tracker.get_activity(anchor_indices)
        idle_mask = targets < 0.1  # 已经很冷的锚点
        
        if idle_mask.any():
            idle_indices = anchor_indices[idle_mask]
            # 对这些锚点附近的参数施加额外衰减
            with torch.no_grad():
                for param in self.landscape.parameters():
                    if param.requires_grad:
                        param.data *= extra_decay
        
        return {
            'status': 'targeted_decay',
            'n_targets': len(anchor_indices),
            'n_idle': idle_mask.sum().item(),
        }
    
    def get_stats(self) -> Dict:
        """获取衰减统计"""
        if not self.decay_history:
            return {'total_decays': 0}
        
        recent = self.decay_history[-10:]
        avg_delta = np.mean([h['norm_delta'] for h in recent])
        
        return {
            'total_decays': len(self.decay_history),
            'recent_avg_norm_delta': avg_delta,
            'current_rate': self.base_decay_rate,
        }


# ============================================================================
# 第五部分：完整学习循环
# ============================================================================

class DragonBallLearner:
    """
    龙珠完整学习循环 —— 整合学习、检测、衰减的端到端系统。
    
    这是龙珠的"大脑可塑性"层，协调三个子系统的运作:
      - HebbianLearner:     用进（强化学习）
      - SelfIgnoranceDetector: 自知无知（元认知）
      - WeightDecayScheduler:  废退（遗忘）
    
    典型使用流程:
      learner = DragonBallLearner(landscape, anchor_field)
      
      # 1. 检查是否知道
      result = learner.check_knowledge("量子纠缠")
      if not result['is_known']:
          print("我不太确定，需要学习...")
      
      # 2. 学习新知识
      learner.learn(query_vec, answer_vec, feedback=1.0)
      
      # 3. 定期衰减
      learner.decay_step()
    """
    
    def __init__(
        self,
        landscape: 'EnergyLandscape',      # type: ignore
        anchor_field: 'HanziAnchorField',  # type: ignore
        hebbian_lr: float = 0.001,
        decay_rate: float = 0.9999,
        decay_interval: int = 100,
        device: str = "cpu",
    ):
        self.landscape = landscape
        self.anchors = anchor_field
        self.device = device
        
        # 初始化追踪器（共享给各个子系统）
        self.tracker = ActivationTracker(anchor_field.num_hanzi)
        
        # 初始化子系统
        self.hebbian = HebbianLearner(
            landscape=landscape,
            anchor_field=anchor_field,
            learning_rate=hebbian_lr,
            device=device,
        )
        self.hebbian.tracker = self.tracker  # 共享追踪器
        
        self.detector = SelfIgnoranceDetector(
            landscape=landscape,
            anchor_field=anchor_field,
            device=device,
        )
        
        self.decay = WeightDecayScheduler(
            landscape=landscape,
            tracker=self.tracker,
            base_decay_rate=decay_rate,
            decay_interval=decay_interval,
            device=device,
        )
        
        # 学习统计
        self.total_learns = 0
        self.total_decays = 0
        self.total_checks = 0

        # ── EWC 弹性权重巩固 ──
        self._ewc_ref_params: Dict[str, torch.Tensor] = {}   # 锚定参数
        self._ewc_fisher: Dict[str, torch.Tensor] = {}        # Fisher 对角
        self._ewc_lambda: float = 0.05                         # 正则强度
        self._ewc_enabled: bool = False
        self._ewc_rounds: int = 0
    
    def calibrate(self):
        """校准自知无知检测器"""
        self.detector.calibrate()
    
    def check_knowledge(self, query: torch.Tensor) -> Dict:
        """
        检查系统对查询的了解程度。
        
        Args:
            query: 查询向量或文本字符串
        
        Returns:
            自知无知检测结果
        """
        self.total_checks += 1
        
        if isinstance(query, str):
            return self.detector.check_text(query)
        return self.detector.check(query)
    
    def learn(
        self,
        query_vec: torch.Tensor,
        answer_vec: torch.Tensor,
        feedback: float = 1.0,
    ) -> Dict:
        """
        学习一次查询-答案关联。
        
        Args:
            query_vec: 查询向量
            answer_vec: 答案向量
            feedback: 反馈强度
        
        Returns:
            学习结果
        """
        self.total_learns += 1
        return self.hebbian.update(query_vec, answer_vec, feedback=feedback)
    
    def learn_chars(
        self,
        query_char: str,
        answer_char: str,
        strength: float = 1.0,
    ) -> Dict:
        """学习两个汉字之间的关联"""
        self.total_learns += 1
        return self.hebbian.reinforce_path(query_char, answer_char, strength=strength)
    
    def learn_pairs_batch(
        self,
        pairs: List[Tuple[int, int]],
        learning_rate: float = 0.05,
    ) -> Dict:
        """
        批量学习字符关联对——对比学习版本。
        
        不再使用MSE全局目标（会抹平景观），改用对比损失：
          - 已知字对 → 降低中点能量（形成盆地）
          - 随机字对 → 保持/升高能量（作为负样本）
        
        损失 = ranking_loss(已知 < 随机 + margin) + push_loss(已知 < target_low)
        """
        import random as _random
        
        if not pairs:
            return {'status': 'ok', 'pairs_learned': 0}
        
        anchors = self.anchors.anchors
        device = next(self.landscape.parameters()).device
        n_anchors = len(anchors)
        n_known = len(pairs)
        
        # 已知字对中点 — 批量索引，一次GPU传输
        idx_a = torch.tensor([p[0] for p in pairs])
        idx_b = torch.tensor([p[1] for p in pairs])
        mid_known = ((anchors[idx_a] + anchors[idx_b]) / 2).to(device).detach()
        
        # 随机字对中点（负样本）— 批量索引
        rand_a = torch.randint(0, n_anchors, (n_known,))
        rand_b = torch.randint(0, n_anchors, (n_known,))
        mid_random = ((anchors[rand_a] + anchors[rand_b]) / 2).to(device).detach()
        
        with torch.no_grad():
            energy_known_before = self.landscape(mid_known).mean().item()
            energy_random_before = self.landscape(mid_random).mean().item()
            separation_before = energy_random_before - energy_known_before
        
        # 对比学习：已知对能量必须低于随机对
        optimizer = torch.optim.Adam(self.landscape.parameters(), lr=learning_rate * 0.01)
        margin = 3.0       # 已知对比随机对至少低3.0
        target_low = -15.0  # 已知对目标能级
        
        for _ in range(5):
            e_known = self.landscape(mid_known)
            e_random = self.landscape(mid_random)
            
            # 排序损失：已知必须低于随机至少margin
            rank_loss = torch.nn.functional.relu(
                e_known - e_random + margin
            ).mean()
            
            # 下推损失：已知对能量要低于target_low
            push_loss = torch.nn.functional.relu(
                e_known - target_low
            ).mean()
            
            loss = rank_loss + 0.3 * push_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.landscape.parameters(), 0.05)
            optimizer.step()
        
        # 批量 Hebbian 精细调节——仅在小批量时逐对执行
        if n_known <= 100:
            for ia, ib in pairs:
                vec_a = anchors[ia].detach().to(device)
                vec_b = anchors[ib].detach().to(device)
                self.hebbian.update(vec_a, vec_b, feedback=0.3)
        # 大批量时跳过逐对 Hebbian（对比学习已覆盖）
        
        with torch.no_grad():
            energy_known_after = self.landscape(mid_known).mean().item()
            energy_random_after = self.landscape(mid_random).mean().item()
            separation_after = energy_random_after - energy_known_after
        
        self.total_learns += len(pairs)
        
        return {
            'status': 'ok',
            'pairs_learned': len(pairs),
            'avg_energy_known_before': energy_known_before,
            'avg_energy_known_after': energy_known_after,
            'avg_energy_random_before': energy_random_before,
            'avg_energy_random_after': energy_random_after,
            'separation_before': separation_before,
            'separation_after': separation_after,
        }
    
    def update_point(self, point_vec: torch.Tensor, target_energy: float) -> Dict:
        """
        对单个点进行能量调整（局部梯度下降）。
        
        用于序列臂训练：在锚点→目标字连线上的插值点建立能量梯度。
        
        Args:
            point_vec: 目标点向量 (embed_dim,)
            target_energy: 目标能量值
        
        Returns:
            更新结果
        """
        device = next(self.landscape.parameters()).device
        x = point_vec.detach().to(device)
        
        self.landscape.train()
        optimizer = torch.optim.SGD(self.landscape.parameters(), lr=0.001)
        
        energy_before = 0.0
        with torch.no_grad():
            energy_before = self.landscape(x.unsqueeze(0)).item()
        
        target = torch.tensor([target_energy], device=device)
        
        for _ in range(3):
            optimizer.zero_grad()
            current_energy = self.landscape(x.unsqueeze(0)).squeeze(-1)
            loss = torch.nn.functional.mse_loss(current_energy, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.landscape.parameters(), 0.1)
            optimizer.step()
        
        with torch.no_grad():
            energy_after = self.landscape(x.unsqueeze(0)).item()
        
        return {
            'status': 'updated',
            'energy_before': energy_before,
            'energy_after': energy_after,
            'delta': energy_after - energy_before,
        }

    def update_ewc_fisher(self, n_samples: int = 200) -> Dict:
        """更新 EWC Fisher 信息矩阵，锚定当前重要参数。

        每 50 守护轮调用一次。采样锚点计算能量梯度平方作为
        Fisher 对角近似，标记当前景观的重要参数。
        后续学习通过 ewc_regularize() 惩罚这些参数的偏移。
        """
        device = next(self.landscape.parameters()).device
        self.landscape.train()

        # 1. 保存参考参数（用于后续计算偏离量）
        self._ewc_ref_params = {
            name: p.clone().detach()
            for name, p in self.landscape.named_parameters()
        }

        # 2. Fisher 对角 ≈ E[(∂E/∂θ)²]，在随机锚点上采样
        fisher = {name: torch.zeros_like(p)
                  for name, p in self.landscape.named_parameters()}

        n_anchors = len(self.anchors.anchors)
        indices = torch.randperm(n_anchors)[:min(n_samples, n_anchors)]
        samples = self.anchors.anchors[indices].to(device)

        for x in samples:
            self.landscape.zero_grad()
            e = self.landscape(x.unsqueeze(0))
            e.backward()
            for name, p in self.landscape.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2

        # 3. 归一化 + 防零
        for name in fisher:
            fisher[name] /= n_samples
            fisher[name].clamp_(min=1e-8)

        self._ewc_fisher = fisher
        self._ewc_enabled = True
        self._ewc_rounds = 0

        total_fisher = sum(f.sum().item() for f in fisher.values())
        return {
            'status': 'ok',
            'n_samples': n_samples,
            'total_fisher_mass': total_fisher,
            'params_anchored': len(fisher),
        }

    def ewc_regularize(self) -> Dict:
        """运行一次 EWC 正则化：用 Adam 将参数拉回锚定点。

        在每次 learn_pairs_batch 后调用，防止新知识推挤旧知识。
        已启用时做全量 Adam 步，未启用时跳过（零开销）。
        """
        if not self._ewc_enabled or not self._ewc_ref_params:
            return {'status': 'skipped', 'reason': 'ewc not enabled'}

        device = next(self.landscape.parameters()).device
        self.landscape.train()
        optimizer = torch.optim.Adam(self.landscape.parameters(), lr=1e-5)

        loss_total = torch.tensor(0.0, device=device)
        for name, p in self.landscape.named_parameters():
            if name in self._ewc_ref_params and name in self._ewc_fisher:
                ref = self._ewc_ref_params[name].to(device)
                fisher = self._ewc_fisher[name].to(device)
                loss_total += (fisher * (p - ref) ** 2).sum()

        if loss_total.item() == 0:
            return {'status': 'skipped', 'reason': 'no ewc params'}

        loss_total = self._ewc_lambda * loss_total
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.landscape.parameters(), 0.01)
        optimizer.step()

        self._ewc_rounds += 1
        return {
            'status': 'regularized',
            'ewc_loss': loss_total.item(),
            'ewc_rounds': self._ewc_rounds,
        }

    def unlearn_chars(
        self,
        query_char: str,
        answer_char: str,
        strength: float = 1.0,
    ) -> Dict:
        """削弱两个汉字之间的关联"""
        self.total_learns += 1
        return self.hebbian.weaken_path(query_char, answer_char, strength=strength)
    
    def decay_step(self) -> Dict:
        """执行一次衰减检查"""
        self.total_decays += 1
        return self.decay.step()
    
    def get_stats(self) -> Dict:
        """获取完整学习统计"""
        return {
            'total_learns': self.total_learns,
            'total_decays': self.total_decays,
            'total_checks': self.total_checks,
            'hebbian': self.hebbian.get_stats(),
            'decay': self.decay.get_stats(),
            'top_active': self.tracker.most_active(5),
            'top_idle': self.tracker.most_idle(5),
        }


# ============================================================================
# 第六部分：测试
# ============================================================================

def test_learner(learner: DragonBallLearner):
    """测试学习系统的完整功能"""
    print("\n" + "=" * 60)
    print("龙珠学习系统测试")
    print("=" * 60)
    
    # 测试1: 校准
    print("\n[测试1] 校准自知无知检测器")
    learner.calibrate()
    print("  ✅ 校准完成")
    
    # 测试2: 已知锚点检测
    print("\n[测试2] 已知锚点检测")
    test_anchor = learner.anchors.anchors[1000]  # 第1000个锚点
    result = learner.check_knowledge(test_anchor)
    print(f"  梯度模长: {result['gradient_norm']:.4f}")
    print(f"  能量值:   {result['energy']:.4f}")
    print(f"  置信度:   {result['confidence']:.3f}")
    print(f"  诊断:     {result['diagnosis']}")
    print(f"  {'✅ 正确识别为已知' if result['is_known'] else '⚠️ 未识别为已知'}")
    
    # 测试3: 随机点检测（应为未知）
    print("\n[测试3] 随机点检测（应为未知）")
    random_vec = F.normalize(torch.randn(learner.anchors.embed_dim), dim=-1)
    result = learner.check_knowledge(random_vec)
    print(f"  梯度模长: {result['gradient_norm']:.4f}")
    print(f"  置信度:   {result['confidence']:.3f}")
    print(f"  诊断:     {result['diagnosis']}")
    print(f"  {'✅ 正确识别为未知' if not result['is_known'] else '⚠️ 误判为已知'}")
    
    # 测试4: Hebbian学习
    print("\n[测试4] Hebbian学习")
    q_vec = learner.anchors.find_by_chars(["火"])[0]
    a_vec = learner.anchors.find_by_chars(["水"])[0]
    
    # 学习前能量
    mid = F.normalize((q_vec + a_vec) / 2, dim=-1)
    e_before = learner.landscape.energy(mid.unsqueeze(0)).item()
    
    result = learner.learn(q_vec, a_vec, feedback=1.0)
    
    e_after = learner.landscape.energy(mid.unsqueeze(0)).item()
    print(f"  '火'→'水' 路径能量: {e_before:.4f} → {e_after:.4f}")
    print(f"  变化: {e_after - e_before:+.4f}")
    print(f"  {'✅ 能量降低' if e_after < e_before else '⚠️ 能量未降低'}")
    
    # 测试5: 衰减
    print("\n[测试5] 权重衰减")
    for _ in range(5):
        learner.decay_step()
    stats = learner.get_stats()
    print(f"  总学习次数: {stats['total_learns']}")
    print(f"  总衰减次数: {stats['total_decays']}")
    print(f"  最活跃锚点: {stats['top_active'][:3]}")
    
    print(f"\n{'=' * 60}")
    print("测试完成")
    print("=" * 60)


# ============================================================================
# 第七部分：主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="龙珠学习系统")
    parser.add_argument('--zichang', '-z', type=str, required=True,
                        help='字场文件路径')
    parser.add_argument('--energy', '-e', type=str, required=True,
                        help='能量景观文件路径')
    parser.add_argument('--calibrate', action='store_true',
                        help='校准检测器')
    parser.add_argument('--test', action='store_true',
                        help='运行测试')
    
    args = parser.parse_args()
    
    import loongpearl.core.zichang
    from loongpearl.core.energy_landscape import EnergyLandscape
    
    # 加载
    zf = zichang.HanziAnchorField.load(args.zichang)
    landscape = EnergyLandscape.load(args.energy)
    
    # 创建学习系统
    learner = DragonBallLearner(landscape, zf)
    
    if args.calibrate:
        learner.calibrate()
    
    if args.test:
        test_learner(learner)
    
    # 交互式演示
    print("\n" + "=" * 60)
    print("学习演示")
    print("=" * 60)
    
    demo_pairs = [
        ("火", "水", 1.0),
        ("天", "地", 1.0),
        ("日", "月", 1.0),
    ]
    
    for q, a, s in demo_pairs:
        r = learner.learn_chars(q, a, strength=s)
        print(f"  学习: '{q}'→'{a}' | 能量变化: {r.get('energy_delta', 0):+.4f}")
    
    print("\n检测:")
    for text in ["火水", "天地", "量子"]:
        r = learner.check_knowledge(text)
        print(f"  '{text}': known={r['is_known']}, conf={r['confidence']:.3f}, {r['diagnosis']}")
