#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
能量景观与推理引擎（energy_landscape.py）—— 龙珠的可微分推理核心
====================================================================
在字场（94117汉字锚点矩阵）之上构建可微分的能量函数，使任意查询向量
通过梯度下降收敛到最近的稳定吸引子（汉字锚点）。

核心思想:
  1. 能量景观: 学习一个 MLP 网络 E(x) → ℝ，使得字场锚点处于能量极小值
  2. 梯度下降推理: 从查询向量出发，沿 -∇E(x) 下降，收敛到最近吸引子
  3. 吸引子映射: 收敛后的向量通过余弦相似度映射回最近的汉字

依赖: torch, numpy, zichang (字场模块)

训练策略:
  - 正样本: 唯一锚点向量（目标能量: -1.0）
  - 负样本: 球面随机点 + 锚点间插值点（目标能量: +1.0）
  - 损失: MSE(energy(sample), target) 使锚点为深谷，其他区域为高坡

作者: Hermes + 李泽坤
版本: 1.0.0 (初代龙珠)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# ============================================================================
# 第一部分：能量景观网络
# ============================================================================

class EnergyLandscape(nn.Module):
    """
    龙珠的能量景观 —— 可微分的吸引子网络。
    
    这是一个多层感知机，将 512 维嵌入向量映射为标量能量值。
    训练后，字场中的汉字锚点处于能量极小值（深谷），
    而锚点之间的区域处于能量高值（山脊），形成清晰的吸引子盆地结构。
    
    网络结构: 512 → 1536 → 1536 → 768 → 1
    激活函数: GELU (比 ReLU 更光滑，利于梯度下降)
    
    训练后的使用方式:
      result = landscape.infer(query_vec, steps=50)
      char, similarity = landscape.resolve(zichang, result.state)
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dims: List[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        """
        初始化能量景观网络。
        
        Args:
            embed_dim: 输入嵌入维度（默认 1024，匹配 BAAI/bge-large-zh）
            hidden_dims: 隐藏层维度列表，默认 [3072, 3072, 1536]
            activation: 激活函数类型 ('gelu', 'relu', 'silu', 'mish')
            dropout: Dropout 比例（训练时使用，推理时为0）
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        if hidden_dims is None:
            hidden_dims = [3072, 3072, 1536]
        
        # 构建网络层
        layers = []
        in_dim = embed_dim
        
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hdim))
            
            # 选择激活函数
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "mish":
                layers.append(nn.Mish())
            else:
                layers.append(nn.GELU())
            
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hdim
        
        # 输出层：标量能量值
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
        # 训练统计
        self.train_history: List[Dict] = []
    
    def _init_weights(self):
        """
        初始化权重，使初始能量景观相对平坦。
        
        使用较小的 gain 值 (0.1) 确保初始输出接近零，
        避免训练初期梯度爆炸或消失。
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算状态向量 x 的能量值。
        
        Args:
            x: 输入向量 (batch_size, embed_dim) 或 (embed_dim,)
        
        Returns:
            能量值 (batch_size,) 或标量
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，等同于 energy()"""
        return self.energy(x)
    
    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    
    def fit(
        self,
        anchors: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 512,
        lr: float = 1e-3,
        noise_std: float = 0.15,
        noise_type: str = "gaussian",
        margin: float = 2.0,
        device: str = "cpu",
        verbose: bool = True,
        save_path: Optional[str] = None,
    ) -> 'EnergyLandscape':
        """
        训练能量景观，使锚点成为能量极小值。
        
        训练策略:
          1. 正样本 = 唯一锚点向量（目标能量 = -margin）
          2. 负样本 = 锚点 + 高斯噪声后归一化（目标能量 = +margin）
          3. 附加负样本 = 两个锚点之间的随机插值（目标能量 = +margin）
        
        这样锚点周围形成深谷，谷间形成能量壁垒，构成清晰的吸引子结构。
        
        Args:
            anchors: 锚点矩阵 (N, embed_dim)，应已归一化
            epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
            noise_std: 负样本噪声标准差
            noise_type: 噪声类型 ('gaussian', 'uniform', 'interpolation')
            margin: 能量目标值（正样本=-margin，负样本=+margin）
            device: 训练设备
            verbose: 是否打印进度
            save_path: 训练完成后保存模型路径
        
        Returns:
            self
        """
        # 去重：只使用唯一锚点训练
        unique_anchors = torch.unique(anchors, dim=0)
        n_unique = unique_anchors.shape[0]
        print(f"训练数据: {n_unique} 个唯一锚点 (原始 {anchors.shape[0]} 个)")
        
        # 确保锚点归一化
        unique_anchors = F.normalize(unique_anchors, p=2, dim=1)
        
        self.to(device)
        unique_anchors = unique_anchors.to(device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        print(f"\n{'='*60}")
        print(f"能量景观训练开始")
        print(f"  锚点数:    {n_unique}")
        print(f"  嵌入维度:  {self.embed_dim}")
        print(f"  训练轮数:  {epochs}")
        print(f"  批量大小:  {batch_size}")
        print(f"  噪声强度:  {noise_std}")
        print(f"  能量边际:  ±{margin}")
        print(f"  设备:      {device}")
        print(f"{'='*60}\n")
        
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            
            # 随机打乱锚点
            perm = torch.randperm(n_unique, device=device)
            
            for i in range(0, n_unique, batch_size):
                batch_idx = perm[i:i + batch_size]
                anchor_batch = unique_anchors[batch_idx]
                bsz = anchor_batch.shape[0]
                
                # 正样本：锚点本身
                pos_energy = self.energy(anchor_batch)
                pos_target = torch.full_like(pos_energy, -margin)
                pos_loss = F.mse_loss(pos_energy, pos_target)
                
                # 负样本1：锚点 + 噪声 → 球面随机扰动
                if noise_type == "gaussian":
                    noise = torch.randn_like(anchor_batch) * noise_std
                elif noise_type == "uniform":
                    noise = (torch.rand_like(anchor_batch) - 0.5) * 2 * noise_std
                else:
                    noise = torch.randn_like(anchor_batch) * noise_std
                
                neg_samples = F.normalize(anchor_batch + noise, p=2, dim=1)
                neg_energy = self.energy(neg_samples)
                neg_target = torch.full_like(neg_energy, margin)
                neg_loss = F.mse_loss(neg_energy, neg_target)
                
                # 负样本2：锚点间插值（防止不同盆地融合）
                if bsz >= 2:
                    # 随机配对
                    idx1 = torch.randperm(bsz, device=device)[:bsz//2]
                    idx2 = torch.randperm(bsz, device=device)[:bsz//2]
                    alpha = torch.rand(bsz//2, 1, device=device) * 0.8 + 0.1  # [0.1, 0.9]
                    interp = F.normalize(
                        alpha * anchor_batch[idx1] + (1 - alpha) * anchor_batch[idx2],
                        p=2, dim=1
                    )
                    interp_energy = self.energy(interp)
                    interp_target = torch.full_like(interp_energy, margin)
                    interp_loss = F.mse_loss(interp_energy, interp_target)
                else:
                    interp_loss = torch.tensor(0.0, device=device)
                
                # 总损失
                loss = pos_loss + neg_loss + 0.5 * interp_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            scheduler.step()
            
            avg_loss = np.mean(epoch_losses)
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': scheduler.get_last_lr()[0],
            })
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_path:
                    self.save(save_path)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch+1:4d}/{epochs}] "
                      f"loss={avg_loss:.6f} "
                      f"best={best_loss:.6f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed:.0f}s")
        
        total_time = time.time() - start_time
        print(f"\n训练完成！最佳loss={best_loss:.6f}, 总耗时={total_time:.0f}s")
        
        if save_path:
            self.save(save_path)
            print(f"模型已保存: {save_path}")
        
        return self
    
    # ------------------------------------------------------------------
    # 梯度下降推理
    # ------------------------------------------------------------------
    
    def infer(
        self,
        query_vec: torch.Tensor,
        steps: int = 50,
        lr: float = 0.02,
        convergence_threshold: float = 1e-5,
        early_stop_patience: int = 5,
        project_to_sphere: bool = True,
        return_trajectory: bool = False,
    ) -> Dict:
        """
        从查询向量出发，沿能量梯度下降到最近的吸引子。
        
        这是龙珠的核心推理操作。给定任意查询向量（可以是文本嵌入、
        知识激活后的融合向量等），通过梯度下降在能量景观中找到最近的
        稳定吸引子。收敛后的向量可通过 resolve() 映射回具体汉字。
        
        注意: 此方法需要梯度计算，不要用 @torch.no_grad() 装饰。
        
        Args:
            query_vec: 查询向量 (embed_dim,)
            steps: 最大迭代步数
            lr: 梯度下降学习率（Adam 自适应）
            convergence_threshold: 能量变化收敛阈值
            early_stop_patience: 早停耐心值（连续N步未改善则停止）
            project_to_sphere: 是否每步投影回单位球面
            return_trajectory: 是否记录完整轨迹
        
        Returns:
            dict: {
                'state': 收敛后的状态向量 (embed_dim,)
                'energy': 最终能量值 (float)
                'steps': 实际迭代步数 (int)
                'converged': 是否收敛 (bool)
                'energy_delta': 最终能量变化量 (float)
                'trajectory': 轨迹列表 (仅当 return_trajectory=True)
            }
        """
        # 切换到训练模式以启用梯度（但保持参数冻结的效果通过优化器只优化 x 实现）
        was_training = self.training
        self.train()
        
        # 克隆查询向量并启用梯度
        x = query_vec.clone().detach().to(next(self.parameters()).device)
        if project_to_sphere:
            x = F.normalize(x, p=2, dim=-1)
        x.requires_grad_(True)
        
        optimizer = torch.optim.Adam([x], lr=lr)
        
        prev_energy = self.energy(x).item()
        no_improvement = 0
        delta = float('inf')
        trajectory = [x.detach().cpu().clone()] if return_trajectory else None
        
        for step in range(steps):
            optimizer.zero_grad()
            e = self.energy(x)
            e.backward()
            optimizer.step()
            
            # 投影回单位球面（保持与锚点在同一流形上）
            if project_to_sphere:
                with torch.no_grad():
                    x.data = F.normalize(x.data, p=2, dim=-1)
            
            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())
            
            # 检查收敛
            current_energy = e.item()
            delta = abs(current_energy - prev_energy)
            
            if current_energy >= prev_energy:
                no_improvement += 1
            else:
                no_improvement = 0
            
            if delta < convergence_threshold:
                break
            
            if no_improvement >= early_stop_patience:
                break
            
            prev_energy = current_energy
        
        final_energy = self.energy(x).item()
        
        # 恢复原始训练状态
        self.train(was_training)
        
        return {
            'state': x.detach().cpu(),
            'energy': final_energy,
            'steps': step + 1,
            'converged': delta < convergence_threshold,
            'energy_delta': delta,
            'trajectory': trajectory,
        }
    
    # ------------------------------------------------------------------
    # 吸引子解析
    # ------------------------------------------------------------------
    
    def resolve(
        self,
        zichang: 'HanziAnchorField',  # type: ignore
        state: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        将收敛后的状态向量映射回最近的汉字。
        
        在能量景观中找到吸引子后，通过余弦相似度在字场中检索
        最匹配的汉字，完成从"连续向量空间"到"离散汉字"的映射。
        
        Args:
            zichang: HanziAnchorField 实例（字场）
            state: 收敛后的状态向量 (embed_dim,)
            top_k: 返回前k个候选
        
        Returns:
            [(汉字, 相似度), ...] 列表
        """
        _, chars, sims = zichang.find_nearest(state.unsqueeze(0), k=top_k)
        return list(zip(chars, sims.tolist()))
    
    # ------------------------------------------------------------------
    # 批量推理
    # ------------------------------------------------------------------
    
    def infer_batch(
        self,
        query_vecs: torch.Tensor,
        steps: int = 50,
        lr: float = 0.02,
        convergence_threshold: float = 1e-5,
        project_to_sphere: bool = True,
    ) -> Dict:
        """
        批量梯度下降推理。
        
        注意: 此方法需要梯度计算，不要用 @torch.no_grad() 装饰。
        
        Args:
            query_vecs: 查询向量矩阵 (batch_size, embed_dim)
            其他参数同 infer()
        
        Returns:
            dict: {
                'states': 收敛状态矩阵 (batch_size, embed_dim)
                'energies': 能量值列表
                'steps': 各样本步数列表
                'converged': 各样本收敛标志
            }
        """
        was_training = self.training
        self.train()
        batch_size = query_vecs.shape[0]
        device = next(self.parameters()).device
        
        x = query_vecs.clone().detach().to(device)
        if project_to_sphere:
            x = F.normalize(x, p=2, dim=-1)
        x.requires_grad_(True)
        
        optimizer = torch.optim.Adam([x], lr=lr)
        
        prev_energies = self.energy(x)
        no_improvement = torch.zeros(batch_size, dtype=torch.int32)
        converged = torch.zeros(batch_size, dtype=torch.bool)
        steps_taken = torch.zeros(batch_size, dtype=torch.int32)
        
        for step in range(steps):
            optimizer.zero_grad()
            e = self.energy(x)
            # 对每个样本独立计算梯度
            e_sum = e.sum()
            e_sum.backward()
            optimizer.step()
            
            if project_to_sphere:
                with torch.no_grad():
                    x.data = F.normalize(x.data, p=2, dim=-1)
            
            # 检查每个样本的收敛情况
            with torch.no_grad():
                current_energies = self.energy(x)
                deltas = (current_energies - prev_energies).abs()
                
                # 标记已收敛的样本
                newly_converged = (deltas < convergence_threshold) & ~converged
                steps_taken[newly_converged] = step + 1
                converged = converged | newly_converged
                
                # 早停计数
                no_improvement = torch.where(
                    current_energies >= prev_energies,
                    no_improvement + 1,
                    torch.zeros_like(no_improvement)
                )
                converged = converged | (no_improvement >= 5)
                
                prev_energies = current_energies
            
            if converged.all():
                break
        
        # 未收敛的标记为总步数
        steps_taken[~converged] = steps
        
        # 恢复原始训练状态
        self.train(was_training)
        
        return {
            'states': x.detach().cpu(),
            'energies': self.energy(x).detach().cpu().tolist(),
            'steps': steps_taken.tolist(),
            'converged': converged.tolist(),
        }
    
    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    
    def save(self, path: str):
        """保存能量景观模型"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # 将 train_history 中的 numpy 值转为 Python 原生类型（兼容 torch.load weights_only=True）
        clean_history = []
        for entry in self.train_history:
            clean_entry = {}
            for k, v in entry.items():
                if hasattr(v, 'item'):
                    clean_entry[k] = v.item()
                else:
                    clean_entry[k] = v
            clean_history.append(clean_entry)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
            'train_history': clean_history,
        }, path)
        print(f"能量景观已保存: {path}")
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'EnergyLandscape':
        """加载能量景观模型"""
        import numpy as np
        # 允许 numpy scalar 以兼容旧格式
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        data = torch.load(path, map_location='cpu', weights_only=True)
        embed_dim = data.get('embed_dim', kwargs.get('embed_dim', 512))
        instance = cls(embed_dim=embed_dim, **kwargs)
        instance.load_state_dict(data['model_state_dict'])
        instance.train_history = data.get('train_history', [])
        print(f"能量景观已加载: {path} (dim={embed_dim})")
        return instance
    
    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def diagnose(self, anchors: torch.Tensor, n_random: int = 1000) -> Dict:
        """
        诊断能量景观质量。
        
        检查锚点能量分布的统计特征，确保训练成功：
        - 锚点能量应显著低于随机点能量
        - 能量分布应该有明显分离
        
        Args:
            anchors: 锚点矩阵 (N, embed_dim)
            n_random: 随机采样点数
        
        Returns:
            诊断统计字典
        """
        # 确保锚点在正确设备上
        anchors = anchors.to(next(self.parameters()).device)
        
        self.eval()
        anchors = F.normalize(anchors, p=2, dim=-1)
        
        # 锚点能量
        anchor_energies = self.energy(anchors)
        
        # 随机点能量
        random_points = F.normalize(torch.randn(n_random, self.embed_dim), p=2, dim=-1)
        random_energies = self.energy(random_points)
        
        # 锚点+噪声能量（锚点附近）
        noisy_anchors = F.normalize(
            anchors[:min(1000, anchors.shape[0])] + torch.randn(min(1000, anchors.shape[0]), self.embed_dim) * 0.1,
            p=2, dim=-1
        )
        noisy_energies = self.energy(noisy_anchors)
        
        return {
            'anchor_energy_mean': anchor_energies.mean().item(),
            'anchor_energy_std': anchor_energies.std().item(),
            'anchor_energy_min': anchor_energies.min().item(),
            'anchor_energy_max': anchor_energies.max().item(),
            'random_energy_mean': random_energies.mean().item(),
            'random_energy_std': random_energies.std().item(),
            'noisy_energy_mean': noisy_energies.mean().item(),
            'separation': (random_energies.mean() - anchor_energies.mean()).item(),
            'energy_ratio': (random_energies.mean() / (anchor_energies.mean().abs() + 1e-8)).item(),
        }
    
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"EnergyLandscape({self.embed_dim}d, {n_params:,} params)"


# ============================================================================
# 第二部分：龙珠推理引擎（封装训练+推理的完整流程）
# ============================================================================

class DragonBallReasoner:
    """
    龙珠推理引擎 —— 整合字场 + 能量景观的端到端推理系统。
    
    使用方式:
      reasoner = DragonBallReasoner(zichang, landscape)
      result = reasoner.query("龙的传人")
      print(result.best_char, result.confidence)
    """
    
    def __init__(
        self,
        zichang: 'HanziAnchorField',  # type: ignore
        landscape: EnergyLandscape,
        device: str = "cpu",
    ):
        self.zichang = zichang
        self.landscape = landscape
        self.device = device
        self.landscape.to(device)
    
    def query(
        self,
        text: str,
        infer_steps: int = 50,
        top_k: int = 5,
    ) -> Dict:
        """
        端到端查询：文本 → 嵌入 → 能量梯度下降 → 汉字解析。
        
        Args:
            text: 输入文本（单字或短词）
            infer_steps: 梯度下降步数
            top_k: 返回候选数
        
        Returns:
            dict: {
                'input': 原始输入
                'initial_embedding': 初始嵌入向量
                'converged_state': 收敛状态
                'energy': 最终能量
                'steps': 梯度下降步数
                'converged': 是否收敛
                'candidates': [(汉字, 相似度), ...]
                'best_char': 最佳匹配汉字
                'confidence': 置信度
            }
        """
        # 步骤1：文本 → 嵌入（取字场中汉字的均值）
        initial = self.zichang.encode_text(text)
        if initial.shape[0] == 0:
            return {'error': f"输入'{text}'中无有效汉字"}
        initial_vec = initial.mean(dim=0)  # 多字取均值
        
        # 步骤2：能量梯度下降
        result = self.landscape.infer(
            initial_vec,
            steps=infer_steps,
            project_to_sphere=True,
        )
        
        # 步骤3：解析为汉字
        candidates = self.landscape.resolve(
            self.zichang,
            result['state'],
            top_k=top_k,
        )
        
        return {
            'input': text,
            'initial_embedding': initial_vec,
            'converged_state': result['state'],
            'energy': result['energy'],
            'steps': result['steps'],
            'converged': result['converged'],
            'candidates': candidates,
            'best_char': candidates[0][0] if candidates else None,
            'confidence': candidates[0][1] if candidates else 0.0,
        }
    
    def query_batch(
        self,
        texts: List[str],
        infer_steps: int = 50,
        top_k: int = 3,
    ) -> List[Dict]:
        """批量查询"""
        results = []
        for text in texts:
            results.append(self.query(text, infer_steps=infer_steps, top_k=top_k))
        return results


# ============================================================================
# 第三部分：便捷函数
# ============================================================================

def create_energy_landscape(
    zichang_path: str = None,
    anchors: torch.Tensor = None,
    embed_dim: int = 1024,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    noise_std: float = 0.15,
    margin: float = 2.0,
    device: str = "cpu",
    save_path: str = None,
    verbose: bool = True,
) -> EnergyLandscape:
    """
    一键创建并训练能量景观。
    
    Args:
        zichang_path: 字场文件路径（.pt），或直接传入 anchors
        anchors: 锚点矩阵（如果已加载）
        embed_dim: 嵌入维度
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
        noise_std: 负样本噪声强度
        margin: 能量边际值
        device: 训练设备
        save_path: 模型保存路径
        verbose: 是否打印进度
    
    Returns:
        训练好的 EnergyLandscape 实例
    """
    # 加载字场
    if anchors is None:
        if zichang_path is None:
            raise ValueError("需要提供 zichang_path 或 anchors")
        sys.path.insert(0, os.path.dirname(zichang_path))
        import zichang
        zf = zichang.HanziAnchorField.load(zichang_path)
        anchors = zf.anchors
    
    # 创建能量景观
    landscape = EnergyLandscape(embed_dim=embed_dim)
    
    # 训练
    landscape.fit(
        anchors=anchors,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        noise_std=noise_std,
        margin=margin,
        device=device,
        verbose=verbose,
        save_path=save_path,
    )
    
    # 诊断
    diag = landscape.diagnose(anchors)
    print(f"\n诊断结果:")
    print(f"  锚点能量: {diag['anchor_energy_mean']:.4f} ± {diag['anchor_energy_std']:.4f}")
    print(f"  随机点能量: {diag['random_energy_mean']:.4f} ± {diag['random_energy_std']:.4f}")
    print(f"  能量分离度: {diag['separation']:.4f} {'✅' if diag['separation'] > 0.5 else '⚠️'}")
    
    return landscape


# ============================================================================
# 第四部分：测试
# ============================================================================

def test_energy_landscape(landscape: EnergyLandscape, anchors: torch.Tensor):
    """测试能量景观的基本功能"""
    print("\n" + "="*60)
    print("能量景观功能测试")
    print("="*60)
    
    # 测试1：锚点能量应低于随机点
    print(f"\n[测试1] 能量分离度:")
    anchor_e = landscape.energy(anchors[:100])
    random_e = landscape.energy(F.normalize(torch.randn(100, landscape.embed_dim), dim=-1))
    print(f"  锚点平均能量: {anchor_e.mean():.4f}")
    print(f"  随机平均能量: {random_e.mean():.4f}")
    print(f"  {'✅ 分离良好' if random_e.mean() > anchor_e.mean() else '⚠️ 需更多训练'}")
    
    # 测试2：从锚点出发不应移动太多
    print(f"\n[测试2] 锚点稳定性:")
    test_anchor = anchors[0]
    result = landscape.infer(test_anchor, steps=30, project_to_sphere=True)
    moved = (result['state'] - test_anchor).norm().item()
    print(f"  锚点位移: {moved:.6f}")
    print(f"  {'✅ 稳定' if moved < 0.01 else '⚠️ 不稳定'}")
    
    # 测试3：从随机点出发应能收敛
    print(f"\n[测试3] 随机点收敛:")
    random_start = F.normalize(torch.randn(landscape.embed_dim), dim=-1)
    result = landscape.infer(random_start, steps=50, project_to_sphere=True)
    print(f"  初始能量: {landscape.energy(random_start).item():.4f}")
    print(f"  最终能量: {result['energy']:.4f}")
    print(f"  收敛步数: {result['steps']}")
    print(f"  {'✅ 收敛' if result['converged'] else '⚠️ 未收敛'}")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print("="*60)


# ============================================================================
# 第五部分：主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="龙珠能量景观训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--zichang', '-z', type=str, required=True,
                        help='字场文件路径 (.pt)')
    parser.add_argument('--output', '-o', type=str, default='energy_landscape_1024d.pt',
                        help='能量景观输出路径')
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch-size', '-b', type=int, default=512,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--noise', type=float, default=0.15,
                        help='噪声强度')
    parser.add_argument('--margin', '-m', type=float, default=2.0,
                        help='能量边际')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--test', action='store_true',
                        help='训练后运行测试')
    
    args = parser.parse_args()
    
    # 加载字场
    sys.path.insert(0, os.path.dirname(args.zichang))
    import zichang
    zf = zichang.HanziAnchorField.load(args.zichang)
    
    # 创建并训练
    landscape = create_energy_landscape(
        anchors=zf.anchors,
        embed_dim=zf.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_std=args.noise,
        margin=args.margin,
        device=args.device,
        save_path=args.output,
    )
    
    if args.test:
        test_energy_landscape(landscape, zf.anchors)
    
    # 创建推理引擎演示
    reasoner = DragonBallReasoner(zf, landscape)
    
    print("\n" + "="*60)
    print("推理演示")
    print("="*60)
    for text in ["龙", "知识", "宇宙", "凤凰"]:
        r = reasoner.query(text, infer_steps=30)
        print(f"\n  '{text}' → '{r['best_char']}' (conf={r['confidence']:.3f}, "
              f"energy={r['energy']:.3f}, steps={r['steps']})")
        print(f"   Top-3: {[(c, f'{s:.3f}') for c, s in r['candidates'][:3]]}")
