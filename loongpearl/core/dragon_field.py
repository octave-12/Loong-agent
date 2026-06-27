#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 认知场核心 — 现代连续 Hopfield 网络
═══════════════════════════════════════════════════════

替代 FreqEnergyLandscape，将知识从「被动调用模块」升级为「持续活跃的认知场」。

核心机制:
  存储: 每个概念图三元组 → BGE 嵌入向量 → 直接堆叠为模式矩阵 Y[N×1024]
  查询: query_vec → 与全部模式 softmax 相似度 → 加权收敛到吸引子
  推理: 梯度下降在能量函数上找最近盆地
  不确定性: Hessian 曲率 / MC Dropout 方差 → 连续的「确定→模糊→盲区」

能量函数:
  E(x) = -logsumexp(β · Y^T x) + ½‖x‖²

收敛属性:
  - 收敛点 x* 一定在存储模式 Y 的凸包内 → 不会凭空幻觉
  - basin 深度 E(x*) → 置信度 (越深越确定)
  - Hessian 最小特征值 → 确定性 (越大越确定)
  - 两个吸引子间鞍点高度 → 冲突强度

技术栈: PyTorch (MultiheadAttention / 手动 Hopfield) + torchdiffeq (ODE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 查询结果
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FieldResult:
    """场收敛结果 — 替代旧的四种离散信号"""
    # 收敛到的向量 (1024d)
    convergent_vector: torch.Tensor

    # 能量值 (越低=越确定, 负数=已掌握)
    energy: float

    # 最近存储模式的索引和元数据
    top_pattern_indices: List[int]
    top_similarities: List[float]

    # 不确定性 (连续值, 替代盲区/冲突/低置信)
    basin_depth: float           # 能量井深度 (越深越确定)
    curvature: float             # Hessian 最小特征值近似 (越大越确定)
    saddle_gap: float            # Top-2 盆地间鞍点差距 (越小越冲突)

    # 距离最近存储模式的距离 → 区分「检索」和「涌现」
    distance_to_nearest: float   # <0.2=检索, 0.2-0.5=涌现, >0.5=不可靠

    # 收敛统计
    convergence_steps: int
    trajectory: Optional[List[torch.Tensor]] = None

    @property
    def is_retrieval(self) -> bool:
        return self.distance_to_nearest < 0.2

    @property
    def is_emergent(self) -> bool:
        return 0.2 <= self.distance_to_nearest < 0.5

    @property
    def is_unreliable(self) -> bool:
        return self.distance_to_nearest >= 0.5

    @property
    def confidence_label(self) -> str:
        """人类可读的置信度标签"""
        if self.is_retrieval and self.basin_depth < -10:
            return "已知 (确定)"
        elif self.is_retrieval:
            return "已知"
        elif self.is_emergent:
            return "推测 (涌现)"
        else:
            return "不确定"


# ═══════════════════════════════════════════════════════════════════
# 龙 认知场
# ═══════════════════════════════════════════════════════════════════

class DragonField(nn.Module):
    """
    现代连续 Hopfield 网络 — 龙的认知场核心。

    存储: Y ∈ R^{N×D} — N个模式, 每个D维 (概念图三元组的BGE嵌入)
    查询: x ∈ R^D → 迭代收敛到最近吸引子

    参数:
        embed_dim: 嵌入维度 (必须匹配字场, 默认1024)
        beta: 逆温度 (越高=更硬的最大值, 默认8.0)
    """

    def __init__(self, embed_dim: int = 1024, beta: float = 8.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta

        # 存储模式矩阵 — 运行时从文件加载
        self.register_buffer(
            'patterns',
            torch.empty(0, embed_dim, dtype=torch.float16)
        )
        self._pattern_ids: List[int] = []       # 对应的三元组ID
        self._pattern_subjects: List[str] = []   # 主体字
        self._pattern_count: int = 0

        # 最近激活计数 (用于热缓存淘汰)
        self.register_buffer(
            '_activation_count',
            torch.zeros(0, dtype=torch.int32)
        )

        # MC Dropout 配置
        self._mc_dropout_rate = 0.1  # MC dropout rate

    # ── 存储操作 ──────────────────────────────────────────────

    @property
    def num_patterns(self) -> int:
        return self._pattern_count

    def store_patterns(
        self,
        vectors: torch.Tensor,
        ids: List[int],
        subjects: List[str],
        device: str = 'cpu'
    ):
        """
        批量存储模式到场的记忆矩阵。

        Args:
            vectors: (N, D) float16 嵌入向量
            ids: 对应的三元组 SQLite rowid
            subjects: 主体字 (用于回溯查询)
        """
        vectors = vectors.to(dtype=torch.float16, device='cpu')
        self.register_buffer('patterns', vectors)
        self._pattern_ids = list(ids)
        self._pattern_subjects = list(subjects)
        self._pattern_count = len(ids)
        self.register_buffer(
            '_activation_count',
            torch.zeros(len(ids), dtype=torch.int32)
        )
        log.info(f"场记忆: {len(ids)} 个模式已存储 ({vectors.element_size() * vectors.numel() / 1024**2:.1f} MB)")

    def append_pattern(
        self,
        vector: torch.Tensor,
        pattern_id: int,
        subject: str
    ):
        """追加单个模式 (增量学习)"""
        vector = vector.to(dtype=torch.float16, device='cpu').unsqueeze(0)
        if self.patterns.numel() == 0:
            self.register_buffer('patterns', vector)
        else:
            self.patterns = torch.cat([self.patterns, vector.cpu()])
        self._pattern_ids.append(pattern_id)
        self._pattern_subjects.append(subject)
        self._pattern_count += 1

    # ── 核心推理: 能量函数 + 收敛 ─────────────────────────────

    def energy(self, x: torch.Tensor, patterns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        现代 Hopfield 能量函数。
        E(x) = -logsumexp(β Y^T x) + ½‖x‖² + β⁻¹log N

        Args:
            x: (D,) 或 (B, D) 查询向量
            patterns: 传入指定模式子集, 默认用全部
        """
        if patterns is None:
            patterns = self.patterns.to(device=x.device, dtype=torch.float32)
        else:
            patterns = patterns.to(device=x.device, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Y^T x → (N, B) 或 (N,)
        sim = (patterns @ x.T) * self.beta  # (N, B)

        # logsumexp over patterns
        lse = torch.logsumexp(sim, dim=0)  # (B,)

        # ½‖x‖²
        norm_term = 0.5 * (x.norm(dim=-1) ** 2)  # (B,)

        # β⁻¹ log N
        log_n = math.log(max(patterns.shape[0], 1)) / self.beta

        return -lse + norm_term + log_n

    def converge(
        self,
        query_vec: torch.Tensor,
        patterns: Optional[torch.Tensor] = None,
        max_steps: int = 30,
        convergence_threshold: float = 1e-4,
        return_trajectory: bool = False,
        project_to_sphere: bool = True,
    ) -> FieldResult:
        """
        从查询向量出发，迭代收敛到最近吸引子。

        现代 Hopfield 更新规则:
          x_new = Y · softmax(β Y^T x)   (投影到存储模式的凸组合)

        Args:
            query_vec: (D,) 查询向量
            patterns: 传入指定模式子集 (用于热缓存在GPU, 温/冷层在CPU)
            max_steps: 最大迭代步数
            convergence_threshold: 能量变化收敛阈值
            return_trajectory: 是否记录轨迹
            project_to_sphere: 是否每步投影回单位球面

        Returns:
            FieldResult — 收敛结果
        """
        if patterns is None:
            patterns = self.patterns.to(
                device=query_vec.device, dtype=torch.float32
            )
        else:
            patterns = patterns.to(
                device=query_vec.device, dtype=torch.float32
            )

        device = query_vec.device
        N = patterns.shape[0]
        x = query_vec.clone().detach().to(dtype=torch.float32)

        prev_energy = float('inf')
        trajectory = [x.clone()] if return_trajectory else None

        for step in range(max_steps):
            x.requires_grad_(True)

            # 计算 softmax 权重
            sim = (patterns @ x) * self.beta  # (N,)
            weights = F.softmax(sim, dim=0)    # (N,)

            # 现代 Hopfield 更新: x ← Y^T softmax(β Y^T x)
            x_new = patterns.T @ weights  # (D,)

            if project_to_sphere:
                x_new = F.normalize(x_new, dim=-1)

            # 能量检查收敛
            with torch.no_grad():
                current_energy = self.energy(x_new, patterns).item()
                delta = abs(current_energy - prev_energy)

            x = x_new.detach()
            prev_energy = current_energy

            if return_trajectory:
                trajectory.append(x.clone())

            if delta < convergence_threshold:
                break

        # ── 收敛后分析 ──
        with torch.no_grad():
            x_final = x.detach()
            final_energy = self.energy(x_final, patterns).item()

            # 找最近存储模式
            sim_scores = (patterns @ x_final).cpu()  # (N,)
            values, indices = torch.topk(sim_scores, k=min(5, N))
            similarities = values.tolist()
            pattern_indices = indices.tolist()

            # 距离最近模式
            distance_to_nearest = (
                1.0 - similarities[0] if similarities else 1.0
            )

            # basin 深度 = -energy (正值越深 = 越确定)
            basin_depth = -final_energy

            # Hessian 曲率近似 (对角 Hessian)
            curvature = self._estimate_curvature(x_final, patterns)

            # 鞍点差距: Top-2 相似度差
            saddle_gap = (
                similarities[0] - similarities[1]
                if len(similarities) >= 2 else 1.0
            )

        return FieldResult(
            convergent_vector=x_final.cpu(),
            energy=final_energy,
            top_pattern_indices=pattern_indices,
            top_similarities=similarities,
            basin_depth=basin_depth,
            curvature=curvature,
            saddle_gap=float(saddle_gap),
            distance_to_nearest=float(distance_to_nearest),
            convergence_steps=step + 1,
            trajectory=trajectory,
        )

    def _estimate_curvature(
        self,
        x: torch.Tensor,
        patterns: torch.Tensor,
        eps: float = 0.01
    ) -> float:
        """对角 Hessian 近似 → basin 曲率 (陡峭度=确定性)"""
        x_flat = x.detach()
        dim = x_flat.shape[0]

        # 采样几个随机方向, 估算 Hessian 对角线
        n_samples = min(20, dim)
        indices = torch.randperm(dim, device=x.device)[:n_samples]
        diag_vals = []

        for i in indices:
            x_plus = x_flat.clone()
            x_minus = x_flat.clone()
            x_plus[i] += eps
            x_minus[i] -= eps

            e_plus = self.energy(x_plus, patterns)
            e_minus = self.energy(x_minus, patterns)
            e_center = self.energy(x_flat, patterns)

            # 二阶中心差分
            h_ii = (e_plus + e_minus - 2 * e_center) / (eps ** 2)
            diag_vals.append(h_ii.item())

        # 最小对角元 → 最平缓方向 = 最不确定
        if diag_vals:
            return float(min(diag_vals))
        return 0.0

    def mc_uncertainty(
        self,
        query_vec: torch.Tensor,
        n_samples: int = 10,
        dropout_rate: float = 0.15,
        patterns: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        MC Dropout 不确定性估计 (轻量替代 Hessian)。

        对查询向量加噪声 + 多次收敛 → 方差 = 不确定性
        """
        if patterns is None:
            patterns = self.patterns.to(
                device=query_vec.device, dtype=torch.float32
            )
        else:
            patterns = patterns.to(
                device=query_vec.device, dtype=torch.float32
            )

        convergent_vectors = []
        energies = []

        for _ in range(n_samples):
            # 加 dropout 风格的噪声
            noise_mask = (
                torch.rand_like(query_vec) > dropout_rate
            ).float()
            noisy = query_vec * noise_mask
            noisy = F.normalize(noisy, dim=-1)

            result = self.converge(
                noisy, patterns, max_steps=15,
                convergence_threshold=1e-4,
            )
            convergent_vectors.append(result.convergent_vector)
            energies.append(result.energy)

        vecs = torch.stack(convergent_vectors)  # (S, D)
        mean_vec = vecs.mean(dim=0)
        variance = ((vecs - mean_vec) ** 2).mean().item()

        return {
            'variance': variance,
            'energy_std': float(torch.tensor(energies).std().item()),
            'energy_mean': float(torch.tensor(energies).mean().item()),
            'is_stable': variance < 0.01,
        }

    # ── 持续演化 (Neural ODE 接口) ────────────────────────────

    def evolution_step(
        self,
        x: torch.Tensor,
        patterns: Optional[torch.Tensor] = None,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        场的一次演化步 — 向最近 basin 下滑。

        用于 ODE solver: dx/dt = -∇E(x)
        可用 torchdiffeq.odeint(field, x0, t) 做连续演化。
        """
        if patterns is None:
            patterns = self.patterns.to(
                device=x.device, dtype=torch.float32
            )
        else:
            patterns = patterns.to(
                device=x.device, dtype=torch.float32
            )

        x_in = x.detach().clone().to(dtype=torch.float32)
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)

        x_in.requires_grad_(True)
        e = self.energy(x_in, patterns)
        grad = torch.autograd.grad(e.sum(), x_in)[0]

        # 梯度下降: dx = -∇E(x) * dt
        x_new = x_in - grad * dt

        if x_in.shape[0] == 1:
            x_new = x_new.squeeze(0)
        return x_new.detach()

    # ── 热缓存管理 ──────────────────────────────────────────

    def get_hot_patterns(
        self,
        max_hot: int = 200_000,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """获取最近经常激活的模式子集 (热缓存)"""
        if self._pattern_count <= max_hot:
            return self.patterns.to(device=device, dtype=torch.float32)

        # 选激活次数最多的
        _, topk = torch.topk(
            self._activation_count.float(), k=max_hot
        )
        return self.patterns[topk].to(device=device, dtype=torch.float32)

    def bump_activation(self, pattern_indices: List[int]):
        """增加指定模式的激活计数"""
        for idx in pattern_indices:
            if 0 <= idx < self._activation_count.shape[0]:
                self._activation_count[idx] += 1

    # ── 保存/加载 ──────────────────────────────────────────

    def save(self, path: str):
        """保存场状态到磁盘"""
        state = {
            'patterns': self.patterns.cpu(),
            'pattern_ids': self._pattern_ids,
            'pattern_subjects': self._pattern_subjects,
            'activation_count': self._activation_count.cpu(),
            'embed_dim': self.embed_dim,
            'beta': self.beta,
        }
        torch.save(state, path)
        log.info(f"场已保存到 {path} ({self._pattern_count} 模式)")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DragonField':
        """从磁盘加载场状态"""
        state = torch.load(path, map_location=device, weights_only=False)
        field = cls(
            embed_dim=state['embed_dim'],
            beta=state.get('beta', 8.0),
        )
        field.store_patterns(
            state['patterns'],
            state['pattern_ids'],
            state['pattern_subjects'],
        )
        if 'activation_count' in state:
            field.register_buffer(
                '_activation_count',
                state['activation_count'].to(torch.int32)
            )
        return field

    def forward(self, x, patterns=None):
        """前向: 计算能量 (兼容旧 FreqEnergyLandscape 接口)"""
        return self.energy(x, patterns)
