"""频率门控能量景观 v2 — 基网络判已知/未知，频率偏移调深度"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FreqEnergyLandscape(nn.Module):
    """
    双通路架构：
      主通路: 嵌入 → MLP → 基础能量（判已知/未知）
      频率通路: freq → MLP → 能量偏移（高频=更深通道）
      最终能量 = 基础能量 + 频率偏移
    
    推理方法:
      infer():  梯度下降 → 收敛到吸引子 → 发出信号
      resolve(): 吸引子 → 最近汉字

    信号系统 (v3):
      梯度下降收敛后，根据收敛质量发出四种信号之一：
        blind_spot    — 梯度平坦，无匹配吸引子 → 双臂去搜索
        conflict      — 多个盆地竞争 → 身体去裁决
        low_confidence — 单一盆地但能量偏高 → 双脚去验证
        certain       — 能量低、梯度陡、无竞争 → 可以直接回答
    """

    # ── 信号阈值（类属性，可在初始化后按需覆盖） ──
    BLIND_GRADIENT_THRESHOLD: float = 0.01   # 梯度范数低于此值→盲区
    CONFLICT_SIMILARITY_GAP: float = 0.05     # Top-2相似度差距小于此值→冲突
    LOW_CONF_ENERGY_THRESHOLD: float = 0.0    # 收敛能量高于此值→低置信

    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 主通路：1024维 → 标量能量
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
        
        # 频率偏移通路：1维频率 → 标量偏移
        # 高频(≈5) → 负偏移(≈-5)，低频(≈0.7) → 近零偏移
        self.freq_shift = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x, freq=None):
        base = self.net(x)  # 基础能量
        if freq is not None:
            f = freq.unsqueeze(-1)  # (bs, 1)
            shift = self.freq_shift(f)  # (bs, 1), 学习到的频率偏移
            return base + shift
        return base
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """计算标量能量值"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.forward(x).squeeze(-1)
    
    # ── 梯度下降推理 ──────────────────────────────────────────

    def infer(
        self,
        query_vec: torch.Tensor,
        steps: int = 50,
        lr: float = 0.02,
        convergence_threshold: float = 1e-5,
        early_stop_patience: int = 5,
        project_to_sphere: bool = True,
        return_trajectory: bool = False,
        zichang: Optional['HanziAnchorField'] = None,  # ★ 新增：用于信号发射
    ) -> Dict:
        """
        从查询向量出发，沿能量梯度下降到最近的吸引子。

        这是龙珠的核心推理：在能量景观中找到「盆地」——即已知概念。

        Args:
            query_vec: 查询向量 (embed_dim,)
            steps: 最大迭代步数
            lr: 梯度下降学习率
            convergence_threshold: 能量变化收敛阈值
            early_stop_patience: 早停耐心值
            project_to_sphere: 是否每步投影回单位球面
            return_trajectory: 是否记录完整轨迹
            zichang: ★ HanziAnchorField 字场实例，传入则启用信号发射

        Returns:
            {
                'state': Tensor,          # 收敛后的状态向量
                'energy': float,           # 最终能量
                'steps': int,              # 实际迭代步数
                'converged': bool,         # 是否收敛
                'energy_delta': float,     # 最后一步能量变化
                'trajectory': list,        # (如 return_trajectory=True)
                # ★ 以下为新增信号字段 (仅当 zichang 传入时)
                'signal': str,             # 'certain'|'blind_spot'|'conflict'|'low_confidence'
                'signal_detail': str,      # 人类可读的信号说明
                'gradient_norm': float,    # 最后一步梯度范数
                'top_candidates': list,    # str 列表: 最近k个汉字
                'top_similarities': list,  # float 列表: 对应的余弦相似度
            }
        """
        was_training = self.training
        self.train()

        device = next(self.parameters()).device
        x = query_vec.clone().detach().to(device)
        if project_to_sphere:
            x = F.normalize(x, p=2, dim=-1)
        x.requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        prev_energy = self.energy(x).item()
        no_improvement = 0
        delta = float('inf')
        trajectory = [x.detach().cpu().clone()] if return_trajectory else None
        step = 0
        final_grad_norm = 0.0  # ★ 记录最后一步梯度范数

        for step in range(steps):
            optimizer.zero_grad()
            e = self.energy(x)
            e.backward()

            # ★ 每步都记录梯度范数，循环结束后保留最后一步的值
            if x.grad is not None:
                final_grad_norm = x.grad.norm().item()

            optimizer.step()

            if project_to_sphere:
                with torch.no_grad():
                    x.data = F.normalize(x.data, p=2, dim=-1)

            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())

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
        self.train(was_training)

        # ★ 信号发射：仅在字场可用时分析收敛质量
        signal = 'certain'
        signal_detail = ''
        top_candidates = []
        top_similarities = []

        if zichang is not None:
            candidates = self.resolve(zichang, x.detach(), top_k=5)
            # candidates = [(汉字, 相似度), ...]
            top_candidates = [ch for ch, _ in candidates]
            top_similarities = [float(s) for _, s in candidates]

            top1_sim = top_similarities[0] if top_similarities else 0.0
            top2_sim = top_similarities[1] if len(top_similarities) > 1 else top1_sim
            sim_gap = abs(top1_sim - top2_sim)

            # ── 信号判断逻辑 ──
            if final_grad_norm < self.BLIND_GRADIENT_THRESHOLD:
                signal = 'blind_spot'
                signal_detail = (f"梯度平坦(grad={final_grad_norm:.5f}<{self.BLIND_GRADIENT_THRESHOLD})，"
                                f"无匹配吸引子，顶候'{top_candidates[0] if top_candidates else '?'}'"
                                f"相似度仅{top1_sim:.4f}")
            elif sim_gap < self.CONFLICT_SIMILARITY_GAP:
                signal = 'conflict'
                signal_detail = (f"多盆地竞争：'{top_candidates[0]}'({top1_sim:.4f})"
                                f" vs '{top_candidates[1] if len(top_candidates)>1 else '?'}'"
                                f"({top2_sim:.4f})，差距{sim_gap:.4f}<{self.CONFLICT_SIMILARITY_GAP}")
            elif final_energy > self.LOW_CONF_ENERGY_THRESHOLD:
                signal = 'low_confidence'
                signal_detail = (f"收敛至'{top_candidates[0] if top_candidates else '?'}'"
                                f"(sim={top1_sim:.4f})，但能量偏高(E={final_energy:.3f}>{{self.LOW_CONF_ENERGY_THRESHOLD}})")
            else:
                signal = 'certain'
                signal_detail = (f"确定收敛至'{top_candidates[0] if top_candidates else '?'}'"
                                f"(sim={top1_sim:.4f}, E={final_energy:.3f}，"
                                f"grad={final_grad_norm:.5f})")

        return {
            'state': x.detach().cpu(),
            'energy': final_energy,
            'steps': step + 1,
            'converged': delta < convergence_threshold,
            'energy_delta': delta,
            'trajectory': trajectory,
            # ★ 信号字段
            'signal': signal,
            'signal_detail': signal_detail,
            'gradient_norm': final_grad_norm,
            'top_candidates': top_candidates,
            'top_similarities': top_similarities,
        }
    
    # ── 吸引子解析 ──────────────────────────────────────────
    
    def resolve(
        self,
        zichang: 'HanziAnchorField',
        state: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        将收敛后的状态向量映射回最近的汉字。
        
        Args:
            zichang: HanziAnchorField 字场实例
            state: 收敛后的状态向量 (embed_dim,)
            top_k: 返回前k个候选
        
        Returns:
            [(汉字, 相似度), ...]
        """
        _, chars, sims = zichang.find_nearest(state.cpu().unsqueeze(0), k=top_k)
        return list(zip(chars, sims.tolist()))
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'embed_dim': self.embed_dim,
        }, path)
        print(f"能量景观已保存: {path}", flush=True)
    
    @classmethod
    def load(cls, path, map_location='cpu'):
        data = torch.load(path, map_location=map_location, weights_only=True)
        instance = cls(embed_dim=data.get('embed_dim', 1024))
        instance.load_state_dict(data['model_state_dict'])
        print(f"能量景观已加载: {path} (dim={instance.embed_dim})", flush=True)
        return instance
