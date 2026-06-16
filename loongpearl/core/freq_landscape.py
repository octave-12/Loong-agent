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
      infer():  梯度下降 → 收敛到吸引子
      resolve(): 吸引子 → 最近汉字
    """
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
        
        Returns:
            {'state': Tensor, 'energy': float, 'steps': int, 'converged': bool, ...}
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
        
        for step in range(steps):
            optimizer.zero_grad()
            e = self.energy(x)
            e.backward()
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
        
        return {
            'state': x.detach().cpu(),
            'energy': final_energy,
            'steps': step + 1,
            'converged': delta < convergence_threshold,
            'energy_delta': delta,
            'trajectory': trajectory,
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
        _, chars, sims = zichang.find_nearest(state.unsqueeze(0), k=top_k)
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
