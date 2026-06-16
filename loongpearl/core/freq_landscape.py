"""频率门控能量景观 v2 — 基网络判已知/未知，频率偏移调深度"""
import torch
import torch.nn as nn


class FreqEnergyLandscape(nn.Module):
    """
    双通路架构：
      主通路: 嵌入 → MLP → 基础能量（判已知/未知）
      频率通路: freq → MLP → 能量偏移（高频=更深通道）
      最终能量 = 基础能量 + 频率偏移
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
