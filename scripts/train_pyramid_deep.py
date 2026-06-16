#!/usr/bin/env python3
"""
深度能量金字塔训练 — RTX 3060 12GB 优化版
============================================
6级抽象: 1024 → 512 → 256 → 128 → 64 → 32

架构:
  L0: 字场锚点 (94117×1024d, ❄️冻结)
  P1: 1024→512  ─→ L1 词级景观
  P2: 512→256   ─→ L2 句级景观
  P3: 256→128   ─→ L3 段级景观 (NEW)
  P4: 128→64    ─→ L4 篇级景观 (NEW)
  P5: 64→32     ─→ L5 主题级景观 (NEW)

训练数据: 概念图 RELATED 边 + 维基百科词对

RTX 3060 12GB 完全足够 — 金字塔仅 ~23M 参数
训练时间估算: ~20分钟/级 × 5级 = ~2小时

用法:
  python scripts/train_pyramid_deep.py           # GPU训练
  python scripts/train_pyramid_deep.py --cpu     # CPU训练(慢10x)
"""
import sys, os, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.concept_graph import ConceptGraph
from loongpearl.core.freq_landscape import FreqEnergyLandscape

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 6级投影块
# ============================================================

class DeepProjection(nn.Module):
    """含残差连接的深度投影块"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.dropout(F.gelu(self.norm(self.linear(x)))).squeeze(0)
        return self.dropout(F.gelu(self.norm(self.linear(x))))


class DeepEnergyPyramid(nn.Module):
    """
    6级深度能量金字塔。
    
    超越 Transformer 的关键差异:
      - 每级降维 (1024→32)，信息逐级压缩提炼
      - 每级独立能量景观，可单独验证
      - 总参数 ~23M (vs LLM 的数十亿)
    """
    
    def __init__(self, field, device='cuda'):
        super().__init__()
        self.field = field
        self.device = device
        
        # 投影层
        self.P1 = DeepProjection(1024, 512)  # 字→词
        self.P2 = DeepProjection(512, 256)   # 词→句
        self.P3 = DeepProjection(256, 128)   # 句→段
        self.P4 = DeepProjection(128, 64)    # 段→篇
        self.P5 = DeepProjection(64, 32)     # 篇→主题
        
        # 6级能量景观
        self.L1 = FreqEnergyLandscape(embed_dim=512)
        self.L2 = FreqEnergyLandscape(embed_dim=256)
        self.L3 = FreqEnergyLandscape(embed_dim=128)
        self.L4 = FreqEnergyLandscape(embed_dim=64)
        self.L5 = FreqEnergyLandscape(embed_dim=32)
        
        self.to(device)
        self.trained_levels = set()
    
    def train_level(self, level, pairs, epochs=200, lr=0.005):
        """训练单级 (GPU batch优化)"""
        projections = [self.P1, self.P2, self.P3, self.P4, self.P5]
        landscapes = [self.L1, self.L2, self.L3, self.L4, self.L5]
        proj = projections[level]
        landscape = landscapes[level]
        
        # 准备张量
        vecs_a = torch.stack([p[0] for p in pairs]).to(self.device)
        vecs_b = torch.stack([p[1] for p in pairs]).to(self.device)
        n = len(pairs)
        
        # 生成负样本
        perm = torch.randperm(n, device=self.device)
        vecs_neg = vecs_b[perm]
        
        optimizer = torch.optim.Adam(
            list(proj.parameters()) + list(landscape.parameters()), lr=lr)
        
        # 通过前面冻结的投影层
        with torch.no_grad():
            for p in projections[:level]:
                proj.eval()
                vecs_a = p(vecs_a)
                vecs_b = p(vecs_b)
                vecs_neg = p(vecs_neg)
        
        proj.train()
        landscape.train()
        
        for epoch in range(epochs):
            # 当前层投影
            ha = proj(vecs_a)
            hb = proj(vecs_b)
            hn = proj(vecs_neg)
            
            mid_pos = (ha + hb) / 2
            mid_neg = (ha + hn) / 2
            
            e_pos = landscape(mid_pos).squeeze(-1)
            e_neg = landscape(mid_neg).squeeze(-1)
            
            loss = F.relu(2.0 + e_pos - e_neg).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + list(landscape.parameters()), 1.0)
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"  L{level+1} e{epoch}: loss={loss.item():.4f} "
                      f"e_pos={e_pos.mean().item():.1f} e_neg={e_neg.mean().item():.1f}")
        
        # 冻结
        proj.eval()
        landscape.eval()
        for p in proj.parameters():
            p.requires_grad = False
        for p in landscape.parameters():
            p.requires_grad = False
        self.trained_levels.add(level)
        
        return loss.item()
    
    def forward_all(self, va, vb):
        """6级前向传播"""
        with torch.no_grad():
            h = [va.to(self.device), vb.to(self.device)]
            
            for i, proj in enumerate([self.P1, self.P2, self.P3, self.P4, self.P5]):
                h[0] = proj(h[0])
                h[1] = proj(h[1])
            
            energies = []
            landscapes = [self.L1, self.L2, self.L3, self.L4, self.L5]
            # 注意: L1在P1之后评估, L2在P2之后评估...
            # 简化: 在最终状态评估
            final_mid = (h[0] + h[1]) / 2
            for ls in landscapes:
                e = ls(final_mid.unsqueeze(0) if final_mid.dim() == 1 else final_mid)
                energies.append(e.item() if e.numel() == 1 else e.mean().item())
            
            return energies
    
    def save(self, path):
        torch.save({
            f'P{i+1}': getattr(self, f'P{i+1}').state_dict() for i in range(5)
        } | {
            f'L{i+1}': getattr(self, f'L{i+1}').state_dict() for i in range(5)
        } | {'trained_levels': list(self.trained_levels)},
        path)
        print(f"深度金字塔已保存: {path}")
    
    @classmethod
    def load(cls, path, field, device='cuda'):
        pyramid = cls(field, device)
        data = torch.load(path, map_location=device)
        for i in range(5):
            getattr(pyramid, f'P{i+1}').load_state_dict(data[f'P{i+1}'])
            getattr(pyramid, f'L{i+1}').load_state_dict(data[f'L{i+1}'])
        pyramid.trained_levels = set(data.get('trained_levels', []))
        pyramid.to(device).eval()
        return pyramid


# ============================================================
# 主训练流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='6级深度能量金字塔训练')
    parser.add_argument('--cpu', action='store_true', help='CPU训练')
    parser.add_argument('--epochs', type=int, default=200, help='每级训练轮数')
    parser.add_argument('--level', type=int, default=0, help='从第几级开始(0-4)')
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(f"🔺 6级深度能量金字塔训练 — {device}")
    print("=" * 60)
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 加载
    print("\n加载字场...")
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)
    
    cg = ConceptGraph(field)
    cg_path = os.path.join(PROJECT, 'data/models/concept_graph')
    if os.path.exists(cg_path + '.json'):
        cg.load(cg_path)
    
    # 创建金字塔
    pyramid = DeepEnergyPyramid(field, device=device)
    params = sum(p.numel() for p in pyramid.parameters())
    print(f"   金字塔参数: {params:,} (~{params/1e6:.1f}M)")
    
    # 准备训练对
    print("\n准备训练数据...")
    pairs = []
    for t in cg.triples.values():
        if t.confidence < 0.3:
            continue
        va = cg.get_embedding(t.subject)
        vb = cg.get_embedding(t.object)
        if va is not None and vb is not None:
            pairs.append((va, vb))
    
    print(f"   训练对: {len(pairs)}")
    
    # 逐级训练
    for level in range(args.level, 5):
        print(f"\n{'='*40}")
        print(f"训练 L{level+1}: {'字→词 词→句 句→段 段→篇 篇→主题'.split()[level]} "
              f"({'1024→512 512→256 256→128 128→64 64→32'.split()[level]})")
        print(f"{'='*40}")
        
        t0 = time.time()
        loss = pyramid.train_level(level, pairs, epochs=args.epochs)
        elapsed = time.time() - t0
        
        print(f"   ✅ L{level+1} 完成: loss={loss:.4f} 耗时={elapsed:.0f}s")
    
    # 保存
    save_path = os.path.join(PROJECT, 'data/models/energy_pyramid_deep.pt')
    pyramid.save(save_path)
    
    # 推理测试
    print(f"\n🔍 6级推理测试:")
    test_pairs = [("电子","原子"), ("细胞","器官"), ("数学","物理")]
    for a, b in test_pairs:
        va = cg.get_embedding(a)
        vb = cg.get_embedding(b)
        if va is not None and vb is not None:
            energies = pyramid.forward_all(va, vb)
            print(f"  {a}→{b}: " + " ".join(f"L{i+1}={e:.1f}" for i, e in enumerate(energies)))
    
    print(f"\n🎉 6级深度金字塔训练完成!")


if __name__ == '__main__':
    main()
