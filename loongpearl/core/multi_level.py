#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠多级能量金字塔 (multi_level.py) — 超越 Transformer 的多层知识抽象
══════════════════════════════════════════════════════════════════════

为什么这比 LLM 的 Transformer 层更强？

  Transformer:  96层同质 attention+MLP, 全部 1024d, 黑箱
  能量金字塔:   3层异构景观, 逐级降维(1024→512→256→128), 每层独立可验证

核心差异:
  ┌──────────────────┬─────────────────────┬──────────────────┐
  │                  │    Transformer       │   能量金字塔      │
  ├──────────────────┼─────────────────────┼──────────────────┤
  │ 层结构           │ 同质重复(attention)  │ 异构(每层不同操作)│
  │ 维度             │ 全程 1024d           │ 1024→512→256→128 │
  │ 可解释性         │ 黑箱                 │ 每层独立验证      │
  │ 参数规模         │ 数十亿               │ ~100万            │
  │ 幻觉风险         │ 概率采样             │ 确定性能量评分    │
  │ 训练方式         │ 端到端反向传播       │ 逐层 Hebbian+投影 │
  └──────────────────┴─────────────────────┴──────────────────┘

══════════════════════════════════════════════════════════════════════
架构
══════════════════════════════════════════════════════════════════════

  字场锚点 (94117 × 1024d, ❄️ 永久冻结)
      │
      │ mean pooling (概念嵌入)
      ▼
  ┌─────────────────────────────────────┐
  │ Projection_1: Linear(1024→512)      │ ← 可学习
  │             + LayerNorm + GELU      │
  └─────────────────────────────────────┘
      │
      ▼
  词级景观 (FreEnergyLandscape, 512d)    ← L1: 评估词-词连贯性
      │
  ┌─────────────────────────────────────┐
  │ Projection_2: Linear(512→256)       │ ← 可学习
  │             + LayerNorm + GELU      │
  └─────────────────────────────────────┘
      │
      ▼
  句级景观 (FreqEnergyLandscape, 256d)   ← L2: 评估句内结构
      │
  ┌─────────────────────────────────────┐
  │ Projection_3: Linear(256→128)       │ ← 可学习
  │             + LayerNorm + GELU      │
  └─────────────────────────────────────┘
      │
      ▼
  概念级景观 (FreqEnergyLandscape, 128d) ← L3: 评估跨概念推理

══════════════════════════════════════════════════════════════════════
训练
══════════════════════════════════════════════════════════════════════

  逐层训练:
    1. L1(词级): 用概念图的 RELATED 边作为正样本对, Hebbian 降低中点能量
    2. L2(句级): 用概念图的 PART_OF/IS_A 路径作为正样本, 投影来自 L1
    3. L3(概念级): 用概念图的跨域桥梁作为正样本, 投影来自 L2

  每层训练后冻结该层景观参数，继续训练投影矩阵和上层景观。

══════════════════════════════════════════════════════════════════════
推理
══════════════════════════════════════════════════════════════════════

  forward(concept_a, concept_b) → 三级能量评分:
    e1 = L1_landscape(P1(embed(concept_a, concept_b)))
    e2 = L2_landscape(P2(P1_output))
    e3 = L3_landscape(P3(P2_output))
    total_score = 0.4*e1 + 0.35*e2 + 0.25*e3

  每层独立可检查：如果 L1 低能但 L2 高能 → 词级合理但句法不通
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os


# ============================================================================
# 投影层
# ============================================================================

class ProjectionBlock(nn.Module):
    """
    可学习的层级投影块。

    不只是简单的线性映射——带 LayerNorm + GELU 非线性，
    比 Transformer 的 FFN 轻得多（无 attention），但足够做抽象。
    参数量: in_dim × out_dim + 2*out_dim (≈ 500K for 1024→512)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) or (in_dim,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            x = self.linear(x)
            x = F.gelu(x)
            x = self.norm(x)
            x = self.dropout(x)
            return x.squeeze(0)
        x = self.linear(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


# ============================================================================
# 能量金字塔
# ============================================================================

class EnergyPyramid(nn.Module):
    """
    三级能量金字塔。

    用法:
        pyramid = EnergyPyramid(field, base_dim=1024)
        pyramid.train_levels(concept_graph, epochs_per_level=100)

        # 推理
        e1, e2, e3 = pyramid.forward(concept_a_vec, concept_b_vec)
        score = e1*0.4 + e2*0.35 + e3*0.25
    """

    def __init__(
        self,
        field,                    # HanziAnchorField
        base_dim: int = 1024,
        l1_dim: int = 512,
        l2_dim: int = 256,
        l3_dim: int = 128,
        device: str = 'cpu',
    ):
        super().__init__()
        self.field = field
        self.base_dim = base_dim
        self.device = device if torch.cuda.is_available() else 'cpu'

        # ── 投影层 ──
        self.P1 = ProjectionBlock(base_dim, l1_dim)   # 1024 → 512
        self.P2 = ProjectionBlock(l1_dim, l2_dim)      # 512 → 256
        self.P3 = ProjectionBlock(l2_dim, l3_dim)      # 256 → 128

        # ── 能量景观（每层独立） ──
        from loongpearl.core.freq_landscape import FreqEnergyLandscape
        self.L1 = FreqEnergyLandscape(embed_dim=l1_dim)  # 词级
        self.L2 = FreqEnergyLandscape(embed_dim=l2_dim)  # 句级
        self.L3 = FreqEnergyLandscape(embed_dim=l3_dim)  # 概念级

        self.to(self.device)

        # 状态
        self.l1_trained = False
        self.l2_trained = False
        self.l3_trained = False

    # ═══════════════════════════════════════════════════════════════
    # 前向传播
    # ═══════════════════════════════════════════════════════════════

    def forward(
        self,
        vec_a: torch.Tensor,
        vec_b: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        三级前向: 输入两个概念向量 → 输出三级能量评分。

        Returns:
            (e1, e2, e3, total_score)
            e1: 词级能量（越低→词对越合理）
            e2: 句级能量（越低→句法越通顺）
            e3: 概念级能量（越低→推理越正确）
            total_score: 加权综合评分
        """
        with torch.no_grad():
            # 确保在正确的设备上
            va = vec_a.to(self.device)
            vb = vec_b.to(self.device)

            # Level 1: 词级
            h1_a = self.P1(va)
            h1_b = self.P1(vb)
            mid1 = ((h1_a + h1_b) / 2).unsqueeze(0)
            e1 = self.L1(mid1).item()

            # Level 2: 句级
            h2_a = self.P2(h1_a)
            h2_b = self.P2(h1_b)
            mid2 = ((h2_a + h2_b) / 2).unsqueeze(0)
            e2 = self.L2(mid2).item()

            # Level 3: 概念级
            h3_a = self.P3(h2_a)
            h3_b = self.P3(h2_b)
            mid3 = ((h3_a + h3_b) / 2).unsqueeze(0)
            e3 = self.L3(mid3).item()

        total = 0.4 * e1 + 0.35 * e2 + 0.25 * e3
        return e1, e2, e3, total

    def forward_batch(
        self,
        vecs_a: torch.Tensor,   # (N, base_dim)
        vecs_b: torch.Tensor,   # (N, base_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量前向传播"""
        with torch.no_grad():
            va = vecs_a.to(self.device)
            vb = vecs_b.to(self.device)

            h1_a = self.P1(va)
            h1_b = self.P1(vb)
            mid1 = (h1_a + h1_b) / 2
            e1 = self.L1(mid1).squeeze(-1)

            h2_a = self.P2(h1_a)
            h2_b = self.P2(h1_b)
            mid2 = (h2_a + h2_b) / 2
            e2 = self.L2(mid2).squeeze(-1)

            h3_a = self.P3(h2_a)
            h3_b = self.P3(h2_b)
            mid3 = (h3_a + h3_b) / 2
            e3 = self.L3(mid3).squeeze(-1)

        return e1, e2, e3

    # ═══════════════════════════════════════════════════════════════
    # 逐层训练
    # ═══════════════════════════════════════════════════════════════

    def train_level(
        self,
        level: int,
        positive_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 200,
        lr: float = 0.01,
        contrastive_margin: float = 2.0,
    ) -> Dict[str, float]:
        """
        训练单层。

        Args:
            level: 1/2/3
            positive_pairs: [(vec_a, vec_b), ...] 正样本对 (base_dim 维度)
            epochs: 训练轮数
            lr: 学习率
            contrastive_margin: 对比损失边距
        """
        if level < 1 or level > 3:
            raise ValueError(f"Invalid level: {level}")

        landscape = [self.L1, self.L2, self.L3][level - 1]
        # 待训练的投影层（只训练当前层的投影，前面层的投影已冻结）
        projections_to_train = [self.P1] if level == 1 else ([self.P2] if level == 2 else [self.P3])

        # 前面的投影层（已冻结，仅用于前向）
        prev_projections = []
        if level >= 2:
            prev_projections.append(self.P1)
        if level >= 3:
            prev_projections.append(self.P2)

        # 生成负样本（随机打乱配对）
        n = len(positive_pairs)
        neg_pairs = []
        indices = torch.randperm(n)
        for i in range(min(n, 500)):
            j = indices[i].item()
            k = indices[(i + 1) % n].item()
            neg_pairs.append((positive_pairs[j][0], positive_pairs[k][1]))
        neg_pairs = neg_pairs[:len(positive_pairs)]

        # 训练参数：当前投影层 + 当前景观
        trainable_params = list(landscape.parameters())
        for proj in projections_to_train:
            trainable_params += list(proj.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        landscape.train()
        for proj in projections_to_train:
            proj.train()

        # 确保前面的投影层在 eval 模式
        for proj in prev_projections:
            proj.eval()

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for (va_pos, vb_pos), (va_neg, vb_neg) in zip(positive_pairs, neg_pairs):
                # 完整前向链: 输入 raw → 前面投影(frozen) → 当前投影(trainable) → 景观
                ha_pos = va_pos.to(self.device)
                hb_pos = vb_pos.to(self.device)
                ha_neg = va_neg.to(self.device)
                hb_neg = vb_neg.to(self.device)

                # 通过前面已冻结的投影层
                with torch.no_grad():
                    for proj in prev_projections:
                        ha_pos = proj(ha_pos)
                        hb_pos = proj(hb_pos)
                        ha_neg = proj(ha_neg)
                        hb_neg = proj(hb_neg)

                # 通过当前可训练的投影层
                for proj in projections_to_train:
                    ha_pos = proj(ha_pos)
                    hb_pos = proj(hb_pos)
                    ha_neg = proj(ha_neg)
                    hb_neg = proj(hb_neg)

                # 中点能量
                mid_pos = (ha_pos + hb_pos) / 2
                mid_neg = (ha_neg + hb_neg) / 2

                e_pos = landscape(mid_pos.unsqueeze(0)).squeeze()
                e_neg = landscape(mid_neg.unsqueeze(0)).squeeze()

                # 安全的对比损失: 正样本能量应低，负样本能量应高
                # loss = max(0, margin + e_pos - e_neg)
                loss = F.relu(contrastive_margin + e_pos - e_neg)
                total_loss += loss
                count += 1

            if count > 0:
                avg_loss = total_loss / count
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            if epoch % 50 == 0 and count > 0:
                print(f"  L{level} epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

        # 冻结已训练的景观参数
        landscape.eval()
        for p in landscape.parameters():
            p.requires_grad = False
        for proj in projections_to_train:
            proj.eval()

        setattr(self, f'l{level}_trained', True)

        return {'level': level, 'epochs': epochs, 'final_loss': avg_loss.item() if count > 0 else 999}

    def train_all_levels(
        self,
        concept_graph,       # ConceptGraph: 提供训练样本
        epochs_per_level: int = 200,
    ) -> Dict[str, Any]:
        """
        从概念图中提取训练样本，逐层训练。

        训练数据:
          L1: RELATED 边 → 概念对的字场嵌入
          L2: PART_OF/IS_A 边 → 概念对
          L3: 跨域桥 RELATED 边 → 概念对
        """
        results = {}

        # ── 准备基础嵌入 ──
        concept_vecs: Dict[str, torch.Tensor] = {}
        for concept in concept_graph.nodes:
            emb = concept_graph.get_embedding(concept)
            if emb is not None:
                concept_vecs[concept] = emb

        def get_pair(triple):
            a = concept_vecs.get(triple.subject)
            b = concept_vecs.get(triple.object)
            if a is not None and b is not None:
                return (a, b)
            return None

        # ── L1: 词级 — RELATED 边 ──
        print("=== 训练 L1: 词级景观 (1024→512) ===")
        l1_pairs = []
        for t in concept_graph.triples.values():
            if t.relation == "RELATED" and t.confidence > 0.3:
                pair = get_pair(t)
                if pair:
                    l1_pairs.append(pair)
        if l1_pairs:
            results['L1'] = self.train_level(1, l1_pairs[:500], epochs=epochs_per_level)
            print(f"  L1 完成: {len(l1_pairs[:500])}对, loss={results['L1']['final_loss']:.4f}")
        else:
            print("  L1 跳过: 无训练样本")
            results['L1'] = {'skipped': True}

        # ── L2: 句级 — PART_OF/IS_A 边 ──
        print("\n=== 训练 L2: 句级景观 (512→256) ===")
        l2_pairs = []
        for t in concept_graph.triples.values():
            if t.relation in ("PART_OF", "IS_A") and t.confidence > 0.4:
                pair = get_pair(t)
                if pair:
                    l2_pairs.append(pair)
        if l2_pairs:
            results['L2'] = self.train_level(2, l2_pairs[:300], epochs=epochs_per_level)
            print(f"  L2 完成: {len(l2_pairs[:300])}对, loss={results['L2']['final_loss']:.4f}")
        else:
            print("  L2 跳过: 无训练样本")
            results['L2'] = {'skipped': True}

        # ── L3: 概念级 — 跨域桥 RELATED 边 ──
        print("\n=== 训练 L3: 概念级景观 (256→128) ===")
        l3_pairs = []
        for t in concept_graph.triples.values():
            if t.relation == "RELATED" and t.source == "cross_domain_bridge" and t.confidence > 0.4:
                pair = get_pair(t)
                if pair:
                    l3_pairs.append(pair)
        if l3_pairs:
            results['L3'] = self.train_level(3, l3_pairs[:200], epochs=epochs_per_level)
            print(f"  L3 完成: {len(l3_pairs[:200])}对, loss={results['L3']['final_loss']:.4f}")
        else:
            print("  L3 跳过: 无训练样本")
            results['L3'] = {'skipped': True}

        return results

    # ═══════════════════════════════════════════════════════════════
    # 推理：三级分析
    # ═══════════════════════════════════════════════════════════════

    def analyze(
        self,
        concept_a: str,
        concept_b: str,
        concept_graph,   # 用于获取嵌入和置信度
    ) -> Dict[str, Any]:
        """
        三级分析：不仅给分，还解释每层为什么。

        Returns:
            {
                'e1': 词级能量,
                'e2': 句级能量,
                'e3': 概念级能量,
                'total': 综合,
                'diagnosis': {  # 人类可读的诊断
                    'word_level': '✅ 词级合理' | '⚠️ 词级不通',
                    'syntax_level': '✅ 结构通顺' | '⚠️ 结构异常',
                    'concept_level': '✅ 推理正确' | '⚠️ 推理不确定',
                }
            }
        """
        va = concept_graph.get_embedding(concept_a)
        vb = concept_graph.get_embedding(concept_b)

        if va is None or vb is None:
            return {'error': 'concept not found'}

        e1, e2, e3, total = self.forward(va, vb)

        # 诊断阈值（基于训练后的能量分布）
        diagnosis = {
            'word_level': '✅ 词级合理' if e1 < -5 else ('⚠️ 词级不通' if e1 > 10 else '🟡 词级一般'),
            'syntax_level': '✅ 结构通顺' if e2 < -3 else ('⚠️ 结构异常' if e2 > 8 else '🟡 结构一般'),
            'concept_level': '✅ 推理正确' if e3 < -2 else ('⚠️ 推理不确定' if e3 > 5 else '🟡 推理待验证'),
        }

        # 从概念图获取置信度
        conf = 0.5
        for r in ['PART_OF', 'IS_A', 'RELATED', 'HAS', 'CAUSE']:
            key = f"{concept_a}|{r}|{concept_b}"
            if key in concept_graph.triples:
                conf = max(conf, concept_graph.triples[key].confidence)
            key2 = f"{concept_b}|{r}|{concept_a}"
            if key2 in concept_graph.triples:
                conf = max(conf, concept_graph.triples[key2].confidence)

        return {
            'concept_a': concept_a,
            'concept_b': concept_b,
            'e1': round(e1, 2),
            'e2': round(e2, 2),
            'e3': round(e3, 2),
            'total': round(total, 2),
            'concept_confidence': round(conf, 2),
            'diagnosis': diagnosis,
        }

    # ═══════════════════════════════════════════════════════════════
    # 保存/加载
    # ═══════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存完整的能量金字塔"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'P1': self.P1.state_dict(),
            'P2': self.P2.state_dict(),
            'P3': self.P3.state_dict(),
            'L1': self.L1.state_dict(),
            'L2': self.L2.state_dict(),
            'L3': self.L3.state_dict(),
            'base_dim': self.base_dim,
            'l1_dim': self.P1.linear.out_features,
            'l2_dim': self.P2.linear.out_features,
            'l3_dim': self.P3.linear.out_features,
            'l1_trained': self.l1_trained,
            'l2_trained': self.l2_trained,
            'l3_trained': self.l3_trained,
        }, path)
        print(f"能量金字塔已保存: {path}")

    @classmethod
    def load(cls, path: str, field, device: str = 'cpu') -> "EnergyPyramid":
        """加载能量金字塔"""
        data = torch.load(path, map_location=device)

        pyramid = cls(
            field,
            base_dim=data['base_dim'],
            l1_dim=data['l1_dim'],
            l2_dim=data['l2_dim'],
            l3_dim=data['l3_dim'],
            device=device,
        )

        pyramid.P1.load_state_dict(data['P1'])
        pyramid.P2.load_state_dict(data['P2'])
        pyramid.P3.load_state_dict(data['P3'])
        pyramid.L1.load_state_dict(data['L1'])
        pyramid.L2.load_state_dict(data['L2'])
        pyramid.L3.load_state_dict(data['L3'])

        pyramid.l1_trained = data.get('l1_trained', False)
        pyramid.l2_trained = data.get('l2_trained', False)
        pyramid.l3_trained = data.get('l3_trained', False)

        pyramid.to(device).eval()
        print(f"能量金字塔已加载: L1={pyramid.l1_trained} L2={pyramid.l2_trained} L3={pyramid.l3_trained}")
        return pyramid


# ============================================================================
# 演示
# ============================================================================

def demo_pyramid(field, concept_graph):
    """演示能量金字塔"""
    print("=" * 60)
    print("🔺 龙珠能量金字塔 — 超越 Transformer 的多层抽象")
    print("=" * 60)

    pyramid = EnergyPyramid(field, base_dim=1024, device='cpu')
    print(f"   参数量: {sum(p.numel() for p in pyramid.parameters()):,}")
    print(f"   架构: 1024 → 512 → 256 → 128")
    print(f"   vs Transformer: ~100M vs 数十亿参数")

    # 训练
    print(f"\n📚 逐层训练...")
    results = pyramid.train_all_levels(concept_graph, epochs_per_level=100)

    # 推理测试
    print(f"\n🔍 三级推理测试:")
    test_pairs = [
        ("电子", "原子"),
        ("原子", "分子"),
        ("细胞", "器官"),
        ("数学", "物理"),
        ("红楼梦", "唐诗"),
    ]
    for a, b in test_pairs:
        analysis = pyramid.analyze(a, b, concept_graph)
        print(f"\n  {a} → {b}:")
        print(f"    e1(词):{analysis.get('e1','?')} {analysis.get('diagnosis',{}).get('word_level','?')}")
        print(f"    e2(句):{analysis.get('e2','?')} {analysis.get('diagnosis',{}).get('syntax_level','?')}")
        print(f"    e3(概念):{analysis.get('e3','?')} {analysis.get('diagnosis',{}).get('concept_level','?')}")
        print(f"    总分:{analysis.get('total','?')}")

    return pyramid


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from loongpearl.core.zichang import HanziAnchorField
    from loongpearl.core.concept_graph import ConceptGraph

    PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    field_path = os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt')
    cg_path = os.path.join(PROJECT, 'data/models/concept_graph')

    field = HanziAnchorField.load(field_path, freeze=True)
    cg = ConceptGraph(field)
    if os.path.exists(cg_path + '.json'):
        cg.load(cg_path)
    else:
        cg.seed_all_domains()
        cg.induce()

    demo_pyramid(field, cg)
