#!/usr/bin/env python3
"""
龙珠多源知识注入器（enrich_landscape.py）
========================================
将三个知识源注入能量景观:
  1. 字形拆解 (Make Me a Hanzi)  — 同部件 = 字形关联边
  2. Unicode 字典 (Unihan)         — 同部首/同义 = 结构关联边
  3. 向量语义 (BAAI 嵌入)         — cos 相似度 = 语义关联边

三路信号融合 → Hebbian 批量植入 → 更新能量景观
"""

import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "energy_landscape_1024d_vector_seeded.pt")
DECOMPOSE = os.path.join(BASE, "dict_decompose.json")
UNIHAN = os.path.join(BASE, "dict_unihan.json")
OUT = os.path.join(BASE, "energy_landscape_1024d_enriched.pt")

def main():
    print("🐉 龙珠多源知识注入器")
    print("="*50)

    # 1. 加载核心组件
    print("\n[1/4] 加载字场 + 能量景观...")
    zc = HanziAnchorField.load(ZICHANG)
    ls = EnergyLandscape.load(LANDSCAPE)
    anchors = zc.anchors  # (94117, 1024) 已归一化
    print(f"  字场: {zc.num_hanzi}字 | 能量景观已加载")

    # 2. 加载字典数据
    print("\n[2/4] 加载多源字典...")
    decompose = json.load(open(DECOMPOSE)) if os.path.exists(DECOMPOSE) else {}
    unihan = json.load(open(UNIHAN)) if os.path.exists(UNIHAN) else {}
    print(f"  字形拆解: {len(decompose)} 字")
    print(f"  Unicode字典: {len(unihan)} 字")

    # 3. 构建多源知识边
    print("\n[3/4] 构建多源知识边...")
    edges = {}  # (a, b) -> {source: weight}

    # ── 来源1: 字形部件拆解 ──
    print("  来源1: 字形拆解 (同部件关联)...")
    n_decomp_edges = 0
    for ch, info in decompose.items():
        if ch not in zc._char_to_idx:
            continue
        for comp in info.get('components', []):
            if comp not in zc._char_to_idx or comp == ch:
                continue
            pair = tuple(sorted([ch, comp]))
            if pair not in edges:
                edges[pair] = {'decompose': 0, 'radical': 0, 'semantic': 0}
            edges[pair]['decompose'] += 0.3  # 同部件 = 0.3 强度
            n_decomp_edges += 1
    print(f"    字形关联边: {n_decomp_edges}")

    # ── 来源2: Unicode 同部首 ──
    print("  来源2: Unicode 同部首/同义关联...")
    # 按部首分组
    radical_groups = {}
    for ch, info in unihan.items():
        if ch not in zc._char_to_idx:
            continue
        rad = info.get('radical_idx', 0)
        if rad > 0:
            if rad not in radical_groups:
                radical_groups[rad] = []
            radical_groups[rad].append(ch)

    n_radical_edges = 0
    for rad, chars in radical_groups.items():
        if len(chars) < 2 or len(chars) > 500:  # 太大的部首组跳过(如"水"部)
            continue
        for i in range(len(chars)):
            for j in range(i+1, len(chars)):
                pair = tuple(sorted([chars[i], chars[j]]))
                if pair not in edges:
                    edges[pair] = {'decompose': 0, 'radical': 0, 'semantic': 0}
                edges[pair]['radical'] += 0.15  # 同部首 = 0.15 强度
                n_radical_edges += 1

    # 限制同部首边 (避免爆炸)
    if n_radical_edges > 100000:
        # 只保留前10万字形边
        sorted_edges = sorted(
            [(k, v) for k, v in edges.items() if v['radical'] > 0],
            key=lambda x: -x[1]['radical']
        )[:50000]
        new_edges = {k: v for k, v in edges.items() if v['radical'] == 0}
        for k, v in sorted_edges:
            new_edges[k] = v
        edges = new_edges
        n_radical_edges = len(sorted_edges)
    print(f"    部首关联边: {n_radical_edges}")

    # ── 来源3: 向量语义近邻 (取 top-3 而非 top-5) ──
    print("  来源3: 向量语义近邻 (top-3)...")
    n_semantic = 0
    batch = 4096
    all_chars = zc.hanzi_list
    for start in range(0, len(all_chars), batch):
        end = min(start + batch, len(all_chars))
        batch_vecs = anchors[start:end]
        sims = batch_vecs @ anchors.T
        # 排除自身
        for bj in range(batch_vecs.shape[0]):
            sims[bj, start + bj] = -float('inf')
        _, top_ids = torch.topk(sims, 3, dim=1)
        for bj in range(batch_vecs.shape[0]):
            ch_a = all_chars[start + bj]
            if ch_a not in zc._char_to_idx:
                continue
            for j in range(3):
                ch_b = all_chars[int(top_ids[bj, j])]
                pair = tuple(sorted([ch_a, ch_b]))
                if pair not in edges:
                    edges[pair] = {'decompose': 0, 'radical': 0, 'semantic': 0}
                edges[pair]['semantic'] += 0.2
                n_semantic += 1
    print(f"    语义关联边: {n_semantic}")

    # 综合权重
    for pair in edges:
        v = edges[pair]
        v['total'] = v['decompose'] + v['radical'] + v['semantic']

    total_edges = len(edges)
    involved_chars = set()
    for a, b in edges:
        involved_chars.add(a); involved_chars.add(b)
    print(f"\n  总知识边: {total_edges}")
    print(f"  涉及汉字: {len(involved_chars)}")

    # 4. 注入能量景观
    print("\n[4/4] 批量梯度下降注入...")
    ls.train()
    optimizer = torch.optim.Adam(ls.parameters(), lr=5e-5)
    steps = 100
    batch_size = 512

    # 取高频边（权重最高的）
    sorted_pairs = sorted(edges.items(), key=lambda x: -x[1]['total'])[:50000]
    pair_chars = set()
    for (a, b), _ in sorted_pairs:
        pair_chars.add(a); pair_chars.add(b)

    pair_indices = [zc._char_to_idx[c] for c in pair_chars if c in zc._char_to_idx]
    print(f"  注入 {len(sorted_pairs)} 条边, {len(pair_indices)} 个锚点")

    t0 = time.time()
    for step in range(steps):
        # 随机采样锚点
        if len(pair_indices) > batch_size:
            idx = torch.randperm(len(pair_indices))[:batch_size]
            batch = anchors[pair_indices][idx]
        else:
            batch = anchors[pair_indices]

        optimizer.zero_grad()
        energy = ls.energy(batch)
        loss = energy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ls.parameters(), 0.5)
        optimizer.step()

        if (step+1) % 20 == 0:
            with torch.no_grad():
                avg_e = ls.energy(anchors[pair_indices]).mean().item()
            print(f"  step {step+1}/{steps} loss={loss.item():.4f} avg_energy={avg_e:.3f}")

    ls.eval()
    ls.save(OUT)
    elapsed = time.time() - t0

    # 验证
    with torch.no_grad():
        e_all = ls.energy(anchors[:1000]).mean().item()
        e_focused = ls.energy(anchors[list(pair_indices)[:500]]).mean().item()
    print(f"\n✅ 注入完成 ({elapsed:.0f}s)")
    print(f"  保存: {OUT}")
    print(f"  全量锚点平均能量: {e_all:.3f}")
    print(f"  注入区平均能量: {e_focused:.3f}")
    print(f"\n知识源: 字形{len(decompose)}字 + Unicode{len(unihan)}字 + BAAI语义")


if __name__ == "__main__":
    main()
