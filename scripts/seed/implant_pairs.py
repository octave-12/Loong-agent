#!/usr/bin/env python3
"""将 vector_pairs.json 批量植入能量景观 — 一次性梯度下降"""
import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "energy_landscape_1024d.pt")
PAIRS_IN = os.path.join(BASE, "vector_pairs.json")

def main():
    print("加载字场...")
    zc = HanziAnchorField.load(ZICHANG)
    anchors = zc.anchors

    print("加载能量景观...")
    ls = EnergyLandscape.load(LANDSCAPE)
    ls.train()

    print("加载关联对...")
    data = json.load(open(PAIRS_IN))
    pairs = data["pairs"]
    print(f"关联对: {len(pairs)}")

    # 收集所有涉及的锚点（去重）
    all_chars = set()
    for a, b in pairs:
        all_chars.add(a); all_chars.add(b)
    print(f"涉及锚点: {len(all_chars)}")

    # 获取锚点索引
    char_indices = []
    for ch in all_chars:
        if ch in zc._char_to_idx:
            char_indices.append(zc._char_to_idx[ch])
    anchor_vecs = anchors[char_indices].clone().detach()

    # 批量梯度下降：降低所有锚点的能量
    optimizer = torch.optim.Adam(ls.parameters(), lr=1e-4)
    total_steps = 200
    batch_size = 1024

    print(f"\n批量植入: {len(char_indices)} 锚点 × {total_steps} 步")
    t0 = time.time()

    for step in range(total_steps):
        # 随机采样一批锚点
        perm = torch.randperm(len(char_indices))[:batch_size]
        batch = anchor_vecs[perm]

        optimizer.zero_grad()
        energy = ls.energy(batch)
        loss = energy.mean()
        loss.backward()
        # 梯度裁剪防止震荡
        torch.nn.utils.clip_grad_norm_(ls.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 20 == 0:
            # 评估所有锚点平均能量
            with torch.no_grad():
                total_e = ls.energy(anchor_vecs).mean().item()
            print(f"  step {step+1}/{total_steps} loss={loss.item():.4f} avg_energy={total_e:.3f}")

    # 保存
    out = LANDSCAPE.replace(".pt", "_vector_seeded.pt")
    ls.save(out)
    elapsed = time.time() - t0
    print(f"\n✅ 植入完成: {out} ({elapsed:.1f}s)")

    # 验证
    ls.eval()
    with torch.no_grad():
        e_before = ls.energy(anchors[:500]).mean().item()
        e_sample = ls.energy(anchor_vecs[:500]).mean().item()
    print(f"  前500字场锚点平均能量: {e_before:.3f}")
    print(f"  植入锚点样本平均能量: {e_sample:.3f}")

if __name__ == "__main__":
    main()
