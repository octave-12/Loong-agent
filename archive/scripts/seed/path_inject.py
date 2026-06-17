#!/usr/bin/env python3
"""
龙珠路径学习注入器（path_inject.py）
====================================
正确做法: 在关联锚点对之间插值采样 → 降低路径上的能量
→ 形成"知识沟壑" → 相关概念间可沿低能路径互达

之前错误: 只挖深每个锚点盆地 → 盆地更深但彼此孤立
正确做法: 在 A↔B 之间创造低能通道 → 相关概念可互达
"""
import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
# 从原始版开始（不用之前均匀挖深的版本）
LANDSCAPE_SRC = os.path.join(BASE, "energy_landscape_1024d.pt")
LANDSCAPE_OUT = os.path.join(BASE, "energy_landscape_1024d_path.pt")

# 知识源
DECOMPOSE = os.path.join(BASE, "dict_decompose.json")
UNIHAN = os.path.join(BASE, "dict_unihan.json")

def main():
    print("🐉 龙珠路径学习 — 在关联锚点间创造低能通道")
    print("="*50)

    # 1. 加载
    print("\n[1/4] 加载原始能量景观...")
    zc = HanziAnchorField.load(ZICHANG)
    ls = EnergyLandscape.load(LANDSCAPE_SRC)
    anchors = zc.anchors
    print(f"  字场: {zc.num_hanzi}字 | 能量景观: 原始版 (锚点≈-2.02)")

    # 2. 构建知识对
    print("\n[2/4] 构建关联对...")
    pairs = set()

    # 来源1: 字形拆解 (同部件) — 权重 1.0
    if os.path.exists(DECOMPOSE):
        decomp = json.load(open(DECOMPOSE))
        for ch, info in decomp.items():
            if ch not in zc._char_to_idx: continue
            for comp in info.get('components', []):
                if comp in zc._char_to_idx and comp != ch:
                    pairs.add(tuple(sorted([ch, comp])))
    print(f"  字形对: {len(pairs)}")

    # 来源2: 向量语义近邻 (top-3, 双向) — 权重 0.5
    batch = 4096
    all_chars = zc.hanzi_list
    n_sem = 0
    for start in range(0, len(all_chars), batch):
        end = min(start+batch, len(all_chars))
        bv = anchors[start:end]
        sims = bv @ anchors.T
        for bj in range(bv.shape[0]):
            sims[bj, start+bj] = -float('inf')
        _, ids = torch.topk(sims, 3, dim=1)
        for bj in range(bv.shape[0]):
            a = all_chars[start+bj]
            if a not in zc._char_to_idx: continue
            for j in range(3):
                b = all_chars[int(ids[bj,j])]
                if b != a:
                    pairs.add(tuple(sorted([a,b])))
                    n_sem += 1
    print(f"  语义对: {n_sem}")
    print(f"  总关联对: {len(pairs)}")

    # 3. 路径学习: 在每对锚点间插值 → 降低路径能量
    print(f"\n[3/4] 路径学习 ({len(pairs)} 对)...")
    ls.train()
    opt = torch.optim.Adam(ls.parameters(), lr=1e-4)
    pairs_list = list(pairs)
    n_path_points = 5  # 每对锚点间的插值点数
    steps = 300
    batch_pairs = 256

    t0 = time.time()
    for step in range(steps):
        # 随机采样一批关联对
        idx = np.random.choice(len(pairs_list), min(batch_pairs, len(pairs_list)), replace=False)
        batch_anchors = []
        for i in idx:
            a, b = pairs_list[i]
            va = anchors[zc._char_to_idx[a]]
            vb = anchors[zc._char_to_idx[b]]
            # 在 a 和 b 之间线性插值 n 个点
            for t in np.linspace(0, 1, n_path_points):
                v_interp = torch.nn.functional.normalize(va * (1-t) + vb * t, dim=-1)
                batch_anchors.append(v_interp)

        batch_tensor = torch.stack(batch_anchors)
        opt.zero_grad()
        energy = ls.energy(batch_tensor)
        loss = energy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ls.parameters(), 0.5)
        opt.step()

        if (step+1) % 30 == 0:
            # 评估: 锚点能量 + 路径能量
            with torch.no_grad():
                e_anchors = ls.energy(anchors[:500]).mean().item()
                e_path = ls.energy(batch_tensor[:200]).mean().item()
            print(f"  step {step+1}/{steps} loss={loss.item():.4f} "
                  f"锚点={e_anchors:.2f} 路径={e_path:.2f}")

    ls.eval()
    ls.save(LANDSCAPE_OUT)
    elapsed = time.time() - t0
    print(f"  完成 ({elapsed:.0f}s) → {LANDSCAPE_OUT}")

    # 4. 验证对比
    print(f"\n[4/4] 验证...")
    with torch.no_grad():
        e_all = ls.energy(anchors).mean().item()
        e_rand = ls.energy(torch.nn.functional.normalize(torch.randn(1000,1024), dim=1)).mean().item()
        sep = (e_rand - e_all) / abs(e_all) if abs(e_all) > 0.01 else 0
    print(f"  全量锚点能量: {e_all:.2f}")
    print(f"  随机点能量: {e_rand:.2f}")
    print(f"  绝对分离: {e_rand - e_all:.2f}")
    print(f"  相对分离度: {sep:.1f}x")

    # 验证一个具体的知识边: "明→日" 是否形成通道
    if '明' in zc._char_to_idx and '日' in zc._char_to_idx:
        va = anchors[zc._char_to_idx['明']]
        vb = anchors[zc._char_to_idx['日']]
        energies_path = []
        for t in np.linspace(0, 1, 10):
            v = torch.nn.functional.normalize(va*(1-t)+vb*t, dim=-1)
            energies_path.append(ls.energy(v.unsqueeze(0)).item())
        print(f"  明→日 路径能量: {energies_path}")
        print(f"    平均: {np.mean(energies_path):.2f} | 明={energies_path[0]:.2f} 日={energies_path[-1]:.2f}")

    print("\n✅ 路径学习完成！与均匀挖深的关键区别:")
    print("  旧方式: 所有锚点 ↓↓↓  → 盆地都深了但彼此孤立")
    print("  新方式: 关联锚点间 === → 形成低能通道，知识可互达")


if __name__ == "__main__":
    main()
