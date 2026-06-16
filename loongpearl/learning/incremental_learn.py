#!/usr/bin/env python3
"""
龙珠增量学习器（incremental_learn.py）
======================================
单一能量景观文件，每次学习叠加在上面，永不分裂。

用法:
    python incremental_learn.py --source vector   # 向量近邻路径学习
    python incremental_learn.py --source dicts    # 字典(字形+部首)路径学习
    python incremental_learn.py --source all      # 全量增量学习

始终读写同一个文件: energy_landscape_1024d.pt
"""
import sys, os, json, time
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根
ZICHANG = os.path.join(BASE, "data/models/zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "data/models/energy_landscape_1024d.pt")
DECOMPOSE = os.path.join(BASE, "data/dicts/dict_decompose.json")
UNIHAN = os.path.join(BASE, "data/dicts/dict_unihan.json")


def build_pairs_vector(zc, k=3) -> set:
    """从向量语义构建关联对"""
    print("  [向量] 构建语义近邻对 (top-3)...")
    anchors = zc.anchors
    all_chars = zc.hanzi_list
    pairs = set()
    batch = 4096
    for start in range(0, len(all_chars), batch):
        end = min(start+batch, len(all_chars))
        bv = anchors[start:end]
        sims = bv @ anchors.T
        for bj in range(bv.shape[0]): sims[bj, start+bj] = -float('inf')
        _, ids = torch.topk(sims, k, dim=1)
        for bj in range(bv.shape[0]):
            a = all_chars[start+bj]
            for j in range(k):
                b = all_chars[int(ids[bj,j])]
                if b != a: pairs.add(tuple(sorted([a,b])))
    return pairs


def build_pairs_decompose(zc) -> set:
    """从字形拆解构建关联对"""
    print("  [字形] 构建部件关联对...")
    if not os.path.exists(DECOMPOSE): return set()
    decomp = json.load(open(DECOMPOSE))
    pairs = set()
    for ch, info in decomp.items():
        if ch not in zc._char_to_idx: continue
        for comp in info.get('components', []):
            if comp in zc._char_to_idx and comp != ch:
                pairs.add(tuple(sorted([ch, comp])))
    return pairs


def build_pairs_unihan(zc) -> set:
    """从 Unicode 部首构建关联对"""
    print("  [部首] 构建同部首关联对...")
    if not os.path.exists(UNIHAN): return set()
    unihan = json.load(open(UNIHAN))
    # 按部首分组
    groups = {}
    for ch, info in unihan.items():
        if ch not in zc._char_to_idx: continue
        rad = info.get('radical_idx', 0)
        if rad > 0:
            groups.setdefault(rad, []).append(ch)
    pairs = set()
    for rad, chars in groups.items():
        if len(chars) < 2 or len(chars) > 300: continue
        for i in range(len(chars)):
            for j in range(i+1, min(i+10, len(chars))):  # 限制组内边数
                pairs.add(tuple(sorted([chars[i], chars[j]])))
    return pairs


def path_learn(ls, zc, pairs, steps=200, lr=1e-4):
    """路径学习: 在每对锚点间插值, 降低路径能量"""
    anchors = zc.anchors
    plist = list(pairs)
    n_points = 5
    batch_pairs = 256

    ls.train()
    opt = torch.optim.Adam(ls.parameters(), lr=lr)

    for step in range(steps):
        idx = np.random.choice(len(plist), min(batch_pairs, len(plist)), replace=False)
        batch = []
        for i in idx:
            a, b = plist[i]
            va = anchors[zc._char_to_idx[a]]
            vb = anchors[zc._char_to_idx[b]]
            for t in np.linspace(0, 1, n_points):
                batch.append(torch.nn.functional.normalize(va*(1-t)+vb*t, dim=-1))

        bt = torch.stack(batch)
        opt.zero_grad()
        loss = ls.energy(bt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ls.parameters(), 0.5)
        opt.step()

        if (step+1) % 40 == 0:
            with torch.no_grad():
                ea = ls.energy(anchors[:200]).mean().item()
                ep = ls.energy(bt[:200]).mean().item()
            print(f"    step {step+1}/{steps} 锚点={ea:.2f} 路径={ep:.2f}")

    ls.eval()
    return ls


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["vector","dicts","all"], default="all")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    print("🐉 龙珠增量学习器")
    print(f"   主文件: {LANDSCAPE}")
    print(f"   学习源: {args.source}")
    print("="*50)

    # 加载
    zc = HanziAnchorField.load(ZICHANG)
    ls = EnergyLandscape.load(LANDSCAPE)
    with torch.no_grad():
        e0 = ls.energy(zc.anchors[:500]).mean().item()
    print(f"   当前锚点能量: {e0:.2f}")

    # 构建关联对
    all_pairs = set()
    if args.source in ("vector", "all"):
        vp = build_pairs_vector(zc)
        print(f"   向量对: {len(vp)}")
        all_pairs |= vp

    if args.source in ("dicts", "all"):
        dp = build_pairs_decompose(zc)
        up = build_pairs_unihan(zc)
        print(f"   字形对: {len(dp)} | 部首对: {len(up)}")
        all_pairs |= dp | up

    print(f"   总关联对: {len(all_pairs)}")

    if not all_pairs:
        print("   无新关联对，跳过")
        return

    # 路径学习
    print(f"\n  路径学习 ({args.steps}步)...")
    t0 = time.time()
    ls = path_learn(ls, zc, all_pairs, steps=args.steps, lr=args.lr)

    # 保存（覆盖主文件）
    ls.save(LANDSCAPE)
    elapsed = time.time() - t0

    with torch.no_grad():
        e1 = ls.energy(zc.anchors[:500]).mean().item()
    print(f"\n✅ 增量完成 ({elapsed:.0f}s)")
    print(f"   锚点能量: {e0:.2f} → {e1:.2f}")
    print(f"   已保存: {LANDSCAPE}")


if __name__ == "__main__":
    main()
