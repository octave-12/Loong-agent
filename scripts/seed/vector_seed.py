#!/usr/bin/env python3
"""龙珠向量近邻播种 — 纯向量计算存对, 秒级全量, 后续批量植入"""
import sys, os, json, time, argparse, gc
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
PAIRS_OUT = os.path.join(BASE, "vector_pairs.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k","-k",type=int,default=5,help="每字关联数")
    ap.add_argument("--batch","-b",type=int,default=4096,help="批量大小")
    args = ap.parse_args()

    print("加载字场...")
    zc = HanziAnchorField.load(ZICHANG)
    anchors = zc.anchors  # (94117, 1024), 已归一化
    all_chars = zc.hanzi_list
    print(f"字场: {len(all_chars)}字 × {anchors.shape[1]}d")
    print(f"k={args.k} batch={args.batch}\n")

    # 收集已播种的字（从之前的 LLM 播种结果）
    seeded = set()
    for f in ["seed_v7_result.json","seed_v7b_result.json","seed_v6_result.json","seed_final.json"]:
        p = os.path.join(BASE, f)
        if os.path.exists(p) and os.path.getsize(p) > 100:
            seeded.update(json.load(open(p)).get("chars",[]))

    pending = [(i, c) for i, c in enumerate(all_chars) if c not in seeded]
    print(f"已播: {len(seeded)} | 待播: {len(pending)}")

    all_pairs = set()
    t0 = time.time()

    for start in range(0, len(pending), args.batch):
        batch = pending[start:start+args.batch]
        indices = [i for i, _ in batch]
        batch_vecs = anchors[indices]  # (B, 1024)

        # 点积 = cos sim (向量已归一化)
        sims = batch_vecs @ anchors.T  # (B, 94117)
        # 排除自身
        for bj, gi in enumerate(indices):
            sims[bj, gi] = -float('inf')

        _, top_ids = torch.topk(sims, args.k, dim=1)

        for bj, (gi, ch) in enumerate(batch):
            for j in range(args.k):
                tgt_i = int(top_ids[bj, j])
                tgt_c = all_chars[tgt_i]
                pair = tuple(sorted([ch, tgt_c]))
                all_pairs.add(pair)

        # 进度 + 释放内存
        done = start + len(batch)
        el = time.time() - t0
        rate = done / max(el, 1)
        eta = (len(pending) - done) / max(rate, 0.01)
        print(f"  [{done}/{len(pending)}] {done/len(pending)*100:.1f}% "
              f"rate={rate:.0f}/s pairs={len(all_pairs)} eta={eta:.0f}s")
        del sims, top_ids, batch_vecs
        gc.collect()

    # 保存
    pairs_list = [list(p) for p in all_pairs]
    json.dump({"pairs": pairs_list, "total": len(pairs_list), "k": args.k},
              open(PAIRS_OUT, "w"), ensure_ascii=False)
    elapsed = time.time() - t0
    print(f"\n✅ 完成: {len(pairs_list)} 对 | {elapsed:.1f}s | {len(pending)/elapsed:.0f}字/s")
    print(f"   保存至: {PAIRS_OUT}")
    print(f"   下一步: python implant_pairs.py 将关联植入能量景观")

if __name__ == "__main__":
    main()
