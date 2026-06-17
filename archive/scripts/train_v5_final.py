#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
龙珠训练 v5 — 真单侧损失 + 频率特征 + GPU加速。
可以作为模块导入（不执行训练），或直接运行。
"""
import torch, torch.nn as nn, numpy as np
import sys, os, json, random, time, signal, math

PROJECT = "/mnt/d/soso/projects/Loong-agent/Loong-pearl"
sys.path.insert(0, PROJECT)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16384
EPOCHS = 60
LR = 0.002
TARGET_ANCHOR = -15.0
TARGET_POS    = -10.0
TARGET_NEG    =   2.0
SAVE_PATH = os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt")

STOP_REQUESTED = False

def signal_handler(sig, frame):
    global STOP_REQUESTED
    print("\n⚠️  安全停止...", flush=True)
    STOP_REQUESTED = True

def log(msg):
    print(msg, flush=True)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    log("=" * 60)
    log(f"🐉 龙珠训练 v5 — 真单侧损失 + 频率感知")
    log(f"   设备: {DEVICE}")
    log("=" * 60)

    # 加载
    log("\n📦 加载...")
    field = HanziAnchorField.load(os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"), freeze=True)
    anchors_all = field.anchors.to(DEVICE)

    with open(os.path.join(PROJECT, "data/dicts/cedict_parsed.json"), encoding="utf-8") as f:
        cedict = json.load(f)

    positive_pairs = set()
    pair_freq = {}
    for word in cedict:
        if not all('\u4e00' <= c <= '\u9fff' for c in word):
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            positive_pairs.add(pair)
            pair_freq[pair] = pair_freq.get(pair, 0) + 1

    pos_indices, pos_freqs = [], []
    for a, b in positive_pairs:
        ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
        if ia is not None and ib is not None:
            pos_indices.append((ia, ib))
            pos_freqs.append(math.log1p(pair_freq.get((a, b), 1)))

    pos_tensor = torch.tensor(pos_indices, dtype=torch.long, device=DEVICE)
    pos_freqs = torch.tensor(pos_freqs, dtype=torch.float32, device=DEVICE)
    total = len(pos_indices)
    log(f"  正样本: {total} 对")

    all_idx = list(range(field.num_hanzi))
    neg_set = set()
    while len(neg_set) < total:
        ia, ib = random.sample(all_idx, 2)
        a_ch, b_ch = field.hanzi_list[ia], field.hanzi_list[ib]
        if (a_ch, b_ch) in positive_pairs or (b_ch, a_ch) in positive_pairs:
            continue
        neg_set.add((ia, ib))
    neg_indices = list(neg_set)
    neg_tensor = torch.tensor(neg_indices, dtype=torch.long, device=DEVICE)
    neg_freqs = torch.zeros(len(neg_indices), device=DEVICE)
    log(f"  负样本: {len(neg_indices)} 对")

    anchor_n = 2000
    anchor_proxy_idx = random.sample(range(min(3725, field.num_hanzi)), anchor_n)
    anchor_proxy = anchors_all[anchor_proxy_idx]
    anchor_freqs = torch.ones(anchor_n, device=DEVICE)

    log("  初始化频率感知能量景观...")
    landscape = FreqEnergyLandscape(embed_dim=field.embed_dim).to(DEVICE)
    landscape.train()
    log(f"  参数: {sum(p.numel() for p in landscape.parameters()):,}")

    optimizer = torch.optim.AdamW(landscape.parameters(), lr=LR, weight_decay=1e-6)

    log(f"\n{'='*70}")
    log(f"🚀 训练: {total}对 × {EPOCHS}轮 | bs={BATCH_SIZE}")
    log(f"{'Epoch':>5} {'Batch':>6} {'正E':>8} {'负E':>8} {'间隔':>8} {'锚E':>8} {'Loss':>8} {'耗时'}")
    log(f"{'─'*70}")

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        pos_perm = torch.randperm(total, device=DEVICE)
        neg_perm = torch.randperm(total, device=DEVICE)
        sum_pos, sum_neg, sum_anchor = 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, total, BATCH_SIZE):
            if STOP_REQUESTED:
                break
            end = min(start + BATCH_SIZE, total)

            p_idx = pos_perm[start:end]
            p_rows = pos_tensor[p_idx]
            p_mids = (anchors_all[p_rows[:,0]] + anchors_all[p_rows[:,1]]) / 2
            p_freq = pos_freqs[p_idx]

            n_idx = neg_perm[start:end]
            n_rows = neg_tensor[n_idx]
            n_mids = (anchors_all[n_rows[:,0]] + anchors_all[n_rows[:,1]]) / 2
            n_freq = neg_freqs[n_idx]

            if n_batches % 50 == 0:
                anchor_proxy_idx = random.sample(range(min(3725, field.num_hanzi)), anchor_n)
                anchor_proxy = anchors_all[anchor_proxy_idx]
            a_freq = anchor_freqs[:anchor_proxy.shape[0]]

            e_pos = landscape(p_mids, p_freq).squeeze(-1)
            e_neg = landscape(n_mids, n_freq).squeeze(-1)
            e_anchor = landscape(anchor_proxy, a_freq).squeeze(-1)

            # 真单侧 + 频率加权目标 + 微弱L2刹车
            # 高频词对目标更深：freq=5(高频)→-20, freq=0.7(低频)→-11
            freq_target = TARGET_POS - p_freq * 2.0
            pos_loss = (torch.clamp(e_pos - freq_target, min=0)**2).mean() + 0.0005*(e_pos**2).mean()
            neg_loss = (torch.clamp(TARGET_NEG - e_neg, min=0)**2).mean() + 0.0005*(e_neg**2).mean()
            anchor_loss = (torch.clamp(e_anchor - TARGET_ANCHOR, min=0)**2).mean() + 0.0005*(e_anchor**2).mean()
            loss = pos_loss * 0.5 + neg_loss * 2.0 + anchor_loss * 1.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(landscape.parameters(), 1.0)
            optimizer.step()

            sum_pos += e_pos.mean().item()
            sum_neg += e_neg.mean().item()
            sum_anchor += e_anchor.mean().item()
            n_batches += 1

            e_pos_m = e_pos.mean().item()
            e_neg_m = e_neg.mean().item()
            elapsed = time.time() - ep_start
            log(f"{epoch:>5} {n_batches:>4}/{len(range(0,total,BATCH_SIZE))} "
                f"{e_pos_m:>8.2f} {e_neg_m:>8.2f} {e_neg_m-e_pos_m:>8.2f} "
                f"{e_anchor.mean().item():>8.2f} {loss.item():>8.2f} {elapsed:>5.1f}s")

        if STOP_REQUESTED:
            break
        dur = time.time() - ep_start
        avg_pos = sum_pos/n_batches
        avg_neg = sum_neg/n_batches
        avg_anchor = sum_anchor/n_batches
        log(f"{'─'*70}")
        log(f"  ✅ E{epoch}: 通道={avg_pos:.2f} 墙={avg_neg:.2f} "
            f"间隔={avg_neg-avg_pos:.2f} 锚={avg_anchor:.2f} ({dur:.1f}s)")
        log("")

    # 保存
    landscape.eval()
    log(f"\n💾 保存...")
    landscape_cpu = FreqEnergyLandscape(embed_dim=field.embed_dim)
    landscape_cpu.load_state_dict(landscape.state_dict())
    landscape_cpu.save(SAVE_PATH)

    # 验证
    anchors_cpu = field.anchors
    log(f"\n🧪 验证:")
    tests = [
        ('中','国', pair_freq.get(('中','国'),0)),
        ('大','学', pair_freq.get(('大','学'),0)),
        ('龙','珠', pair_freq.get(('龙','珠'),0)),
        ('中','龘', 0), ('龙','𰻝', 0),
    ]
    with torch.no_grad():
        for a,b,freq in tests:
            ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
            if ia is not None and ib is not None:
                mid = (anchors_cpu[ia] + anchors_cpu[ib]) / 2
                f = torch.tensor([math.log1p(freq)], dtype=torch.float32)
                e = landscape_cpu(mid.unsqueeze(0), f).item()
                tag = "🔵通道" if e < -5 else ("🟡" if e < 0 else "🔴墙")
                log(f"  {a}↔{b} (f={freq}): {e:+.2f} {tag}")
    log("✅ 完成!")

if __name__ == "__main__":
    main()
