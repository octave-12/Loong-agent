#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POETIC_NEXT 低强度批量注入 — 全唐诗字邻接 → 能量景观微调"""

import sys, os, json, time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import torch
import numpy as np

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.concept_graph import ConceptGraph
from loongpearl.learning.learner import DragonBallLearner

# ── 注入锁 ──
LOCK_PATH = os.path.join(PROJECT, 'data', 'runtime', 'inject.lock')
os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
with open(LOCK_PATH, 'w') as lf:
    lf.write(str(os.getpid()))
print(f"🔒 写入注入锁: {LOCK_PATH}")

try:
    device = torch.device('cpu')
    print("🐉 POETIC_NEXT 低强度批量注入")
    print(f"   策略: conf>0.001 + 单汉字 + 跳过已有负能量 + feedback=0.05")
    print(f"   设备: {device}\n")

    # ── 1. 加载模型 ──
    t_total = time.time()
    t0 = time.time()
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)
    landscape = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')).to(device)
    cg = ConceptGraph(field, landscape)
    cg.load(os.path.join(PROJECT, 'data/models/concept_graph.json'))
    learner = DragonBallLearner(landscape, field)
    print(f"   加载耗时: {time.time() - t0:.1f}s\n")

    # ── 2. 提取 + 过滤 ──
    print("提取 POETIC_NEXT 字对 (conf>0.001, 单汉字)...")
    t0 = time.time()
    
    char_to_idx = field._char_to_idx
    anchors = field.anchors
    raw_pairs = []
    seen_raw = set()

    for t in cg.triples.values():
        if t.relation != 'POETIC_NEXT':
            continue
        if t.confidence <= 0.001:
            continue
        s, o = t.subject, t.object
        if len(s) != 1 or len(o) != 1:
            continue
        if not ('\u4e00' <= s <= '\u9fff' and '\u4e00' <= o <= '\u9fff'):
            continue
        ia = char_to_idx.get(s)
        ib = char_to_idx.get(o)
        if ia is None or ib is None or ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key not in seen_raw:
            seen_raw.add(key)
            raw_pairs.append((ia, ib))

    print(f"   过滤后保留: {len(raw_pairs)} 对 ({time.time() - t0:.1f}s)")

    # ── 3. 能量预检: 跳过已为负的中 ──
    print("\n能量预检 (跳过已有强关联的字对)...")
    t0 = time.time()
    
    to_inject = []
    skipped = 0
    batch_size = 5000
    
    landscape.eval()
    for i in range(0, len(raw_pairs), batch_size):
        batch = raw_pairs[i:i + batch_size]
        idx_a = torch.tensor([p[0] for p in batch])
        idx_b = torch.tensor([p[1] for p in batch])
        mids = (anchors[idx_a] + anchors[idx_b]) / 2.0
        with torch.no_grad():
            energies = landscape(mids).squeeze()
        
        for j, e in enumerate(energies):
            if e.item() < 0:
                skipped += 1
            else:
                to_inject.append(batch[j])
    
    print(f"   跳过(已有负能量): {skipped}")
    print(f"   实际注入: {len(to_inject)} 对 ({time.time() - t0:.1f}s)")

    if not to_inject:
        print("\n⚠️ 所有字对已有强关联，无需注入")
        sys.exit(0)

    # ── 4. 分批注入 ──
    BATCH = 500
    total_batches = (len(to_inject) + BATCH - 1) // BATCH
    print(f"\n开始分批注入 (共 {total_batches} 批, feedback=0.05, lr=0.00005)...")
    t0 = time.time()
    
    # 注入前采样评估
    sample_n = min(2000, len(to_inject))
    sample_idx = np.random.choice(len(to_inject), sample_n, replace=False)
    sample_pairs = [to_inject[i] for i in sample_idx]
    sample_a = torch.tensor([p[0] for p in sample_pairs])
    sample_b = torch.tensor([p[1] for p in sample_pairs])
    sample_mids = (anchors[sample_a] + anchors[sample_b]) / 2.0
    with torch.no_grad():
        e_before = landscape(sample_mids).mean().item()
    rand_vecs = torch.randn(sample_n, 1024)
    rand_vecs = rand_vecs / rand_vecs.norm(dim=1, keepdim=True)
    with torch.no_grad():
        e_random = landscape(rand_vecs).mean().item()
    sep_before = e_random - e_before
    print(f"   注入前: 已知能量={e_before:+.2f} 随机能量={e_random:+.2f} 分离度={sep_before:.2f}")

    landscape.train()
    injected_count = 0
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH
        end = min(start + BATCH, len(to_inject))
        batch_pairs = to_inject[start:end]
        
        result = learner.learn_pairs_batch(
            batch_pairs,
            learning_rate=0.00005,  # feedback=0.05: 1/10 of normal strength
        )
        
        batch_injected = result.get('pairs_learned', 0)
        injected_count += batch_injected
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t0
            progress = (batch_idx + 1) / total_batches * 100
            sep_cur = result.get('separation_after', 0)
            print(f"  [{batch_idx+1}/{total_batches}] {progress:.0f}% | "
                  f"累计注入 {injected_count} | sep={sep_cur:.1f} | {elapsed:.0f}s")

    inject_time = time.time() - t0
    
    # ── 5. 注入后评估 ──
    landscape.eval()
    with torch.no_grad():
        e_after = landscape(sample_mids).mean().item()
    sep_after = e_random - e_after
    
    print(f"\n{'='*50}")
    print(f"注入后评估:")
    print(f"   已知字对能量: {e_before:+.2f} → {e_after:+.2f}")
    print(f"   分离度: {sep_before:.2f} → {sep_after:.2f} "
          f"({'↑' if sep_after > sep_before else '↓'}{abs(sep_after - sep_before):.2f})")
    print(f"   总耗时: {inject_time:.1f}s")
    print(f"   总耗时(含加载): {time.time() - t_total:.1f}s")

    # ── 6. 保存 ──
    model_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    landscape.save(model_path)
    print(f"\n💾 模型已保存: {model_path}")

    # ── 7. 汇总 ──
    print(f"\n{'='*50}")
    print(f"✅ POETIC_NEXT 低强度注入完成")
    print(f"   过滤后保留: {len(raw_pairs)} 对")
    print(f"   跳过(已有强关联): {skipped} 对")
    print(f"   实际注入: {len(to_inject)} 对")
    print(f"   总耗时: {time.time() - t_total:.1f}s")

finally:
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)
        print(f"🔓 注入锁已释放: {LOCK_PATH}")
