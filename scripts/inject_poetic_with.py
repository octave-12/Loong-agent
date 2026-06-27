#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POETIC_WITH 字对注入 — 从概念图提取对仗关系注入能量景观"""

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
    print("🐉 POETIC_WITH 对仗字对注入")
    print(f"   设备: {device}\n")

    # ── 1. 加载模型 ──
    t0 = time.time()
    print("加载字场...")
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
        freeze=True
    )

    print("加载能量景观...")
    landscape = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    ).to(device)

    print("加载概念图...")
    cg = ConceptGraph(field, landscape)
    cg.load(os.path.join(PROJECT, 'data/models/concept_graph.json'))

    print("初始化学习器...")
    learner = DragonBallLearner(landscape, field)

    print(f"   加载耗时: {time.time() - t0:.1f}s\n")

    # ── 2. 提取 POETIC_WITH 字对 ──
    print("从概念图提取 POETIC_WITH 字对...")
    t0 = time.time()
    
    pairs = []
    seen = set()
    for t in cg.triples.values():
        if t.relation != 'POETIC_WITH':
            continue
        s = t.subject
        o = t.object
        if len(s) != 1 or len(o) != 1:
            continue
        if not ('\u4e00' <= s <= '\u9fff' and '\u4e00' <= o <= '\u9fff'):
            continue
        ia = field._char_to_idx.get(s)
        ib = field._char_to_idx.get(o)
        if ia is None or ib is None or ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key not in seen:
            seen.add(key)
            pairs.append((ia, ib))

    print(f"   提取完成: {len(pairs)} 对 ({time.time() - t0:.1f}s)\n")

    if not pairs:
        print("❌ 未找到 POETIC_WITH 字对")
        sys.exit(1)

    # ── 3. 注入前评估 ──
    print("注入前评估...")
    anchors = field.anchors
    sample_size = min(2000, len(pairs))
    sample = np.random.choice(len(pairs), sample_size, replace=False)
    
    idx_a = torch.tensor([pairs[i][0] for i in sample])
    idx_b = torch.tensor([pairs[i][1] for i in sample])
    mids = (anchors[idx_a] + anchors[idx_b]) / 2.0

    with torch.no_grad():
        e_before = landscape(mids).mean().item()

    random_vecs = torch.randn(sample_size, 1024)
    random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)
    with torch.no_grad():
        e_random = landscape_eval = landscape(random_vecs).mean().item()

    sep_before = e_random - e_before
    print(f"   已知字对能量: {e_before:+.2f}")
    print(f"   随机向量能量: {e_random:+.2f}")
    print(f"   分离度: {sep_before:.2f}\n")

    # ── 4. 注入 ──
    print(f"开始注入 {len(pairs)} 对 POETIC_WITH 字对 (feedback=0.5)...")
    t0 = time.time()

    result = learner.learn_pairs_batch(
        pairs,
        learning_rate=0.0005,
    )

    inject_time = time.time() - t0
    print(f"   注入耗时: {inject_time:.1f}s")
    print(f"   注入结果: {result}\n")

    # ── 5. 注入后评估 ──
    print("注入后评估...")
    with torch.no_grad():
        e_after = landscape(mids).mean().item()

    sep_after = e_random - e_after
    print(f"   已知字对能量: {e_after:+.2f}")
    print(f"   随机向量能量: {e_random:+.2f}")
    print(f"   分离度: {sep_before:.2f} → {sep_after:.2f} "
          f"({'↑' if sep_after > sep_before else '↓'}"
          f"{abs(sep_after - sep_before):.2f})\n")

    # ── 6. 保存 ──
    model_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    landscape.save(model_path)
    print(f"💾 模型已保存: {model_path}")

    print(f"\n✅ POETIC_WITH 注入完成: {len(pairs)} 对, 分离度 {sep_before:.1f}→{sep_after:.1f}")

finally:
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)
        print(f"🔓 注入锁已释放: {LOCK_PATH}")
