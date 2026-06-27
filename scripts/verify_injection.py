#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""概念图注入效果验证 — 5项测试"""

import sys, os, random, torch, time
import numpy as np

PROJECT = '/mnt/d/soso/projects/Loong-agent'
sys.path.insert(0, os.path.join(PROJECT, 'loongpearl', 'core'))

from zichang import HanziAnchorField
from freq_landscape import FreqEnergyLandscape

print("=" * 60)
print("🐉 概念图注入效果验证")
print("=" * 60)

# ── 加载模型 ──
t0 = time.time()
zf = HanziAnchorField.load(os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'))
landscape = FreqEnergyLandscape.load(os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt'))
landscape.eval()
print(f"\n📦 模型加载: {time.time()-t0:.1f}s | 字场={zf.num_hanzi}字 | 维度={zf.embed_dim}d\n")

# ═══════════════════════════════════════════════════
# 测试1: 能量分离度
# ═══════════════════════════════════════════════════
print("=" * 60)
print("测试1: 能量分离度")
print("=" * 60)

anchors = zf.anchors[:1000]
random_pts = torch.nn.functional.normalize(torch.randn(1000, 1024), dim=1)

with torch.no_grad():
    anchor_e = landscape.energy(anchors).mean().item()
    random_e = landscape.energy(random_pts).mean().item()

separation = abs(random_e) / (abs(anchor_e) + 1e-8)
print(f"  锚点平均能量: {anchor_e:+.2f}")
print(f"  随机点平均能量: {random_e:+.2f}")
print(f"  能量分离度: {separation:.2f}x")
if separation > 2.0:
    print(f"  评估: ✅ 锚点形成明显低能盆地，知识结构良好")
elif separation > 1.0:
    print(f"  评估: ⚠️ 锚点能量略低，建议继续在线学习")
else:
    print(f"  评估: ❌ 分离度不足")

# ═══════════════════════════════════════════════════
# 测试2: 对仗关联验证
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("测试2: 对仗关联验证 (POETIC_WITH)")
print("=" * 60)

test_pairs = [
    ("天", "地"), ("山", "海"), ("日", "月"),
    ("风", "雨"), ("花", "鸟"), ("春", "秋"),
    ("龙", "凤"), ("金", "玉"), ("红", "绿"),
    ("江", "河"),
]
print(f"  {'字对':<10} {'中点能量':>10} {'状态'}")
print(f"  {'─'*10} {'─'*10} {'─'*20}")

strong = weak = failed = 0
results = []
for src, tgt in test_pairs:
    ia = zf._char_to_idx.get(src)
    ib = zf._char_to_idx.get(tgt)
    if ia is None or ib is None:
        results.append((src, tgt, None, '不在字场'))
        continue
    mid = (zf.anchors[ia] + zf.anchors[ib]) / 2.0
    with torch.no_grad():
        e = landscape.energy(mid.unsqueeze(0)).item()
    
    # 对比随机字对能量作为参考
    ri = random.randint(0, len(zf.anchors)-1)
    rj = random.randint(0, len(zf.anchors)-1)
    rand_mid = (zf.anchors[ri] + zf.anchors[rj]) / 2.0
    with torch.no_grad():
        e_rand = landscape.energy(rand_mid.unsqueeze(0)).item()
    
    if e < e_rand:  # 对仗字对比随机字对能量更低
        strong += 1
        status = "✅ 强关联"
    elif e < 0:
        weak += 1
        status = "⚠️ 中关联"
    else:
        failed += 1
        status = "❌ 弱关联"
    
    results.append((src, tgt, e, status))
    print(f"  {src} ↔ {tgt:<6} {e:>+10.2f} {status} (随机参考={e_rand:+.1f})")

print(f"\n  汇总: 强关联={strong} 中关联={weak} 弱关联={failed}")

# ═══════════════════════════════════════════════════
# 测试3: 诗词补全能力
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("测试3: 诗词补全能力 (POETIC_NEXT)")
print("=" * 60)

test_phrases = {
    "床前明月": "光",
    "白日依山": "尽", 
    "黄河入海": "流",
    "春眠不觉": "晓",
    "锄禾日当": "午",
    "举头望明": "月",
}

hits = 0
for phrase, expected in test_phrases.items():
    last_char = phrase[-1]
    ia = zf._char_to_idx.get(last_char)
    if ia is None:
        print(f"  '{phrase}' → '{last_char}' 不在字场")
        continue
    
    start_vec = zf.anchors[ia].clone().detach()
    
    try:
        result = landscape.infer(start_vec, steps=30, lr=0.01)
        state = result.get('state', start_vec)
        
        _, chars, sims = zf.find_nearest(state.unsqueeze(0), k=10)
        new_chars = [c for c in chars if c not in phrase]
        top3 = new_chars[:3]
        
        if expected in top3:
            hits += 1
            status = f"✅ 命中"
        elif expected in new_chars[:5]:
            status = f"⚠️ Top-5"
        else:
            status = f"❌ 未命中"
        
        print(f"  '{phrase}' → 期望'{expected}' {status} (Top-3: {top3})")
    except Exception as e:
        print(f"  '{phrase}' → 推理异常: {e}")

print(f"\n  命中率: {hits}/{len(test_phrases)} ({hits/len(test_phrases)*100:.0f}%)")

# ═══════════════════════════════════════════════════
# 测试4: 知识密度统计
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("测试4: 知识密度统计")
print("=" * 60)

# 在常用3500字中随机采样1000个字对
hanzi_sample = zf.hanzi_list[:3500]
random_pairs = []
for _ in range(1000):
    src = random.choice(hanzi_sample)
    tgt = random.choice(hanzi_sample)
    while src == tgt:
        tgt = random.choice(hanzi_sample)
    random_pairs.append((src, tgt))

negative_count = 0
energies_sample = []

# 批量计算（更快）
batch = 200
for i in range(0, len(random_pairs), batch):
    chunk = random_pairs[i:i+batch]
    idx_a = torch.tensor([zf._char_to_idx[s] for s, _ in chunk])
    idx_b = torch.tensor([zf._char_to_idx[t] for _, t in chunk])
    mids = (zf.anchors[idx_a] + zf.anchors[idx_b]) / 2.0
    with torch.no_grad():
        e = landscape.energy(mids)
    negative_count += (e < 0).sum().item()
    energies_sample.extend(e.tolist())

density = negative_count / 1000 * 100
avg_e = np.mean(energies_sample)
print(f"  采样: 1000 随机字对 (常用3500字)")
print(f"  平均中点能量: {avg_e:+.2f}")
print(f"  负能量比例: {negative_count}/1000 = {density:.1f}%")
if density > 15:
    print(f"  评估: ✅ 知识密度高 — 注入有效扩大了低能区域")
elif density > 5:
    print(f"  评估: ⚠️ 知识密度中等 — 注入产生了部分关联")
else:
    print(f"  评估: ❌ 知识密度低 — 建议增加注入数据量")

# ═══════════════════════════════════════════════════
# 测试5: 冲突检测
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("测试5: 冲突检测")
print("=" * 60)

conflict_count = 0
for i in range(10):
    random_vec = torch.nn.functional.normalize(torch.randn(1024), dim=0)
    
    try:
        result = landscape.infer(random_vec, steps=30, lr=0.01,
                                  zichang=zf)
        signal = result.get('signal', 'unknown')
        
        if signal == 'conflict':
            conflict_count += 1
            detail = result.get('signal_detail', '')
            print(f"  查询{i+1}: ⚠️ 冲突 — {detail}")
        elif signal == 'certain':
            top_char = result.get('top_char', '?')
            print(f"  查询{i+1}: ✅ 确定 → '{top_char}'")
        else:
            print(f"  查询{i+1}: 🔍 {signal}")
    except Exception as e:
        print(f"  查询{i+1}: ❌ 异常 — {e}")

print(f"\n  冲突比例: {conflict_count}/10")

# ═══════════════════════════════════════════════════
# 整体评估
# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 整体评估")
print("=" * 60)

score = 0
if separation > 2.0: score += 1
if strong >= 7: score += 1
if hits >= 3: score += 1
if density > 10: score += 1
if conflict_count <= 2: score += 1

grades = {
    5: "🏆 优秀 — 知识注入效果显著，所有指标达标",
    4: "✅ 良好 — 大部分指标达标，个别维度可优化",
    3: "⚠️ 一般 — 注入产生效果但需补充训练",
    2: "🔧 待改进 — 部分维度未达预期",
    1: "❌ 差 — 注入效果不明显",
    0: "💥 失败 — 注入可能未能生效",
}

print(f"  分离度: {'✅' if separation>2 else '❌'} ({separation:.1f}x)")
print(f"  对仗关联: {'✅' if strong>=7 else '❌'} ({strong}/10)")
print(f"  诗词补全: {'✅' if hits>=3 else '❌'} ({hits}/{len(test_phrases)})")
print(f"  知识密度: {'✅' if density>10 else '❌'} ({density:.0f}%)")
print(f"  冲突控制: {'✅' if conflict_count<=2 else '❌'} ({conflict_count}/10)")
print(f"  综合评分: {score}/5")
print(f"  {grades[score]}")
