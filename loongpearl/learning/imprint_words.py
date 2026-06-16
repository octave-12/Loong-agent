#!/usr/bin/env python3
"""
印刻词语关联到能量景观 (imprint_words.py)
=========================================
将 99K 词语 + 17K 成语的字符关联注入能量景观
→ 相关字之间形成低能通道

原理:
  "中国" → 中↔国 低能路径
  "画龙点睛" → 画↔龙, 龙↔点, 点↔睛 低能路径
  路径学习 → 梯度下降降低路径中点能量
"""
import sys, os, json, time
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根
ZICHANG = os.path.join(BASE, "data/models/zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "data/models/energy_landscape_1024d.pt")
CEDICT = os.path.join(BASE, "data/dicts/cedict_parsed.json")

def build_word_pairs(zc, cedict, known_chars):
    """从词语构建字符关联对"""
    pairs = set()
    for word in cedict:
        if len(word) < 2: continue
        if not all(ch in known_chars and ch in zc._char_to_idx for ch in word):
            continue
        chars = list(word)
        for i in range(len(chars)-1):
            a, b = chars[i], chars[i+1]
            if a != b:
                pairs.add((a, b) if a < b else (b, a))
        if len(word) == 4:
            a, b = chars[0], chars[-1]
            if a != b:
                pairs.add((a, b) if a < b else (b, a))
    return pairs

def build_decompose_pairs(zc, known_chars):
    """从部件拆解构建关联对 (明↔日, 明↔月)"""
    decompose = json.load(open(os.path.join(BASE, "data/dicts/dict_decompose.json")))
    pairs = set()
    for ch, info in decompose.items():
        if ch not in known_chars or ch not in zc._char_to_idx:
            continue
        comps = info.get('components', [])
        for comp in comps:
            if comp in zc._char_to_idx:
                pairs.add((ch, comp) if ch < comp else (comp, ch))
    return pairs

def path_learn(ls, zc, pairs, steps=300, lr=5e-5):
    """路径学习: 降低关联字之间的能量"""
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
            if a not in zc._char_to_idx or b not in zc._char_to_idx:
                continue
            va = anchors[zc._char_to_idx[a]]
            vb = anchors[zc._char_to_idx[b]]
            for t in np.linspace(0, 1, n_points):
                batch.append(torch.nn.functional.normalize(va*(1-t)+vb*t, dim=-1))
        
        if not batch: continue
        bt = torch.stack(batch)
        opt.zero_grad()
        loss = ls.energy(bt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ls.parameters(), 0.5)
        opt.step()
        
        if (step+1) % 50 == 0:
            with torch.no_grad():
                ea = ls.energy(anchors[:200]).mean().item()
                ep = ls.energy(bt[:200]).mean().item()
            print(f"  step {step+1}/{steps}  锚点能量={ea:.2f}  路径能量={ep:.2f}")
    
    ls.eval()
    return ls


# ── Main ──
print("🐉 印刻词语关联到能量景观")
print("=" * 50)

# 加载
zc = HanziAnchorField.load(ZICHANG)
ls = EnergyLandscape.load(LANDSCAPE)

# 加载已知字
from loongpearl.learning.curriculum import BabyCurriculum
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR
baby = BabyCurriculum()
known = baby.known_chars
print(f"已知字: {len(known)}")

# 加载词典
cedict = json.load(open(CEDICT))
print(f"词典词条: {len(cedict)}")

# 构建关联对 (词语 + 部件拆解)
print("\n构建关联对...")
wp = build_word_pairs(zc, cedict, known)
dp = build_decompose_pairs(zc, known)
all_pairs = wp | dp
print(f"  词语关联: {len(wp)} 对")
print(f"  部件关联: {len(dp)} 对")
print(f"  总计:     {len(all_pairs)} 对")

# 评估当前状态
with torch.no_grad():
    e0 = ls.energy(zc.anchors[:500]).mean().item()
print(f"\n当前锚点能量: {e0:.2f}")

# 路径学习
print(f"\n路径学习 (300步)...")
t0 = time.time()
ls = path_learn(ls, zc, all_pairs, steps=300, lr=5e-5)

# 保存
ls.save(LANDSCAPE)
elapsed = time.time() - t0

with torch.no_grad():
    e1 = ls.energy(zc.anchors[:500]).mean().item()
print(f"\n✅ 印刻完成 ({elapsed:.0f}s)")
print(f"   锚点能量: {e0:.2f} → {e1:.2f}")
print(f"   已保存: {LANDSCAPE}")
