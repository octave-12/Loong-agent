#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""龙珠智能接龙 v3 — 频率感知 + 死路避让 + 优先常见词"""
import torch, json, random, math, sys, os

PROJECT = "/mnt/d/soso/projects/Loong-agent/Loong-pearl"


from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

print("🐉 龙珠智能接龙 v3\n" + "=" * 50)

field = HanziAnchorField.load(os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"), freeze=True)
ls = FreqEnergyLandscape.load(os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt"))
ls.eval()

# 加载 + 统计
with open(os.path.join(PROJECT, "data/dicts/cedict_parsed.json"), encoding="utf-8") as f:
    cedict = json.load(f)

four_char = []
word_freq = {}
pair_freq = {}
for k in cedict:
    is_cn = all('\u4e00' <= c <= '\u9fff' for c in k)
    if is_cn:
        for i in range(len(k) - 1):
            pair_freq[(k[i], k[i+1])] = pair_freq.get((k[i], k[i+1]), 0) + 1
    if len(k) == 4 and is_cn:
        four_char.append(k)
        word_freq[k] = word_freq.get(k, 0) + 1

# 首字索引 + 死路检测
head_idx = {}
dead_ends = set()
for w in four_char:
    head_idx.setdefault(w[0], []).append(w)
# 检测死路：尾字没有任何词以它开头
for w in four_char:
    if w[-1] not in head_idx:
        dead_ends.add(w[-1])

known_dead = {w for w in four_char if w[-1] in dead_ends}

print(f"  四字词: {len(four_char)} | 死路尾字: {len(dead_ends)} | 死路词: {len(known_dead)}")
print(f"  锚点能量: {ls.net[-1].bias.item():.2f}\n")

# ===== 打分 =====
def cidx(ch):
    return field._char_to_idx.get(ch)

def pair_e(a, b):
    ia, ib = cidx(a), cidx(b)
    if ia is None or ib is None:
        return None
    f = pair_freq.get((a, b), 0)
    with torch.no_grad():
        mid = (field.anchors[ia] + field.anchors[ib]) / 2
        ft = torch.tensor([math.log1p(f)], dtype=torch.float32)
        return ls(mid.unsqueeze(0), ft).item()

def score_word(w):
    scores = [pair_e(w[i], w[i+1]) for i in range(len(w)-1)]
    scores = [s for s in scores if s is not None]
    return sum(scores)/len(scores) if scores else 999

def rank_candidates(candidates, tail):
    """智能排名：
       - 内部能量（越低越好，权重0.6）
       - 词频奖励（越常见越好，权重0.3）
       - 死路惩罚（尾字无出路，权重0.1）
    """
    scored = []
    for w in candidates:
        ie = score_word(w)  # 内部能量
        wf = min(word_freq.get(w, 1), 10)  # 词频，封顶10
        # 词频转换：频率越高，奖励越多（负数表示拉低能量）
        freq_bonus = -math.log1p(wf) * 5  # freq=1→0, freq=10→-12
        # 死路惩罚
        dead_penalty = 10.0 if w[-1] in dead_ends else 0.0
        # 综合
        total = ie * 0.6 + freq_bonus * 0.3 + dead_penalty * 0.1
        scored.append((w, ie, wf, freq_bonus, dead_penalty > 0, total))
    scored.sort(key=lambda x: x[-1])
    return scored

# ===== 接龙 =====
def play(start, steps=8):
    print(f"\n{'─'*50}")
    print(f"🎮 {start}")
    chain = [start]
    tail = start[-1]
    
    for i in range(steps):
        candidates = [w for w in head_idx.get(tail, []) if w not in chain]
        if not candidates:
            print(f"  ❌ '{tail}' 死胡同\n")
            break
        
        ranked = rank_candidates(candidates, tail)
        best = ranked[0]
        chain.append(best[0])
        
        alts = ""
        if len(ranked) > 1:
            as_ = [f"{w}({s:.0f})" for w,_,_,_,dead,s in ranked[1:4]]
            alts = f" ← {', '.join(as_)}"
        
        dead_tag = "💀" if best[4] else ""
        print(f"  {i+1}. {best[0]} {dead_tag} [内={best[1]:+.1f} 频={best[2]} 综={best[5]:+.1f}]{alts}")
        tail = best[0][-1]
    
    print(f"  → {' → '.join(chain)} ({len(chain)}步)")

# 跑
for s in ["龙飞凤舞", "学无止境", "心想事成", "万紫千红", "一见钟情"]:
    play(s, 8)
