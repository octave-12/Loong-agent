#!/usr/bin/env python3
"""
纯能量景观成语接龙 — 修复版
═══════════════════════════════════════════════
设计原则:
  1. 能量景观的角色是「精排」而非「海选」
  2. 候选池由尾字匹配规则约束（head_idx[tail_char]）
  3. 能量景观为每条候选成语打分（连续字对能量之和）
  4. 温度采样保证多样性

之前 bug: best_next("天") 把「天」对所有首字（一/万/罔…）打分，
        而不是只看以「天」开头的成语。等于让景观在全字典里猜，
        而非在匹配池里排序。
"""
import sys, os, json, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 加载 ──
field = HanziAnchorField.load(f'{P}/data/models/zichang_94117_1024d.pt', freeze=True)
ls = FreqEnergyLandscape.load(f'{P}/data/models/energy_landscape_1024d.pt')
ls = ls.to(DEVICE)
ls.eval()

idioms = json.load(open(f'{P}/data/dicts/idioms.json'))
head_idx = {}  # {'天': ['天衣无缝', '天长地久', ...], '一': [...], ...}
for w in idioms:
    if len(w) == 4:
        head_idx.setdefault(w[0], []).append(w)

ci = field._char_to_idx.get  # char → index

@torch.no_grad()
def score_idiom(idiom_str):
    """计算一条成语的能量分数：连续字对能量之和（越低越「自然」）"""
    chars = list(idiom_str)
    total_energy = 0.0
    n_pairs = 0
    for a, b in zip(chars, chars[1:]):
        ia, ib = ci(a), ci(b)
        if ia is None or ib is None:
            return float('inf')
        mid = (field.anchors[ia] + field.anchors[ib]) / 2
        mid = mid.to(DEVICE)
        e = ls(mid.unsqueeze(0)).item()
        total_energy += e
        n_pairs += 1
    return total_energy / max(n_pairs, 1)  # 平均每对能量


def best_next(tail_char, exclude=set(), k=5, temperature=1.5):
    """
    能量景观精排：在匹配尾字的候选池中，按能量打分，温度采样。
    
    返回: [(idiom_str, avg_energy), ...]
    """
    candidates = head_idx.get(tail_char, [])
    if not candidates:
        return []
    
    # 去重 + 排除已用
    candidates = [w for w in candidates if w not in exclude]
    if not candidates:
        return []
    
    # GPU 批量打分 — 计算每条成语的所有连续字对中点，一次前向
    scores = []
    for idiom in candidates:
        e = score_idiom(idiom)
        if e != float('inf'):
            scores.append((idiom, e))
    
    if not scores:
        return []
    
    # 按能量排序（低能量 = 更自然的成语）
    scores.sort(key=lambda x: x[1])
    
    # 温度采样：转换为概率分布
    energies = torch.tensor([e for _, e in scores], device=DEVICE)
    # 能量越低 → 分数越高
    scaled = -energies / temperature
    probs = torch.softmax(scaled, dim=0)
    
    # 采样 k*2 个，去重取前 k
    n_sample = min(k * 2, len(scores))
    sampled_idx = torch.multinomial(probs, n_sample, replacement=False)
    
    seen = set()
    results = []
    for i in sampled_idx:
        idiom, e = scores[i.item()]
        if idiom not in seen:
            seen.add(idiom)
            results.append((idiom, e))
        if len(results) >= k:
            break
    
    return results


def dump_candidates(tail_char, top_n=8):
    """调试: 列出尾字匹配的候选成语及能量排名"""
    candidates = head_idx.get(tail_char, [])
    if not candidates:
        print(f"  '{tail_char}' → 无匹配成语")
        return
    
    scored = []
    for w in candidates:
        e = score_idiom(w)
        if e != float('inf'):
            scored.append((w, e))
    
    scored.sort(key=lambda x: x[1])
    print(f"  '{tail_char}' 候选 Top{top_n}:")
    for i, (w, e) in enumerate(scored[:top_n]):
        bar = '▒' * max(1, int(abs(e) * 2))
        print(f"    {i+1}. {w}  能量={e:+.2f} {bar}")
    print()


# ── 接龙 ──
print("🐉 纯能量景观成语接龙（修复版）")
print("═" * 50)
print(f"候选池: {len(head_idx)} 首字索引 | 设备: {DEVICE}")
print()

# 先看一组尾字的候选排名
for ch in ['天', '龙', '成', '红', '境']:
    dump_candidates(ch, top_n=6)

print("接龙结果:")
print("─" * 50)

seeds = ['一飞冲天', '龙飞凤舞', '心想事成', '万紫千红', '学无止境']
for start in seeds:
    chain = [start]
    tail = start[-1]
    
    for _ in range(10):
        next_candidates = best_next(tail, exclude=set(chain), k=5, temperature=1.2)
        if not next_candidates:
            break
        # 取第一个（已按能量排序+温度采样）
        chosen, energy = next_candidates[0]
        chain.append(chosen)
        tail = chosen[-1]
    
    arrow = ' → '.join(chain)
    print(f"  {len(chain)}步 | {arrow[:150]}")

print(f"\n纯能量景观精排 | LLM: 0")
