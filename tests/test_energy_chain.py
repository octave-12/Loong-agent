#!/usr/bin/env python3
"""纯能量景观接龙 — 不依赖JSON词典查表"""
import sys,os,json,math,torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

P=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

field=HanziAnchorField.load(f'{P}/data/models/zichang_94117_1024d.pt',freeze=True)
ls=FreqEnergyLandscape.load(f'{P}/data/models/energy_landscape_1024d.pt')
ls.eval()

idioms=json.load(open(f'{P}/data/dicts/idioms.json'))
head_idx={}
for w in idioms: head_idx.setdefault(w[0],[]).append(w)

ci=field._char_to_idx.get

def best_next(tail_char, exclude=set(), k=5, temperature=2.0):
    """能量景观排序 + 温度采样: 避免所有链收敛到同一路径"""
    if tail_char not in field._char_to_idx: return None
    idx = field._char_to_idx[tail_char]
    src = field.anchors[idx]
    
    candidates = [(ch, field._char_to_idx[ch]) for ch in head_idx if ch not in exclude]
    if not candidates: return None
    
    with torch.no_grad():
        src_batch = src.unsqueeze(0).expand(len(candidates), -1)
        tgt_indices = [ci for _, ci in candidates]
        tgt_batch = field.anchors[torch.tensor(tgt_indices)]
        mids = (src_batch + tgt_batch) / 2
        energies = ls(mids).squeeze()
        
        # 温度缩放 + softmax → 概率分布 → 加权随机采样
        scores = -energies / temperature  # 能量越低分数越高
        probs = torch.softmax(scores, dim=0)
        
        # 按概率采样 k 个不重复的
        sampled = torch.multinomial(probs, min(k*3, len(candidates)), replacement=False)
        results = []
        for i in sampled:
            ch = candidates[i][0]
            results.append((ch, energies[i].item()))
        return results[:k]

for start in ['一飞冲天','龙飞凤舞','心想事成','万紫千红','学无止境']:
    chain=[start]; tail=start[-1]
    for i in range(10):
        next_chars=best_next(tail, exclude=set(chain))
        if not next_chars: break
        found=False
        for ch,e in next_chars:
            if ch in head_idx:
                cands=[w for w in head_idx[ch] if w not in chain]
                if cands:
                    chain.append(cands[0]); tail=cands[0][-1]; found=True
                    break
        if not found: break
    arrow=' → '.join(chain)
    print(f'{start}: {len(chain)}步')
    print(f'  {arrow[:120]}')
print(f'\n纯能量景观 | LLM:0')
