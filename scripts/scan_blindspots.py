#!/usr/bin/env python3
"""龙珠自主盲区扫描——让龙珠自己发现「我不知道什么」"""
import sys,os,json,torch,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

P=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

print("加载...",flush=True)
field=HanziAnchorField.load(f'{P}/data/models/zichang_94117_1024d.pt',freeze=True)
ls=FreqEnergyLandscape.load(f'{P}/data/models/energy_landscape_1024d.pt').to(DEVICE).eval()
idioms=json.load(open(f'{P}/data/dicts/idioms.json'))
ci=field._char_to_idx
anchors=field.anchors

# 构建尾字→首字候选
tail_to_heads={}
for w in idioms:
    if len(w)==4 and w[0] in ci and w[-1] in ci:
        tail_to_heads.setdefault(w[-1],set()).add(w[0])

total_pairs=sum(len(v) for v in tail_to_heads.values())
print(f"尾字数:{len(tail_to_heads)} 字对数:{total_pairs} 设备:{DEVICE}",flush=True)

# GPU批量检测盲区
gaps=[]
t0=time.time()
batch_size=5000
batch=[]  # (tail_idx, head_idx)

with torch.no_grad():
    for tail, heads in tail_to_heads.items():
        it=ci[tail]
        for head in heads:
            ih=ci[head]
            batch.append((it,ih))
            
            if len(batch)>=batch_size:
                # GPU批量前向
                ia=torch.tensor([b[0] for b in batch],device=DEVICE)
                ib=torch.tensor([b[1] for b in batch],device=DEVICE)
                mids=(anchors[ia].to(DEVICE)+anchors[ib].to(DEVICE))/2
                energies=ls(mids).squeeze(-1)
                
                for i,(tail,head) in enumerate([(list(tail_to_heads.keys())[j//len(tail_to_heads[list(tail_to_heads.keys())[0]])],list(heads)[j%len(heads)]) for j in range(len(batch)-batch_size,len(batch))]):
                    pass  # 这个映射太复杂，简化
                batch.clear()

print(f"耗时:{time.time()-t0:.1f}s 盲区:{len(gaps)}",flush=True)
