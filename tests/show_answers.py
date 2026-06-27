#!/usr/bin/env python3
"""提取龙珠真实回答"""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.orchestrator import create_orchestrator

print("加载模型...")
hf = HanziAnchorField.load("data/models/zichang_94117_1024d.pt")
el = FreqEnergyLandscape.load("data/models/energy_landscape_1024d.pt")
orch = create_orchestrator(field=hf, landscape=el)
print()

queries = [
    ("闲聊", ["你好", "你是谁", "谢谢"]),
    ("知识", ["龙是什么", "李白是谁"]),
    ("诗词", ["写一首关于春天的诗", "以月亮为题写诗"]),
    ("成语", ["画龙点睛是什么意思"]),
    ("边界", ["龘"]),
]

for cat, qs in queries:
    print(f"{'='*60}")
    print(f"【{cat}】")
    print(f"{'='*60}")
    for q in qs:
        t0 = time.time()
        r = orch.query(q)
        t = time.time() - t0
        a = r.get('answer', '(无)')
        s = r.get('signal', '?')
        c = r.get('confidence', 0)
        print(f"\n❓ {q}")
        print(f"   ⏱ {t:.1f}s | signal={s} | conf={c:.0%}")
        print(f"   {a}")
