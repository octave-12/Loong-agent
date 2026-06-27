#!/usr/bin/env python3
"""验证 Phase 1+2: DragonField + FieldNLG 端到端"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.orchestrator import create_orchestrator
from loongpearl.core.dragon_field import DragonField
from loongpearl.core.field_nlg import FieldNLG
import torch

print("加载...")
hf = HanziAnchorField.load("data/models/zichang_94117_1024d.pt")
el = FreqEnergyLandscape.load("data/models/energy_landscape_1024d.pt")
orch = create_orchestrator(field=hf, landscape=el)

# 加载 DragonField + FieldNLG
df_cache = "data/models/dragon_field_patterns.pt"
db_path = "data/models/concept_graph.db"
data = torch.load(df_cache, map_location='cpu')
orch._dragon_field = DragonField(embed_dim=1024, beta=8.0)
orch._dragon_field.store_patterns(data['vectors'], data['ids'], data['subjects'])
orch._field_nlg = FieldNLG(db_path=db_path, pattern_ids=orch._dragon_field._pattern_ids)

print(f"\nDragonField: {orch._dragon_field.num_patterns} 模式")
print(f"FieldNLG: {'✅' if orch._field_nlg else '❌'}")

# 测试
tests = [
    ("龙", "知识-概念"),
    ("量子力学", "知识-复合"),
    ("火", "知识-简单"),
]

print("\n" + "="*60)
for q, cat in tests:
    r = orch.query(q)
    engine = r['debug']['infer']['engine']
    answer = r.get('answer', '')[:200]
    print(f"\n[{cat}] '{q}'")
    print(f"  引擎: {engine}")
    print(f"  回答: {answer}")
