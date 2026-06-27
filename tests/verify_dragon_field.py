#!/usr/bin/env python3
"""验证 DragonField 是否在 query() 中被实际调用"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.orchestrator import create_orchestrator

print("加载...")
hf = HanziAnchorField.load("data/models/zichang_94117_1024d.pt")
el = FreqEnergyLandscape.load("data/models/energy_landscape_1024d.pt")
orch = create_orchestrator(field=hf, landscape=el)

# 手动加载 DragonField (create_orchestrator 不自动加载)
from loongpearl.core.dragon_field import DragonField
import torch
df_cache = "data/models/dragon_field_patterns.pt"
if os.path.exists(df_cache):
    orch._dragon_field = DragonField(embed_dim=1024, beta=8.0)
    data = torch.load(df_cache, map_location='cpu')
    orch._dragon_field.store_patterns(data['vectors'], data['ids'], data['subjects'])
    print(f"DragonField: {orch._dragon_field.num_patterns} 模式手动加载")
else:
    print("DragonField: 缓存不存在")

# 检查 DragonField 状态
df = getattr(orch, '_dragon_field', None)
print(f"\nDragonField: {'✅ ' + str(df.num_patterns) + ' 模式' if df and df.num_patterns > 0 else '❌ 未加载'}")

# 测试知识查询（经 stage4 → 能量推理路径）
tests = ["龙", "量子力学", "火"]
for q in tests:
    r = orch.query(q)
    engine = r.get('debug', {}).get('infer', {}).get('engine', '?')
    signal = r.get('signal', '?')
    conf = r.get('confidence', 0)
    print(f"\nquery('{q}'):")
    print(f"  engine: {engine}")
    print(f"  signal: {signal}  conf: {conf:.0%}")
    if engine == 'dragon_field':
        basin = r['debug']['infer'].get('basin_depth', 0)
        saddle = r['debug']['infer'].get('saddle_gap', 0)
        label = r['debug']['infer'].get('confidence_label', '?')
        print(f"  basin: {basin:.1f}  saddle: {saddle:.3f}  label: {label}")
