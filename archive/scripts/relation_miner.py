#!/usr/bin/env python3
"""
关系挖掘器 — 从中文文本中自动发现新关系类型
==============================================
从维基百科/百度百科文本中自动提取高频谓词模式，
超越预定义的6种关系类型。
"""
import sys, os, re, json, time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RelationMiner:
    def __init__(self, field, concept_graph, min_freq=3):
        self.field = field
        self.cg = concept_graph
        self.min_freq = min_freq
        self.discovered_relations = {}
    
    def mine_from_text(self, texts):
        patterns = [
            (re.compile(r'([\u4e00-\u9fff]{2,4})([\u4e00-\u9fff]{1,2})(?:了|出|到|起)?([\u4e00-\u9fff]{2,6})'), "fwd"),
            (re.compile(r'([\u4e00-\u9fff]{2,4})由([\u4e00-\u9fff]{2,8})(?:组成|构成|创建|发明|发现|建立)'), "rev"),
            (re.compile(r'([\u4e00-\u9fff]{2,6})是([\u4e00-\u9fff]{2,4})的'), "rev"),
        ]
        pred_counter = Counter()
        pred_examples = defaultdict(list)
        for text in texts:
            if not text or len(text) < 6:
                continue
            sentences = re.split(r'[。！？；\n]', text)
            for sent in sentences:
                if len(sent) < 6:
                    continue
                for pat, direction in patterns:
                    for m in pat.finditer(sent):
                        g = m.groups()
                        if len(g) < 3: continue
                        if direction == "fwd":
                            subj, pred, obj = g[0], g[1], g[2]
                        else:
                            obj, subj, pred = g[0], g[1], g[2]
                        if not all(c in self.field._char_to_idx for c in subj): continue
                        if not all(c in self.field._char_to_idx for c in obj): continue
                        pred_counter[pred] += 1
                        if len(pred_examples[pred]) < 5:
                            pred_examples[pred].append((subj, pred, obj))
        results = []
        for pred, freq in pred_counter.most_common(50):
            if freq < self.min_freq: break
            examples = pred_examples[pred][:3]
            energy_sum = sum(self.cg.triple_energy(s, o) for s, _, o in examples)
            avg_e = energy_sum / len(examples)
            conf = max(0.1, min(0.9, 1.0 - (avg_e + 50) / 80.0))
            results.append((pred, conf, f"频{freq}能{avg_e:.1f}例:{examples[0][0]}{pred}{examples[0][2]}"))
        return results

if __name__ == '__main__':
    from loongpearl.core.zichang import HanziAnchorField
    from loongpearl.core.concept_graph import ConceptGraph
    print("=" * 50)
    print("🔍 关系挖掘器测试")
    print("=" * 50)
    field = HanziAnchorField.load(os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)
    cg = ConceptGraph(field)
    cg_path = os.path.join(PROJECT, 'data/models/concept_graph')
    if os.path.exists(cg_path + '.json'): cg.load(cg_path)
    miner = RelationMiner(field, cg, min_freq=1)
    samples = [
        "爱因斯坦提出了相对论。相对论改变了物理学。",
        "秦始皇统一了中国，建立了秦朝。",
        "水由氢和氧组成。氢是宇宙中最丰富的元素。",
        "达尔文提出了进化论。进化论解释了物种起源。",
        "牛顿发现了万有引力。万有引力定律是经典力学的基础。",
        "屠呦呦发现了青蒿素。青蒿素用于治疗疟疾。",
    ]
    results = miner.mine_from_text(samples)
    print(f"\n发现 {len(results)} 个候选谓词:")
    for pred, conf, reason in results[:15]:
        print(f"  {pred} (信:{conf:.2f}) — {reason}")
    print("\n✅ 关系挖掘器就绪")
