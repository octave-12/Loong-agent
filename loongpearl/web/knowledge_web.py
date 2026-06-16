#!/usr/bin/env python3
"""龙珠联网检索与学习模块 — 未知概念自动搜→学→推"""
import sys, os, re, json, time
import requests
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根
ZICHANG = os.path.join(BASE, "data/models/zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "energy_landscape_1024d_vector_seeded.pt")
LANDSCAPE_FB = os.path.join(BASE, "data/models/energy_landscape_1024d.pt")


class WebSearcher:
    """DuckDuckGo/Bing 搜索 + 汉字关键词提取"""

    def search(self, query: str) -> list:
        texts = self._ddg(query) or self._bing(query)
        if not texts: return []
        # 提取汉字 + 频率统计
        combined = " ".join(texts)
        hanzi = re.findall(r'[\u4e00-\u9fff]', combined)
        freq = {}
        for ch in hanzi: freq[ch] = freq.get(ch, 0) + 1
        # 排除查询字
        for qc in set(re.findall(r'[\u4e00-\u9fff]', query)):
            freq.pop(qc, None)
        return sorted(freq.items(), key=lambda x: -x[1])[:20]

    def _ddg(self, q):
        try:
            r = requests.get("https://lite.duckduckgo.com/lite/",
                params={"q": q}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', r.text, re.DOTALL)
        except: return []

    def _bing(self, q):
        try:
            r = requests.get("https://www.bing.com/search",
                params={"q": q}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return re.findall(r'<p[^>]*>(.*?)</p>', r.text, re.DOTALL)[:10]
        except: return []


class WebAwareLoongPearl:
    """联网龙珠: 查询→自知无知→联网搜索→植入→重查"""

    def __init__(self):
        print("加载字场...")
        self.zc = HanziAnchorField.load(ZICHANG)
        
        path = LANDSCAPE if os.path.exists(LANDSCAPE) else LANDSCAPE_FB
        print("加载能量景观...")
        self.ls = EnergyLandscape.load(path)
        self.ls.eval()
        
        print("初始化学习器...")
        self.lr = DragonBallLearner(landscape=self.ls, anchor_field=self.zc, hebbian_lr=0.001)
        
        self.searcher = WebSearcher()
        self.web_learned = 0
        self.web_implanted = 0
        self.queries = 0

    def query(self, text: str, auto_web: bool = True, verbose: bool = False) -> dict:
        """查询 + 自动联网学习"""
        self.queries += 1

        # 1. 编码 → 自知无知
        vec = self._encode(text)
        check = self.lr.check_knowledge(vec)
        is_known = check['is_known']
        conf = check['confidence']

        # 2. 已知 → 直接推理
        if is_known:
            self.ls.train()
            result = self.ls.infer(vec, steps=50)
            self.ls.eval()
            _, chars, energies = self.zc.find_nearest(result['state'], k=5)
            return {
                'answer': f"「{''.join(chars)}」与「{text}」最相关 (能量={result['energy']:.2f})",
                'known': True, 'confidence': conf,
                'web_learned': False, 'nearest': chars,
            }

        # 3. 未知 → 联网搜索
        if auto_web and not is_known:
            keywords = self.searcher.search(text)
            if keywords:
                imp = self._implant(vec, text, keywords)
                self.web_learned += 1
                self.web_implanted += imp
                # 重查
                self.ls.train()
                result2 = self.ls.infer(vec, steps=50)
                self.ls.eval()
                _, chars2, _ = self.zc.find_nearest(result2['state'], k=5)
                return {
                    'answer': f"「{''.join(chars2)}」与「{text}」最相关 (已联网学习)",
                    'known': True, 'confidence': 0.8,
                    'web_learned': True, 'web_keywords': [kw[0] for kw in keywords[:8]],
                    'web_implanted': imp, 'nearest': chars2,
                }

        return {
            'answer': f'「{text}」尚未学到，联网也未找到相关信息。',
            'known': False, 'confidence': conf,
            'web_learned': False, 'nearest': check.get('nearest_chars', [])[:5],
        }

    def _encode(self, text):
        v = self.zc.encode_text(text)
        if v.shape[0] == 0:
            return torch.zeros(self.zc.embed_dim)
        return torch.nn.functional.normalize(v.mean(dim=0), dim=-1)

    def _vec_to_answer(self, result, query):
        chars = result.get('nearest_chars', [])
        energy = result.get('energy', 0)
        if chars:
            return f"「{''.join(chars[:3])}」与「{query}」最相关 (能量={energy:.2f})"
        return f"未找到「{query}」的关联汉字"

    def _implant(self, query_vec, query_text, keywords):
        imp = 0
        for hanzi, freq in keywords[:10]:
            if hanzi not in self.zc._char_to_idx: continue
            try:
                r = self.lr.hebbian.update(
                    query_vec, self.zc.anchors[self.zc._char_to_idx[hanzi]],
                    feedback=min(0.5, freq / 20))
                if r.get('status') != 'skipped': imp += 1
            except: pass
        return imp

    def save(self, path=None):
        self.ls.save(path or LANDSCAPE)

    def stats(self):
        return {'queries': self.queries, 'web_learned': self.web_learned,
                'web_implanted': self.web_implanted}


if __name__ == "__main__":
    print("🐉 龙珠 · 联网感知测试")
    lp = WebAwareLoongPearl()

    for q in ["量子纠缠", "赛博朋克", "龙珠"]:
        print(f"\n🔍 {q}")
        r = lp.query(q, verbose=True)
        print(f"  已知:{r['known']} 置信:{r['confidence']:.2f} 联网:{r['web_learned']}")
        if r['web_learned']:
            print(f"  关键词:{r.get('web_keywords',[])}")
        print(f"  → {r['answer']}")

    print(f"\n统计: {lp.stats()}")
    lp.save()
    print("✅ 已保存")
