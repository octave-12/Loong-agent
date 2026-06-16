#!/usr/bin/env python3
"""
龙珠原生回答引擎 (native_answer.py)
====================================
三步修复:
  1. 释义截断 → 完整释义 + 智能分段
  2. CC-CEDICT 词典 → 12万中英词条
  3. 复合词检测 → 最长匹配优先, "量子"不再拆成"量+子"

架构:
  输入 → 复合词最长匹配 → 单字降级 → 字典释义(Decompose→Unihan→CC-CEDICT)
  → 能量近邻 → 拼装答案 → (可选) LLM 润色
"""

import sys, os, json, re
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根
ZICHANG = os.path.join(BASE, "data/models/zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "data/models/energy_landscape_1024d.pt")
DECOMPOSE = os.path.join(BASE, "data/dicts/dict_decompose.json")
UNIHAN = os.path.join(BASE, "data/dicts/dict_unihan.json")
CEDICT = os.path.join(BASE, "data/dicts/cedict_parsed.json")


class NativeAnswerEngine:
    """龙珠原生回答: 字典 + 能量关联 → 拼出答案, 零 LLM"""

    def __init__(self):
        self.zc = HanziAnchorField.load(ZICHANG)
        self.ls = EnergyLandscape.load(LANDSCAPE)
        self.ls.eval()

        # 加载字典 (优先级: Decompose > Unihan)
        self.decompose = json.load(open(DECOMPOSE)) if os.path.exists(DECOMPOSE) else {}
        self.unihan = json.load(open(UNIHAN)) if os.path.exists(UNIHAN) else {}

        # 加载 CC-CEDICT (复合词词典, 12万词条)
        # 结构: {word: {pinyin, definitions: [str], traditional}}
        self.cedict = json.load(open(CEDICT)) if os.path.exists(CEDICT) else {}
        self._build_cedict_index()

        self.anchors = self.zc.anchors

        # 停用词 (虚词, 不提取)
        self.stop = set('的了吗呢嘛啊吧呀是的不一在了有和就都也这那个什么怎么')

    # ── CC-CEDICT 索引 ─────────────────────────

    def _build_cedict_index(self):
        """构建复合词前缀树索引, 支持最长匹配"""
        # 按词长降序排列, 用于贪婪最长匹配
        self.cedict_words = sorted(
            [(w, info) for w, info in self.cedict.items() if len(w) > 1],
            key=lambda x: -len(x[0])  # 长词优先
        )
        # 单字索引 (CC-CEDICT 中也有单字释义, 作为补充)
        self.cedict_single = {
            w: info for w, info in self.cedict.items() if len(w) == 1
        }

    # ── 复合词最长匹配 ──────────────────────────

    def _extract_tokens(self, text: str) -> list:
        """
        提取语义单元: 复合词优先, 剩余单字降级
        
        "什么是量子纠缠" → ["量子", "纠缠"]  (而非 "什","么","是","量","子","纠","缠")
        """
        # 只保留中文
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        if not chinese:
            return []
        
        full = ''.join(chinese)
        tokens = []
        pos = 0
        
        while pos < len(full):
            matched = False
            
            # 尝试最长匹配复合词 (从最长到最短)
            for word, info in self.cedict_words:
                wlen = len(word)
                if pos + wlen <= len(full) and full[pos:pos + wlen] == word:
                    if word not in self.stop_raw:
                        tokens.append(('word', word, info))
                    pos += wlen
                    matched = True
                    break
            
            if not matched:
                # 没匹配到复合词, 取单字
                ch = full[pos]
                if ch not in self.stop:
                    tokens.append(('char', ch, None))
                pos += 1
        
        # 去重但保序
        seen = set()
        result = []
        for ttype, token, info in tokens:
            if token not in seen:
                seen.add(token)
                result.append((ttype, token, info))
        
        return result[:12]  # 最多取12个语义单元

    @property
    def stop_raw(self):
        """复合词停用 (包含单字虚词组成的双字虚词)"""
        return {'什么', '怎么', '这个', '那个', '我们', '他们', '你们', '这里', '那里',
                '可以', '没有', '已经', '因为', '所以', '但是', '而且', '虽然', '如果'}

    # ── 主入口 ──────────────────────────────────

    def answer(self, question: str) -> dict:
        """
        用龙珠自己的知识回答。

        Returns:
            {
                'native': str,          # 龙珠原生回答
                'source': str,          # 'dictionary' | 'association' | 'hybrid'
                'anchors': [(字/词, 释义), ...],  # 答案的证据锚点
                'related': [(字, 释义), ...],     # 关联概念
                'tokens': [(type, token, ...)],   # 提取的语义单元
            }
        """
        # 1. 提取语义单元 (复合词优先)
        tokens = self._extract_tokens(question)
        if not tokens:
            return self._fallback(question)

        # 2. 对每个语义单元查释义 + 找关联
        evidence = []   # 直接证据
        related = []    # 能量关联

        for ttype, token, cedict_info in tokens:
            if ttype == 'word':
                # 复合词: 直接取 CC-CEDICT 释义
                defn = self._format_cedict_def(cedict_info)
                if defn:
                    evidence.append((token, defn, 'cedict'))
                # 复合词: 用词向量找能量近邻 (而非逐字)
                self._collect_related_compound(token, related)
            else:
                # 单字: 查字典 + 能量近邻
                defn = self._lookup_definition(ch=token)
                if defn:
                    evidence.append((token, defn, 'dict'))
                self._collect_related(token, related)

        # 3. 拼出答案
        if evidence:
            native, source = self._build_answer(question, tokens, evidence, related)
        elif related:
            native, source = self._build_from_assoc(question, tokens, related)
        else:
            return self._fallback(question)

        return {
            'native': native,
            'source': source,
            'anchors': [(t, d, s) for t, d, s in evidence[:8]],
            'related': [(ch, self._lookup_definition(ch=ch)) for ch, _ in related[:8]],
            'tokens': [(t, tok) for t, tok, _ in tokens],
        }

    def _collect_related(self, char: str, related: list):
        """收集一个字的能量近邻 (去重)"""
        if char in self.zc._char_to_idx:
            vec = self.anchors[self.zc._char_to_idx[char]]
            _, chars, _ = self.zc.find_nearest(vec, k=5)
            for ch in chars[1:]:  # 排除自己
                if ch not in [r[0] for r in related]:
                    related.append((ch, 0.0))

    def _collect_related_compound(self, word: str, related: list):
        """复合词的能量近邻: 平均字向量 → 找最近汉字"""
        vecs = []
        for ch in word:
            if ch in self.zc._char_to_idx:
                vecs.append(self.anchors[self.zc._char_to_idx[ch]])
        if not vecs:
            return
        avg_vec = torch.stack(vecs).mean(dim=0)
        _, chars, _ = self.zc.find_nearest(avg_vec, k=6)
        # 排除组成词本身的字
        word_chars = set(word)
        for ch in chars:
            if ch not in word_chars and ch not in [r[0] for r in related]:
                related.append((ch, 0.0))
                if len([r for r in related]) >= 5:
                    break

    # ── 查字典 (三层降级) ──────────────────────

    def _lookup_definition(self, ch: str = None, word: str = None) -> str:
        """
        查释义, 优先级:
          1. MakeMeAHanzi (部件+释义, 最完整)
          2. Unihan (42K字释义)
          3. CC-CEDICT 单字 (兜底)
        
        Returns 完整释义, 不做截断
        """
        target = ch or word

        # 1. Decompose
        if target in self.decompose:
            d = self.decompose[target].get('definition', '')
            if d:
                return d

        # 2. Unihan
        if target in self.unihan:
            d = self.unihan[target].get('definition', '')
            if d:
                return d

        # 3. CC-CEDICT 单字
        if target in self.cedict_single:
            defs = self.cedict_single[target].get('definitions', [])
            if defs:
                return '; '.join(defs)

        return ''

    def _format_cedict_def(self, info: dict) -> str:
        """格式化 CC-CEDICT 复合词释义"""
        if not info:
            return ''
        defs = info.get('definitions', [])
        if not defs:
            return ''
        pinyin = info.get('pinyin', '')
        return '; '.join(defs[:5])

    def _lookup_components(self, char: str) -> list:
        """查一个字的部件拆解"""
        if char in self.decompose:
            return self.decompose[char].get('components', [])
        return []

    # ── 答案拼装 ────────────────────────────────

    def _build_answer(self, question, tokens, evidence, related) -> tuple:
        """基于证据拼出中文回答"""
        lines = []

        for i, (token, defn, source) in enumerate(evidence[:5]):
            # 翻译释义为中文关键词
            defn_cn = self._translate_def(defn)

            if source == 'cedict':
                # 复合词
                if i == 0:
                    lines.append(f"「{token}」: {defn_cn}")
                else:
                    lines.append(f"「{token}」: {defn_cn}")
            else:
                # 单字
                if i == 0:
                    lines.append(f"「{token}」: {defn_cn}")

                    # 部件拆解 (只有单字才有)
                    comps = self._lookup_components(token)
                    if comps:
                        comp_strs = []
                        for c in comps:
                            cd = self._lookup_definition(ch=c)
                            if cd:
                                comp_strs.append(f"{c}({self._translate_def(cd)[:12]})")
                            else:
                                comp_strs.append(c)
                        if comp_strs:
                            lines.append(f"字形: {token} = {' + '.join(comp_strs)}")
                else:
                    lines.append(f"相关: {defn_cn}")

        # 能量关联
        if related:
            rel_items = []
            for ch, _ in related[:6]:
                d = self._lookup_definition(ch=ch)
                rel_items.append(f"{ch}({self._translate_def(d)[:10] if d else ''})")
            if rel_items:
                lines.append(f"能量近邻: {'、'.join(rel_items)}")

        return "\n".join(lines), 'dictionary'

    def _build_from_assoc(self, question, tokens, related) -> tuple:
        """纯能量关联回答 (无字典释义时)"""
        token_strs = [tok for _, tok, _ in tokens]
        kw_str = "、".join(token_strs[:6])
        rel_strs = []
        for ch, _ in related[:6]:
            d = self._lookup_definition(ch=ch)
            rel_strs.append(f"{ch}({self._translate_def(d)[:10] if d else ''})")
        lines = [
            f"「{kw_str}」在字场中关联到:",
            "、\n".join(rel_strs),
        ]
        return "\n".join(lines), 'association'

    def _fallback(self, question: str) -> dict:
        return {
            'native': f"「{question}」暂未收录，建议联网检索",
            'source': 'none',
            'anchors': [],
            'related': [],
            'tokens': [],
        }

    # ── 英文释义转中文关键词 (修复截断) ─────────

    # 更大、更完整的英→中映射表
    DEF_MAP = {
        # 基础概念
        'dragon': '龙', 'imperial': '帝王', 'emperor': '皇帝',
        'bright': '明亮', 'light': '光', 'clear': '清晰',
        'sun': '太阳', 'moon': '月亮', 'star': '星',
        'water': '水', 'fire': '火', 'mountain': '山',
        'king': '王', 'power': '力量', 'heaven': '天',
        'earth': '地', 'human': '人', 'heart': '心',
        'spirit': '精神', 'knowledge': '知识', 'wisdom': '智慧',
        'love': '爱', 'beautiful': '美', 'good': '好',
        'big': '大', 'small': '小', 'god': '神',
        'deity': '神', 'jade': '玉', 'gold': '金',
        'tree': '木', 'speech': '言', 'word': '言',
        'language': '语言', 'culture': '文化',
        'symbol': '象征', 'luck': '吉祥', 'auspicious': '吉祥',
        'cloud': '云', 'rain': '雨', 'wind': '风',
        'thunder': '雷', 'electric': '电', 'river': '河',
        'sea': '海', 'sky': '天', 'bird': '鸟',
        'fish': '鱼', 'insect': '虫',
        # 抽象概念
        'reason': '理性', 'logic': '逻辑', 'manage': '管理',
        'science': '科学', 'measure': '测量', 'quantity': '数量',
        'volume': '体积', 'amount': '数量', 'capacity': '容量',
        'offspring': '后代', 'child': '孩子', 'seed': '种子',
        'fruit': '果实', 'branch': '分支', 'earthly': '地支',
        'literature': '文学', 'writing': '文字',
        'consciousness': '意识', 'awareness': '觉察',
        'universe': '宇宙', 'cosmos': '宇宙', 'space': '空间',
        'philosophy': '哲学', 'intelligence': '智能',
        'artificial': '人工', 'quantum': '量子',
        'physics': '物理', 'energy': '能量',
        'tangle': '纠缠', 'nag': '纠缠', 'entanglement': '纠缠',
        'tradition': '传统', 'custom': '习俗',
        'transformation': '转化', 'change': '变化',
        'nature': '自然', 'life': '生命', 'death': '死亡',
        'peace': '和平', 'war': '战争', 'time': '时间',
    }

    def _translate_def(self, defn: str) -> str:
        """
        英文释义 → 中文关键词, 不做暴力截断
        
        策略:
          1. 按 ; / 分段
          2. 每段尝试关键词匹配
          3. 兜底: 取前60字符 (而非20), 保证完整性
        """
        if not defn:
            return ''

        # 分段
        segments = defn.lower().replace(';', '/').replace('|', '/').split('/')
        found = []

        for seg in segments:
            seg = seg.strip().strip('.').strip()
            if not seg:
                continue
            # 精确匹配
            if seg in self.DEF_MAP:
                if self.DEF_MAP[seg] not in found:
                    found.append(self.DEF_MAP[seg])
                continue
            # 子串匹配: 收集所有匹配的关键词 (而非只保留最长)
            seen_cn = set(found)
            for eng, cn in sorted(self.DEF_MAP.items(), key=lambda x: -len(x[0])):
                if eng in seg and cn not in seen_cn:
                    found.append(cn)
                    seen_cn.add(cn)

        if found:
            return "/".join(found)

        # 兜底: 取语义完整的一段 (最多60字符, 在句号或分号处截断)
        full = defn.strip()
        if len(full) <= 60:
            return full
        # 尝试在标点处截断
        for sep in [';', '.', ',', '|']:
            idx = full[:60].rfind(sep)
            if idx > 30:
                return full[:idx]
        return full[:60]


# ====================================================================
# 测试：婴儿学语 — 龙珠用自己的声音回答，不打粉不化妆
# ====================================================================

if __name__ == "__main__":
    engine = NativeAnswerEngine()

    tests = [
        "龙是什么?",
        "明字怎么来的?",
        "什么是量子?",
        "量子纠缠是什么?",
        "人工智能是什么?",
        "宇宙有多大?",
        "爱是什么意思?",
    ]

    print("╔" + "═" * 58 + "╗")
    print("║  龙珠学语 · 零 LLM · 婴儿自己的声音" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")

    for q in tests:
        print(f"\n👶 问：{q}")
        r = engine.answer(q)
        print(f"   🧠 懂了这些词：{'、'.join(tok for _, tok in r.get('tokens', []))}")
        if r['source'] == 'none':
            print(f"   💬 {r['native']}")
        else:
            for line in r['native'].split('\n'):
                print(f"   {line}")
