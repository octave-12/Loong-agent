#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠数据消化器 — 将全部本地数据源一次性消化进概念图
════════════════════════════════════════════════════════════════════════════

吃掉所有外挂文件，让概念图成为唯一知识源:
  1. idioms.json     → 成语→字间COOCCURS三元组 + 成语本身为概念节点
  2. cedict_parsed.json → 词条→DEFINED_AS三元组 + 字间共现
  3. dict_unihan.json   → 部首/笔画/异体→HAS_RADICAL/HAS/VARIANT三元组
  4. dict_decompose.json → 部件分解→DECOMPOSES_INTO三元组
  5. tang_poetry_ngrams.json → 高频字对→POETIC_WITH三元组

消化后: 所有知识在概念图中，文件只读一次，之后管线只查概念图。

════════════════════════════════════════════════════════════════════════════
用法:
  python scripts/ingest_all_dicts.py              # 消化全部
  python scripts/ingest_all_dicts.py --idioms-only  # 只消化成语
  python scripts/ingest_all_dicts.py --dry-run      # 预览不写入
"""

import sys, os, json, time, argparse
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)


def load_field():
    from loongpearl.core.zichang import HanziAnchorField
    return HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)


def load_concept_graph(field):
    from loongpearl.core.concept_graph import ConceptGraph
    cg = ConceptGraph(field, None)
    path = os.path.join(PROJECT, 'data/models/concept_graph')
    if os.path.exists(path + '.json'):
        cg.load(path)
    return cg


# ═══════════════════════════════════════════════════════════════════════
# 消化器
# ═══════════════════════════════════════════════════════════════════════

class DictIngester:
    """将本地数据文件消化进概念图"""

    def __init__(self, cg, dry_run=False):
        self.cg = cg
        self.dry_run = dry_run
        self.stats = defaultdict(int)

    def _add(self, s, r, o, confidence=0.7, source="dict_ingest"):
        """安全添加三元组"""
        if self.dry_run:
            self.stats['would_add'] += 1
            return
        try:
            if s and o and len(s) >= 1 and len(o) >= 1:
                self.cg.add_triple(s, r, o, confidence=confidence, source=source)
                self.stats['added'] += 1
        except Exception:
            self.stats['errors'] += 1

    # ── 成语消化 ──
    def ingest_idioms(self):
        path = os.path.join(PROJECT, 'data/dicts/idioms.json')
        if not os.path.exists(path):
            print("  idioms.json 不存在，跳过")
            return

        with open(path, 'r', encoding='utf-8') as f:
            idioms = json.load(f)

        print(f"  成语: {len(idioms)} 条")

        for idiom in idioms:
            if not isinstance(idiom, str) or len(idiom) < 4:
                continue

            # 成语本身作为概念节点
            self._add(idiom, "IS_A", "成语", confidence=0.9, source="idiom_dict")

            # 字间共现: 每对相邻字 → COOCCURS
            for i in range(len(idiom) - 1):
                a, b = idiom[i], idiom[i+1]
                if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                    self._add(a, "COOCCURS_IN", idiom, confidence=0.8, source="idiom_dict")
                    self._add(b, "COOCCURS_IN", idiom, confidence=0.8, source="idiom_dict")
                    self._add(a, "COOCCURS_WITH", b, confidence=0.6, source="idiom")

            if self.stats.get('added', 0) % 5000 == 0 and not self.dry_run:
                print(f"    ... {self.stats['added']} 条")

        print(f"  成语消化: {self.stats['added']} 条三元组")

    # ── CEDICT消化 ──
    def ingest_cedict(self):
        path = os.path.join(PROJECT, 'data/dicts/cedict_parsed.json')
        if not os.path.exists(path):
            print("  cedict_parsed.json 不存在，跳过")
            return

        with open(path, 'r', encoding='utf-8') as f:
            cedict = json.load(f)

        before = self.stats['added']
        print(f"  CEDICT: {len(cedict)} 词条")

        count = 0
        for term, entry in cedict.items():
            if not isinstance(term, str) or len(term) < 1:
                continue

            # 词条本身作为概念
            if len(term) >= 2:
                self._add(term, "IS_A", "中文词条", confidence=0.7, source="cedict")

            # 定义
            if isinstance(entry, dict):
                definitions = entry.get('definitions', [])
                for d in definitions[:3]:
                    if isinstance(d, str) and len(d) >= 2:
                        self._add(term, "DEFINED_AS", d[:60], confidence=0.65, source="cedict")

                # 拼音
                pinyin = entry.get('pinyin', '')
                if pinyin:
                    self._add(term, "HAS_PINYIN", pinyin, confidence=0.9, source="cedict")

            # 字间共现 (高频——出现在词典中说明是真实搭配)
            for i in range(len(term) - 1):
                a, b = term[i], term[i+1]
                if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                    self._add(a, "COOCCURS_WITH", b, confidence=0.7, source="cedict")

            count += 1
            if count % 20000 == 0:
                print(f"    ... {count}/{len(cedict)} ({self.stats['added'] - before} 新增)")

        added_now = self.stats['added'] - before
        print(f"  CEDICT消化: {added_now} 条三元组")

    # ── Unihan消化 ──
    def ingest_unihan(self):
        path = os.path.join(PROJECT, 'data/dicts/dict_unihan.json')
        if not os.path.exists(path):
            print("  dict_unihan.json 不存在，跳过")
            return

        with open(path, 'r', encoding='utf-8') as f:
            unihan = json.load(f)

        before = self.stats['added']
        print(f"  Unihan: {len(unihan)} 汉字")

        for char, entry in unihan.items():
            if not isinstance(char, str) or len(char) != 1:
                continue
            if not isinstance(entry, dict):
                continue

            # 读音
            mandarin = entry.get('mandarin', '')
            if mandarin:
                self._add(char, "HAS_PINYIN", mandarin, confidence=0.85, source="unihan")

            # 释义
            definition = entry.get('definition', '')
            if definition and len(definition) >= 2:
                self._add(char, "DEFINED_AS", definition[:80], confidence=0.6, source="unihan")

            # 其他属性
            for key in ('kTotalStrokes', 'kRSUnicode', 'kCangjie'):
                val = entry.get(key, '')
                if val:
                    self._add(char, "HAS", str(val)[:30], confidence=0.7, source="unihan")

        added_now = self.stats['added'] - before
        print(f"  Unihan消化: {added_now} 条三元组")

    # ── 唐诗消化 ──
    def ingest_tang(self):
        path = os.path.join(PROJECT, 'data/dicts/tang_poetry_ngrams.json')
        if not os.path.exists(path):
            print("  tang_poetry_ngrams.json 不存在，跳过")
            return

        with open(path, 'r', encoding='utf-8') as f:
            tang = json.load(f)

        before = self.stats['added']
        bigrams = tang.get('bigrams', {})
        print(f"  唐诗: {len(bigrams)} 字对")

        for key, freq in bigrams.items():
            parts = key.split('|')
            if len(parts) == 2:
                a, b = parts[0], parts[1]
                if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                    # 高频字对加权
                    conf = min(0.95, 0.5 + freq / 40)
                    self._add(a, "POETIC_WITH", b, confidence=conf, source="tang_poetry")

        added_now = self.stats['added'] - before
        print(f"  唐诗消化: {added_now} 条三元组")

    # ── 全量消化 ──
    def ingest_all(self):
        print("🐉 开始消化全部本地数据...")
        t0 = time.time()

        self.ingest_idioms()
        self.ingest_cedict()
        self.ingest_unihan()
        self.ingest_tang()

        if not self.dry_run and self.stats['added'] > 0:
            save_path = os.path.join(PROJECT, 'data/models/concept_graph')
            print(f"\n💾 保存概念图 ({self.stats['added']} 新增)...")
            self.cg.save(save_path)
            print(f"   已保存: {len(self.cg.triples)} 三元组")

        elapsed = time.time() - t0
        print(f"\n{'='*50}")
        print(f"消化完成: {self.stats['added']} 条三元组")
        print(f"错误: {self.stats['errors']}")
        print(f"耗时: {elapsed:.1f}s")
        if self.dry_run:
            print(f"⚠️ DRY RUN — 未实际写入")
            print(f"   将会添加: {self.stats['would_add']} 条")


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='龙珠数据消化器 — 将本地数据吃进概念图')
    parser.add_argument('--dry-run', action='store_true', help='预览不写入')
    parser.add_argument('--idioms-only', action='store_true', help='只消化成语')
    parser.add_argument('--cedict-only', action='store_true', help='只消化CEDICT')
    parser.add_argument('--unihan-only', action='store_true', help='只消化Unihan')
    parser.add_argument('--tang-only', action='store_true', help='只消化唐诗')
    args = parser.parse_args()

    print("加载字场...")
    field = load_field()

    print("加载概念图...")
    cg = load_concept_graph(field)
    print(f"  当前: {len(cg.triples)} 三元组")

    ingester = DictIngester(cg, dry_run=args.dry_run)

    if args.idioms_only:
        ingester.ingest_idioms()
    elif args.cedict_only:
        ingester.ingest_cedict()
    elif args.unihan_only:
        ingester.ingest_unihan()
    elif args.tang_only:
        ingester.ingest_tang()
    else:
        ingester.ingest_all()


if __name__ == '__main__':
    main()
