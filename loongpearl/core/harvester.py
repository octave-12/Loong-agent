#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠万象收 (Harvester) — 大规模确定性知识采集引擎
════════════════════════════════════════════════════════════════════════════

三路并行知识采集，不依赖 LLM，全链路确定性：
  源A: 结构化数据库 (Wikidata SPARQL)
  源B: 百科全书文本 (Wikipedia dumps → 正则提取)
  源C: 学术论文 (arXiv API → 标题/摘要实体识别)

设计目标: 数小时内追平 GPT-4 的知识量级（从10万→5000万三元组）

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.harvester import KnowledgeHarvester

    h = KnowledgeHarvester(concept_graph)
    h.harvest_wikidata(limit=10000)    # Wikidata 批量采集
    h.harvest_wikipedia("物理学")      # Wikipedia 条目采集
    h.harvest_arxiv("quantum", max_results=50)  # arXiv 论文采集

"""
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 知识提取正则库 — 从自由文本提取三元组的确定性模式
# ═══════════════════════════════════════════════════════════════════════════

# 中文模式
_ZH_PATTERNS = [
    # (正则, 关系类型, 主体组, 客体组)
    # IS_A 模式
    (r'(.{1,10})是(一种|一个|一类|个)(.{1,15})', 'IS_A', 0, 2),
    (r'(.{1,10})属于(.{1,15})(的)?(一种|范畴|类别)', 'IS_A', 0, 1),
    (r'(.{1,10})被归类为(.{1,15})', 'IS_A', 0, 1),

    # PART_OF 模式
    (r'(.{1,10})是(.{1,15})的(组成)?部分', 'PART_OF', 0, 1),
    (r'(.{1,10})由(.{1,15})组成', 'PART_OF', 1, 0),
    (r'(.{1,10})包含(.{1,15})', 'HAS', 0, 1),
    (r'(.{1,10})包括(.{1,15})', 'HAS', 0, 1),

    # CAUSE 模式
    (r'(.{1,10})导致(.{1,15})', 'CAUSE', 0, 1),
    (r'(.{1,10})引起(.{1,15})', 'CAUSE', 0, 1),
    (r'(.{1,10})造成(.{1,15})', 'CAUSE', 0, 1),
    (r'因为(.{1,15})[，,]?所以(.{1,15})', 'CAUSE', 0, 1),

    # OPPOSITE 模式
    (r'(.{1,10})与(.{1,10})相反', 'OPPOSITE', 0, 1),
    (r'(.{1,10})是(.{1,10})的对立面', 'OPPOSITE', 0, 1),

    # HAS 模式
    (r'(.{1,10})具有(.{1,15})', 'HAS', 0, 1),
    (r'(.{1,10})拥有(.{1,15})', 'HAS', 0, 1),

    # RELATED 模式
    (r'(.{1,10})与(.{1,10})相关', 'RELATED', 0, 1),
    (r'(.{1,10})和(.{1,10})有关', 'RELATED', 0, 1),
]

# 英文模式
_EN_PATTERNS = [
    (r'(\w+(?:\s+\w+){0,3}) is (?:a|an|the) (\w+(?:\s+\w+){0,3})', 'IS_A', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) (?:consists of|is composed of|comprises) (\w+(?:\s+\w+){0,3})', 'PART_OF', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) causes? (\w+(?:\s+\w+){0,3})', 'CAUSE', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) (?:leads to|results in) (\w+(?:\s+\w+){0,3})', 'CAUSE', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) has (\w+(?:\s+\w+){0,3})', 'HAS', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) is related to (\w+(?:\s+\w+){0,3})', 'RELATED', 0, 1),
    (r'(\w+(?:\s+\w+){0,3}) (?:is opposed to|is the opposite of) (\w+(?:\s+\w+){0,3})', 'OPPOSITE', 0, 1),
]


# ═══════════════════════════════════════════════════════════════════════════
# 万象收主类
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeHarvester:
    """
    万象收 — 大规模多源知识采集引擎。

    三条采集管线:
      1. Wikidata SPARQL → 结构化三元组批量导入
      2. Wikipedia API → 百科文本正则提取
      3. arXiv API → 学术论文实体识别

    所有管线都是确定性的，不依赖 LLM。
    """

    def __init__(self, concept_graph=None):
        self.cg = concept_graph
        self.stats = {
            "total_harvested": 0,
            "by_source": defaultdict(int),
            "by_relation": defaultdict(int),
            "errors": 0,
            "start_time": None,
        }
        # 停用词过滤
        self._stop_words = {
            '这个', '那个', '一个', '一种', '这些', '那些', '所有', '每个',
            '可以', '可能', '应该', '必须', '需要', '能够', '会', '能',
            '的', '了', '是', '在', '和', '与', '或', '及',
        }

    # ═════════════════════════════════════════════════════════════════════
    # 源A: Wikidata SPARQL
    # ═════════════════════════════════════════════════════════════════════

    def harvest_wikidata(self, limit: int = 5000,
                         property_filter: List[str] = None) -> int:
        """
        从 Wikidata SPARQL 端点采集结构化知识。

        常用属性:
          P31 (instance of) → IS_A
          P279 (subclass of) → IS_A
          P361 (part of) → PART_OF
          P527 (has part) → HAS
          P828 (has cause) → CAUSE

        Args:
            limit: 最大采集条数
            property_filter: 只采集特定属性

        Returns:
            采集到的三元组数量
        """
        print(f"[万象收/Wikidata] 开始采集 (limit={limit})...")

        if property_filter is None:
            # 默认采集最常用的知识属性
            property_filter = ['P31', 'P279', 'P361', 'P527']

        # Wikidata SPARQL endpoint
        endpoint = "https://query.wikidata.org/sparql"

        # 属性→关系映射
        PROP_MAP = {
            'P31': 'IS_A',     # instance of
            'P279': 'IS_A',    # subclass of
            'P361': 'PART_OF', # part of
            'P527': 'HAS',     # has part
            'P828': 'CAUSE',   # has cause
        }

        total = 0
        batch_size = min(limit, 500)

        for prop in property_filter:
            relation = PROP_MAP.get(prop, 'RELATED')

            # SPARQL 查询: 获取 (subject, property, object) 三元组
            # 要求 subject 和 object 都有中文标签
            query = f"""
            SELECT ?subjectLabel ?objectLabel WHERE {{
              ?subject wdt:{prop} ?object.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh". }}
            }}
            LIMIT {batch_size}
            """

            try:
                params = urllib.parse.urlencode({
                    'format': 'json',
                    'query': query,
                })
                url = f"{endpoint}?{params}"
                headers = {'User-Agent': 'LoongAgent/1.0 (KnowledgeHarvester)'}

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode('utf-8'))

                bindings = data.get('results', {}).get('bindings', [])
                for binding in bindings:
                    subj = binding.get('subjectLabel', {}).get('value', '')
                    obj = binding.get('objectLabel', {}).get('value', '')
                    if subj and obj and len(subj) >= 2 and len(obj) >= 2:
                        if self.cg:
                            self.cg.add_triple(subj, relation, obj,
                                               confidence=0.85,
                                               source="wikidata")
                        total += 1
                        self.stats["by_relation"][relation] += 1

                self.stats["by_source"]["wikidata"] += total
                print(f"  [Wikidata/{prop}] 采集 {len(bindings)} 条")
                time.sleep(0.5)  # 速率限制

            except Exception as e:
                print(f"  [Wikidata/{prop}] 查询失败: {e}")
                self.stats["errors"] += 1

        self.stats["total_harvested"] += total
        return total

    # ═════════════════════════════════════════════════════════════════════
    # 源B: Wikipedia
    # ═════════════════════════════════════════════════════════════════════

    def harvest_wikipedia(self, titles: List[str] = None,
                          lang: str = "zh",
                          max_per_page: int = 50) -> int:
        """
        从 Wikipedia API 采集百科条目文本并用正则提取三元组。

        Args:
            titles: 要采集的条目名称列表
            lang: 语言 (zh/en)
            max_per_page: 每页最多提取的三元组数

        Returns:
            采集到的三元组数量
        """
        if titles is None:
            titles = [
                "物理学", "化学", "生物学", "数学", "计算机科学",
                "哲学", "历史", "经济学", "天文学", "地理学",
                "量子力学", "相对论", "进化论", "遗传学", "人工智能",
            ]

        print(f"[万象收/Wikipedia] 开始采集 {len(titles)} 个条目...")
        total = 0

        for title in titles:
            try:
                triples = self._fetch_and_extract_wiki(title, lang, max_per_page)
                for subj, rel, obj in triples:
                    if self.cg:
                        self.cg.add_triple(subj, rel, obj,
                                           confidence=0.7,
                                           source=f"wikipedia:{title}")
                    total += 1
                    self.stats["by_relation"][rel] += 1

                print(f"  [Wikipedia] {title}: 提取 {len(triples)} 个三元组")
                time.sleep(0.3)  # 速率限制

            except Exception as e:
                print(f"  [Wikipedia] {title}: 失败 - {e}")
                self.stats["errors"] += 1

        self.stats["by_source"]["wikipedia"] += total
        self.stats["total_harvested"] += total
        return total

    def _fetch_and_extract_wiki(self, title: str, lang: str,
                                 max_triples: int) -> List[Tuple[str, str, str]]:
        """获取 Wikipedia 文本并提取三元组"""
        # Wikipedia API
        encoded_title = urllib.parse.quote(title)
        url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=1&explaintext=1"
            f"&titles={encoded_title}&format=json"
        )

        headers = {'User-Agent': 'LoongAgent/1.0 (KnowledgeHarvester)'}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        pages = data.get('query', {}).get('pages', {})
        texts = []
        for page_id, page_data in pages.items():
            text = page_data.get('extract', '')
            if text:
                texts.append(text)

        # 用正则从文本中提取三元组
        return self._extract_triples_from_text('\n'.join(texts), lang, max_triples)

    def _extract_triples_from_text(self, text: str, lang: str,
                                    max_triples: int) -> List[Tuple[str, str, str]]:
        """从自由文本中用正则提取三元组"""
        patterns = _ZH_PATTERNS if lang == "zh" else _EN_PATTERNS
        triples = []

        # 按句号分割为句子
        sentences = re.split(r'[。！？.!?\n]', text)

        for sentence in sentences:
            if len(triples) >= max_triples:
                break
            sentence = sentence.strip()
            if len(sentence) < 4:
                continue

            for pattern, relation, s_group, o_group in patterns:
                for match in re.finditer(pattern, sentence):
                    if len(triples) >= max_triples:
                        break
                    try:
                        subj = match.group(s_group + 1).strip()
                        obj = match.group(o_group + 1).strip()

                        # 过滤噪音
                        if self._is_valid_concept(subj) and self._is_valid_concept(obj):
                            triples.append((subj, relation, obj))
                    except IndexError:
                        continue

        return triples

    def _is_valid_concept(self, text: str) -> bool:
        """验证是否为有效概念（非停用词，非纯标点，非太长）"""
        text = text.strip()
        if len(text) < 2 or len(text) > 20:
            return False
        if text in self._stop_words:
            return False
        if re.match(r'^[\s\d\W]+$', text):
            return False
        return True

    # ═════════════════════════════════════════════════════════════════════
    # 源C: arXiv
    # ═════════════════════════════════════════════════════════════════════

    def harvest_arxiv(self, query: str = "",
                       category: str = "",
                       max_results: int = 50) -> int:
        """
        从 arXiv API 采集学术论文标题和摘要，提取概念三元组。

        Args:
            query: 搜索关键词
            category: arXiv 分类 (如 cs.AI, quant-ph)
            max_results: 最大结果数

        Returns:
            采集到的三元组数量
        """
        print(f"[万象收/arXiv] 搜索 '{query or category}'...")

        params = {
            'search_query': f"all:{query}" if query else f"cat:{category}",
            'start': 0,
            'max_results': min(max_results, 100),
            'sortBy': 'relevance',
        }
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_text = resp.read().decode('utf-8')

            total = self._parse_arxiv_xml(xml_text, max_results)
            self.stats["by_source"]["arxiv"] += total
            self.stats["total_harvested"] += total
            return total

        except Exception as e:
            print(f"  [arXiv] 采集失败: {e}")
            self.stats["errors"] += 1
            return 0

    def _parse_arxiv_xml(self, xml_text: str, max_results: int) -> int:
        """解析 arXiv XML 响应"""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return 0

        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom',
        }

        total = 0
        entries = root.findall('atom:entry', ns)

        for entry in entries[:max_results]:
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)

            if title_elem is None:
                continue

            title = title_elem.text.strip()
            summary = summary_elem.text.strip() if summary_elem is not None else ""

            # 从标题中提取概念
            triples = []

            # 标题中的 "X: Y" 模式 → X IS_A Y 的弱版本
            if ':' in title or '：' in title:
                parts = re.split(r'[:：]', title, 1)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()[:50]
                    if self._is_valid_concept(left) and self._is_valid_concept(right):
                        triples.append((right, 'RELATED', left))

            # 从摘要中提取
            if summary:
                short_summary = summary[:500]
                text_triples = self._extract_triples_from_text(
                    short_summary, "en", max(3, 20 - len(triples))
                )
                triples.extend(text_triples)

            # 导入概念图
            for subj, rel, obj in triples:
                if self.cg:
                    self.cg.add_triple(subj, rel, obj, confidence=0.5,
                                       source="arxiv")
                total += 1

        print(f"  [arXiv] 采集 {total} 个三元组")
        return total

    # ═════════════════════════════════════════════════════════════════════
    # 本地文本采集
    # ═════════════════════════════════════════════════════════════════════

    def harvest_from_text(self, text: str, lang: str = "zh",
                          source: str = "unknown",
                          max_triples: int = 100) -> int:
        """从任意文本中提取三元组"""
        triples = self._extract_triples_from_text(text, lang, max_triples)
        total = 0
        for subj, rel, obj in triples:
            if self.cg:
                self.cg.add_triple(subj, rel, obj, confidence=0.6, source=source)
            total += 1
            self.stats["by_relation"][rel] += 1

        self.stats["by_source"][source] += total
        self.stats["total_harvested"] += total
        return total

    def harvest_from_file(self, filepath: str, lang: str = "zh",
                          source: str = "file",
                          max_triples: int = 500) -> int:
        """从文本文件中提取三元组"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.harvest_from_text(text, lang, source, max_triples)
        except Exception as e:
            print(f"[万象收] 文件读取失败: {e}")
            return 0

    # ═════════════════════════════════════════════════════════════════════
    # 批处理
    # ═════════════════════════════════════════════════════════════════════

    def harvest_all(self,
                    wikidata_limit: int = 2000,
                    wiki_titles: List[str] = None,
                    arxiv_query: str = "quantum",
                    arxiv_max: int = 50) -> Dict[str, int]:
        """
        三路并行采集。

        Returns:
            {source: count}
        """
        self.stats["start_time"] = time.time()
        results = {}

        # 源A: Wikidata
        results['wikidata'] = self.harvest_wikidata(limit=wikidata_limit)

        # 源B: Wikipedia
        results['wikipedia'] = self.harvest_wikipedia(titles=wiki_titles)

        # 源C: arXiv
        results['arxiv'] = self.harvest_arxiv(query=arxiv_query, max_results=arxiv_max)

        elapsed = time.time() - (self.stats["start_time"] or time.time())
        print(f"\n[万象收] 采集完成: {self.stats['total_harvested']} 三元组 "
              f"耗时 {elapsed:.1f}s")

        return results

    def print_stats(self):
        """打印统计"""
        s = self.stats
        print(f"═══ 万象收统计 ═══")
        print(f"  总三元组: {s['total_harvested']:>8}")
        print(f"  按来源:")
        for src, count in s['by_source'].items():
            print(f"    {src:20s}: {count:>8}")
        print(f"  按关系:")
        for rel, count in s['by_relation'].items():
            print(f"    {rel:10s}: {count:>8}")
        print(f"  错误:     {s['errors']:>8}")


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_harvester():
    """自测万象收 — 文本提取模式（不联网）"""
    h = KnowledgeHarvester()

    # 测试中文文本提取
    text_zh = """
    电子是原子的一种组成部分。原子由质子和中子组成。
    质子具有正电荷。中子是电中性的粒子。
    量子力学是物理学的一个分支。光电效应导致电子从金属表面逸出。
    热与冷是相反的概念。熵与信息论密切相关。
    细胞是生物体的基本单位。基因位于染色体上。
    """

    count = h.harvest_from_text(text_zh, lang="zh", source="test")
    print(f"中文文本提取: {count} 个三元组")
    h.print_stats()

    # 测试英文文本提取
    h2 = KnowledgeHarvester()
    text_en = """
    Quantum mechanics is a branch of physics. The electron is a fundamental particle.
    Gravity causes objects to fall. Heat is the opposite of cold.
    DNA has a double helix structure. Evolution leads to species diversity.
    """
    count2 = h2.harvest_from_text(text_en, lang="en", source="test")
    print(f"\n英文文本提取: {count2} 个三元组")
    h2.print_stats()


if __name__ == "__main__":
    test_harvester()
