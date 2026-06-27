#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia 知识注入脚本 — 从 zhwiki.db 提取概念关系，注入 concept_graph.db
═══════════════════════════════════════════════════════════════════════════

流程:
  1. 加载 concept_graph.db，按主题频次选出 top-N 高优先级概念
  2. 对每个概念调用 WikipediaLookup 搜索 zhwiki.db:
     - char_pairs 表 → co_concepts (字共现关系)
     - FTS5 文章检索 → 获取概念文章正文
  3. 从 Wikipedia 文本中提取三元组 (中文模式匹配 + 章节标题)
  4. 将新三元组注入 concept_graph.db (跳过已有重复)
  5. 输出每概念的注入统计

用法:
    python scripts/inject_wiki_knowledge.py --max-concepts 100 --dry-run   # 预览模式
    python scripts/inject_wiki_knowledge.py --max-concepts 100             # 实际注入
    python scripts/inject_wiki_knowledge.py --max-concepts 2000 --batch    # 全量批量
"""

import sys
import os
import re
import time
import sqlite3
import argparse
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

# ── 确保项目路径 ──
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from loongpearl.core.wiki_lookup import WikipediaLookup

log = logging.getLogger(__name__)

# ── 路径 ──
CG_DB_PATH = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')
WIKI_DB_PATH = os.path.join(PROJECT, 'data', 'wikipedia', 'zhwiki.db')

# ── 关系类型 ──
REL_IS_A    = "IS_A"
REL_RELATED = "RELATED"
REL_PART_OF = "PART_OF"
REL_HAS     = "HAS"

# ── 置信度常量 ──
CONF_HIGH   = 0.75   # 强模式匹配
CONF_MEDIUM = 0.55   # 中等模式匹配
CONF_LOW    = 0.35   # 弱模式/共现推断
CONF_COOCCUR = 0.25  # 纯字共现推断

# ── 噪声/停用概念 ──
STOP_CONCEPTS = {
    '是', '的', '了', '在', '和', '与', '或', '及', '而', '以', '为', '所', '其',
    '不', '也', '都', '就', '还', '要', '能', '会', '可', '很', '更', '最', '极',
    '这', '那', '哪', '什', '怎', '多', '少', '几', '每', '各', '某', '全', '整',
    '我', '你', '他', '她', '它', '们', '自', '己', '人', '者', '谁', '什么',
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿',
    '个', '种', '类', '些', '点', '些', '次', '回', '年', '月', '日', '时', '分', '秒',
    '上', '下', '左', '右', '前', '后', '内', '外', '中', '间', '里', '旁', '边',
    '来', '去', '出', '进', '过', '到', '从', '向', '对', '给', '把', '被', '让',
    '但', '却', '只', '仅', '另', '再', '又', '已', '将', '正', '并', '且', '虽', '然',
    '因', '此', '所', '以', '如', '果', '虽', '然', '于', '由', '之',
    '可以', '能够', '需要', '必须', '应该', '已经', '没有', '不是',
    '这个', '那个', '这些', '那些', '它们', '我们', '他们', '你们',
    '一种', '一个', '一些', '所有', '每个', '任何', '什么', '怎么',
    '因为', '所以', '但是', '而且', '或者', '如果', '虽然', '然而',
    '其中', '之间', '之后', '之前', '之上', '之下', '之外', '之内',
    '通过', '根据', '按照', '关于', '对于', '由于', '随着', '为了',
    '可能', '也许', '大约', '大概', '几乎', '完全', '非常', '十分',
    '一样', '同样', '不同', '相似', '相同', '相关', '相应', '相当',
    '包括', '称为', '比如', '例如', '其他', '主要', '特别', '尤其',
    '表示', '使用', '利用', '采用', '应用', '研究', '发展', '形成',
    '具有', '存在', '发生', '产生', '出现', '进行', '实现', '完成',
    '方面', '领域', '部分', '作用', '影响', '关系', '问题', '情况',
}


def is_valid_concept(text: str) -> bool:
    """检查是否为有效概念词（非停用词、长度合理、含中文）"""
    if not text:
        return False
    text = text.strip()
    if len(text) < 1 or len(text) > 20:
        return False
    if text in STOP_CONCEPTS:
        return False
    # 必须包含至少一个中文字符
    if not re.search(r'[\u4e00-\u9fff]', text):
        return False
    # 纯标点/空白
    if re.match(r'^[\s\.,;:!?，。；：！？"""''（）()【】\[\]《》<>、…—\-·]+$', text):
        return False
    return True


def _decode_html_entities(text: str) -> str:
    """简单解码常见 HTML 实体"""
    entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
        '&apos;': "'", '&nbsp;': ' ', '&#39;': "'",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    # 数字实体 &#xxxx;
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))) if int(m.group(1)) < 65536 else '', text)
    return text


def clean_concept(text: str) -> Optional[str]:
    """清洗概念名：去标点、去空白、去引号、去 Wiki 残留"""
    text = _decode_html_entities(text)
    # 移除 Wiki 残留标记
    text = re.sub(r"'''?", '', text)           # 粗体/斜体标记
    text = re.sub(r'={2,}', '', text)         # 章节标题残留
    text = re.sub(r'\[\[|\]\]', '', text)     # Wiki 链接残留
    text = re.sub(r'<[^>]+>', '', text)       # HTML 标签残留
    text = re.sub(r'[_/\\|]', '', text)       # Wiki 管道符等
    # 去标点
    text = re.sub(r'[\s\.,;:!?，。；：！？"""''（）()【】\[\]《》<>、…—\-·「」『』\u200b\u3000]+', '', text)
    text = text.strip()
    if is_valid_concept(text):
        # 长度检查：太长的非概念
        if len(text) <= 12:
            return text
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 中文模式 → 三元组提取
# ═══════════════════════════════════════════════════════════════════════════

# 提取模式: (正则, 关系类型, 置信度, 方向: 's_o'=组1→组2, 'o_s'=组2→组1)
EXTRACTION_RULES = [
    # ── IS_A ──
    (re.compile(r'(.{2,8})是一种(.{2,12})'),           REL_IS_A,    CONF_HIGH, 's_o'),
    (re.compile(r'(.{2,8})属于(.{2,8})类'),            REL_IS_A,    CONF_HIGH, 's_o'),
    (re.compile(r'(.{2,8})[是为属](.{2,12})(?:的)?一种'), REL_IS_A,    CONF_HIGH, 's_o'),
    (re.compile(r'(.{2,12})[是属为](.{2,12})(?:的)?一类'), REL_IS_A,    CONF_HIGH, 's_o'),
    # ── RELATED — "X，又称Y" / "X也称Y" / "X又名Y" ──
    (re.compile(r'(.{2,12})[，,](?:也)?(?:又?称|又名|又叫|亦[称叫])(.{2,12})'), REL_RELATED, CONF_HIGH, 's_o'),
    (re.compile(r'(.{2,12})(?:又|也|亦)(?:称|名|叫)(.{2,12})'), REL_RELATED, CONF_MEDIUM, 's_o'),
    # ── RELATED — "X与Y" / "X和Y" (必须带关联词) ──
    (re.compile(r'(.{2,6})(?:与|和|跟|同)(.{2,6})(?:相?关|有关|相关|联系|结合|组合|配合|搭配)'), REL_RELATED, CONF_MEDIUM, 's_o'),
    # ── PART_OF — "X是Y的组成部分" / "X属于Y的一部分" ──
    (re.compile(r'(.{2,8})是(.{2,12})的(?:组成部分|一部分|分支|子集|成员)'), REL_PART_OF, CONF_HIGH, 's_o'),
    (re.compile(r'(.{2,8})属于(.{2,12})的(?:一部分|组成部分)'), REL_PART_OF, CONF_HIGH, 's_o'),
    # ── PART_OF — "X由Y组成" ──
    (re.compile(r'(.{2,8})由(.{2,8})(?:和(.{2,8}))?(?:共同)?(?:所)?(?:组?成|构[成建])'), REL_PART_OF, CONF_HIGH, 'o_s'),
    # ── HAS — "X包含Y" / "X包括Y" / "X含有Y" ──
    (re.compile(r'(.{2,8})(?:包[含括]|[含擁]有|具[有备])(.{2,8})'), REL_HAS, CONF_MEDIUM, 's_o'),
    # ── RELATED — "X与Y" / "X和Y" (宽泛共现，低置信度) ──
    (re.compile(r'(.{2,6})与(.{2,6})'), REL_RELATED, CONF_LOW, 's_o'),
    (re.compile(r'(.{2,6})和(.{2,6})'), REL_RELATED, CONF_LOW, 's_o'),
]

# 章节标题模式 — 从标题推断关系
CHAPTER_PATTERNS = [
    # "== 定义 ==" / "== 概念 ==" 之后的第一次出现的实体与该概念 IS_A
    # "== 历史 ==" → 概念 RELATED 历史
    (re.compile(r'={2,}\s*(?:概述|定义|概念|简介|介绍)\s*={2,}'), REL_IS_A, 'chapter_def'),
    (re.compile(r'={2,}\s*(?:历史|起源|由来|发展|沿革)\s*={2,}'), REL_RELATED, 'chapter_history'),
    (re.compile(r'={2,}\s*(?:分类|种类|类型|品种)\s*={2,}'), REL_IS_A, 'chapter_class'),
    (re.compile(r'={2,}\s*(?:结构|组成|构造|成分)\s*={2,}'), REL_PART_OF, 'chapter_structure'),
    (re.compile(r'={2,}\s*(?:特点|特征|特性|性质|属性)\s*={2,}'), REL_HAS, 'chapter_props'),
    (re.compile(r'={2,}\s*(?:作用|功能|用途|应用)\s*={2,}'), REL_RELATED, 'chapter_usage'),
    (re.compile(r'={2,}\s*(?:关系|关联|相关)\s*={2,}'), REL_RELATED, 'chapter_relation'),
    (re.compile(r'={2,}\s*(?:影响|意义|价值|地位)\s*={2,}'), REL_RELATED, 'chapter_impact'),
]


def extract_triples_from_text(text: str, parent_concept: str = None,
                              max_results: int = 80) -> List[Tuple[str, str, str, float]]:
    """从中文文本中提取 (subject, relation, object, confidence) 四元组。

    Args:
        text: 中文文本
        parent_concept: 父概念（用于章节标题推断）
        max_results: 最大提取数

    Returns:
        [(subject, relation, object, confidence), ...]
    """
    if not text or len(text) < 10:
        return []

    results = []
    seen = set()

    # ── 1. 正则模式提取 ──
    for pattern, rel_type, conf, direction in EXTRACTION_RULES:
        for match in pattern.finditer(text):
            if len(results) >= max_results:
                break
            groups = match.groups()
            if len(groups) < 2:
                continue

            if direction == 's_o':
                s = clean_concept(groups[0])
                o = clean_concept(groups[1])
                if s and o and s != o:
                    key = f"{s}|{rel_type}|{o}"
                    if key not in seen:
                        seen.add(key)
                        results.append((s, rel_type, o, conf))
                # 如果有第三个组 (如 "X由Y和Z组成")
                if len(groups) > 2 and groups[2]:
                    s2 = clean_concept(groups[0])
                    o2 = clean_concept(groups[2])
                    if s2 and o2 and s2 != o2:
                        key = f"{s2}|{rel_type}|{o2}"
                        if key not in seen:
                            seen.add(key)
                            results.append((s2, rel_type, o2, conf))
            elif direction == 'o_s':
                # 反向: "X由Y组成" → Y PART_OF X
                o = clean_concept(groups[0])
                s = clean_concept(groups[1])
                if s and o and s != o:
                    key = f"{s}|{rel_type}|{o}"
                    if key not in seen:
                        seen.add(key)
                        results.append((s, rel_type, o, conf))
                if len(groups) > 2 and groups[2]:
                    o2 = clean_concept(groups[0])
                    s2 = clean_concept(groups[2])
                    if s2 and o2 and s2 != o2:
                        key = f"{s2}|{rel_type}|{o2}"
                        if key not in seen:
                            seen.add(key)
                            results.append((s2, rel_type, o2, conf))

        if len(results) >= max_results:
            break

    # ── 2. 章节标题推断 ──
    if parent_concept and is_valid_concept(parent_concept):
        for chap_pattern, rel_type, chap_type in CHAPTER_PATTERNS:
            chap_match = chap_pattern.search(text)
            if not chap_match:
                continue
            chap_start = chap_match.end()

            # 获取章节后续文本 (500字)
            chap_text = text[chap_start:chap_start + 500]

            # 在此文本中查找其他概念词
            # 提取章节后的第一个主要名词
            found_concepts = _extract_key_concepts(chap_text, exclude={parent_concept})
            for other_concept in found_concepts[:5]:
                if other_concept == parent_concept:
                    continue
                key = f"{parent_concept}|{rel_type}|{other_concept}"
                if key not in seen:
                    seen.add(key)
                    results.append((parent_concept, rel_type, other_concept, CONF_LOW))

            if len(results) >= max_results:
                break

    return results


def _extract_key_concepts(text: str, exclude: Set[str] = None, max_items: int = 10) -> List[str]:
    """从文本中提取关键概念词（非停用词的名词性短语）"""
    exclude = exclude or set()
    concepts = []
    seen = set()

    # 匹配中文词汇（2-6字）
    words = re.findall(r'[\u4e00-\u9fff]{2,6}', text)

    for w in words:
        w_clean = clean_concept(w)
        if w_clean and w_clean not in seen and w_clean not in exclude:
            seen.add(w_clean)
            concepts.append(w_clean)
            if len(concepts) >= max_items:
                break

    return concepts


# ═══════════════════════════════════════════════════════════════════════════
# 概念选择
# ═══════════════════════════════════════════════════════════════════════════

def get_top_concepts(cg_conn: sqlite3.Connection, limit: int = 2000) -> List[Tuple[str, int]]:
    """从 concept_graph.db 中获取出现频次最高的主题概念。

    Returns:
        [(concept, frequency), ...] 按频次降序
    """
    rows = cg_conn.execute(
        "SELECT s, COUNT(*) as cnt FROM triples "
        "GROUP BY s ORDER BY cnt DESC LIMIT ?",
        (limit * 2,)  # 取2倍，过滤后保留limit
    ).fetchall()

    concepts = []
    for s, cnt in rows:
        if not s:
            continue
        # 过滤掉非概念（纯标点、单字非中文、停用词）
        if is_valid_concept(s) and len(s) >= 1:
            # 优先多字概念（更有语义）
            concepts.append((s, cnt))

    # 按频次排序，优先多字概念
    concepts.sort(key=lambda x: (-x[1], -len(x[0])))
    return concepts[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# 核心注入逻辑
# ═══════════════════════════════════════════════════════════════════════════

def triple_exists(cg_conn: sqlite3.Connection, s: str, r: str, o: str) -> bool:
    """检查三元组是否已存在于 concept_graph.db"""
    row = cg_conn.execute(
        "SELECT 1 FROM triples WHERE s=? AND r=? AND o=? LIMIT 1",
        (s, r, o)
    ).fetchone()
    return row is not None


def insert_triple(cg_conn: sqlite3.Connection, s: str, r: str, o: str,
                  confidence: float, source: str = "wikipedia_extract") -> bool:
    """向 concept_graph.db 插入三元组（跳过重复）。

    Returns:
        True 如果成功插入，False 如果已存在
    """
    if triple_exists(cg_conn, s, r, o):
        return False

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        cg_conn.execute(
            "INSERT INTO triples(s, r, o, c, src, ev, learned_at, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (s, r, o, round(confidence, 4), source, '', now, now)
        )
        return True
    except sqlite3.IntegrityError:
        return False


def process_concept(wiki: WikipediaLookup, cg_conn: sqlite3.Connection,
                    concept: str, freq: int) -> Dict:
    """处理单个概念：搜索 Wikipedia → 提取三元组 → 注入 concept_graph.db

    Returns:
        统计字典
    """
    result = {
        'concept': concept,
        'freq': freq,
        'article_found': False,
        'co_concepts_count': 0,
        'triples_extracted': 0,
        'triples_added': 0,
        'triples_skipped': 0,
        'new_triples': [],
    }

    # ── 1. FTS5 搜索文章 ──
    search_results = wiki.search_articles(concept, limit=1)
    article_text = ""

    if search_results:
        article = wiki.get_article(search_results[0]['title'])
        if article:
            article_text = article.get('text', '')
            result['article_found'] = True

    if not article_text:
        # 降级：直接用 FTS5 snippet
        if search_results:
            article_text = search_results[0].get('snippet', '')

    # ── 2. char_pairs 共现 ──
    # 取概念的首个汉字查询共现字
    first_char = None
    for ch in concept:
        if '\u4e00' <= ch <= '\u9fff':
            first_char = ch
            break

    co_concepts = []
    if first_char:
        try:
            co_concepts = wiki.get_co_concepts(first_char, min_count=5, limit=30)
        except Exception:
            pass
    result['co_concepts_count'] = len(co_concepts)

    # 从共现字推导 RELATED 三元组
    co_triples = []
    for co_char, co_count in co_concepts:
        if is_valid_concept(co_char) and co_char != concept and co_char != first_char:
            # 只对多字概念生成完整的 RELATED
            if len(concept) >= 1:
                co_triples.append((concept, REL_RELATED, co_char,
                                   min(CONF_COOCCUR + co_count * 0.001, 0.4)))

    # ── 3. 从文章文本提取三元组 ──
    text_triples = []
    if article_text:
        text_triples = extract_triples_from_text(
            article_text, parent_concept=concept, max_results=60
        )

    # 合并所有三元组（去重）
    all_triples = []
    seen = set()
    for s, r, o, c in text_triples + co_triples:
        key = f"{s}|{r}|{o}"
        if key not in seen:
            seen.add(key)
            all_triples.append((s, r, o, c))
    result['triples_extracted'] = len(all_triples)

    # ── 4. 注入 concept_graph.db ──
    added_count = 0
    skipped_count = 0
    new_triples_detail = []

    for s, r, o, c in all_triples:
        if insert_triple(cg_conn, s, r, o, c, source="wikipedia_extract"):
            added_count += 1
            new_triples_detail.append((s, r, o, round(c, 3)))
        else:
            skipped_count += 1

    result['triples_added'] = added_count
    result['triples_skipped'] = skipped_count
    result['new_triples'] = new_triples_detail

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia → concept_graph.db 知识注入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/inject_wiki_knowledge.py --max-concepts 100 --dry-run
  python scripts/inject_wiki_knowledge.py --max-concepts 100
  python scripts/inject_wiki_knowledge.py --max-concepts 2000 --commit-interval 50
        """
    )
    parser.add_argument('--max-concepts', type=int, default=2000,
                        help='最大处理概念数 (default: 2000)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式：只评估不写入 concept_graph.db')
    parser.add_argument('--commit-interval', type=int, default=100,
                        help='每 N 个概念提交一次 (default: 100)')
    parser.add_argument('--min-freq', type=int, default=3,
                        help='最低概念频次阈值 (default: 3)')
    parser.add_argument('--cg-db', type=str, default=CG_DB_PATH,
                        help='concept_graph.db 路径')
    parser.add_argument('--wiki-db', type=str, default=WIKI_DB_PATH,
                        help='zhwiki.db 路径')
    parser.add_argument('--concept', type=str, default=None,
                        help='指定单个概念测试 (覆盖 --max-concepts)')
    args = parser.parse_args()

    # ── 验证路径 ──
    if not os.path.exists(args.cg_db):
        print(f"❌ concept_graph.db 不存在: {args.cg_db}")
        sys.exit(1)
    if not os.path.exists(args.wiki_db):
        print(f"❌ zhwiki.db 不存在: {args.wiki_db}")
        sys.exit(1)

    print("=" * 70)
    print("📖 Wikipedia 知识注入 → concept_graph.db")
    print("=" * 70)
    print(f"   概念图 DB: {args.cg_db}")
    print(f"   Wiki DB:   {args.wiki_db}")
    print(f"   最大概念数: {args.max_concepts}")
    print(f"   最低频次:   {args.min_freq}")
    print(f"   模式:       {'🔍 干运行 (预览)' if args.dry_run else '✍️  实际写入'}")
    print()

    # ── 连接数据库 ──
    cg_conn = sqlite3.connect(args.cg_db)
    cg_conn.execute("PRAGMA journal_mode=WAL")
    cg_conn.execute("PRAGMA synchronous=NORMAL")
    cg_conn.execute("PRAGMA cache_size=-64000")

    wiki = WikipediaLookup(args.wiki_db)

    # ── 统计基线 ──
    initial_count = cg_conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    print(f"📊 concept_graph.db 基线: {initial_count:,} 条三元组")
    print()

    # ── 获取概念列表 ──
    if args.concept:
        concepts = [(args.concept, 1)]
        print(f"🎯 指定概念: {args.concept}")
        print()
    else:
        print("🔍 获取高频概念...")
        t0 = time.time()
        raw_concepts = get_top_concepts(cg_conn, limit=args.max_concepts)
        concepts = [(c, f) for c, f in raw_concepts if f >= args.min_freq]
        elapsed = time.time() - t0
        print(f"   选出 {len(concepts)} 个高频概念 (阈值≥{args.min_freq}) ({elapsed:.1f}s)")
        print(f"   Top-10: {', '.join(c for c, _ in concepts[:10])}")
        print()

    # ── 逐概念处理 ──
    print("🚀 开始处理...")
    print()

    t_start = time.time()
    total_added = 0
    total_skipped = 0
    total_extracted = 0
    articles_found = 0
    concept_results = []

    for i, (concept, freq) in enumerate(concepts):
        result = process_concept(wiki, cg_conn, concept, freq)

        if result['article_found']:
            articles_found += 1
        total_added += result['triples_added']
        total_skipped += result['triples_skipped']
        total_extracted += result['triples_extracted']
        concept_results.append(result)

        # 输出进度
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (len(concepts) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(concepts)}] {concept:<12s} "
                  f"freq={freq:<5d} article={'✓' if result['article_found'] else '✗'} "
                  f"extracted={result['triples_extracted']:<3d} "
                  f"added={result['triples_added']:<3d} "
                  f"| {rate:.0f}con/s ETA={eta:.0f}s",
                  flush=True)

        # 定期提交
        if not args.dry_run and (i + 1) % args.commit_interval == 0:
            cg_conn.commit()
            print(f"  💾 已提交 ({i+1} 概念)", flush=True)

    # ── 最终提交 ──
    if not args.dry_run:
        cg_conn.commit()

    total_elapsed = time.time() - t_start

    # ── 最终统计 ──
    print()
    print("=" * 70)
    print("📊 注入统计报告")
    print("=" * 70)
    print(f"   处理概念数:     {len(concepts)}")
    print(f"   找到文章:       {articles_found} / {len(concepts)} "
          f"({articles_found/max(len(concepts),1)*100:.1f}%)")
    print(f"   提取三元组:     {total_extracted}")
    print(f"   新增三元组:     {total_added}")
    print(f"   跳过(重复):     {total_skipped}")
    print(f"   总耗时:         {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f}min)")
    if len(concepts) > 0:
        print(f"   平均每概念:     {total_elapsed/len(concepts):.2f}s")
    print()

    if not args.dry_run:
        final_count = cg_conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        growth = final_count - initial_count
        print(f"   concept_graph.db: {initial_count:,} → {final_count:,} "
              f"(+{growth:,}, +{growth/max(initial_count,1)*100:.2f}%)")
        print()

    # ── Top 新增概念 ──
    if concept_results:
        print("   Top-20 新增最多的概念:")
        concept_results.sort(key=lambda r: -r['triples_added'])
        for r in concept_results[:20]:
            if r['triples_added'] > 0:
                sample = ", ".join(
                    f"({s} {rel} {o})"
                    for s, rel, o, _ in r['new_triples'][:3]
                ) if r['new_triples'] else ""
                print(f"     {r['concept']:<12s} +{r['triples_added']:<4d} {sample}")
    print()

    if args.dry_run:
        print("🔍 干运行完成 — 未修改 concept_graph.db")
    else:
        print("✅ Wikipedia 知识注入完成")
        print(f"   concept_graph.db 已更新: {args.cg_db}")

    # ── 清理 ──
    wiki.close()
    cg_conn.close()


if __name__ == '__main__':
    main()
