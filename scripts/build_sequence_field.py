#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建序列场 — Wikipedia 首段句子 → 按文章主题分区 → 存 bigram

双场架构的下半场: 序列场不存 BGE 向量，只存每个语义盆内的字间转移频率。

流程:
  1. 加载 DragonField 概念模式 → 获取已知概念集合
  2. 读取 Wikipedia 文章 → 取首段句子
  3. 以文章标题为盆标识 → 存 bigram
  4. 剪枝去噪 → 保存为 JSON
"""

import sys, os, re, time, sqlite3, json, math
from collections import Counter, defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from loongpearl.core.sequence_field import SequenceField

# ── 配置 ──────────────────────────────────────────────
DB_PATH = os.path.join(PROJECT, "data/wikipedia/zhwiki.db")
OUT_PATH = os.path.join(PROJECT, "data/models/sequence_field.json")
MAX_ARTICLES = 50000       # 读多少篇文章
MIN_BIGRAM_COUNT = 2       # bigram 最少出现次数 (去噪)


def load_known_concepts():
    """从 DragonField 概念模式加载已知概念集合"""
    import torch
    cache = os.path.join(PROJECT, "data/models/dragon_field_patterns.pt")
    if not os.path.exists(cache):
        print("⚠️  DragonField 缓存不存在，将使用文章标题作为盆标识")
        return set()
    data = torch.load(cache, map_location='cpu')
    # 只取概念模式 (非 sequence_*)
    types = data.get('pattern_types', ['concept'] * len(data['subjects']))
    subjects = data['subjects']
    concepts = set()
    for i, t in enumerate(types):
        if not t.startswith('sequence'):
            concepts.add(subjects[i])
    print(f"  DragonField 已知概念: {len(concepts):,} 个")
    return concepts


def is_valid_sentence(s: str) -> bool:
    """判断是否为有效句子 (至少4个汉字)"""
    chars = re.findall(r'[\u4e00-\u9fff]', s)
    return len(chars) >= 4


def split_sentences(text: str) -> list:
    """将文本按句号/问号/感叹号/换行切分为句子"""
    sentences = re.split(r'[。！？!?\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def main():
    print("═" * 60)
    print("序列场构建 — 双场架构下半场")
    print("═" * 60)

    # ── 1. 加载已知概念 ──
    known_concepts = load_known_concepts()

    # ── 2. 读取 Wikipedia 首段 ──
    print(f"\n📖 读取 Wikipedia 首段 (上限 {MAX_ARTICLES:,} 篇)...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT title, text FROM articles WHERE char_count > 300 "
        "ORDER BY char_count DESC LIMIT ?",
        (MAX_ARTICLES,)
    )

    sf = SequenceField()
    total_sentences = 0
    total_bigrams_stored = 0
    skipped_no_concept = 0

    for i, (title, text) in enumerate(cursor):
        # 取首段
        paragraphs = text.split('\n')
        first_para = paragraphs[0] if paragraphs else text
        # 截取前500字 (定义通常在开头)
        first_para = first_para[:500]

        sentences = split_sentences(first_para)

        # 确定盆标识: 优先用文章标题
        basin = title.strip()
        # 如果标题在已知概念中, 用概念作为盆
        if basin in known_concepts:
            pass  # 直接用标题
        elif len(basin) >= 2:
            # 标题不在概念中但长度>=2, 仍可用
            pass
        else:
            skipped_no_concept += 1
            continue

        # 存入所有有效句子
        article_bigrams = 0
        for sent in sentences:
            if is_valid_sentence(sent):
                sf.ingest_sentence(sent, basin, weight=1.0)
                total_sentences += 1
                article_bigrams += len(re.findall(r'[\u4e00-\u9fff]', sent)) - 1

        total_bigrams_stored += article_bigrams

        if (i + 1) % 5000 == 0:
            stats = sf.global_stats()
            print(f"  {i+1:,}/{MAX_ARTICLES:,} 篇 | "
                  f"{total_sentences:,} 句 | "
                  f"{stats['num_basins']:,} 盆 | "
                  f"{stats['total_bigrams']:,} bigrams")

    conn.close()
    print(f"\n✅ 摄入完成: {i+1:,} 篇文章 → {total_sentences:,} 个句子 → "
          f"{sf.global_stats()['num_basins']:,} 个盆")

    # ── 3. 剪枝去噪 ──
    print(f"\n🔧 剪枝去噪 (min_count={MIN_BIGRAM_COUNT})...")
    sf.prune_rare(min_count=MIN_BIGRAM_COUNT)

    # ── 4. 保存 ──
    stats = sf.global_stats()
    print(f"\n📊 最终统计:")
    print(f"   盆数:     {stats['num_basins']:,}")
    print(f"   bigram总数: {stats['total_bigrams']:,}")
    print(f"   全局字表:   {stats['vocab_size']:,}")
    print(f"   前10大盆:")
    for basin, count in stats['top_basins']:
        bs = sf.basin_stats(basin)
        print(f"     {basin}: {count:,} bigrams, {bs['vocab_size']} 字")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sf.save(OUT_PATH)
    print(f"\n✅ 序列场已保存: {OUT_PATH}")


if __name__ == '__main__':
    main()
