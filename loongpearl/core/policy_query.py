#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策应器搜索策略 — 因子→关键词模板（纯规则驱动，零 LLM 依赖）。
=============================================================

根据盲区检测结果，为每个因子生成针对性的搜索关键词。
不依赖 LLM 生成查询——因子分类确定，模板确定。

设计原则:
  - 每个因子有 2-3 个查询模板，增加搜索覆盖面
  - 模板包含 {char}、{char_a}、{char_b} 占位符
  - 搜索结果由 trafilatura 提取正文 → DualExtractor 提取字对
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchStrategy:
    """搜索策略"""
    queries: List[str]           # 搜索查询列表
    target_knowledge: str        # 目标知识域
    expected_source: str         # 期望来源: 'wikipedia' | 'wikidata' | 'web_search' | 'any'
    priority: int = 5            # 1=最优先


# ═══════════════════════════════════════════════════════════════════
# 因子 → 查询模板映射
# ═══════════════════════════════════════════════════════════════════

FACTOR_QUERY_TEMPLATES: Dict[str, Dict] = {
    'statistical': {
        'templates': [
            '"{char}" 字在词语中的用法',
            '包含"{char}"字的常见词语有哪些',
            '"{char}" 汉字 组词',
        ],
        'target': '词语搭配',
        'source': 'wikipedia',
    },
    'energy': {
        'templates': [
            '"{char}" 是什么意思',
            '"{char}" 的定义和解释',
            '汉字"{char}" 的含义 读音',
        ],
        'target': '字义解释',
        'source': 'any',
    },
    'coverage': {
        'templates': [
            '包含"{char}"的词语 常见搭配',
            '"{char}" 在中文中的组合',
            '与"{char}"相关的概念',
        ],
        'target': '连接广度',
        'source': 'wikipedia',
    },
    'dead_end': {
        'templates': [
            '"{char}"字开头的成语',
            '"{char}" 打头的词语',
            '以"{char}"开头的词汇有哪些',
        ],
        'target': '后续候选',
        'source': 'any',
    },
    'gradient': {
        'templates': [
            '与"{char}"相关的概念和知识',
            '"{char}" 相关知识领域',
            '关于"{char}"的百科介绍',
        ],
        'target': '关联概念',
        'source': 'wikipedia',
    },
    'semantic': {
        'templates': [
            '"{char_a}"和"{char_b}"有什么关系',
            '"{char_a}" "{char_b}" 区别 联系',
            '"{char_a}"与"{char_b}" 关联',
        ],
        'target': '语义关联',
        'source': 'wikidata',
    },
    'freshness': {
        'templates': [
            '"{char}" 的常见用法',
            '汉字"{char}"怎么用',
            '"{char}" 用法举例',
        ],
        'target': '使用频率',
        'source': 'any',
    },
}


def build_search_strategy(factor_name: str, char: str,
                          char_b: str = None) -> SearchStrategy:
    """
    为盲区因子生成搜索策略。
    
    Args:
        factor_name: 因子名称 (statistical/energy/coverage/dead_end/gradient/semantic/freshness)
        char: 主字符
        char_b: 语义因子需要的第二个字符
    
    Returns:
        SearchStrategy 对象
    """
    config = FACTOR_QUERY_TEMPLATES.get(factor_name)
    if config is None:
        # 未知因子，通用回退
        return SearchStrategy(
            queries=[f'"{char}" 中文'],
            target_knowledge='通用',
            expected_source='any',
            priority=10,
        )
    
    queries = []
    for tmpl in config['templates']:
        q = tmpl.format(char=char, char_a=char, char_b=char_b or char)
        queries.append(q)
    
    return SearchStrategy(
        queries=queries,
        target_knowledge=config['target'],
        expected_source=config['source'],
        priority={'wikipedia': 1, 'wikidata': 2, 'web_search': 3, 'any': 4}.get(
            config['source'], 5
        ),
    )


def build_multi_strategy(gaps: list, max_per_factor: int = 3) -> List[SearchStrategy]:
    """
    批量生成搜索策略（从盲区队列）。
    
    Args:
        gaps: BlindSpot 对象列表
        max_per_factor: 每个因子最多生成几个策略
    
    Returns:
        SearchStrategy 列表（已去重）
    """
    strategies = []
    seen_queries = set()
    
    # 按因子分组，每个因子取前 N 个
    factor_counts: Dict[str, int] = {}
    
    for gap in gaps:
        factor = getattr(gap, 'factor', 'unknown')
        char = getattr(gap, 'char', '')
        
        if not char:
            continue
        
        if factor_counts.get(factor, 0) >= max_per_factor:
            continue
        
        # 语义因子需要两个字符
        char_b = None
        if factor == 'semantic':
            evidence = getattr(gap, 'evidence', {})
            char_b = evidence.get('char_b', '')
            if not char_b:
                continue
        
        strategy = build_search_strategy(factor, char, char_b)
        
        # 去重
        unique_queries = []
        for q in strategy.queries:
            if q not in seen_queries:
                seen_queries.add(q)
                unique_queries.append(q)
        
        if unique_queries:
            strategy.queries = unique_queries
            strategies.append(strategy)
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    return strategies


# ═══════════════════════════════════════════════════════════════════
# Wikipedia 文章标题生成
# ═══════════════════════════════════════════════════════════════════

def wikipedia_title(char: str) -> Optional[str]:
    """
    根据汉字推测可能的 Wikipedia 文章标题。
    某些单字有独立词条（如"道"、"气"），某些没有。
    
    Returns: 文章标题 或 None
    """
    # 常见有独立词条的汉字
    KNOWN_CHAR_ARTICLES = {
        '道', '德', '气', '理', '法', '天', '地', '人', '心', '性',
        '仁', '义', '礼', '智', '信', '忠', '孝', '勇', '善', '恶',
        '阴', '阳', '易', '禅', '佛', '儒', '墨', '兵', '农', '医',
        '金', '木', '水', '火', '土', '风', '雷', '云', '雨', '雪',
        '龙', '凤', '虎', '鹤', '龟', '鱼', '鸟', '马', '牛', '羊',
        '诗', '词', '曲', '赋', '书', '画', '琴', '棋', '舞', '乐',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
        '春', '夏', '秋', '冬', '日', '月', '星', '山', '河', '海',
    }
    
    if char in KNOWN_CHAR_ARTICLES:
        return char
    return None
