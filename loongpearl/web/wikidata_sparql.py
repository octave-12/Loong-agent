#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata SPARQL 客户端 — 查询中文实体结构化三元组。
========================================================

只查中文标签实体，不下载 130GB dump。
通过 SPARQL 端点在线查询，返回可直接注入概念图的三元组。

用法:
    client = WikidataClient()
    triples = client.query_concept("量子力学")
    → [("量子力学", "IS_A", "物理学分支", 0.9, "wikidata"), ...]
"""

import re
import time
import logging
from typing import List, Tuple, Dict, Optional
from urllib.parse import quote
from urllib.error import HTTPError

from SPARQLWrapper import SPARQLWrapper, JSON

log = logging.getLogger('wikidata')


class WikidataClient:
    """Wikidata SPARQL 查询客户端"""
    
    ENDPOINT = 'https://query.wikidata.org/sparql'
    TIMEOUT = 30
    MAX_RETRIES = 2
    
    # 中文标签过滤
    ZH_LABEL_FILTER = 'FILTER(LANG(?itemLabel) = "zh" || LANG(?valueLabel) = "zh")'
    
    def __init__(self, timeout: int = 30, user_agent: str = None):
        self._sparql = SPARQLWrapper(self.ENDPOINT)
        self._sparql.setTimeout(timeout)
        self._sparql.setReturnFormat(JSON)
        if user_agent:
            self._sparql.addCustomHttpHeader('User-Agent', user_agent)
        else:
            self._sparql.addCustomHttpHeader(
                'User-Agent', 'LoongAgent/2.3 (https://gitee.com/octave-12/Loong-agent)'
            )
    
    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════
    
    def query_concept(self, keyword: str, max_results: int = 20) -> List[Tuple[str, str, str, float, str]]:
        """
        查询一个中文概念的结构化关系。
        
        Returns: [(subject, relation, object, confidence, source), ...]
        """
        triples = []
        
        # 先搜索实体 ID
        entity_id = self._search_entity(keyword)
        if not entity_id:
            log.debug(f"  Wikidata: 未找到实体 '{keyword}'")
            return triples
        
        # 查询关系
        triples = self._query_relations(entity_id, keyword)
        return triples[:max_results]
    
    def query_char_pairs(self, char_a: str, char_b: str) -> List[Tuple[str, str, str, float, str]]:
        """查询两个字之间是否存在结构化关系"""
        triples = []
        
        # 查询包含这两个字的实体
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label ?itemLabel.
          FILTER(LANG(?itemLabel) = "zh")
          FILTER(CONTAINS(?itemLabel, "{char_a}") && CONTAINS(?itemLabel, "{char_b}"))
        }}
        LIMIT 10
        """
        
        try:
            results = self._execute(query)
            for binding in results.get('bindings', []):
                label = binding.get('itemLabel', {}).get('value', '')
                if label and char_a in label and char_b in label:
                    triples.append((char_a, 'RELATED', char_b, 0.6, 'wikidata_cooccur'))
                if len(triples) >= 2:
                    break
        except Exception:
            pass
        
        return triples
    
    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════
    
    def _search_entity(self, keyword: str) -> Optional[str]:
        """搜索中文实体 ID"""
        # 转义特殊字符
        safe_keyword = keyword.replace('"', '\\"')
        
        query = f"""
        SELECT ?item WHERE {{
          ?item rdfs:label ?label.
          FILTER(LANG(?label) = "zh")
          FILTER(STR(?label) = "{safe_keyword}")
        }}
        LIMIT 1
        """
        
        try:
            results = self._execute(query)
            bindings = results.get('bindings', [])
            if bindings:
                item_uri = bindings[0].get('item', {}).get('value', '')
                # 提取 Q-ID
                m = re.search(r'(Q\d+)', item_uri)
                if m:
                    return m.group(1)
        except Exception as e:
            log.debug(f"  Wikidata 搜索失败: {e}")
        
        # 回退：模糊搜索
        return self._fuzzy_search(keyword)
    
    def _fuzzy_search(self, keyword: str) -> Optional[str]:
        """模糊搜索中文实体"""
        safe_keyword = keyword.replace('"', '\\"')
        
        query = f"""
        SELECT ?item WHERE {{
          ?item rdfs:label ?label.
          FILTER(LANG(?label) = "zh")
          FILTER(CONTAINS(?label, "{safe_keyword}"))
        }}
        LIMIT 1
        """
        
        try:
            results = self._execute(query)
            bindings = results.get('bindings', [])
            if bindings:
                item_uri = bindings[0].get('item', {}).get('value', '')
                m = re.search(r'(Q\d+)', item_uri)
                if m:
                    return m.group(1)
        except Exception:
            pass
        
        return None
    
    def _query_relations(self, entity_id: str, keyword: str) -> List[Tuple[str, str, str, float, str]]:
        """查询实体的语义关系"""
        triples = []
        
        # 核心关系: P31(instance of), P279(subclass of), P361(part of),
        #          P138(named after), P527(has part), P921(main subject)
        #          P828(has cause), P1552(has quality)
        query = f"""
        SELECT ?propLabel ?valueLabel WHERE {{
          wd:{entity_id} ?prop ?value.
          ?prop wikibase:directClaim ?propClaim.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "zh". }}
          FILTER(?prop IN (
            wdt:P31, wdt:P279, wdt:P361, wdt:P138,
            wdt:P527, wdt:P921, wdt:P828, wdt:P1552
          ))
        }}
        LIMIT 30
        """
        
        try:
            results = self._execute(query)
            for binding in results.get('bindings', []):
                prop_label = binding.get('propLabel', {}).get('value', '')
                value_label = binding.get('valueLabel', {}).get('value', '')
                
                if not prop_label or not value_label:
                    continue
                
                # 映射 Wikidata 关系到 Loong-agent 关系类型
                rel = self._map_relation(prop_label)
                if rel:
                    triples.append((keyword, rel, value_label, 0.9, 'wikidata'))
        except Exception as e:
            log.debug(f"  Wikidata 关系查询失败: {e}")
        
        return triples
    
    def _map_relation(self, wikidata_prop: str) -> Optional[str]:
        """Wikidata 属性 → Loong-agent 关系类型"""
        mapping = {
            'instance of': 'IS_A',
            'subclass of': 'IS_A',
            'part of': 'PART_OF',
            'has part': 'HAS',
            'named after': 'RELATED',
            'main subject': 'RELATED',
            'has cause': 'CAUSE',
            'has quality': 'HAS',
            '属于': 'IS_A',
            '子类': 'IS_A',
            '组成部分': 'PART_OF',
            '主题': 'RELATED',
            '起因': 'CAUSE',
        }
        
        wikidata_prop_lower = wikidata_prop.lower().strip()
        for key, rel in mapping.items():
            if key in wikidata_prop_lower:
                return rel
        
        return 'RELATED'  # 兜底
    
    def _execute(self, query: str) -> Dict:
        """执行 SPARQL 查询（带重试）"""
        self._sparql.setQuery(query)
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                result = self._sparql.query().convert()
                return result.get('results', {})
            except HTTPError as e:
                if e.code == 429 and attempt < self.MAX_RETRIES:
                    wait = min(2 ** attempt, 8)
                    time.sleep(wait)
                    continue
                raise
            except Exception:
                if attempt < self.MAX_RETRIES:
                    time.sleep(1)
                    continue
                raise
        
        return {'bindings': []}
