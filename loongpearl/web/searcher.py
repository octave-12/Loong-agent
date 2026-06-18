#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠知识搜索引擎 v3 — 适配国内网络环境
==========================================

网络现状（WSL 环境实测）:
  ✅ Bing CN (cn.bing.com)      — 国内可访问，主力
  ✅ Baidu (www.baidu.com)      — 可用，需反爬处理
  ❌ Wikipedia API              — 阻断，转离线 Dump
  ❌ DuckDuckGo / Google        — 阻断
  ❌ Wikidata SPARQL            — 阻断

四层漏斗（修正后）:
  L1 本地: SQLite 1.93M + 概念图邻接索引 (零网络)
  L2 本地: Wikipedia 中文 Dump → wikitextprocessor → SQLite 索引
  L3 在线: Bing CN + Baidu 双引擎并发 (国内仅有的可用源)
  L4 本地: idioms.json + 本地词典

并发策略:
  - Bing + Baidu 双引擎同时发出
  - 熔断: 单引擎连续3次失败 → 冷却5分钟
  - 抖动: random.uniform(0.3, 1.2) 延迟
  - UA 轮转: 4 个 User-Agent 随机选取
  - trafilatura 正文提取: 对非搜索引擎的 URL 提取全文
"""

import re
import json
import time
import hashlib
import random
import threading
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from loongpearl.data_config import RUNTIME_DIR
from loongpearl.web.rate_limiter import (
    random_ua, random_delay, EngineRequestManager,
)


# ═══════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════

class SearchResult:
    def __init__(self, title: str = "", url: str = "", snippet: str = "",
                 source: str = "", relevance: float = 0.0):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.relevance = relevance
    
    def __repr__(self):
        return f"SearchResult({self.source}: {self.title[:40]})"


class SearchResponse:
    def __init__(self):
        self.results: List[SearchResult] = []
        self.answer: str = ""
        self.sources: List[str] = []
        self.confidence: float = 0.0
        self.search_time: float = 0.0
        self.fulltext_used: bool = False


# ═══════════════════════════════════════════════════════════════════
# WebSearcher v3
# ═══════════════════════════════════════════════════════════════════

class WebSearcher:
    """
    四层漏斗知识搜索引擎（国内网络适配版）。
    
    L3 仅 Bing CN + Baidu 双引擎（WSL 网络环境下仅此二源可用）。
    等 Wikipedia Dump 本地化完成后，L2 将作为主要来源。
    """
    
    DOMAIN_PATTERNS = {
        'idiom': [r'^[\u4e00-\u9fff]{4}$', r'成语|是什么意思|怎么读|含义'],
        'character': [r'^[\u4e00-\u9fff]$', r'怎么读|拼音|部首|笔画'],
        'math': [r'[\d+\-*/%^]'],
    }
    
    # 国内可用的搜索引擎配置
    ENGINES = [
        {
            'name': 'bing_cn',
            'url': 'https://cn.bing.com/search',
            'params': lambda q: {'q': q, 'ensearch': '1', 'count': '10'},
            'source_label': 'bing',
            'priority': 1,
            'max_per_minute': 10,
            'cooldown': 120,
        },
        {
            'name': 'baidu',
            'url': 'https://www.baidu.com/s',
            'params': lambda q: {'wd': q, 'rn': '10', 'ie': 'utf-8'},
            'source_label': 'baidu',
            'priority': 2,
            'max_per_minute': 5,
            'cooldown': 300,
        },
    ]
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 10):
        self.cache_enabled = cache_enabled
        self.timeout = timeout
        
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': random_ua(),
            'Accept': 'text/html,application/xhtml+xml,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        # 请求管理器
        self._req_mgr = EngineRequestManager(global_max_per_minute=20)
        for eng in self.ENGINES:
            self._req_mgr.register_engine(
                eng['name'],
                max_failures=3,
                cooldown_seconds=eng['cooldown'],
                max_per_minute=eng['max_per_minute'],
            )
        
        # 缓存
        self._cache = {}
        self._cache_path = RUNTIME_DIR / "web_search_cache.json"
        if cache_enabled:
            self._load_cache()
        
        # 统计
        self._stats_lock = threading.Lock()
        self._stats = {'api_calls': 0, 'fulltext_fetches': 0, 'failures': 0}
    
    # ═══════════════════════════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════════════════════════
    
    def search(self, query: str, max_results: int = 8,
               prefer_fulltext: bool = True) -> SearchResponse:
        """
        全网搜索 — 国内双引擎并发。
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            prefer_fulltext: 是否尝试正文提取
        """
        start = time.time()
        domain = self._detect_domain(query)
        
        # 缓存检查
        cache_key = self._cache_key(query, domain)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.search_time = time.time() - start
            return cached
        
        response = SearchResponse()
        all_results = []
        
        # ═══ 双引擎并发 ═══
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for eng in self.ENGINES:
                name = eng['name']
                if not self._req_mgr.can_request(name):
                    continue
                futures[pool.submit(self._query_one, eng, query, max_results)] = name
            
            for future in as_completed(futures, timeout=self.timeout + 3):
                name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    self._req_mgr.record_success(name)
                except Exception:
                    self._req_mgr.record_failure(name)
                    with self._stats_lock:
                        self._stats['failures'] += 1
        
        # 延迟抖动
        time.sleep(random_delay(0.1, 0.5))
        
        # ═══ 正文提取 ═══
        if prefer_fulltext and all_results:
            fulltext_items = self._extract_fulltext_batch(all_results[:3])
            if fulltext_items:
                response.fulltext_used = True
                all_results[0].snippet += ' [全文] ' + fulltext_items[0][:200]
        
        # 去重 + 排序
        seen_urls = set()
        unique = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)
        
        source_order = {'bing': 0, 'baidu': 1}
        unique.sort(key=lambda r: source_order.get(r.source, 99))
        
        response.results = unique[:max_results]
        response.sources = list(set(r.source for r in response.results))
        response.answer = self._synthesize_answer(query, response.results, domain)
        response.confidence = self._estimate_confidence(response.results, domain)
        response.search_time = time.time() - start
        
        if self.cache_enabled:
            self._cache[cache_key] = response
            if len(self._cache) > 500:
                self._save_cache()
        
        return response
    
    def search_idiom(self, idiom: str) -> Optional[Dict]:
        response = self.search(f"{idiom} 成语 释义")
        if response.results:
            return {
                'word': idiom,
                'definition': response.answer[:300] if response.answer else '',
                'source': '+'.join(response.sources),
                'examples': [],
            }
        return None
    
    def search_fact(self, question: str) -> SearchResponse:
        return self.search(question)
    
    @property
    def stats(self) -> Dict:
        with self._stats_lock:
            s = dict(self._stats)
        s['engine_status'] = self._req_mgr.status_summary()
        return s
    
    # ═══════════════════════════════════════════════════════════════
    # 搜索引擎查询
    # ═══════════════════════════════════════════════════════════════
    
    def _query_one(self, engine: Dict, query: str,
                   max_results: int) -> List[SearchResult]:
        """查询单个搜索引擎"""
        name = engine['name']
        
        if name == 'bing_cn':
            return self._search_bing_cn(query, max_results)
        elif name == 'baidu':
            return self._search_baidu(query, max_results)
        
        return []
    
    def _search_bing_cn(self, query: str, max_results: int) -> List[SearchResult]:
        """Bing CN HTML 搜索"""
        try:
            r = self._session.get(
                'https://cn.bing.com/search',
                params={'q': query, 'ensearch': '1', 'count': max_results},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            html = r.text
            results = []
            
            # Bing 结果格式: <li class="b_algo"><h2><a href="URL">TITLE</a></h2><p>SNIPPET</p>
            blocks = re.findall(
                r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>',
                html, re.DOTALL
            )
            
            for block in blocks[:max_results]:
                m = re.search(
                    r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                    block, re.DOTALL
                )
                if not m:
                    continue
                url = m.group(1)
                title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                
                snippet = ''
                m2 = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
                if m2:
                    snippet = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:300]
                
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet,
                    source='bing', relevance=0.7,
                ))
            
            with self._stats_lock:
                self._stats['api_calls'] += 1
            return results
        except Exception:
            return []
    
    def _search_baidu(self, query: str, max_results: int) -> List[SearchResult]:
        """
        百度搜索 — 带反爬处理。
        百度对非浏览器请求敏感，返回短页面时需要重试。
        """
        for attempt in range(2):
            try:
                headers = {
                    'User-Agent': random_ua(),
                    'Accept': 'text/html,application/xhtml+xml,*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9',
                    'Referer': 'https://www.baidu.com/',
                }
                r = self._session.get(
                    'https://www.baidu.com/s',
                    params={'wd': query, 'rn': max_results},
                    headers=headers,
                    timeout=self.timeout,
                )
                
                if r.status_code != 200:
                    continue
                
                html = r.text
                
                # 检测反爬页面（百度会返回极短 HTML）
                if len(html) < 3000:
                    if attempt == 0:
                        time.sleep(random_delay(1.0, 2.0))
                        continue
                    return []
                
                results = []
                
                # 百度结果格式
                blocks = re.findall(
                    r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
                    html, re.DOTALL
                )
                
                # 备选格式
                if not blocks:
                    blocks = re.findall(
                        r'<div[^>]*class="[^"]*c-container[^"]*"[^>]*>(.*?)</div>',
                        html, re.DOTALL
                    )
                
                for block in blocks[:max_results]:
                    m = re.search(
                        r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                        block, re.DOTALL
                    )
                    if not m:
                        continue
                    url = m.group(1)
                    title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                    
                    snippet = ''
                    m2 = re.search(
                        r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>',
                        block, re.DOTALL
                    )
                    if m2:
                        snippet = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:300]
                    
                    results.append(SearchResult(
                        title=title, url=url, snippet=snippet,
                        source='baidu', relevance=0.6,
                    ))
                
                with self._stats_lock:
                    self._stats['api_calls'] += 1
                return results
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
                continue
        
        return []
    
    # ═══════════════════════════════════════════════════════════════
    # 正文提取
    # ═══════════════════════════════════════════════════════════════
    
    def _extract_fulltext_batch(self, results: List[SearchResult]) -> List[str]:
        """对搜索结果 URL 批量提取正文（trafilatura）"""
        texts = []
        for result in results[:2]:
            try:
                url = result.url
                if not url or any(s in url for s in (
                    'baidu.com', 'bing.com', 'google.com',
                )):
                    continue
                
                r = self._session.get(
                    url,
                    headers={'User-Agent': random_ua()},
                    timeout=self.timeout,
                )
                if r.status_code != 200:
                    continue
                
                # 延迟导入 trafilatura（可选依赖）
                try:
                    import trafilatura
                    text = trafilatura.extract(
                        r.text,
                        include_comments=False,
                        include_tables=False,
                        no_fallback=False,
                        favor_precision=True,
                    )
                except ImportError:
                    # trafilatura 未安装 → 用简单正则提取
                    text = self._simple_extract_text(r.text)
                
                if text and len(text) > 50:
                    texts.append(text[:2000])
                    with self._stats_lock:
                        self._stats['fulltext_fetches'] += 1
            except Exception:
                continue
            
            time.sleep(random_delay(0.2, 0.6))
        
        return texts
    
    @staticmethod
    def _simple_extract_text(html: str) -> str:
        """简单正则正文提取（trafilatura 的回退方案）"""
        # 移除 script/style
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
        # 提取 <p> 标签内容
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
        text = ' '.join(re.sub(r'<[^>]+>', ' ', p) for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) > 50 else ''
    
    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════
    
    def _detect_domain(self, query: str) -> str:
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query):
                    return domain
        return 'general'
    
    def _synthesize_answer(self, query: str, results: List[SearchResult],
                            domain: str) -> str:
        if not results:
            return ""
        best = results[0]
        if best.relevance >= 0.8 and best.snippet:
            return best.snippet
        snippets = [r.snippet for r in results[:3] if r.snippet]
        return ' | '.join(snippets[:2]) if snippets else best.title or ""
    
    def _estimate_confidence(self, results: List[SearchResult],
                              domain: str) -> float:
        if not results:
            return 0.0
        sources = set(r.source for r in results)
        source_bonus = min(len(sources) * 0.15, 0.4)
        high_quality = sum(1 for r in results if r.relevance >= 0.7)
        quality_score = min(high_quality / max(len(results), 1), 0.6)
        return min(source_bonus + quality_score, 1.0)
    
    def _cache_key(self, query: str, domain: str) -> str:
        return hashlib.md5(f"{domain}:{query}".encode()).hexdigest()[:12]
    
    def _load_cache(self):
        try:
            if self._cache_path.exists():
                with open(self._cache_path, encoding='utf-8') as f:
                    self._cache = json.load(f)
        except Exception:
            self._cache = {}
    
    def _save_cache(self):
        try:
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("🐉 龙珠知识搜索引擎 v3 (国内版) — 自测\n")
    searcher = WebSearcher(timeout=10)
    
    print(f"引擎状态: {searcher._req_mgr.status_summary()}\n")
    
    tests = ["量子力学", "人工智能 定义", "境由心生 是什么意思"]
    
    for q in tests:
        print(f"{'='*60}")
        print(f"🔍 {q}")
        r = searcher.search(q)
        print(f"  来源: {r.sources}")
        print(f"  结果数: {len(r.results)}")
        for res in r.results[:3]:
            print(f"    [{res.source}] {res.title[:60]}")
            print(f"      {res.snippet[:100]}")
        print(f"  正文提取: {'✅' if r.fulltext_used else '❌'}")
        print(f"  置信度: {r.confidence:.0%}")
        print(f"  耗时: {r.search_time:.1f}s")
    
    print(f"\n统计: {searcher.stats}")
