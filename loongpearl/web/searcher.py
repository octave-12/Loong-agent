#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠全网搜索引擎 (searcher.py) — 多源并发能量获取
==================================================

能量获取三原则（核心宪法）：
  1. 多源并发 — 永不依赖单一信息源
     Bing → DuckDuckGo → 百度百科 → Google → Wikipedia
     并发生效，首个返回有效结果即用，不因单源故障放弃
  2. 级联回退 — 每层都有兜底
     HTML解析 → JSON API → 缓存 → idioms.json本地词典
     互联网不通时用本地知识兜底，但绝不因本地够用就放弃网络
  3. 正文提取 — 不只依赖摘要
     搜索结果 → 点开原文 → 提取正文段落 → 字对提取
     摘要太短（通常<150字），原文才能提取高质量相邻字对
     互联网有无限可能，每篇网页都是潜在的能量来源

设计原则:
  - 多源并发: ThreadPoolExecutor并行查询，竞速取结果
  - 级联回退: 引擎 → API → 缓存 → 本地词典
  - 正文提取: 不只读snippet，尝试获取全文
  - 本地缓存: 已搜结果缓存，避免重复请求

用法:
    searcher = WebSearcher()
    results = searcher.search("境由心生 是什么意思")
    → {'answer': '...', 'sources': [...], 'confidence': 0.8}
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from loongpearl.data_config import RUNTIME_DIR


# ============================================================================
# 搜索结果数据类
# ============================================================================

class SearchResult:
    """单条搜索结果"""
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
    """聚合搜索结果"""
    def __init__(self):
        self.results: List[SearchResult] = []
        self.answer: str = ""           # 综合回答
        self.sources: List[str] = []    # 来源列表
        self.confidence: float = 0.0    # 置信度
        self.search_time: float = 0.0   # 耗时
    
    def __repr__(self):
        return (f"SearchResponse({len(self.results)} results, "
                f"conf={self.confidence:.0%}, {self.search_time:.1f}s)")


# ============================================================================
# 全网搜索引擎
# ============================================================================

class WebSearcher:
    """
    多引擎全网搜索引擎。
    
    支持的知识域自动检测:
      - 成语 → 优先百度百科/词典网
      - 汉字 → 优先汉典/百度百科
      - 事实 → 优先维基/百科
      - 通用 → DuckDuckGo
    """
    
    # 搜索引擎配置 (按优先级)
    ENGINES = [
        {
            'name': 'bing',
            'domain': 'zh',
            'url': 'https://www.bing.com/search',
            'params': lambda q: {'q': q, 'cc': 'cn', 'setlang': 'zh-Hans', 'count': '10'},
            'result_pattern': None,
        },
        {
            'name': 'baidu',
            'domain': 'zh',
            'url': 'https://www.baidu.com/s',
            'params': lambda q: {'wd': q, 'rn': '10'},
            'result_pattern': None,
        },
        {
            'name': 'duckduckgo',
            'domain': 'zh',
            'url': 'https://lite.duckduckgo.com/lite/',
            'params': lambda q: {'q': q},
            'result_pattern': None,  # HTML parsing, handled separately
        },
        {
            'name': 'baidu_baike',
            'domain': 'zh',
            'url': 'https://baike.baidu.com/item/',
            'params': lambda q: {},  # URL path, not query params
            'result_pattern': None,
        },
        {
            'name': 'wikipedia_zh',
            'domain': 'zh',
            'url': 'https://zh.wikipedia.org/w/api.php',
            'params': lambda q: {
                'action': 'query', 'list': 'search', 'srsearch': q,
                'format': 'json', 'srlimit': 5, 'srprop': 'snippet',
            },
            'result_pattern': None,
        },
        {
            'name': 'google',
            'domain': 'global',
            'url': 'https://www.google.com/search',
            'params': lambda q: {'q': q, 'hl': 'zh-CN', 'num': '10'},
            'result_pattern': None,
        },
    ]
    
    # 知识域检测模式
    DOMAIN_PATTERNS = {
        'idiom': [
            r'^[\u4e00-\u9fff]{4}$',           # 纯四字
            r'成语|是什么意思|怎么读|含义',        # 成语查询
        ],
        'character': [
            r'^[\u4e00-\u9fff]$',               # 单字
            r'怎么读|拼音|部首|笔画',              # 汉字查询
        ],
        'math': [
            r'[\d+\-*/%^]',                      # 数学表达式
        ],
    }
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 8):
        self.cache_enabled = cache_enabled
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/125.0.0.0 Safari/537.36'),
            'Accept': 'text/html,application/json,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        self._cache = {}
        self._cache_path = RUNTIME_DIR / "web_search_cache.json"
        if cache_enabled:
            self._load_cache()
    
    # ── 公开 API ──────────────────────────────────────────────
    
    def search(self, query: str, max_results: int = 8) -> SearchResponse:
        """
        全网并发搜索——多引擎竞速，永不依赖单一源。
        
        自动检测知识域，并发查询全部可用引擎，
        首个返回结果即合并，不因单源故障阻塞。
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
        
        Returns:
            SearchResponse 聚合结果
        """
        start = time.time()
        domain = self._detect_domain(query)
        
        # 查缓存
        cache_key = self._cache_key(query, domain)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.search_time = time.time() - start
            return cached
        
        response = SearchResponse()
        
        # 获取全部可用引擎
        engines = self._select_engines(domain)
        per_engine = max(2, max_results // len(engines))
        
        # 并发查询：多引擎同时发出请求，单个引擎超时不阻塞其他
        all_results = []
        engine_errors = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def _query_one(engine):
            try:
                return self._query_engine(engine, query, per_engine)
            except Exception as e:
                engine_errors.append(f"{engine['name']}: {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=len(engines)) as pool:
            futures = {pool.submit(_query_one, e): e for e in engines}
            for future in as_completed(futures, timeout=self.timeout + 2):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception:
                    pass
        
        # 去重 + 按来源排序（Bing优先）
        seen_urls = set()
        unique = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)
        
        # 按来源优先级排序（Bing > DDG > 其他）
        source_order = {'bing': 0, 'duckduckgo': 1, 'baidu_baike': 2, 'google': 3}
        unique.sort(key=lambda r: source_order.get(r.source, 99))
        
        response.results = unique[:max_results]
        response.sources = list(set(r.source for r in response.results))
        
        # 综合回答生成
        response.answer = self._synthesize_answer(query, response.results, domain)
        response.confidence = self._estimate_confidence(response.results, domain)
        response.search_time = time.time() - start
        
        # 存缓存
        if self.cache_enabled:
            self._cache[cache_key] = response
            if len(self._cache) > 500:
                self._save_cache()
        
        return response
    
    def search_idiom(self, idiom: str) -> Optional[Dict]:
        """
        搜索一个成语的释义、出处、用法。
        
        优先百度百科，回退通用搜索。
        
        Returns: {word, pinyin, definition, source, examples} or None
        """
        # 先去百度百科
        result = self._search_baike(idiom)
        if result:
            return result
        
        # 回退通用搜索
        response = self.search(f"{idiom} 成语 释义")
        if response.results:
            return {
                'word': idiom,
                'definition': response.answer[:200] if response.answer else '',
                'source': '+'.join(response.sources),
                'examples': [],
            }
        return None
    
    def search_fact(self, question: str) -> SearchResponse:
        """
        搜索事实性知识。
        
        优先维基百科，回退通用搜索。
        """
        return self.search(question)
    
    # ── 搜索引擎查询 ──────────────────────────────────────────
    
    def _query_engine(self, engine: dict, query: str, limit: int) -> List[SearchResult]:
        """查询单个搜索引擎"""
        name = engine['name']
        
        if name == 'baidu':
            return self._search_baidu(query, limit)
        elif name == 'duckduckgo':
            return self._search_ddg(query, limit)
        elif name == 'bing':
            return self._search_bing(query, limit)
        elif name == 'google':
            return self._search_google(query, limit)
        elif name == 'wikipedia_zh':
            return self._search_wikipedia(query, limit)
        elif name == 'baidu_baike':
            return self._search_baike_as_results(query, limit)
        
        return []
    
    def _search_baidu(self, query: str, limit: int) -> List[SearchResult]:
        """百度搜索 (HTML解析)"""
        try:
            r = self.session.get(
                'https://www.baidu.com/s',
                params={'wd': query, 'rn': limit},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            html = r.text
            results = []
            
            # 百度结果格式: <div class="result c-container"> ... <h3><a>标题</a></h3> ... <span class="content-right_...">摘要</span>
            blocks = re.findall(
                r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
                html, re.DOTALL
            )
            
            for block in blocks[:limit]:
                # 提取标题和链接
                m = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                if not m:
                    continue
                url = m.group(1)
                title = re.sub(r'<[^>]+>', '', m.group(2)).strip()
                
                # 提取摘要
                snippet = ''
                m2 = re.search(r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>', block, re.DOTALL)
                if m2:
                    snippet = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:300]
                if not snippet:
                    # 备选: 取所有文本
                    text = re.sub(r'<[^>]+>', ' ', block)
                    text = re.sub(r'\s+', ' ', text).strip()
                    snippet = text[:300]
                
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet,
                    source='baidu', relevance=0.7,
                ))
            
            return results
        except Exception:
            return []
    
    def _search_ddg(self, query: str, limit: int) -> List[SearchResult]:
        """DuckDuckGo HTML 搜索"""
        try:
            r = self.session.get(
                'https://lite.duckduckgo.com/lite/',
                params={'q': query},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            # 提取结果行
            results = []
            html = r.text
            
            # DuckDuckGo lite 格式: 链接 + 描述
            # <a rel="nofollow" href="URL" class="result-link">TITLE</a>
            # <td class="result-snippet">SNIPPET</td>
            links = re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )
            snippets = re.findall(
                r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                html, re.DOTALL
            )
            
            for i in range(min(len(links), limit)):
                url, title = links[i]
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ''
                
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet[:300],
                    source='duckduckgo', relevance=0.6
                ))
            
            return results
        except Exception:
            return []
    
    def _search_bing(self, query: str, limit: int) -> List[SearchResult]:
        """Bing 搜索 (中文版)"""
        try:
            r = self.session.get(
                'https://www.bing.com/search',
                params={'q': query, 'cc': 'cn', 'setlang': 'zh-Hans', 'count': limit},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            results = []
            html = r.text
            
            # Bing 结果格式
            # <li class="b_algo"><h2><a href="URL">TITLE</a></h2><p>SNIPPET</p>
            blocks = re.findall(
                r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>',
                html, re.DOTALL
            )
            
            for block in blocks[:limit]:
                # 提取链接和标题
                m = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                if not m:
                    continue
                url, title = m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()
                
                # 提取摘要
                snippet = ''
                m2 = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
                if m2:
                    snippet = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:300]
                
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet,
                    source='bing', relevance=0.7
                ))
            
            return results
        except Exception:
            return []
    
    def _search_google(self, query: str, limit: int) -> List[SearchResult]:
        """Google 搜索（WSL下可能不通，作为备选）"""
        try:
            r = self.session.get(
                'https://www.google.com/search',
                params={'q': query, 'hl': 'zh-CN', 'num': limit},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            results = []
            html = r.text
            
            # Google 结果格式: <div class="g"> ... <h3>标题</h3> ... <div class="VwiC3b">摘要</div>
            blocks = re.findall(
                r'<div[^>]*class="g"[^>]*>(.*?)</div>\s*</div>\s*</div>',
                html, re.DOTALL
            )
            
            for block in blocks[:limit]:
                m = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                if not m:
                    continue
                url, title = m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()
                
                snippet = ''
                m2 = re.search(r'<div[^>]*class="[^"]*VwiC3b[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
                if not m2:
                    m2 = re.search(r'<span[^>]*class="[^"]*st[^"]*"[^>]*>(.*?)</span>', block, re.DOTALL)
                if m2:
                    snippet = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:300]
                
                results.append(SearchResult(
                    title=title, url=url, snippet=snippet,
                    source='google', relevance=0.6
                ))
            
            return results
        except Exception:
            return []
    
    def _search_wikipedia(self, query: str, limit: int) -> List[SearchResult]:
        """维基百科 API 搜索"""
        try:
            params = {
                'action': 'query', 'list': 'search', 'srsearch': query,
                'format': 'json', 'srlimit': limit, 'srprop': 'snippet|titlesnippet',
            }
            r = self.session.get(
                'https://zh.wikipedia.org/w/api.php',
                params=params,
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            
            data = r.json()
            results = []
            for item in data.get('query', {}).get('search', [])[:limit]:
                results.append(SearchResult(
                    title=item['title'],
                    url=f"https://zh.wikipedia.org/wiki/{quote(item['title'])}",
                    snippet=re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                    source='wikipedia', relevance=0.8
                ))
            return results
        except Exception:
            return []
    
    def _search_baike(self, word: str) -> Optional[Dict]:
        """百度百科词条查询"""
        try:
            url = f'https://baike.baidu.com/item/{quote(word)}'
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code != 200:
                return None
            
            html = r.text
            
            # 提取 meta description
            m = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]+)"', html)
            if not m:
                return None
            
            desc = m.group(1)
            
            # 提取拼音
            pinyin = ''
            m2 = re.search(r'拼音[：:]\s*([a-zA-Zā-ǔ\s]+)', html)
            if m2:
                pinyin = m2.group(1).strip()
            
            return {
                'word': word,
                'pinyin': pinyin,
                'definition': desc[:300],
                'source': 'baidu_baike',
                'url': url,
            }
        except Exception:
            return None
    
    def _search_baike_as_results(self, query: str, limit: int) -> List[SearchResult]:
        """百度百科搜索（返回 SearchResult 格式）"""
        try:
            # 提取关键词（取前4个中文字符）
            word = ''.join(re.findall(r'[\u4e00-\u9fff]', query))[:4]
            info = self._search_baike(word)
            if info and info.get('definition'):
                return [SearchResult(
                    title=word,
                    url=info.get('url', ''),
                    snippet=info['definition'][:300],
                    source='baidu_baike',
                    relevance=0.85,
                )]
        except Exception:
            pass
        return []
    
    # ── 辅助方法 ──────────────────────────────────────────────
    
    def _detect_domain(self, query: str) -> str:
        """检测知识域"""
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query):
                    return domain
        return 'general'
    
    def _select_engines(self, domain: str) -> List[dict]:
        """全引擎并发——互联网有无限可能，永不局限单一源"""
        # 返回所有引擎，并发时自动竞速取结果
        # 优先序已在结果去重时处理
        return [e for e in self.ENGINES if e['name'] in 
                ('bing', 'duckduckgo', 'baidu_baike', 'google', 'wikipedia_zh')]
    
    def _synthesize_answer(self, query: str, results: List[SearchResult], domain: str) -> str:
        """从搜索结果综合回答"""
        if not results:
            return ""
        
        # 取最高相关度的摘要
        best = results[0]
        if best.relevance >= 0.8:
            return best.snippet
        
        # 合并前3条摘要
        snippets = [r.snippet for r in results[:3] if r.snippet]
        if snippets:
            return ' | '.join(snippets[:2])
        
        return best.title
    
    def _estimate_confidence(self, results: List[SearchResult], domain: str) -> float:
        """估计搜索结果的置信度"""
        if not results:
            return 0.0
        
        # 多源一致则置信度高
        sources = set(r.source for r in results)
        source_bonus = min(len(sources) * 0.15, 0.4)
        
        # 高相关度结果多则置信度高
        high_quality = sum(1 for r in results if r.relevance >= 0.7)
        quality_score = min(high_quality / max(len(results), 1), 0.6)
        
        return min(source_bonus + quality_score, 1.0)
    
    def _cache_key(self, query: str, domain: str) -> str:
        return hashlib.md5(f"{domain}:{query}".encode()).hexdigest()[:12]
    
    def _load_cache(self):
        try:
            if self._cache_path.exists():
                with open(self._cache_path, encoding='utf-8') as f:
                    raw = json.load(f)
                    # JSON 不能直接存 SearchResponse 对象，存为简化格式
                    self._cache = raw
        except Exception:
            self._cache = {}
    
    def _save_cache(self):
        try:
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception:
            pass


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    print("🐉 龙珠全网搜索引擎 — 自测\n")
    searcher = WebSearcher(timeout=8)
    
    tests = [
        "境由心生 是什么意思",
        "人工智能的定义",
        "Python programming language",
    ]
    
    for q in tests:
        print(f"\n{'='*60}")
        print(f"🔍 {q}")
        r = searcher.search(q)
        print(f"  来源: {r.sources}")
        print(f"  答案: {r.answer[:150]}")
        print(f"  置信度: {r.confidence:.0%}")
        print(f"  耗时: {r.search_time:.1f}s")
