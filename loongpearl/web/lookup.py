#!/usr/bin/env python3
"""
字符联网学习器 (web_lookup.py)
===============================
本地字典查不到的字 → 上网搜 → 提取释义/拼音 → 缓存本地

用法:
    lookup = CharWebLookup()
    info = lookup.learn("龘")
    → {'definition': '龙飞的样子', 'pinyin': 'dá', 'source': 'web'}
"""

import re, json, os, time
import requests
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE, 'data/runtime/web_char_cache.json')


class CharWebLookup:
    """汉字联网查询器 — 本地没有就上网学"""

    def __init__(self):
        self.cache = {}
        self._load_cache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        })

    def _load_cache(self):
        if os.path.exists(CACHE_PATH):
            try:
                self.cache = json.load(open(CACHE_PATH, encoding='utf-8'))
            except:
                self.cache = {}

    def _save_cache(self):
        json.dump(self.cache, open(CACHE_PATH, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    def learn(self, char: str) -> dict:
        """
        学一个字。先查本地缓存，没有再上网。
        
        Returns: {'definition': str, 'pinyin': str, 'source': 'cache'|'web'|'none'}
        """
        if char in self.cache:
            info = self.cache[char]
            info['source'] = 'cache'
            return info

        # 上网搜
        result = self._search(char)
        if result:
            self.cache[char] = result
            self._save_cache()
            result['source'] = 'web'
            return result

        # 什么都没找到
        empty = {'definition': '', 'pinyin': '', 'source': 'none'}
        self.cache[char] = empty
        self._save_cache()
        return empty

    def _search(self, char: str) -> dict:
        """多源搜索一个字的信息"""
        # 源1: 百度百科 (中文释义最全)
        result = self._baidu_baike(char)
        if result and result.get('definition'):
            return result

        # 源2: 汉典 (zdic.net)
        result = self._zdic(char)
        if result and result.get('definition'):
            return result

        # 源3: DuckDuckGo 兜底
        return self._ddg_search(char)

    def _baidu_baike(self, char: str) -> dict:
        """从百度百科提取释义"""
        try:
            url = f'https://baike.baidu.com/item/{char}'
            r = self.session.get(url, timeout=8)
            if r.status_code != 200:
                return {}

            html = r.text

            # 提取拼音 (百度百科格式: <span class="pinyin">lóng</span>
            # 或 <h2>中直接包含拼音)
            pinyin = ''
            # 尝试多种拼音格式
            for pat in [
                r'拼音[：:]\s*([a-zA-Zā-ǔ]+)',
                r'<span[^>]*class="?pinyin"?[^>]*>([a-zA-Zā-ǔ\s]+)</span>',
                r'读音[：:]\s*([a-zA-Zā-ǔ]+)',
                r'读[作音][“"]?\s*([a-zA-Zā-ǔ]{1,8})',
            ]:
                m = re.search(pat, html)
                if m:
                    pinyin = m.group(1).strip().lower()
                    break

            # 提取释义 (百度百科第一段)
            # 通常在 <meta name="description" 或 <div class="lemma-summary"
            definition = ''
            for pat in [
                r'<meta[^>]*name="description"[^>]*content="([^"]{20,200})"',
                r'<div[^>]*class="?lemma-summary"?[^>]*>(.*?)</div>',
                r'<div[^>]*class="?para"?[^>]*>(.*?)</div>',
            ]:
                m = re.search(pat, html, re.DOTALL)
                if m:
                    text = m.group(1)
                    # 清洗 HTML 标签
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\[.*?\]', '', text)  # 去掉引用标记 [1]
                    text = text.strip()
                    if len(text) >= 10:
                        definition = text[:200]
                        break

            if definition or pinyin:
                return {
                    'definition': definition,
                    'pinyin': pinyin,
                    'source_name': 'baidu_baike',
                }
        except Exception:
            pass
        return {}

    def _zdic(self, char: str) -> dict:
        """从汉典网提取"""
        try:
            url = f'https://www.zdic.net/hans/{char}'
            r = self.session.get(url, timeout=8)
            if r.status_code != 200:
                return {}

            html = r.text
            definition = ''
            pinyin = ''

            # 拼音
            m = re.search(r'拼音[：:]\s*<[^>]*>\s*([a-zA-Zā-ǔ]+)', html)
            if not m:
                m = re.search(r'<span[^>]*class="?pinyin"?[^>]*>([a-zA-Zā-ǔ]+)', html)
            if m:
                pinyin = m.group(1).strip().lower()

            # 释义
            m = re.search(
                r'<p[^>]*>[^<]*?(?:基本解释|详细解释|字义)[^<]*</p>.*?<p[^>]*>(.*?)</p>',
                html, re.DOTALL
            )
            if m:
                text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                if len(text) >= 8:
                    definition = text[:200]

            if definition or pinyin:
                return {
                    'definition': definition,
                    'pinyin': pinyin,
                    'source_name': 'zdic',
                }
        except Exception:
            pass
        return {}

    def _ddg_search(self, char: str) -> dict:
        """DuckDuckGo 兜底搜索"""
        try:
            queries = [
                f'{char} 是什么意思',
                f'{char} 汉字 释义 读音',
                f'汉字{char}的意思',
            ]

            for q in queries:
                r = self.session.get(
                    'https://lite.duckduckgo.com/lite/',
                    params={'q': q},
                    timeout=8
                )
                if r.status_code != 200:
                    continue

                # 提取结果摘要
                snippets = re.findall(
                    r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                    r.text, re.DOTALL
                )
                if not snippets:
                    continue

                # 合并摘要
                text = ' '.join(snippets)
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\s+', ' ', text).strip()

                if not text or len(text) < 10:
                    continue

                # 提取拼音
                pinyin = ''
                m = re.search(r'[（(]([a-zA-Zā-ǔ]{1,8})[）)]', text)
                if m:
                    pinyin = m.group(1).lower()
                else:
                    # 尝试匹配拼音模式
                    m = re.search(r'(?:读[音作]?|拼音)[：:]\s*([a-zA-Zā-ǔ]{1,8})', text)
                    if m:
                        pinyin = m.group(1).lower()

                # 提取前200字符作为释义
                definition = text[:200]

                return {
                    'definition': definition,
                    'pinyin': pinyin,
                    'source_name': 'ddg',
                }

        except Exception:
            pass
        return {}

    def batch_learn(self, chars: list, delay: float = 0.5) -> dict:
        """
        批量学一批字 (带延迟防止被封)
        
        Returns: {char: info_dict, ...}
        """
        results = {}
        for i, ch in enumerate(chars):
            # 只查本地没数据的字
            info = self.learn(ch)
            if info.get('source') == 'web':
                results[ch] = info
                print(f"  🌐 {ch} → {info.get('pinyin','?')}: {info.get('definition','')[:40]}")
            if delay and i < len(chars) - 1:
                time.sleep(delay)
        return results


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    lookup = CharWebLookup()

    tests = ['龘', '靐', '爨', '龙']  # 前三个是生僻字, 龙已有本地数据做对比
    for ch in tests:
        info = lookup.learn(ch)
        print(f"\n{ch}:")
        print(f"  来源: {info.get('source')} ({info.get('source_name', 'local')})")
        print(f"  拼音: {info.get('pinyin', '?')}")
        print(f"  释义: {info.get('definition', '未找到')[:120]}")
