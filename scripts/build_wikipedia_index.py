#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia 中文 Dump 解析器 → 本地 SQLite 索引 (纯 Python，零外部依赖)
====================================================================

输入: zhwiki-latest-pages-articles.xml.bz2 (~3.2GB 压缩, ~7GB 解压)
输出: data/wikipedia/zhwiki.db

处理策略:
  1. 流式 bz2 解压 → 逐 <page> 解析 (避免全量 DOM)
  2. 纯正则剥离 Wiki 标记 → 提取中文正文
  3. 过滤: 跳重定向/消歧义/非主命名空间/正文<200字
  4. SQLite: articles(title,text) + FTS5 + char_pairs 字对统计

用法:
  # 全量构建 (10-20分钟)
  python scripts/build_wikipedia_index.py --build
  
  # 测试 (5000篇)
  python scripts/build_wikipedia_index.py --build --max 5000
  
  # 查询
  python scripts/build_wikipedia_index.py --query "量子力学"
  python scripts/build_wikipedia_index.py --char "龙"
"""

import bz2
import re
import sys
import os
import time
import sqlite3
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

DATA_DIR = os.path.join(PROJECT, 'data', 'wikipedia')
DUMP_PATH = os.path.join(DATA_DIR, 'zhwiki-latest-pages-articles.xml.bz2')
DB_PATH = os.path.join(DATA_DIR, 'zhwiki.db')

MIN_TEXT_LENGTH = 200
MAX_ARTICLES = None
BATCH_SIZE = 5000

# 非主命名空间前缀
SKIP_NAMESPACES = {
    'Wikipedia', 'Category', 'Template', 'File', 'Media',
    'Help', 'Portal', 'Draft', 'Module', 'TimedText',
    'User', 'Talk', 'User talk', 'Wikipedia talk',
    'Template talk', 'Category talk', 'File talk',
    'MediaWiki', 'MediaWiki talk', 'Module talk',
    'Gadget', 'Gadget talk', 'Gadget definition',
    'Topic', 'MOS', 'WT',
}


def iter_pages(bz2_path: str) -> Iterator[Tuple[str, str]]:
    """
    流式解析 bz2 XML，逐 <page> 产出 (title, wikitext)。
    不做全量 DOM，逐行扫描避免内存爆炸。
    """
    with bz2.open(bz2_path, 'rt', encoding='utf-8', errors='replace') as f:
        page_text = ""
        in_page = False
        
        for line in f:
            if '<page>' in line:
                in_page = True
                page_text = line
                continue
            
            if in_page:
                page_text += line
            
            if in_page and '</page>' in line:
                in_page = False
                
                # 提取 title
                m = re.search(r'<title>(.*?)</title>', page_text)
                if not m:
                    continue
                title = m.group(1)
                
                # 提取 text
                m = re.search(r'<text[^>]*>(.*?)</text>', page_text, re.DOTALL)
                if not m:
                    continue
                text = m.group(1)
                
                yield (title, text)


def clean_wikitext(wikitext: str) -> Optional[str]:
    """
    纯正则剥离 Wiki 标记，提取中文正文。
    返回 None = 跳过（重定向/太短/无中文）。
    """
    text = wikitext
    
    # 跳過重定向
    if text.strip().startswith('#REDIRECT') or text.strip().startswith('#重定向'):
        return None
    
    # 移除模板 {{...}} (嵌套处理)
    text = _remove_templates(text)
    
    # 移除 HTML 注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 移除 Wiki 表格 {| ... |}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    
    # 移除引用 <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除 Wiki 链接 [[...]] (保留显示文本)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    
    # 移除外链 [http://... text]
    text = re.sub(r'\[https?://[^\]]*\]', '', text)
    
    # 移除 Wiki 标记
    text = re.sub(r"''+", '', text)           # 粗体/斜体
    text = re.sub(r'={2,}', '', text)         # 章节标题
    text = re.sub(r'^[*#:;]+', '', text, flags=re.MULTILINE)  # 列表
    text = re.sub(r'__\w+__', '', text)       # 行为开关
    
    # 移除文件/图片占位
    text = re.sub(r'\[\[(?:File|Image|文件|图像):[^\]]+\]\]', '', text)
    
    # 移除分类
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    
    # 移除控制字符和多余空白
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 检查中文含量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < MIN_TEXT_LENGTH:
        return None
    
    return text


def _remove_templates(text: str) -> str:
    """移除 Wiki 模板 {{...}}，处理嵌套"""
    result = []
    depth = 0
    i = 0
    
    while i < len(text):
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}' and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    
    return ''.join(result)


def extract_char_pairs(text: str, max_pairs: int = 100) -> List[Tuple[str, str]]:
    """从文本提取相邻汉字对"""
    hanzi = re.findall(r'[\u4e00-\u9fff]', text)
    pairs = []
    seen = set()
    
    for i in range(len(hanzi) - 1):
        a, b = hanzi[i], hanzi[i + 1]
        if a == b:
            continue
        pair = (a, b)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                break
    
    return pairs


def build_index(dump_path: str = DUMP_PATH, db_path: str = DB_PATH,
                max_articles: int = MAX_ARTICLES,
                batch_size: int = BATCH_SIZE):
    """构建 Wikipedia SQLite 索引"""
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-128000")
    conn.execute("PRAGMA mmap_size=268435456")
    
    conn.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL,
            char_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE article_fts USING fts5(
            title, text,
            content='articles', content_rowid='id',
            tokenize='unicode61 remove_diacritics 1'
        )
    """)
    conn.execute("""
        CREATE TABLE char_pairs (
            char_a TEXT NOT NULL, char_b TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (char_a, char_b)
        )
    """)
    conn.execute("CREATE INDEX idx_articles_title ON articles(title)")
    conn.execute("CREATE INDEX idx_char_pairs_a ON char_pairs(char_a)")
    conn.execute("CREATE INDEX idx_char_pairs_b ON char_pairs(char_b)")
    
    t0 = time.time()
    total = 0
    kept = 0
    skipped_redirect = 0
    skipped_short = 0
    skipped_ns = 0
    
    articles_batch = []
    pairs_counter = defaultdict(int)
    
    print(f"📖 解析 Wikipedia Dump: {dump_path}")
    print(f"   过滤: 正文≥{MIN_TEXT_LENGTH}字, 跳重定向/非主空间")
    print()
    
    for title, wikitext in iter_pages(dump_path):
        total += 1
        
        if max_articles and total >= max_articles:
            break
        
        if total % 100000 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1)
            print(f"  ... {total:>8d} 篇 ({rate:.0f}篇/s) | "
                  f"保留 {kept} | 跳NS {skipped_ns} | "
                  f"跳定向 {skipped_redirect} | 跳过短 {skipped_short}")
        
        # 过滤非主命名空间
        if ':' in title:
            ns = title.split(':')[0]
            if ns in SKIP_NAMESPACES:
                skipped_ns += 1
                continue
        
        clean_text = clean_wikitext(wikitext)
        
        if clean_text is None:
            if wikitext.strip().startswith(('#REDIRECT', '#重定向')):
                skipped_redirect += 1
            else:
                skipped_short += 1
            continue
        
        kept += 1
        pairs = extract_char_pairs(clean_text)
        
        articles_batch.append((title, clean_text, len(clean_text)))
        for a, b in pairs:
            pairs_counter[(a, b)] += 1
        
        if len(articles_batch) >= batch_size:
            _flush_batch(conn, articles_batch, pairs_counter)
            articles_batch.clear()
            pairs_counter.clear()
    
    if articles_batch:
        _flush_batch(conn, articles_batch, pairs_counter)
    
    elapsed = time.time() - t0
    total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    total_pairs = conn.execute("SELECT COUNT(*) FROM char_pairs").fetchone()[0]
    db_size = os.path.getsize(db_path) / (1024**3)
    
    print(f"\n{'='*60}")
    print(f"✅ Wikipedia 索引构建完成")
    print(f"   扫描:   {total:>8d} 篇")
    print(f"   保留:   {total_articles:>8d} 篇")
    print(f"   字对:   {total_pairs:>8d} 对")
    print(f"   数据库: {db_size:.1f} GB")
    print(f"   耗时:   {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
    print(f"   路径:   {db_path}")
    print(f"{'='*60}")
    
    conn.close()
    return {'total_articles': total_articles, 'total_pairs': total_pairs,
            'db_size_gb': db_size, 'elapsed': elapsed}


def _flush_batch(conn, articles_batch, pairs_counter):
    """批量写入"""
    conn.executemany(
        "INSERT OR IGNORE INTO articles (title, text, char_count) VALUES (?, ?, ?)",
        [(t, txt, cnt) for t, txt, cnt in articles_batch]
    )
    for (a, b), cnt in pairs_counter.items():
        conn.execute(
            "INSERT INTO char_pairs (char_a, char_b, count) VALUES (?, ?, ?) "
            "ON CONFLICT(char_a, char_b) DO UPDATE SET count = count + ?",
            (a, b, cnt, cnt)
        )
    conn.commit()


def query_index(db_path: str = DB_PATH, keyword: str = None,
                char: str = None, limit: int = 10):
    """查询索引"""
    conn = sqlite3.connect(db_path)
    
    if keyword:
        rows = conn.execute(
            "SELECT a.title, a.text, rank FROM article_fts f "
            "JOIN articles a ON f.rowid = a.id "
            "WHERE article_fts MATCH ? ORDER BY rank LIMIT ?",
            (keyword, limit)
        ).fetchall()
        for title, text, rank in rows:
            print(f"\n📄 {title} (rank={rank:.1f})")
            print(f"   {text[:200]}...")
    
    elif char:
        rows = conn.execute(
            "SELECT title, text FROM articles WHERE text LIKE ? LIMIT ?",
            (f'%{char}%', limit)
        ).fetchall()
        for title, text in rows:
            pos = text.find(char)
            start = max(0, pos - 30)
            end = min(len(text), pos + 30)
            print(f"\n📄 {title}")
            print(f"   ...{text[start:end]}...")
    
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Wikipedia 中文索引构建')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--query', type=str)
    parser.add_argument('--char', type=str)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    
    if args.build:
        build_index(max_articles=args.max)
    elif args.query:
        query_index(keyword=args.query, limit=args.limit)
    elif args.char:
        query_index(char=args.char, limit=args.limit)
    else:
        parser.print_help()
