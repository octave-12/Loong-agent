#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia 中文 Dump 解析器 → 本地 SQLite 索引
==============================================

输入: zhwiki-latest-pages-articles.xml.bz2 (~2.1GB)
输出: data/wikipedia/zhwiki.db (SQLite 索引)

处理策略:
  1. 流式解压 bz2 → 按 <page> 分块解析
  2. wikitextprocessor 提取纯文本 (去 Wiki 标记)
  3. 过滤: 只保留正文≥200字的文章
  4. 建立 SQLite: (title, text, categories, char_pairs)
  5. 建立 FTS5 全文索引

用法:
  python scripts/build_wikipedia_index.py
  
预计耗时: ~10-15分钟 (CPU密集型, 解析140万篇)
"""

import bz2
import re
import sys
import os
import time
import sqlite3
import xml.etree.ElementTree as ET
from typing import Iterator, Dict, List, Optional, Tuple
from collections import defaultdict

# 加项目根路径
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from wikitextprocessor import Wtp

DATA_DIR = os.path.join(PROJECT, 'data', 'wikipedia')
DUMP_PATH = os.path.join(DATA_DIR, 'zhwiki-latest-pages-articles.xml.bz2')
DB_PATH = os.path.join(DATA_DIR, 'zhwiki.db')

# 过滤条件
MIN_TEXT_LENGTH = 200          # 文章正文最少字数
MAX_ARTICLES = None             # None=全部, 设置数字限制用于测试
BATCH_SIZE = 5000               # 每批提交数据库


def iter_pages(bz2_path: str) -> Iterator[Tuple[str, str]]:
    """
    流式解析 bz2 XML，逐 <page> 产出 (title, wikitext)。
    不做全量 DOM 解析，避免内存爆炸。
    """
    with bz2.open(bz2_path, 'rt', encoding='utf-8', errors='replace') as f:
        buffer = []
        in_page = False
        in_title = False
        in_text = False
        title = ""
        text = ""
        
        for line in f:
            if '<page>' in line:
                in_page = True
                buffer = [line]
                title = ""
                text = ""
                continue
            
            if in_page:
                buffer.append(line)
            
            if in_page and '<title>' in line and '</title>' in line:
                m = re.search(r'<title>(.*?)</title>', line)
                if m:
                    title = m.group(1)
                continue
            
            if in_page and '<text' in line:
                in_text = True
                # 提取 <text ...> 到 </text>
                text_start = buffer.index(line) if line in buffer else len(buffer) - 1
                continue
            
            if in_text and '</text>' in line:
                in_text = False
                # 合并 text 内容
                text_lines = buffer[text_start:]
                full_text = ''.join(text_lines)
                m = re.search(r'<text[^>]*>(.*?)</text>', full_text, re.DOTALL)
                if m:
                    text = m.group(1)
            
            if in_page and '</page>' in line:
                in_page = False
                if title and text:
                    yield (title, text)


def clean_wikitext(wtp: Wtp, title: str, wikitext: str) -> Optional[str]:
    """
    用 wikitextprocessor 提取纯文本。
    返回 None 表示跳过（重定向/消歧义/太短）。
    """
    # 跳過重定向
    if wikitext.strip().startswith('#REDIRECT') or wikitext.strip().startswith('#重定向'):
        return None
    
    try:
        # wikitextprocessor 解析
        wtp.start_page(title)
        tree = wtp.parse(wikitext)
        
        # 提取纯文本节点
        text_parts = []
        _extract_text(tree, text_parts)
        text = ' '.join(text_parts)
        
        # 清洗
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < MIN_TEXT_LENGTH:
            return None
        
        return text
    except Exception:
        return None


def _extract_text(node, parts: List[str]):
    """递归提取 wikitextprocessor AST 的纯文本节点"""
    if node is None:
        return
    
    if isinstance(node, str):
        parts.append(node)
        return
    
    if hasattr(node, 'children'):
        for child in node.children:
            _extract_text(child, parts)
    elif hasattr(node, 'args') and hasattr(node.args, '__iter__'):
        for child in node.args:
            _extract_text(child, parts)


def extract_char_pairs(text: str, max_pairs: int = 100) -> List[Tuple[str, str]]:
    """从文本中提取相邻汉字对 (字对)"""
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
    """
    构建 Wikipedia SQLite 索引。
    
    表结构:
      articles (id, title, text, char_count)
      article_fts (title, text)        — FTS5 全文索引
      char_pairs (char_a, char_b, count)  — 字对统计
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 删除旧数据库
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    
    # 建表
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
            content='articles',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 1'
        )
    """)
    
    conn.execute("""
        CREATE TABLE char_pairs (
            char_a TEXT NOT NULL,
            char_b TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (char_a, char_b)
        )
    """)
    
    conn.execute("CREATE INDEX idx_articles_title ON articles(title)")
    conn.execute("CREATE INDEX idx_char_pairs_a ON char_pairs(char_a)")
    conn.execute("CREATE INDEX idx_char_pairs_b ON char_pairs(char_b)")
    
    # 初始化 wikitextprocessor
    wtp = Wtp()
    
    t0 = time.time()
    total = 0
    kept = 0
    skipped_redirect = 0
    skipped_short = 0
    skipped_error = 0
    
    # 批量缓冲区
    articles_batch = []
    pairs_counter = defaultdict(int)
    
    print(f"📖 解析 Wikipedia Dump: {dump_path}")
    print(f"   过滤: 正文≥{MIN_TEXT_LENGTH}字, 跳过重定向")
    print()
    
    for title, wikitext in iter_pages(dump_path):
        total += 1
        
        if max_articles and total >= max_articles:
            break
        
        # 进度
        if total % 100000 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1)
            print(f"  ... {total:>8d} 篇 ({rate:.0f}篇/s) | "
                  f"保留 {kept} | 跳重定向 {skipped_redirect} | "
                  f"跳过短 {skipped_short} | 错误 {skipped_error}")
        
        # 跳過非主命名空間 (保留 title 不含 ':' 的)
        if ':' in title:
            # 允许 "量子力学" 这样的，跳過 "Wikipedia:", "Category:", "Template:" 等
            ns = title.split(':')[0]
            if ns in ('Wikipedia', 'Category', 'Template', 'File', 'Media',
                      'Help', 'Portal', 'Draft', 'Module', 'TimedText',
                      'User', 'Talk', 'User talk', 'Wikipedia talk'):
                skipped_redirect += 1
                continue
        
        clean_text = clean_wikitext(wtp, title, wikitext)
        
        if clean_text is None:
            if wikitext.strip().startswith(('#REDIRECT', '#重定向')):
                skipped_redirect += 1
            else:
                skipped_short += 1
            continue
        
        kept += 1
        
        # 提取字对
        pairs = extract_char_pairs(clean_text)
        
        articles_batch.append((title, clean_text, len(clean_text)))
        for a, b in pairs:
            pairs_counter[(a, b)] += 1
        
        # 批量提交
        if len(articles_batch) >= batch_size:
            _flush_batch(conn, articles_batch, pairs_counter)
            articles_batch.clear()
            pairs_counter.clear()
    
    # 最终提交
    if articles_batch:
        _flush_batch(conn, articles_batch, pairs_counter)
    
    elapsed = time.time() - t0
    
    # 统计
    total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    total_pairs = conn.execute("SELECT COUNT(*) FROM char_pairs").fetchone()[0]
    db_size = os.path.getsize(db_path) / (1024**3)
    
    print(f"\n{'='*60}")
    print(f"✅ Wikipedia 索引构建完成")
    print(f"   总文章: {total:>8d}")
    print(f"   保留:   {total_articles:>8d} 篇")
    print(f"   字对:   {total_pairs:>8d} 对")
    print(f"   数据库: {db_size:.1f} GB")
    print(f"   耗时:   {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
    print(f"   路径:   {db_path}")
    print(f"{'='*60}")
    
    conn.close()
    return {
        'total_articles': total_articles,
        'total_pairs': total_pairs,
        'db_size_gb': db_size,
        'elapsed': elapsed,
    }


def _flush_batch(conn, articles_batch, pairs_counter):
    """批量写入数据库"""
    # 文章
    conn.executemany(
        "INSERT OR IGNORE INTO articles (title, text, char_count) VALUES (?, ?, ?)",
        [(t, txt, cnt) for t, txt, cnt in articles_batch]
    )
    
    # 字对 (增量更新计数)
    for (a, b), cnt in pairs_counter.items():
        conn.execute(
            "INSERT INTO char_pairs (char_a, char_b, count) VALUES (?, ?, ?) "
            "ON CONFLICT(char_a, char_b) DO UPDATE SET count = count + ?",
            (a, b, cnt, cnt)
        )
    
    conn.commit()


def query_index(db_path: str = DB_PATH, keyword: str = None,
                char: str = None, limit: int = 10):
    """查询 Wikipedia 索引"""
    conn = sqlite3.connect(db_path)
    
    if keyword:
        # FTS5 全文搜索
        rows = conn.execute(
            "SELECT a.title, a.text, rank FROM article_fts f "
            "JOIN articles a ON f.rowid = a.id "
            "WHERE article_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (keyword, limit)
        ).fetchall()
        
        for title, text, rank in rows:
            print(f"\n📄 {title} (rank={rank:.1f})")
            print(f"   {text[:200]}...")
    
    elif char:
        # 查询包含某字的所有文章
        rows = conn.execute(
            "SELECT title, text FROM articles "
            "WHERE text LIKE ? "
            "LIMIT ?",
            (f'%{char}%', limit)
        ).fetchall()
        
        for title, text in rows:
            # 找到字的位置
            pos = text.find(char)
            start = max(0, pos - 30)
            end = min(len(text), pos + 30)
            ctx = text[start:end]
            print(f"\n📄 {title}")
            print(f"   ...{ctx}...")
    
    conn.close()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Wikipedia 中文 Dump 索引构建')
    parser.add_argument('--build', action='store_true', help='构建索引')
    parser.add_argument('--query', type=str, help='全文搜索关键词')
    parser.add_argument('--char', type=str, help='查询包含某字的文章')
    parser.add_argument('--max', type=int, default=None, help='最大文章数(测试用)')
    parser.add_argument('--limit', type=int, default=10, help='查询结果数')
    
    args = parser.parse_args()
    
    if args.build:
        build_index(max_articles=args.max)
    elif args.query:
        query_index(keyword=args.query, limit=args.limit)
    elif args.char:
        query_index(char=args.char, limit=args.limit)
    else:
        parser.print_help()
