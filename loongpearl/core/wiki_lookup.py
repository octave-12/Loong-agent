#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia FTS5 查询封装 — 为龙珠系统提供维基百科全文检索能力。

依赖: data/wikipedia/zhwiki.db (4.0GB SQLite, 430K 文章)
  - articles(id, title, text, char_count)
  - article_fts: FTS5 虚拟表 (title, text, tokenize='unicode61')
  - char_pairs(char_a, char_b, count)

用法:
    from loongpearl.core.wiki_lookup import WikipediaLookup
    wiki = WikipediaLookup("data/wikipedia/zhwiki.db")
    results = wiki.search_articles("量子力学", limit=5)
    article = wiki.get_article("量子力学")
    copairs = wiki.get_co_concepts("龙")
    evidence = wiki.verify_relation("李白", "写", "静夜思")
"""

import os
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)


class WikipediaLookup:
    """Wikipedia 中文索引查询器。

    封装 FTS5 全文检索、文章获取、字共现查询、关系验证等功能。
    连接采用惰性初始化 + WAL 模式 + 64MB 缓存，适合多并发读场景。
    """

    def __init__(self, db_path: str):
        """初始化 Wikipedia 查询器。

        Args:
            db_path: SQLite 数据库文件路径（如 data/wikipedia/zhwiki.db）
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── 连接管理 ──

    @property
    def conn(self) -> sqlite3.Connection:
        """惰性获取数据库连接（WAL + 64MB cache）。"""
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Wikipedia 数据库不存在: {self.db_path}")
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 全文搜索 ──

    def search_articles(
        self, keyword: str, limit: int = 10
    ) -> List[Dict[str, object]]:
        """FTS5 全文检索文章。

        使用 article_fts 虚拟表的 MATCH 查询，按 BM25 rank 排序。

        Args:
            keyword: 搜索关键词（支持 FTS5 查询语法，多词默认 AND 匹配）
            limit: 返回结果数量上限

        Returns:
            列表，每项包含 title, snippet (前200字摘要), char_count
        """
        # 转义 FTS5 特殊字符，避免语法错误
        safe_keyword = self._escape_fts5(keyword)
        # 用双引号包裹实现短语匹配，多词时更精准
        fts_query = f'"{safe_keyword}"'

        rows = self.conn.execute(
            "SELECT a.title, a.text, a.char_count, rank "
            "FROM article_fts f "
            "JOIN articles a ON f.rowid = a.id "
            "WHERE article_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (fts_query, limit),
        ).fetchall()

        results = []
        for row in rows:
            text = row["text"] or ""
            results.append({
                "title": row["title"],
                "snippet": text[:200],
                "char_count": row["char_count"] or 0,
            })
        return results

    # ── 文章获取 ──

    def get_article(self, title: str) -> Optional[Dict[str, object]]:
        """按标题精确获取单篇文章。

        Args:
            title: 文章标题（精确匹配）

        Returns:
            {'title', 'text', 'char_count'} 或 None（未找到）
        """
        row = self.conn.execute(
            "SELECT title, text, char_count FROM articles WHERE title = ?",
            (title,),
        ).fetchone()

        if row is None:
            return None

        return {
            "title": row["title"],
            "text": row["text"] or "",
            "char_count": row["char_count"] or 0,
        }

    # ── 字共现查询 ──

    def get_co_concepts(
        self, char: str, min_count: int = 3, limit: int = 50
    ) -> List[Tuple[str, int]]:
        """查询与指定字共现的其他字（来自 char_pairs 表）。

        统计在 Wikipedia 文章中与给定字成对出现的其他汉字及其共现次数。
        char_pairs 表存储的是无序字对 (char_a, char_b)，需要双向查询。

        Args:
            char: 查询的字（单个汉字）
            min_count: 最小共现次数阈值，低于此值的对忽略
            limit: 返回结果数量上限

        Returns:
            [(共现字, 共现次数), ...] 按次数降序排列
        """
        # 双向查询：当 char 出现在 char_a 或 char_b 位置时
        rows = self.conn.execute(
            """
            SELECT other_char, total_count FROM (
                SELECT char_b AS other_char, count AS total_count
                FROM char_pairs WHERE char_a = ? AND count >= ?
                UNION ALL
                SELECT char_a AS other_char, count AS total_count
                FROM char_pairs WHERE char_b = ? AND count >= ?
            )
            ORDER BY total_count DESC
            LIMIT ?
            """,
            (char, min_count, char, min_count, limit),
        ).fetchall()

        return [(row["other_char"], row["total_count"]) for row in rows]

    # ── 关系验证 ──

    def verify_relation(
        self, subject: str, relation: str, object_: str
    ) -> Dict[str, object]:
        """验证三元组关系 (subject, relation, object) 在 Wikipedia 中的证据。

        策略:
        1. 用 FTS5 搜索同时包含 subject 和 object 的文章（AND 查询）
        2. 从匹配文章中提取包含两者的上下文片段作为证据
        3. 基于共现文章数 vs 单独出现文章数计算置信度

        Args:
            subject: 主体（如 "李白"）
            relation: 关系描述（如 "写"），当前仅用于返回结构，不参与匹配
            object_: 客体（如 "静夜思"），使用 object_ 避免与 Python 关键字冲突

        Returns:
            {
                'found': bool,           # 是否找到共现证据
                'evidence': str,         # 证据文本片段
                'confidence': float,     # [0, 1] 置信度
                'co_count': int,         # 共现文章数
                'subj_count': int,       # subject 单独出现文章数
                'obj_count': int,        # object 单独出现文章数
            }
        """
        # 转义 FTS5 特殊字符
        safe_subj = self._escape_fts5(subject)
        safe_obj = self._escape_fts5(object_)

        # FTS5 AND 查询：同时包含 subject 和 object
        fts_query = f'"{safe_subj}" AND "{safe_obj}"'

        co_rows = self.conn.execute(
            "SELECT a.title, a.text "
            "FROM article_fts f "
            "JOIN articles a ON f.rowid = a.id "
            "WHERE article_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT 5",
            (fts_query,),
        ).fetchall()

        # 收集证据片段
        evidence_parts = []
        for row in co_rows:
            text = row["text"] or ""
            snippet = self._extract_cooccurrence_snippet(
                text, subject, object_, window=80
            )
            if snippet:
                evidence_parts.append(f"[{row['title']}] {snippet}")

        evidence = "\n".join(evidence_parts) if evidence_parts else ""

        # 统计单独出现次数
        subj_count = self._count_articles(safe_subj)
        obj_count = self._count_articles(safe_obj)

        co_count = len(co_rows)

        # 置信度：基于共现比例
        # 如果 subj 和 obj 经常一起出现 → 高置信
        # max() 防止除以零
        max_alone = max(subj_count, obj_count, 1)
        confidence = min(co_count / max_alone, 1.0)

        return {
            "found": co_count > 0,
            "evidence": evidence,
            "confidence": round(confidence, 4),
            "co_count": co_count,
            "subj_count": subj_count,
            "obj_count": obj_count,
        }

    # ── 内部辅助方法 ──

    @staticmethod
    def _escape_fts5(query: str) -> str:
        """转义 FTS5 特殊字符，防止查询语法错误。

        FTS5 特殊字符: * " - ( ) 以及列名前缀如 title:
        我们将双引号替换为两个双引号（SQL FTS5 转义方式），
        并移除可能引起歧义的 * 前缀。
        """
        # 去除首尾空白
        q = query.strip()
        # 双引号转义： " → ""
        q = q.replace('"', '""')
        return q

    def _count_articles(self, keyword: str) -> int:
        """统计包含指定关键词的文章数。"""
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM article_fts "
            "WHERE article_fts MATCH ?",
            (f'"{keyword}"',),
        ).fetchone()
        return row["cnt"] if row else 0

    @staticmethod
    def _extract_cooccurrence_snippet(
        text: str, subj: str, obj: str, window: int = 80
    ) -> str:
        """从文本中提取 subject 和 object 共现的上下文片段。

        Args:
            text: 文章全文
            subj: 主体关键词
            obj: 客体关键词
            window: 上下文窗口大小（字符数）

        Returns:
            包含 subj 和 obj 的上下文片段，或空字符串
        """
        # 找到 subj 和 obj 在文本中的位置
        pos_subj = text.find(subj)
        pos_obj = text.find(obj)

        if pos_subj == -1 or pos_obj == -1:
            return ""

        # 取两者所在的合并窗口
        start = min(pos_subj, pos_obj)
        end = max(pos_subj + len(subj), pos_obj + len(obj))

        # 扩展窗口
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)

        snippet = text[ctx_start:ctx_end]

        # 添加省略号标记
        prefix = "…" if ctx_start > 0 else ""
        suffix = "…" if ctx_end < len(text) else ""

        return f"{prefix}{snippet}{suffix}"


# ── 模块级便捷函数 ──


def quick_search(
    keyword: str,
    db_path: str = None,
    limit: int = 10,
) -> List[Dict[str, object]]:
    """快速搜索 Wikipedia（使用默认数据库路径）。

    Args:
        keyword: 搜索关键词
        db_path: 数据库路径，默认 data/wikipedia/zhwiki.db
        limit: 结果数上限

    Returns:
        搜索结果列表
    """
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "wikipedia", "zhwiki.db",
        )
    wiki = WikipediaLookup(db_path)
    try:
        return wiki.search_articles(keyword, limit=limit)
    finally:
        wiki.close()


# ── 自检 ──

if __name__ == "__main__":
    import sys

    db = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "wikipedia", "zhwiki.db",
    )

    wiki = WikipediaLookup(db)

    print("=" * 60)
    print("WikipediaLookup 自检")
    print("=" * 60)

    # 1. 搜索
    print("\n🔍 search_articles('龙', limit=3):")
    for r in wiki.search_articles("龙", limit=3):
        print(f"  📄 {r['title']} ({r['char_count']}字)")
        print(f"     {r['snippet'][:100]}...")

    # 2. 获取文章
    print("\n📖 get_article('龙'):")
    article = wiki.get_article("龙")
    if article:
        print(f"  标题: {article['title']}")
        print(f"  字数: {article['char_count']}")
        print(f"  前100字: {article['text'][:100]}...")
    else:
        print("  (未找到)")

    # 3. 字共现
    print("\n🔗 get_co_concepts('龙', min_count=5, limit=10):")
    pairs = wiki.get_co_concepts("龙", min_count=5, limit=10)
    for ch, cnt in pairs:
        print(f"  {ch}: {cnt}次共现")

    # 4. 关系验证
    print("\n✅ verify_relation('李白', '写', '静夜思'):")
    result = wiki.verify_relation("李白", "写", "静夜思")
    print(f"  found: {result['found']}")
    print(f"  confidence: {result['confidence']}")
    print(f"  co_count/subj_count/obj_count: {result['co_count']}/{result['subj_count']}/{result['obj_count']}")
    if result['evidence']:
        print(f"  evidence: {result['evidence'][:200]}...")

    wiki.close()
    print("\n✅ 自检完成")
