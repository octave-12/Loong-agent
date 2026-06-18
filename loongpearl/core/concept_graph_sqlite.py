#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念图 SQLite 查询加速层。

解决 268MB JSON 全量加载 + O(N) 遍历的性能瓶颈。
SQLite 支持索引查询，将 O(N) 降为 O(log N)。

设计:
  - JSON 保留为权威数据源（save 时同步写 SQLite）
  - SQLite 作为查询加速层（启动时从 JSON 迁移，或从已有 db 加载）
  - 渐进迁移：不改 JSON 写入逻辑，只加速读路径
"""

import os
import json
import sqlite3
import logging
from typing import List, Tuple, Optional, Dict

log = logging.getLogger(__name__)


class ConceptGraphSQLite:
    """概念图 SQLite 查询加速器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── 连接管理 ──

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 表结构 ──

    def create_tables(self):
        """创建三元组表和索引"""
        c = self.conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                s TEXT NOT NULL,
                r TEXT NOT NULL,
                o TEXT NOT NULL,
                c REAL DEFAULT 1.0,
                src TEXT DEFAULT '',
                ev TEXT DEFAULT ''
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_s ON triples(s)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_sr ON triples(s, r)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_r ON triples(r)")
        c.commit()

    # ── 迁移 ──

    def migrate_from_json(self, json_path: str, batch_size: int = 50000):
        """从 JSON 概念图一次性迁移到 SQLite"""
        if not os.path.exists(json_path):
            log.warning(f"概念图 JSON 不存在: {json_path}，跳过迁移")
            return 0

        self.create_tables()

        # 检查是否已迁移（count 匹配）
        with open(json_path, 'r') as f:
            data = json.load(f)
        json_count = data.get('total_triples', len(data.get('triples', [])))

        existing = self.conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        if existing >= json_count:
            log.info(f"SQLite 已有 {existing} 条三元组 (JSON={json_count})，跳过迁移")
            return existing

        # 清空重建
        self.conn.execute("DELETE FROM triples")
        log.info(f"从 JSON 迁移 {json_count} 条三元组到 SQLite (batch={batch_size})...")

        triples = data.get('triples', [])
        total = 0
        batch = []
        for t in triples:
            batch.append((
                t.get('s', ''), t.get('r', ''), t.get('o', ''),
                t.get('c', 1.0), t.get('src', ''), t.get('ev', '')
            ))
            if len(batch) >= batch_size:
                self.conn.executemany(
                    "INSERT INTO triples(s, r, o, c, src, ev) VALUES(?,?,?,?,?,?)",
                    batch
                )
                total += len(batch)
                batch = []
                if total % 200000 == 0:
                    log.info(f"  迁移进度: {total}/{json_count}")

        if batch:
            self.conn.executemany(
                "INSERT INTO triples(s, r, o, c, src, ev) VALUES(?,?,?,?,?,?)",
                batch
            )
            total += len(batch)

        self.conn.commit()
        self.conn.execute("ANALYZE triples")  # 更新查询优化器统计
        log.info(f"✅ SQLite 迁移完成: {total} 条三元组")
        return total

    # ── 查询 ──

    def query_by_subject(self, s: str, limit: int = 100) -> List[Tuple[str, str, float, str]]:
        """O(log N): 按主语查询三元组 → [(relation, object, confidence, source)]"""
        rows = self.conn.execute(
            "SELECT r, o, c, src FROM triples WHERE s=? LIMIT ?",
            (s, limit)
        ).fetchall()
        return [(r, o, c, src) for r, o, c, src in rows]

    def query_by_subject_relation(self, s: str, r: str, limit: int = 100) -> List[Tuple[str, float, str]]:
        """O(log N): 按主语+关系查询 → [(object, confidence, source)]"""
        rows = self.conn.execute(
            "SELECT o, c, src FROM triples WHERE s=? AND r=? LIMIT ?",
            (s, r, limit)
        ).fetchall()
        return [(o, c, src) for o, c, src in rows]

    def query_poetic_next(self, char: str, min_conf: float = 0.01, limit: int = 20) -> List[Tuple[str, float]]:
        """查询某个字的 POETIC_NEXT 后续字 → [(next_char, confidence)]"""
        rows = self.conn.execute(
            "SELECT o, c FROM triples WHERE s=? AND r='POETIC_NEXT' AND c>? AND length(o)=1 ORDER BY c DESC LIMIT ?",
            (char, min_conf, limit)
        ).fetchall()
        return [(o, c) for o, c in rows]

    def count_triples(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def stats(self) -> Dict:
        c = self.conn
        total = c.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        relations = c.execute(
            "SELECT r, COUNT(*) as cnt FROM triples GROUP BY r ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        return {
            'total': total,
            'top_relations': [(r, cnt) for r, cnt in relations],
        }
