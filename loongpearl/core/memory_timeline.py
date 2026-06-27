#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1: 记忆时序层 (Memory Timeline)

给概念图每条三元组加上时间维度:
  - learned_at:       首次学习时间
  - last_verified_at: 最后验证时间
  - verify_count:     验证次数

回答: "这个概念我什么时候学的?" "上次验证是什么时候?"
驱动: 时间衰减 D-S (旧证据权重降低)

Schema 已通过 ALTER TABLE 添加, 无需额外迁移。

使用:
  mt = MemoryTimeline("data/models/concept_graph.db")
  mt.mark_learned("龙", "DEFINED_AS", "dragon")  # 标记学习
  info = mt.when_did_i_learn("龙")               # 查询学习历史
  old = mt.get_stale_knowledge(days=30)           # 找30天未验证的知识
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

log = logging.getLogger(__name__)

# 迁移前的默认时间 (系统首次运行的日期)
_DEFAULT_LEARNED = "2026-06-01T00:00:00"


class MemoryTimeline:
    """概念图三元组的时间记忆"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "data", "models", "concept_graph.db"
            )
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # ── 写入 ──

    def mark_learned(self, subject: str, relation: str, obj: str,
                     source: str = "", confidence: float = 0.5):
        """
        标记一条知识为"已学习"。
        如果三元组已存在，更新验证时间；否则插入。

        Args:
            subject, relation, obj: 三元组
            source: 知识来源
            confidence: 初始置信度
        """
        now = datetime.now().isoformat()

        # 检查是否存在
        existing = self.conn.execute(
            "SELECT id, learned_at, verify_count FROM triples "
            "WHERE s=? AND r=? AND o=? LIMIT 1",
            (subject, relation, obj)
        ).fetchone()

        if existing:
            # 更新验证
            new_count = (existing[2] or 0) + 1
            learned = existing[1] if existing[1] else now
            self.conn.execute(
                "UPDATE triples SET last_verified_at=?, verify_count=?, "
                "learned_at=CASE WHEN learned_at='' THEN ? ELSE learned_at END "
                "WHERE id=?",
                (now, new_count, now, existing[0])
            )
        else:
            # 插入新三元组
            self.conn.execute(
                "INSERT INTO triples(s, r, o, c, src, learned_at, verify_count) "
                "VALUES(?,?,?,?,?,?,1)",
                (subject, relation, obj, confidence, source, now)
            )

        self.conn.commit()

    def mark_verified(self, subject: str, relation: str, obj: str):
        """标记一条知识为"已验证"（更新验证时间）"""
        now = datetime.now().isoformat()
        self.conn.execute(
            "UPDATE triples SET last_verified_at=?, "
            "verify_count=verify_count+1, "
            "learned_at=CASE WHEN learned_at='' THEN ? ELSE learned_at END "
            "WHERE s=? AND r=? AND o=?",
            (now, now, subject, relation, obj)
        )
        self.conn.commit()

    # ── 查询 ──

    def when_did_i_learn(self, concept: str) -> Dict:
        """
        查询关于某个概念的学习历史。

        Returns:
            {
                "concept": str,
                "total_triples": int,
                "earliest_learned": str,
                "latest_verified": str,
                "avg_verify_count": float,
                "triples_by_source": {source: count},
                "timeline": [(learned_at, relation, object), ...],
            }
        """
        rows = self.conn.execute(
            "SELECT learned_at, last_verified_at, verify_count, src, r, o "
            "FROM triples WHERE s=? "
            "ORDER BY learned_at DESC LIMIT 100",
            (concept,)
        ).fetchall()

        if not rows:
            return {"concept": concept, "total_triples": 0}

        learned_times = []
        verified_times = []
        verify_counts = []
        sources = {}
        timeline = []

        for learned, verified, vc, src, r, o in rows:
            lt = learned if learned else _DEFAULT_LEARNED
            learned_times.append(lt)
            if verified:
                verified_times.append(verified)
            verify_counts.append(vc or 0)
            sources[src] = sources.get(src, 0) + 1
            timeline.append((lt, r, o))

        return {
            "concept": concept,
            "total_triples": len(rows),
            "earliest_learned": min(learned_times) if learned_times else None,
            "latest_verified": max(verified_times) if verified_times else None,
            "avg_verify_count": sum(verify_counts) / len(verify_counts) if verify_counts else 0,
            "triples_by_source": sources,
            "timeline": sorted(timeline, key=lambda x: x[0])[:20],
        }

    def get_stale_knowledge(self, days: int = 30, limit: int = 100) -> List[Dict]:
        """
        找出超过 N 天未验证的知识（需要复查）。

        时间衰减的基础：旧知识权重降低。
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        rows = self.conn.execute(
            "SELECT s, r, o, c, src, learned_at, last_verified_at, verify_count "
            "FROM triples "
            "WHERE (last_verified_at < ? OR (last_verified_at='' AND learned_at < ?)) "
            "AND verify_count > 0 "
            "ORDER BY last_verified_at ASC LIMIT ?",
            (cutoff, cutoff, limit)
        ).fetchall()

        return [
            {
                "subject": s, "relation": r, "object": o,
                "confidence": c, "source": src,
                "learned_at": learned,
                "last_verified": verified,
                "verify_count": vc,
                "stale_days": (
                    datetime.now() - datetime.fromisoformat(
                        verified if verified else learned if learned else _DEFAULT_LEARNED
                    )
                ).days,
            }
            for s, r, o, c, src, learned, verified, vc in rows
        ]

    def time_decay_mass(self, concept: str, half_life_days: float = 90.0) -> float:
        """
        计算概念知识的时间衰减因子。

        公式: mass(t) = exp(-ln(2) * days_since_verified / half_life)

        Returns:
            衰减因子 [0, 1] — 1 = 刚刚验证, 0 = 完全过期
        """
        row = self.conn.execute(
            "SELECT MAX(last_verified_at), MAX(learned_at) FROM triples WHERE s=?",
            (concept,)
        ).fetchone()

        if not row:
            return 0.0

        last_time = row[0] if row[0] else row[1] if row[1] else _DEFAULT_LEARNED
        try:
            last_dt = datetime.fromisoformat(last_time)
        except ValueError:
            last_dt = datetime.fromisoformat(_DEFAULT_LEARNED)

        days = (datetime.now() - last_dt).days
        if days <= 0:
            return 1.0

        import math
        return math.exp(-math.log(2) * days / half_life_days)

    def stats(self) -> Dict:
        """时间线统计"""
        total = self.conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        with_time = self.conn.execute(
            "SELECT COUNT(*) FROM triples WHERE learned_at != ''"
        ).fetchone()[0]
        verified = self.conn.execute(
            "SELECT COUNT(*) FROM triples WHERE verify_count > 0"
        ).fetchone()[0]
        recent = self.conn.execute(
            "SELECT COUNT(*) FROM triples WHERE last_verified_at > datetime('now', '-7 days')"
        ).fetchone()[0]

        return {
            "total_triples": total,
            "with_timestamps": with_time,
            "verified_ever": verified,
            "verified_recent_7d": recent,
            "stale_30d": len(self.get_stale_knowledge(days=30, limit=999999)),
        }


# ── CLI ──
if __name__ == "__main__":
    import sys

    mt = MemoryTimeline()

    print("=== 记忆时序统计 ===")
    for k, v in mt.stats().items():
        print(f"  {k}: {v}")

    concept = sys.argv[1] if len(sys.argv) > 1 else "龙"
    print(f"\n=== [{concept}] 学习历史 ===")
    info = mt.when_did_i_learn(concept)
    for k, v in info.items():
        if k != "timeline":
            print(f"  {k}: {v}")

    if info["timeline"]:
        print(f"  最近5条:")
        for t, r, o in info["timeline"][-5:]:
            print(f"    {t[:19]}  {r:15s} → {o[:30]}")

    decay = mt.time_decay_mass(concept)
    print(f"\n  时间衰减因子: {decay:.3f} (半衰期=90天)")
