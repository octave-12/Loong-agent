#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L6: 验证驱动自主学习闭环 (Verified Learner)

整合 L3(terrain) + L4(tester) + L5(fuzzy D-S) + wiki_lookup，
形成完整的"学→验→融→记"闭环，防幻觉机制贯穿全流程。

核心原则:
  - 所有写入必须经过验证，绝不自我预测→自我写入
  - 至少2个独立源验证才写入 CG
  - wiki验证置信度<0.1 的直接丢弃
  - 冲突知识标记为争议，不写入

用法:
    from loongpearl.core.verified_learner import VerifiedLearner

    vl = VerifiedLearner(
        db_path="data/models/concept_graph.db",
        wiki_path="data/wikipedia/zhwiki.db",
    )
    result = vl.learn_concept("龙")
    tick_result = vl.autonomous_tick(max_concepts=3)
"""

import os
import sqlite3
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime

from loongpearl.core.stage_query import hard_anchor, bfs_traverse
from loongpearl.core.wiki_lookup import WikipediaLookup
from loongpearl.core.hypothesis_tester import HypothesisTester
from loongpearl.core.fuzzy_graph import FuzzyGraph
from loongpearl.core.memory_timeline import MemoryTimeline
from loongpearl.core.cognitive_terrain import CognitiveTerrain

log = logging.getLogger(__name__)

# ── 多源证据来源名称 ──
SOURCE_WIKI = "wikipedia_cross_ref"
SOURCE_CG = "concept_graph"
SOURCE_DS_FUSE = "ds_multi_source_fuse"

# ── 防幻觉阈值 ──
MIN_WIKI_CONFIDENCE = 0.1      # wiki 置信度低于此值直接丢弃
MIN_SOURCES_FOR_WRITE = 2       # 至少 N 个独立源才写入 CG
MIN_DS_BELIEF_FOR_WRITE = 0.3   # D-S 融合信念低于此值不写入


class VerifiedLearner:
    """验证驱动的自主学习闭环。

    整合 L3(认知地形) + L4(假设检验) + L5(模糊格D-S) + Wikipedia交叉验证，
    形成完整的"学→验→融→记"闭环。

    Attributes:
        db_path: concept_graph.db 路径
        wiki: WikipediaLookup 实例
        tester: HypothesisTester 实例
        fg: FuzzyGraph 实例（内存证据库）
        memory: MemoryTimeline 实例
        terrain: CognitiveTerrain 实例
        _controversies: 争议知识列表（未写入 CG 的冲突知识）
    """

    def __init__(self, db_path: str, wiki_path: str):
        """初始化验证学习器。

        Args:
            db_path: concept_graph.db 的绝对/相对路径
            wiki_path: zhwiki.db 的路径
        """
        self.db_path = os.path.abspath(db_path)
        self.wiki_path = os.path.abspath(wiki_path)

        # 初始化所有依赖
        self.wiki = WikipediaLookup(self.wiki_path)
        self.tester = HypothesisTester(db_path=self.db_path)
        self.fg = FuzzyGraph(concept_graph=None)
        self.memory = MemoryTimeline(db_path=self.db_path)
        self.terrain = CognitiveTerrain(db_path=self.db_path)

        # 尝试加载 terrain（可能因 torch 依赖失败，非致命）
        try:
            self.terrain.load()
            self._terrain_loaded = True
            log.info("CognitiveTerrain 加载成功")
        except Exception as e:
            self._terrain_loaded = False
            log.warning(f"CognitiveTerrain 加载失败 (非致命): {e}")

        # 争议知识——在内存中追踪，不写入 CG
        self._controversies: List[Dict[str, Any]] = []

        # 确保 triples 表有 status 列（用于标记 verified/controversial）
        self._ensure_schema()

    def _ensure_schema(self):
        """确保数据库表结构包含必要列。"""
        conn = sqlite3.connect(self.db_path)
        try:
            info = conn.execute("PRAGMA table_info(triples)").fetchall()
            columns = {row[1] for row in info}
            if "status" not in columns:
                conn.execute("ALTER TABLE triples ADD COLUMN status TEXT DEFAULT ''")
                conn.commit()
                log.info("已添加 triples.status 列")
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # 核心方法: learn_concept
    # ══════════════════════════════════════════════════════════════

    def learn_concept(self, concept: str, max_rounds: int = 3) -> Dict[str, Any]:
        """对单个概念执行验证驱动的学习循环。

        流程:
          Stage 1: hard_anchor — 若知识充足则直接返回
          Stage 2: bfs_traverse — 找关联概念
          Stage 3: wiki 交叉验证 — 对关联概念做 Wikipedia 证据检索
          Stage 4: hypothesis_tester — 检查新知识是否与已有知识冲突
          Stage 5: D-S multi_source_fuse — 融合多源证据
          验证通过 → mark_verified; 未通过 → 标记为争议

        Args:
            concept: 要学习的概念（如 "龙"）
            max_rounds: 最大学习轮次（每轮处理一批关联概念）

        Returns:
            {
                "concept": str,
                "learned_count": int,      # 尝试学习的三元组数
                "verified_count": int,     # 通过验证并写入 CG 的三元组数
                "conflicts": [dict, ...],  # 冲突详情
                "new_blind_spots": [str, ...],
                "controversies": [dict, ...],  # 标记为争议的知识
                "rounds": int,             # 实际执行轮数
                "anchor_sufficient": bool, # Stage 1 是否直接返回
            }
        """
        result = {
            "concept": concept,
            "learned_count": 0,
            "verified_count": 0,
            "conflicts": [],
            "new_blind_spots": [],
            "controversies": [],
            "rounds": 0,
            "anchor_sufficient": False,
        }

        # ── Stage 1: 硬锚定 ──
        anchor = hard_anchor(concept, db_path=self.db_path)
        if anchor.get("sufficient"):
            result["anchor_sufficient"] = True
            result["verified_count"] = anchor["count"]
            log.info(f"[{concept}] Stage 1 充足 (avg_conf={anchor['avg_confidence']})，跳过学习")
            return result

        # ── Stage 2: BFS 找关联概念 ──
        bfs = bfs_traverse(concept, max_hops=2, min_conf=0.3, db_path=self.db_path)

        # 收集所有关联概念
        related_concepts: Set[str] = set()
        for direction in ["taxonomy", "association"]:
            for path in bfs.get(direction, {}).get("paths", []):
                # path[0] 是起始节点，path[1:] 是后续跳
                for step in path[1:]:
                    obj = step.get("object", "")
                    if obj and obj != concept:
                        related_concepts.add(obj)

        if not related_concepts:
            log.info(f"[{concept}] BFS 未找到关联概念，无法推进")
            return result

        # 按置信度排序，取 top N 进行 wiki 验证
        # 优先选择短概念（长概念/成语很难在 wiki 中共现验证）
        related_sorted = sorted(
            related_concepts,
            key=lambda c: (
                # 负长度（短的优先） + 高置信度加权
                -len(c) * 0.01 + self._max_bfs_confidence(bfs, c)
            ),
            reverse=True,
        )

        # 分批学习（每轮处理最多 10 个关联概念）
        batch_size = min(10, len(related_sorted))
        for round_num in range(max_rounds):
            batch = related_sorted[round_num * batch_size:(round_num + 1) * batch_size]
            if not batch:
                break

            result["rounds"] = round_num + 1
            round_result = self._learn_round(concept, batch)
            result["learned_count"] += round_result["learned_count"]
            result["verified_count"] += round_result["verified_count"]
            result["conflicts"].extend(round_result["conflicts"])
            result["controversies"].extend(round_result["controversies"])

            # 如果本轮没学到任何东西，提前退出
            if round_result["learned_count"] == 0:
                break

        # 寻找新盲区
        if self._terrain_loaded:
            result["new_blind_spots"] = self._find_new_blind_spots(concept, related_concepts)

        return result

    def _gather_wiki_evidence(
        self, concept: str, related: List[str]
    ) -> List[Dict[str, Any]]:
        """Stage 3: 多策略 Wiki 交叉验证。

        策略:
          A. verify_relation: 精确匹配概念和关联概念共现（适合短词）
          B. search_articles + 文章摘要提取: 搜索概念，从摘要中找关联术语
          C. get_co_concepts (char_pairs): 单字共现查询（适合汉字级概念）

        所有策略结果汇总，按置信度排序返回。
        """
        evidence_list: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # ── 策略 A: 精确 verify_relation ──
        # 只对短于6字的关联概念做精确匹配（长概念很少直接共现）
        short_related = [r for r in related if len(r) <= 6]
        for rel in short_related[:20]:
            if (concept, rel) in seen_pairs:
                continue
            seen_pairs.add((concept, rel))
            try:
                verify = self.wiki.verify_relation(concept, "RELATED", rel)
                if verify.get("confidence", 0) >= MIN_WIKI_CONFIDENCE:
                    evidence_list.append({
                        "subject": concept,
                        "relation": "COOCCURS_WITH",
                        "object": rel,
                        "confidence": verify["confidence"],
                        "found": verify.get("found", False),
                        "evidence_text": verify.get("evidence", ""),
                        "source": SOURCE_WIKI,
                        "strategy": "verify_relation",
                    })
            except Exception as e:
                log.debug(f"Wiki verify_relation 失败 [{concept} ↔ {rel}]: {e}")

        # ── 策略 B: search_articles 提取关联概念 ──
        try:
            articles = self.wiki.search_articles(concept, limit=5)
            for article in articles:
                title = article.get("title", "")
                snippet = article.get("snippet", "")
                # 标题本身可能就是关系概念的来源
                if title and title != concept and len(title) <= 15:
                    pair = (concept, title)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        evidence_list.append({
                            "subject": concept,
                            "relation": "RELATED",
                            "object": title,
                            "confidence": 0.3,  # 搜索排名提供的弱信号
                            "found": True,
                            "evidence_text": snippet,
                            "source": SOURCE_WIKI,
                            "strategy": "article_title",
                        })
        except Exception as e:
            log.debug(f"Wiki search_articles 失败 [{concept}]: {e}")

        # ── 策略 C: get_co_concepts (单字共现) ──
        if len(concept) == 1:
            try:
                co_pairs = self.wiki.get_co_concepts(concept, min_count=5, limit=30)
                for co_char, count in co_pairs:
                    if co_char == concept:
                        continue
                    # 将字共现转为置信度（归一化）
                    conf = min(0.5, count / 1000.0)
                    pair = (concept, co_char)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        evidence_list.append({
                            "subject": concept,
                            "relation": "COOCCURS_WITH",
                            "object": co_char,
                            "confidence": conf,
                            "found": True,
                            "evidence_text": f"Wikipedia 字共现 (count={count})",
                            "source": SOURCE_WIKI,
                            "strategy": "char_pairs",
                            "co_count": count,
                        })
            except Exception as e:
                log.debug(f"Wiki get_co_concepts 失败 [{concept}]: {e}")

        # 按置信度降序排序
        evidence_list.sort(key=lambda e: e["confidence"], reverse=True)

        log.info(
            f"[{concept}] Wiki 交叉验证: 共获得 {len(evidence_list)} 条证据 "
            f"(A精确={sum(1 for e in evidence_list if e['strategy']=='verify_relation')}, "
            f"B搜索={sum(1 for e in evidence_list if e['strategy']=='article_title')}, "
            f"C共现={sum(1 for e in evidence_list if e['strategy']=='char_pairs')})"
        )

        return evidence_list

    def _max_bfs_confidence(self, bfs: Dict, concept: str) -> float:
        """获取某概念在 BFS 结果中的最高置信度。"""
        max_conf = 0.0
        for direction in ["taxonomy", "association"]:
            for path in bfs.get(direction, {}).get("paths", []):
                for step in path[1:]:
                    if step.get("object") == concept:
                        max_conf = max(max_conf, step.get("confidence", 0.0))
        return max_conf

    def _learn_round(self, concept: str, related: List[str]) -> Dict[str, Any]:
        """单轮学习：对一批关联概念执行验证→融合→写入。"""
        round_result = {
            "learned_count": 0,
            "verified_count": 0,
            "conflicts": [],
            "controversies": [],
        }

        # ── Stage 3: Wiki 交叉验证（多策略）──
        wiki_evidence = self._gather_wiki_evidence(concept, related)

        if not wiki_evidence:
            log.info(f"[{concept}] 本轮无 Wiki 验证通过的关联概念")
            return round_result

        # ── Stage 4: 假设检验 ──
        for ev in wiki_evidence:
            s, r, o, c = ev["subject"], ev["relation"], ev["object"], ev["confidence"]

            # 检查是否已存在于 CG 中（已有知识不算"学习"）
            if self._triple_exists(s, r, o):
                # 已有 → 只做验证刷新
                try:
                    self.memory.mark_verified(s, r, o)
                except Exception:
                    pass
                round_result["learned_count"] += 1
                continue

            # 假设检验
            try:
                test_result = self.tester.test_triple(s, r, o, c)
            except Exception as e:
                log.warning(f"假设检验失败 [{s} {r} {o}]: {e}")
                continue

            round_result["learned_count"] += 1

            # 检测冲突
            if test_result.get("conflicts"):
                # 标记为争议，不写入 CG
                controversy = {
                    "triple": (s, r, o, c),
                    "type": "hypothesis_conflict",
                    "conflicts": test_result["conflicts"],
                    "impact_score": test_result.get("impact_score", 0),
                    "timestamp": datetime.now().isoformat(),
                }
                round_result["conflicts"].append(controversy)
                round_result["controversies"].append(controversy)
                self._controversies.append(controversy)
                self._mark_controversial(s, r, o, c, str(test_result["conflicts"]))
                log.info(f"[{concept}] ⚠ 冲突: {s} {r} {o} → 标记为争议")
                continue

            # ── Stage 5: D-S 多源融合 ──
            # 添加 wiki 证据到模糊格
            self.fg.add_evidence(
                s, r, o,
                source=SOURCE_WIKI,
                mass=c,
                description=ev.get("evidence_text", "")[:200],
            )

            # 检查 CG 中是否已有来自其他来源的同一三元组
            existing_conf = self._get_triple_confidence(s, r, o)
            if existing_conf is not None and existing_conf > 0:
                # 已有 CG 证据 → 作为第二个独立源加入
                self.fg.add_evidence(
                    s, r, o,
                    source=SOURCE_CG,
                    mass=existing_conf,
                    description=f"概念图已有 (c={existing_conf})",
                )

            # 多源融合
            fuse_result = self.fg.multi_source_fuse(
                s, r, o,
                memory_timeline=self.memory,
            )

            evidence_count = fuse_result.get("evidence_count", 0)

            # ── 防幻觉机制 ──
            # 1. 至少 2 个独立源验证才能写入
            if evidence_count < MIN_SOURCES_FOR_WRITE:
                log.info(
                    f"[{concept}] 证据不足: {s} {r} {o} "
                    f"(只有 {evidence_count} 个来源，需 ≥{MIN_SOURCES_FOR_WRITE})"
                )
                continue

            # 2. D-S 融合信念低于阈值不写入
            combined_mass = fuse_result.get("combined_mass", 0)
            if combined_mass < MIN_DS_BELIEF_FOR_WRITE:
                log.info(
                    f"[{concept}] 信念过低: {s} {r} {o} "
                    f"(combined_mass={combined_mass:.3f} < {MIN_DS_BELIEF_FOR_WRITE})"
                )
                continue

            # ── 验证通过 → 写入 CG ──
            try:
                self.memory.mark_learned(
                    s, r, o,
                    source=f"{SOURCE_DS_FUSE}({','.join(fuse_result.get('sources', []))})",
                    confidence=combined_mass,
                )
                self.memory.mark_verified(s, r, o)
                self._set_status(s, r, o, "verified")
                round_result["verified_count"] += 1
                log.info(
                    f"[{concept}] ✅ 验证通过: {s} {r} {o} "
                    f"(Bel={fuse_result.get('belief', 0):.3f}, "
                    f"Pl={fuse_result.get('plausibility', 0):.3f}, "
                    f"来源数={evidence_count})"
                )
            except Exception as e:
                log.error(f"写入 CG 失败 [{s} {r} {o}]: {e}")

        return round_result

    # ══════════════════════════════════════════════════════════════
    # 守护进程: autonomous_tick
    # ══════════════════════════════════════════════════════════════

    def autonomous_tick(self, max_concepts: int = 5) -> Dict[str, Any]:
        """守护进程每轮调用：找盲区 → 学习 → 统计。

        Args:
            max_concepts: 每轮最多学习的盲区概念数

        Returns:
            {
                "concepts_learned": int,   # 至少学到一条知识的概念数
                "new_triples": int,        # 新增的已验证三元组数
                "conflicts_found": int,    # 发现的冲突数
                "details": [dict, ...],    # 每个概念的学习详情
            }
        """
        result = {
            "concepts_learned": 0,
            "new_triples": 0,
            "conflicts_found": 0,
            "details": [],
        }

        # a. 从 terrain 找盲区
        if self._terrain_loaded:
            blind_spots = self.terrain.top_blind_spots(n=max_concepts)
            targets = [(tp.concept, tp.energy) for tp in blind_spots]
        else:
            # terrain 未加载 → 从 CG 随机取样
            targets = self._random_blind_spot_fallback(max_concepts)

        if not targets:
            log.info("autonomous_tick: 未找到盲区概念")
            return result

        # b. 对每个盲区调用 learn_concept
        for concept, energy in targets:
            try:
                learn_result = self.learn_concept(concept, max_rounds=2)
                detail = {
                    "concept": concept,
                    "energy": energy,
                    "learned": learn_result.get("learned_count", 0),
                    "verified": learn_result.get("verified_count", 0),
                    "conflicts": len(learn_result.get("conflicts", [])),
                    "anchor_sufficient": learn_result.get("anchor_sufficient", False),
                }
                result["details"].append(detail)

                if learn_result.get("verified_count", 0) > 0:
                    result["concepts_learned"] += 1
                result["new_triples"] += learn_result.get("verified_count", 0)
                result["conflicts_found"] += len(learn_result.get("conflicts", []))
            except Exception as e:
                log.error(f"autonomous_tick 学习 [{concept}] 失败: {e}")

        return result

    def _random_blind_spot_fallback(self, n: int) -> List[Tuple[str, float]]:
        """Terrain 不可用时的盲区回退——从 CG 随机取样。"""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT s FROM triples WHERE "
                "status != 'verified' OR status = '' "
                "ORDER BY RANDOM() LIMIT ?",
                (n,)
            ).fetchall()
            return [(row[0], 999.0) for row in rows if row[0]]
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════════

    def _triple_exists(self, subject: str, relation: str, obj: str) -> bool:
        """检查三元组是否已在 CG 中。"""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM triples WHERE s=? AND r=? AND o=? LIMIT 1",
                (subject, relation, obj),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _get_triple_confidence(self, subject: str, relation: str, obj: str) -> Optional[float]:
        """获取 CG 中某三元组的置信度（若存在）。"""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT c FROM triples WHERE s=? AND r=? AND o=? LIMIT 1",
                (subject, relation, obj),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _set_status(self, subject: str, relation: str, obj: str, status: str):
        """设置三元组的 status 字段。"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE triples SET status=? WHERE s=? AND r=? AND o=?",
                (status, subject, relation, obj),
            )
            conn.commit()
        finally:
            conn.close()

    def _mark_controversial(self, subject: str, relation: str, obj: str,
                            confidence: float, reason: str):
        """标记三元组为争议（写入 CG 但标记为 controversial 状态）。"""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO triples(s, r, o, c, src, ev, "
                "learned_at, last_verified_at, verify_count, status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (subject, relation, obj, confidence,
                 "controversial",
                 f"[CONTROVERSIAL] {reason[:500]}",
                 now, now, 1, "controversial"),
            )
            conn.commit()
        finally:
            conn.close()

    def _find_new_blind_spots(self, concept: str,
                               related: Set[str]) -> List[str]:
        """在相关概念中找出新盲区。"""
        blind_spots = []
        for rel in related:
            try:
                energy = self.terrain.score(rel)
                zone = self.terrain.classify(energy)
                if zone == "blind_spot":
                    blind_spots.append(rel)
            except Exception:
                pass

        # 也检查 concept 自身
        try:
            energy = self.terrain.score(concept)
            zone = self.terrain.classify(energy)
            if zone == "blind_spot":
                blind_spots.append(concept)
        except Exception:
            pass

        return sorted(set(blind_spots))[:20]

    # ══════════════════════════════════════════════════════════════
    # 统计与查询
    # ══════════════════════════════════════════════════════════════

    def get_controversies(self) -> List[Dict[str, Any]]:
        """获取所有争议知识。"""
        return list(self._controversies)

    def stats(self) -> Dict[str, Any]:
        """学习器统计。"""
        mem_stats = self.memory.stats()
        fg_stats = self.fg.stats()
        return {
            "memory": mem_stats,
            "fuzzy_graph": fg_stats,
            "controversies": len(self._controversies),
            "terrain_loaded": self._terrain_loaded,
        }

    def close(self):
        """清理资源。"""
        self.wiki.close()
        self.tester.close()
        # memory 和 terrain 无显式 close，连接由 sqlite3 自动管理


# ══════════════════════════════════════════════════════════════
# 模块自检 / CLI
# ══════════════════════════════════════════════════════════════

def _resolve_default_paths():
    """解析默认数据库路径。"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    db_path = os.path.join(project_root, "data", "models", "concept_graph.db")
    wiki_path = os.path.join(project_root, "data", "wikipedia", "zhwiki.db")
    return db_path, wiki_path


if __name__ == "__main__":
    import json
    import sys

    db_path, wiki_path = _resolve_default_paths()

    print("=" * 70)
    print("  L6 验证驱动自主学习闭环 — VerifiedLearner 测试")
    print("=" * 70)
    print(f"  concept_graph.db: {db_path}")
    print(f"  zhwiki.db:        {wiki_path}")
    print()

    # 检查文件存在
    if not os.path.exists(db_path):
        print(f"  ❌ concept_graph.db 不存在: {db_path}")
        sys.exit(1)

    wiki_available = os.path.exists(wiki_path)
    if not wiki_available:
        print(f"  ⚠ Wikipedia db 不存在: {wiki_path}")
        print(f"     Wiki 验证将不可用，仅测试 Stage 1-2")
        print()

    # 初始化
    print("[初始化] VerifiedLearner...")
    t0 = time.time()
    vl = VerifiedLearner(db_path=db_path, wiki_path=wiki_path)
    print(f"  耗时: {time.time() - t0:.1f}s")
    print(f"  Terrain: {'✅ 已加载' if vl._terrain_loaded else '⚠ 未加载'}")
    print()

    # ── 测试 1: learn_concept("龙") ──
    concept = sys.argv[1] if len(sys.argv) > 1 else "龙"
    print(f"{'─' * 50}")
    print(f"  测试 1: learn_concept(\"{concept}\")")
    print(f"{'─' * 50}")

    t1 = time.time()
    result = vl.learn_concept(concept, max_rounds=3)
    elapsed = time.time() - t1

    print(f"  耗时: {elapsed:.1f}s")
    print(f"  锚定充足: {result['anchor_sufficient']}")
    print(f"  执行轮数: {result['rounds']}")
    print(f"  尝试学习: {result['learned_count']} 条")
    print(f"  验证通过: {result['verified_count']} 条")
    print(f"  冲突数:   {len(result['conflicts'])}")
    print(f"  争议数:   {len(result['controversies'])}")

    if result.get("conflicts"):
        print(f"\n  ⚠ 冲突详情:")
        for c in result["conflicts"][:5]:
            triple = c.get("triple", ())
            print(f"    {triple[0] if len(triple)>0 else '?'} "
                  f"{triple[1] if len(triple)>1 else '?'} "
                  f"{triple[2] if len(triple)>2 else '?'}")
            for detail in c.get("conflicts", [])[:3]:
                print(f"      [{detail.get('severity','?')}] {detail.get('reason','')[:120]}")

    if result.get("new_blind_spots"):
        print(f"\n  🔍 新盲区: {', '.join(result['new_blind_spots'][:10])}")

    print()

    # ── 测试 2: autonomous_tick ──
    print(f"{'─' * 50}")
    print(f"  测试 2: autonomous_tick(max_concepts=3)")
    print(f"{'─' * 50}")

    t2 = time.time()
    tick_result = vl.autonomous_tick(max_concepts=3)
    elapsed2 = time.time() - t2

    print(f"  耗时: {elapsed2:.1f}s")
    print(f"  学到概念: {tick_result['concepts_learned']}")
    print(f"  新增三元组: {tick_result['new_triples']}")
    print(f"  发现冲突: {tick_result['conflicts_found']}")

    if tick_result.get("details"):
        print(f"\n  详情:")
        for d in tick_result["details"]:
            anchor_tag = " [锚定充足]" if d.get("anchor_sufficient") else ""
            print(f"    {d['concept']:20s} "
                  f"尝试={d['learned']} "
                  f"验证={d['verified']} "
                  f"冲突={d['conflicts']}"
                  f"{anchor_tag}")

    print()

    # ── 统计 ──
    print(f"{'─' * 50}")
    print(f"  统计")
    print(f"{'─' * 50}")
    stats = vl.stats()
    print(f"  Memory: {stats['memory']['total_triples']} 条总知识, "
          f"{stats['memory']['verified_ever']} 条已验证")
    print(f"  FuzzyGraph: {stats['fuzzy_graph']['propositions']} 命题, "
          f"{stats['fuzzy_graph']['total_evidences']} 条证据")
    print(f"  争议知识: {stats['controversies']}")

    vl.close()

    print()
    print("=" * 70)
    print("  测试完成 ✅")
    print("=" * 70)
