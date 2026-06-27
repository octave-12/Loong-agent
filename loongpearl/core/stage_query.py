#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1+2: 硬锚定查询 + 多方向BFS图遍历

使用 sqlite3 直连 concept_graph.db，不依赖 ConceptGraph 类。

Stage 1 (hard_anchor): 硬锚定 — 仅查询 DEFINED_AS + IS_A
Stage 2 (bfs_traverse): 多方向BFS — 分类/关联/字场三方向并行遍历
"""

import os
import sqlite3
from typing import List, Dict, Tuple, Optional, Any

# ── 数据库路径解析 ──

def _resolve_db_path() -> str:
    """解析 concept_graph.db 的绝对路径。优先项目内 data/models/concept_graph.db。"""
    # 尝试相对于项目根目录
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'concept_graph.db'),
        os.path.join(os.getcwd(), 'data', 'models', 'concept_graph.db'),
    ]
    for p in candidates:
        abspath = os.path.abspath(p)
        if os.path.exists(abspath):
            return abspath
    raise FileNotFoundError(
        "找不到 concept_graph.db，请确认项目根目录下 data/models/concept_graph.db 存在"
    )

# ── 关系分组 ──

TAXONOMY_RELATIONS = {'IS_A', 'DEFINED_AS', 'PART_OF'}
ASSOCIATION_RELATIONS = {'COOCCURS_WITH', 'COOCCURS_IN', 'CAUSE'}
# 字场暂不实现，留接口
CHARFIELD_RELATIONS = {'POETIC_NEXT', 'POETIC_WITH', 'FOLLOWS', 'OCCURS_IN'}  # 预留

# ── Stage 1: 硬锚定 ──

def hard_anchor(
    concept: str,
    min_conf: float = 0.6,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    硬锚定查询：仅查询 DEFINED_AS + IS_A 关系。

    这是最高置信度的锚定——定义关系与上位词关系，
    用于确定"这个概念到底是什么"。

    Args:
        concept: 概念/字（如 "龙"）
        min_conf: 最低置信度阈值（默认 0.6）
        db_path: SQLite 路径，默认自动解析

    Returns:
        {
            "concept": str,
            "results": [{"relation": str, "object": str, "confidence": float}, ...],
            "count": int,
            "avg_confidence": float | None,
            "sufficient": bool,   # avg_conf > 0.7 且有结果 → True
            "min_conf": float,
        }
    """
    if db_path is None:
        db_path = _resolve_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ','.join(['?'] * len(TAXONOMY_RELATIONS))
        # 只用 DEFINED_AS + IS_A（PART_OF 不走 hard_anchor）
        anchor_rels = ['DEFINED_AS', 'IS_A']
        ph = ','.join(['?'] * len(anchor_rels))
        rows = conn.execute(
            f"SELECT r, o, c, src, ev FROM triples WHERE s=? AND r IN ({ph}) AND c >= ? ORDER BY c DESC",
            [concept] + anchor_rels + [min_conf]
        ).fetchall()

        results = []
        total_conf = 0.0
        for row in rows:
            results.append({
                "relation": row["r"],
                "object": row["o"],
                "confidence": row["c"],
                "source": row["src"],
                "evidence": row["ev"],
            })
            total_conf += row["c"]

        avg_conf = total_conf / len(results) if results else None
        sufficient = avg_conf is not None and avg_conf > 0.7 and len(results) > 0

        return {
            "concept": concept,
            "results": results,
            "count": len(results),
            "avg_confidence": round(avg_conf, 4) if avg_conf else None,
            "sufficient": sufficient,
            "min_conf": min_conf,
        }
    finally:
        conn.close()


# ── Stage 2: 多方向BFS ──

def _bfs_one_direction(
    conn: sqlite3.Connection,
    start_concept: str,
    relations: List[str],
    max_hops: int,
    min_conf: float,
) -> List[List[Dict]]:
    """
    单方向 BFS 遍历。

    Returns:
        路径列表，每条路径是 [{"concept": str, "relation": str, "object": str, "confidence": float}, ...]
        路径长度 = hops（边数）
    """
    if not relations:
        return []

    ph = ','.join(['?'] * len(relations))
    all_paths = []
    visited = {start_concept}  # 已访问的 concept
    # 当前层：待扩展的路径（每条路径末尾的 object 即下一跳的起点）
    # 第 0 层：只含起始概念
    frontier: List[List[Dict]] = [[{"concept": start_concept, "relation": None, "object": start_concept, "confidence": 1.0}]]

    for hop in range(1, max_hops + 1):
        next_frontier = []
        for path in frontier:
            current_node = path[-1]["object"]

            rows = conn.execute(
                f"SELECT r, o, c, src, ev FROM triples WHERE s=? AND r IN ({ph}) AND c >= ? ORDER BY c DESC",
                [current_node] + relations + [min_conf]
            ).fetchall()

            for row in rows:
                obj = row[1]
                if obj in visited:
                    continue
                visited.add(obj)

                new_step = {
                    "concept": current_node,
                    "relation": row[0],
                    "object": obj,
                    "confidence": row[2],
                }
                new_path = path + [new_step]
                next_frontier.append(new_path)
                all_paths.append(new_path)

        frontier = next_frontier
        if not frontier:
            break

    return all_paths


def bfs_traverse(
    concept: str,
    max_hops: int = 3,
    min_conf: float = 0.3,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    多方向 BFS 图遍历。

    三个方向并行展开：
      - 方向1 (分类):    IS_A + DEFINED_AS + PART_OF
      - 方向2 (关联):    COOCCURS_WITH + COOCCURS_IN + CAUSE
      - 方向3 (字场):    暂不实现（留接口，返回空）

    Args:
        concept: 起始概念
        max_hops: 最大跳数（默认 3）
        min_conf: 最低置信度（默认 0.3）
        db_path: SQLite 路径，默认自动解析

    Returns:
        {
            "concept": str,
            "max_hops": int,
            "min_conf": float,
            "taxonomy": {        # 方向1: 分类
                "relations": [...],
                "paths": [...],
                "path_count": int,
            },
            "association": {     # 方向2: 关联
                "relations": [...],
                "paths": [...],
                "path_count": int,
            },
            "charfield": {       # 方向3: 字场 (暂未实现)
                "relations": [...],
                "paths": [],
                "path_count": 0,
            },
        }
    """
    if db_path is None:
        db_path = _resolve_db_path()

    conn = sqlite3.connect(db_path)
    try:
        taxonomy_paths = _bfs_one_direction(
            conn, concept,
            list(TAXONOMY_RELATIONS),
            max_hops, min_conf,
        )

        association_paths = _bfs_one_direction(
            conn, concept,
            list(ASSOCIATION_RELATIONS),
            max_hops, min_conf,
        )

        # 字场暂不实现
        charfield_paths = []

        return {
            "concept": concept,
            "max_hops": max_hops,
            "min_conf": min_conf,
            "taxonomy": {
                "relations": sorted(TAXONOMY_RELATIONS),
                "paths": taxonomy_paths,
                "path_count": len(taxonomy_paths),
            },
            "association": {
                "relations": sorted(ASSOCIATION_RELATIONS),
                "paths": association_paths,
                "path_count": len(association_paths),
            },
            "charfield": {
                "relations": sorted(CHARFIELD_RELATIONS),
                "paths": charfield_paths,
                "path_count": 0,
            },
        }
    finally:
        conn.close()


# ── 便捷查询 ──

def query_combined(
    concept: str,
    anchor_min_conf: float = 0.6,
    bfs_min_conf: float = 0.3,
    bfs_max_hops: int = 2,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    组合查询：先 hard_anchor 再 bfs_traverse。

    便于一键获取完整结果。
    """
    anchor = hard_anchor(concept, min_conf=anchor_min_conf, db_path=db_path)
    bfs = bfs_traverse(concept, max_hops=bfs_max_hops, min_conf=bfs_min_conf, db_path=db_path)
    return {
        "concept": concept,
        "anchor": anchor,
        "bfs": bfs,
    }


# ── CLI / 验证入口 ──

if __name__ == "__main__":
    import json
    import sys

    concept = sys.argv[1] if len(sys.argv) > 1 else "龙"

    print(f"{'='*60}")
    print(f"Stage 1: hard_anchor(\"{concept}\")")
    print(f"{'='*60}")
    result = hard_anchor(concept)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"Stage 2: bfs_traverse(\"{concept}\", max_hops=2)")
    print(f"{'='*60}")
    paths = bfs_traverse(concept, max_hops=2)
    # 摘要输出（完整路径太长）
    for direction in ["taxonomy", "association", "charfield"]:
        d = paths[direction]
        print(f"\n[{direction}] relations={d['relations']} path_count={d['path_count']}")
        for i, path in enumerate(d["paths"][:5]):
            steps = " → ".join(
                f"{s['concept']}--[{s['relation']}]--{s['object']}({s['confidence']:.2f})"
                for s in path[1:]  # 跳过第0跳（只有起始概念）
            )
            print(f"  path[{i}]: {steps}")
        if d["path_count"] > 5:
            print(f"  ... ({d['path_count'] - 5} more)")
        if d["path_count"] == 0:
            print(f"  (no paths — 字场方向暂未实现)" if direction == "charfield" else "  (no paths)")
