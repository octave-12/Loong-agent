#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: 结构化组装 — 汇总 Stage 1-3 结果，生成最终输出。

三路输入:
  Stage 1: hard_anchor()      → 确定性知识图谱查询
  Stage 2: bfs_traverse()      → 多方向图遍历
  Stage 3: WikipediaLookup     → 跨源验证 (独立来源)

输出: 结构化定义 → 分类位置 → 特征属性 → 关联概念 → 来源标注
"""

import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def answer_question(
    query: str,
    parsed: dict,
    min_anchor_conf: float = 0.6,
    bfs_max_hops: int = 2,
    bfs_min_conf: float = 0.3,
    wiki_verify_top: int = 10,
) -> dict:
    """
    四阶段流水线: 解析 → 锚定 → 遍历 → 验证 → 组装

    Args:
        query: 原始用户输入
        parsed: sem_parser 解析结果 (含 concepts, question_type 等)
        min_anchor_conf: Stage 1 最低置信度
        bfs_max_hops: Stage 2 最大跳数
        bfs_min_conf: Stage 2 边过滤阈值
        wiki_verify_top: Stage 3 验证的关系对数量上限

    Returns:
        {
            "query": str,
            "stage": "anchor" | "traverse",
            "sufficient": bool,
            "definition": [...],
            "taxonomy": [...],
            "attributes": [...],
            "associations": [...],
            "wiki_verified": [...],
            "sources": [...],
            "summary": str,
        }
    """
    from loongpearl.core.stage_query import hard_anchor, bfs_traverse, _resolve_db_path

    result = {
        "query": query,
        "stage": "anchor",
        "sufficient": False,
        "definition": [],
        "taxonomy": [],
        "attributes": [],
        "associations": [],
        "wiki_verified": [],
        "sources": [],
        "summary": "",
    }

    concepts = parsed.get("concepts", [parsed.get("subject", query.strip())])
    if not concepts:
        result["summary"] = "未能识别查询目标概念。"
        return result

    target = concepts[0]

    # ── Stage 1: 硬锚定 ──
    anchor = hard_anchor(target, min_conf=min_anchor_conf)

    for t in anchor["results"]:
        entry = {
            "relation": t["relation"],
            "object": t["object"],
            "confidence": t["confidence"],
            "source": t["source"],
        }
        if t["relation"] == "DEFINED_AS":
            entry["lang"] = "en" if any(
                c.isascii() and c.isalpha() for c in t["object"]
            ) else "zh"
            result["definition"].append(entry)
        elif t["relation"] == "IS_A":
            result["taxonomy"].append(entry)
        else:
            result["attributes"].append(entry)

    if anchor["sufficient"]:
        result["stage"] = "anchor"
        result["sufficient"] = True
        result["summary"] = _build_summary(result, target)
        return result

    # ── Stage 2: 多方向图遍历 ──
    traverse = bfs_traverse(target, max_hops=bfs_max_hops, min_conf=bfs_min_conf)

    # 分类方向: IS_A, DEFINED_AS, PART_OF
    for path in traverse["taxonomy"]["paths"][:20]:
        for step in path[1:]:  # skip self-loop
            result["taxonomy"].append({
                "relation": step["relation"],
                "object": step["object"],
                "confidence": step["confidence"],
                "source": "concept_graph",
            })

    # 关联方向: 根据 question_type 选择 relation 类型
    question_type = parsed.get("question_type", "")
    if "DEFINE" in question_type or "WHAT_IS" in question_type:
        # 定义类问题: 只用 COOCCURS_WITH，过滤 COOCCURS_IN (成语名) + CAUSE (不可靠)
        assoc_relations = ["COOCCURS_WITH"]
        assoc_min_conf = 0.6  # 定义类问题提高阈值
    else:
        assoc_relations = ["COOCCURS_WITH", "COOCCURS_IN", "CAUSE"]
        assoc_min_conf = bfs_min_conf

    assoc_paths = traverse["association"]["paths"]
    # 按置信度排序 + relation 过滤，取 top 20
    assoc_paths_filtered = [
        p for p in assoc_paths
        if any(s["relation"] in assoc_relations for s in p[1:])
    ]
    assoc_paths_sorted = sorted(
        assoc_paths_filtered,
        key=lambda p: sum(s["confidence"] for s in p[1:] if s["relation"] in assoc_relations),
        reverse=True,
    )[:20]

    for path in assoc_paths_sorted:
        for step in path[1:]:
            if step["relation"] in assoc_relations:
                result["associations"].append({
                    "relation": step["relation"],
                    "object": step["object"],
                    "confidence": step["confidence"],
                    "source": "concept_graph",
                })

    # 去重
    result["taxonomy"] = _dedupe(result["taxonomy"])
    result["associations"] = _dedupe(result["associations"])

    # ── Stage 3: 跨源验证 ──
    result["wiki_verified"] = _verify_top_relations(
        target, result, top_n=wiki_verify_top
    )

    # ── 组装 ──
    result["stage"] = "traverse"
    result["sources"] = _collect_sources(result)
    result["summary"] = _build_summary(result, target)

    return result


def _verify_top_relations(target: str, result: dict, top_n: int = 10) -> List[dict]:
    """对关键关系对进行 Wikipedia 验证"""
    try:
        from loongpearl.core.wiki_lookup import WikipediaLookup
        wiki = WikipediaLookup("data/wikipedia/zhwiki.db")
    except Exception as e:
        log.warning(f"Wiki lookup 初始化失败: {e}")
        return []

    verified = []
    candidates = []

    # 收集待验证的关系对（优先 taxonomy，再取 association）
    for entry in result["taxonomy"][:5]:
        candidates.append((entry["object"], entry["relation"], entry["confidence"]))
    for entry in result["associations"][:top_n]:
        candidates.append((entry["object"], entry["relation"], entry["confidence"]))

    seen = set()
    for obj, rel, cg_conf in candidates:
        if obj in seen:
            continue
        seen.add(obj)

        vr = wiki.verify_relation(target, rel, obj)
        if vr["found"]:
            verified.append({
                "object": obj,
                "relation": rel,
                "cg_confidence": cg_conf,
                "wiki_confidence": vr["confidence"],
                "evidence": vr["evidence"][:300] if vr["evidence"] else "",
                "co_count": vr["co_count"],
            })

        if len(verified) >= top_n:
            break

    return verified


def _dedupe(entries: List[dict]) -> List[dict]:
    """按 object 去重，保留最高置信度"""
    seen = {}
    for e in entries:
        key = e["object"]
        if key not in seen or e["confidence"] > seen[key]["confidence"]:
            seen[key] = e
    return sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)


def _collect_sources(result: dict) -> List[str]:
    """收集所有来源"""
    sources = set()
    for section in ["definition", "taxonomy", "attributes", "associations"]:
        for entry in result.get(section, []):
            src = entry.get("source", "unknown")
            if src and src != "unknown":
                sources.add(src)
    for v in result.get("wiki_verified", []):
        sources.add("wikipedia_dump")
    return sorted(sources)


def _build_summary(result: dict, target: str) -> str:
    """生成最终摘要文本"""
    parts = []

    # 定义
    if result["definition"]:
        defs = [d["object"] for d in result["definition"][:3]]
        parts.append(f"{target}的定义：{'；'.join(defs)}")

    # 分类
    if result["taxonomy"]:
        tax = [f'{t["object"]}({t["relation"]})' for t in result["taxonomy"][:5]]
        parts.append(f"分类归属：{'，'.join(tax)}")

    # 关联
    if result["associations"]:
        assoc = [a["object"] for a in result["associations"][:8]]
        parts.append(f"关联概念：{'，'.join(assoc)}")

    # Wiki 验证
    if result["wiki_verified"]:
        wiki_ok = [v["object"] for v in result["wiki_verified"][:3]]
        parts.append(f"已由Wikipedia交叉验证：{'，'.join(wiki_ok)}")

    # 置信度
    if result["stage"] == "anchor":
        parts.append("(来源：知识图谱直接命中)")
    else:
        parts.append("(来源：知识图谱多跳遍历 + Wikipedia 交叉验证)")

    return "\n".join(parts)


# ── CLI 快速测试 ──
if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "龙是什么"

    from loongpearl.core.sem_parser import SemParser
    parser = SemParser()
    frame = parser.parse(query)
    parsed = {
        "subject": frame.subject,
        "concepts": frame.concepts,
        "question_type": str(frame.question_type),
    }
    print(f"查询: {query}")
    print(f"解析: subject={frame.subject}, concepts={frame.concepts}, type={frame.question_type}")
    print()

    res = answer_question(query, parsed)

    print(f"阶段: {res['stage']}")
    print(f"充分: {res['sufficient']}")
    print(f"来源: {res['sources']}")
    print()
    print(res["summary"])
    print()

    print("--- 详情 ---")
    print(f"定义: {len(res['definition'])} 条")
    for d in res["definition"][:3]:
        print(f"  {d}")
    print(f"分类: {len(res['taxonomy'])} 条")
    for t in res["taxonomy"][:5]:
        print(f"  {t['relation']:15s} -> {t['object']:25s} conf={t['confidence']:.2f}")
    print(f"关联: {len(res['associations'])} 条 (截断top30)")
    for a in res["associations"][:5]:
        print(f"  {a['relation']:15s} -> {a['object']:25s} conf={a['confidence']:.2f}")
    print(f"Wiki验证: {len(res['wiki_verified'])} 条")
    for v in res["wiki_verified"][:5]:
        print(f"  {v['object']:25s} CG={v['cg_confidence']:.2f} Wiki={v['wiki_confidence']:.3f}")
