#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 场→文本解码器 — 检索+组装 NLG
═══════════════════════════════════════════════════════

场负责「该说什么」(激活哪些知识), 组装规则负责「怎么说」(排列措辞)。

从 FieldResult 提取激活的三元组 → 按关系类型分组 → 组装成自然语言回答。
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any

from .dragon_field import FieldResult

log = logging.getLogger(__name__)


class FieldNLG:
    """
    场→文本翻译层。

    策略: 检索+组装 (路径 C)
      - 场收敛 → 激活的模式ID → 回查 SQLite 拿三元组
      - 按关系类型分组 (定义/分类/属性/关联)
      - 用轻量规则组装为自然语言
      - 涌现标记: 距离 0.2~0.5 → 标注"推测"
    """

    # 语义关系类型 (优先级从高到低)
    SEMANTIC_RELATIONS = [
        'DEFINED_AS',    # 定义 (最重要)
        'IS_A',           # 分类
        'HAS',            # 属性
        'PART_OF',        # 组成
        'CAUSE',          # 因果
        'COOCCURS_WITH',  # 共现
        'RELATED',        # 相关
        'FOLLOWS',        # 顺序
        'OCCURS_IN',      # 语境
    ]

    # 需过滤的噪音关系
    NOISE_RELATIONS = {
        'HAS_PINYIN', 'POETIC_NEXT', 'POETIC_WITH',
        'COOCCURS_IN',  # 成语名共现，噪音多
    }

    # 元标签 (IS_A 中的空泛分类)
    META_LABELS = {"中文词条", "成语", "词语", "词条", "词汇", "汉字"}

    def __init__(self, db_path: str, pattern_ids: List[int] = None):
        """
        Args:
            db_path: concept_graph.db 路径
            pattern_ids: 模式索引 → SQLite rowid 映射 (0-based)
                         如果不传, 默认 pattern_idx + 1 = rowid
        """
        self.db_path = db_path
        self._pattern_ids = pattern_ids or []

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def render(
        self,
        field_result: FieldResult,
        query_text: str = "",
    ) -> str:
        """
        将场收敛结果渲染为自然语言文本。

        优先级: DEFINED_AS > RELATED/IS_A > COOCCURS_WITH
        COOCCURS_WITH 仅在无更好来源时降级使用。
        """
        triples = self._lookup_triples(field_result.top_pattern_indices)
        grouped = self._group_by_relation(triples)
        parts = []
        has_good_source = False

        # 提取主概念
        main_subject = self._extract_main_subject(triples, query_text)

        # 1. 定义 (DEFINED_AS — 最确定)
        if grouped.get('DEFINED_AS'):
            defs = grouped['DEFINED_AS']
            parts.append(self._format_definitions(defs, field_result))
            has_good_source = True

        # 2. 语义关系 (IS_A, RELATED, PART_OF, CAUSE)
        semantic_parts = []
        if grouped.get('IS_A'):
            cats = self._format_categories(grouped['IS_A'])
            if cats:
                semantic_parts.append(cats)
                has_good_source = True
        if grouped.get('RELATED'):
            rels = self._format_related(grouped['RELATED'], main_subject)
            if rels:
                semantic_parts.append(rels)
                has_good_source = True
        if grouped.get('HAS') or grouped.get('PART_OF'):
            attrs = self._format_attributes(grouped)
            if attrs:
                semantic_parts.append(attrs)
                has_good_source = True

        if semantic_parts:
            parts.append("；".join(semantic_parts))

        # 3. COOCCURS_WITH — 仅在无更好来源时降级使用
        if not has_good_source and grouped.get('COOCCURS_WITH'):
            assoc = self._format_cooccurrence(
                grouped['COOCCURS_WITH'], main_subject
            )
            if assoc:
                parts.append(assoc)

        # 4. 涌现标记
        if field_result.is_emergent:
            parts.append("（以上信息部分为基于已有知识的推测）")

        # 5. 回退
        if not parts:
            if field_result.is_unreliable:
                return f"关于「{query_text}」，目前没有足够可靠的知识。"
            return f"关于「{query_text}」，尚未掌握相关知识。"

        return "。".join(parts) + "。"

    def _extract_main_subject(
        self, triples: List[Dict], query_text: str
    ) -> str:
        """从激活的三元组中提取主概念"""
        subjects = [t['subject'] for t in triples if t['subject']]
        if not subjects:
            return query_text
        # 最频繁出现的 subject
        from collections import Counter
        return Counter(subjects).most_common(1)[0][0]

    # ── 内部方法 ──────────────────────────────────────────────

    def _lookup_triples(
        self,
        pattern_indices: List[int],
        max_triples: int = 20,
    ) -> List[Dict]:
        """根据模式索引回查 SQLite 三元组"""
        if not pattern_indices:
            return []

        conn = self._get_conn()
        rows = []

        for pid in pattern_indices[:max_triples]:
            try:
                # 将模式矩阵索引映射到 SQLite rowid
                if self._pattern_ids and pid < len(self._pattern_ids):
                    rowid = self._pattern_ids[pid] + 1  # 转回 1-based
                else:
                    rowid = pid + 1
                cur = conn.execute(
                    "SELECT s, r, o, c, src FROM triples WHERE id = ?",
                    (rowid,)
                )
                row = cur.fetchone()
                if row:
                    rows.append({
                        'subject': row[0],
                        'relation': row[1],
                        'object': row[2],
                        'confidence': row[3],
                        'source': row[4],
                        'pattern_id': pid,
                    })
            except Exception:
                continue

        return rows

    def _group_by_relation(self, triples: List[Dict]) -> Dict[str, List[Dict]]:
        """按关系类型分组, 过滤噪音"""
        grouped: Dict[str, List[Dict]] = {}
        for t in triples:
            rel = t['relation']
            if rel in self.NOISE_RELATIONS:
                continue
            if rel == 'IS_A' and t['object'] in self.META_LABELS:
                continue  # 过滤元标签
            if t['confidence'] < 0.3:
                continue  # 过滤低置信
            grouped.setdefault(rel, []).append(t)
        return grouped

    def _format_definitions(
        self,
        defs: List[Dict],
        result: FieldResult,
    ) -> str:
        """渲染定义"""
        if not defs:
            return ""
        # 选置信度最高的最多3条
        defs_sorted = sorted(defs, key=lambda x: x['confidence'], reverse=True)[:3]

        subject = defs_sorted[0]['subject']
        objs = [d['object'] for d in defs_sorted]

        if len(objs) == 1:
            text = f"{subject}是{objs[0]}"
        elif len(objs) == 2:
            text = f"{subject}是{objs[0]}，也指{objs[1]}"
        else:
            text = f"{subject}是{objs[0]}，也指{'、'.join(objs[1:])}"

        if result.is_emergent:
            text += "（推测）"

        return text

    def _format_categories(self, cats: List[Dict]) -> str:
        """渲染分类"""
        if not cats:
            return ""
        valid = [c for c in cats if c['object'] not in self.META_LABELS]
        if not valid:
            return ""

        subject = valid[0]['subject']
        cat_names = list(dict.fromkeys([c['object'] for c in valid[:3]]))
        return f"{subject}属于{'、'.join(cat_names)}"

    def _format_attributes(self, grouped: Dict) -> str:
        """渲染属性 — 过滤自引用，使用实际主语"""
        attrs = []
        for rel in ['HAS', 'PART_OF']:
            for t in grouped.get(rel, [])[:3]:
                if t['object'] != t['subject']:
                    if rel == 'HAS':
                        attrs.append(f"{t['subject']}具有{t['object']}")
                    elif rel == 'PART_OF':
                        attrs.append(f"{t['subject']}是{t['object']}的组成部分")

        if not attrs:
            return ""
        return '、'.join(attrs)

    def _format_related(self, related: List[Dict], main_subject: str) -> str:
        """渲染 RELATED 语义关联"""
        objects = [t['object'] for t in related[:10]
                   if t['object'] != main_subject]  # 排除自引用
        if not objects:
            return ""
        unique = list(dict.fromkeys(objects))[:5]
        joined = '、'.join(unique)
        return f"{main_subject}与{joined}相关"

    def _format_cooccurrence(
        self, cooc: List[Dict], main_subject: str
    ) -> str:
        """渲染 COOCCURS_WITH — 字符级共现，仅在无更好来源时使用"""
        chars = [t['object'] for t in cooc[:6]]
        unique = list(dict.fromkeys(chars))

        # 粘合成多字概念: "头门王类虎"→"头门王类虎"
        joined = ''.join(unique[:6])
        if len(joined) <= 6:
            return f"{main_subject}的常见关联字符: {joined}"

        return f"{main_subject}与{'、'.join(unique[:5])}共现"

    def render_structured(
        self,
        field_result: FieldResult,
        query_text: str = "",
    ) -> Dict[str, Any]:
        """
        结构化渲染 — 返回 JSON 而非自然语言。
        用于调试和 API 输出。
        """
        triples = self._lookup_triples(field_result.top_pattern_indices)
        grouped = self._group_by_relation(triples)

        return {
            'query': query_text,
            'answer': self.render(field_result, query_text),
            'confidence_label': field_result.confidence_label,
            'distance_to_nearest': field_result.distance_to_nearest,
            'basin_depth': field_result.basin_depth,
            'curvature': field_result.curvature,
            'activated_triples': [
                {
                    'subject': t['subject'],
                    'relation': t['relation'],
                    'object': t['object'],
                    'confidence': t['confidence'],
                }
                for t in triples[:10]
            ],
            'emergent': field_result.is_emergent,
            'groups': {
                rel: [t['object'] for t in items[:5]]
                for rel, items in grouped.items()
            },
        }
