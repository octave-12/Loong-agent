#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠概念图 (concept_graph.py) — 多关系知识图谱 + 多跳推理 + 自主扩展
════════════════════════════════════════════════════════════════════════════

在字场嵌入之上，构建学科概念的多关系有向图。完全独立于LLM：
  - 节点嵌入从字场锚点组合而来（汉字 → 嵌入向量）
  - 边能量由能量景观评估（越低越可信）
  - 知识提取通过中文模式匹配（正则），不依赖 NLP/LLM

════════════════════════════════════════════════════════════════════════════
核心能力
════════════════════════════════════════════════════════════════════════════

1. 三元组存储    6种关系类型，置信度评分
2. 提取引擎     从搜索文本中挖掘 (概念, 关系, 概念) 三元组
3. 归纳推理     传递闭包: A→B + B→C → 推断 A→C (较低置信度)
4. 矛盾检测     环形 PART_OF、IS_A 环路检测
5. 多跳推理     BFS 遍历，能量排序
6. 自评估       覆盖率、一致性、连通性指标
7. 增量持久化   嵌入(pt) + 图结构(json)

════════════════════════════════════════════════════════════════════════════
关系类型
════════════════════════════════════════════════════════════════════════════

  IS_A      — 上位词 (猫 IS_A 动物)
  PART_OF   — 组成部分 (电子 PART_OF 原子)
  HAS       — 拥有属性 (细胞 HAS 细胞核)
  CAUSE     — 因果关系 (燃烧 CAUSE 热)
  OPPOSITE  — 对立关系 (热 OPPOSITE 冷)
  RELATED   — 一般相关 (量子 RELATED 物理)

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.concept_graph import ConceptGraph, Relation

    cg = ConceptGraph(field, landscape)

    # 1. 种子注入
    cg.seed_all_domains()  # 多学科种子

    # 2. 手动添加
    cg.add_triple("电子", "PART_OF", "原子", confidence=0.9)

    # 3. 推理
    paths = cg.reason("电子", max_hops=3, direction="both")
    for p in paths[:5]:
        print(" → ".join(p))

    # 4. 知识提取 (从搜索文本)
    triples = cg.extract_triples("原子由质子和中子组成，电子围绕原子核运动")
    # → [("质子", "PART_OF", "原子"), ("中子", "PART_OF", "原子"), ...]

    # 5. 归纳推理
    inferred = cg.induce(relation="PART_OF")
    for s, r, o in inferred:
        print(f"[推断] {s} {r} {o}")

    # 6. 矛盾检测
    conflicts = cg.detect_contradictions()
    for c in conflicts:
        print(f"[冲突] {c}")

    # 7. 自评估
    report = cg.evaluate()
    print(report)

    # 8. 保存/加载
    cg.save("data/models/concept_graph")
"""

import re
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ============================================================================
# 关系类型
# ============================================================================

class Relation:
    """六种关系类型常量"""
    IS_A     = "IS_A"       # 上位词 (猫 IS_A 动物)
    PART_OF  = "PART_OF"    # 组成部分 (电子 PART_OF 原子)
    HAS      = "HAS"        # 拥有属性 (细胞 HAS 细胞核)
    CAUSE    = "CAUSE"      # 因果关系 (燃烧 CAUSE 热)
    OPPOSITE = "OPPOSITE"   # 对立关系 (热 OPPOSITE 冷)
    RELATED  = "RELATED"    # 一般相关 (量子 RELATED 物理)

    ALL = [IS_A, PART_OF, HAS, CAUSE, OPPOSITE, RELATED]

    # 可传递的关系（A→B, B→C → A→C）
    TRANSITIVE = {IS_A, PART_OF}

    # 对称关系 (A→B → B→A)
    SYMMETRIC = {OPPOSITE, RELATED}

    # 关系中文描述
    ZH = {
        IS_A:     "是一种",
        PART_OF:  "是…的组成部分",
        HAS:      "拥有/包含",
        CAUSE:    "导致",
        OPPOSITE: "对立于",
        RELATED:  "相关于",
    }

    @classmethod
    def to_index(cls, rel: str) -> int:
        return cls.ALL.index(rel) if rel in cls.ALL else len(cls.ALL) - 1

    @classmethod
    def from_zh(cls, zh_pattern: str) -> Optional[str]:
        """从中文模式匹配关系类型"""
        mappings = [
            (["是", "属于", "是一种", "为"],             cls.IS_A),
            (["组成", "构成", "包含于", "属于.*一部分"],    cls.PART_OF),
            (["拥有", "包含", "含有", "具有", "有"],       cls.HAS),
            (["导致", "引起", "产生", "造成", "引发"],     cls.CAUSE),
            (["相反", "对立", "相对"],                    cls.OPPOSITE),
        ]
        for patterns, rel_type in mappings:
            for p in patterns:
                if re.search(p, zh_pattern):
                    return rel_type
        return cls.RELATED

    # 关系组合表: compose(rel_a, rel_b) → (rel_result, confidence_decay)
    # 例: IS_A ○ PART_OF = RELATED  (猫 IS_A 动物, 动物 PART_OF 生态 → 猫 RELATED 生态)
    #     PART_OF ○ HAS = RELATED   (电池 PART_OF 手机, 手机 HAS 屏幕 → 电池 RELATED 屏幕)
    COMPOSITION = {
        # (rel_a, rel_b) → (result_rel, decay_factor)
        # 同一关系的传递(已在induce中处理，此处仅跨关系组合)
        (IS_A, PART_OF):    (RELATED, 0.6),
        (IS_A, HAS):        (RELATED, 0.5),
        (IS_A, CAUSE):      (RELATED, 0.4),
        (PART_OF, IS_A):    (RELATED, 0.6),
        (PART_OF, HAS):     (RELATED, 0.5),
        (PART_OF, CAUSE):   (RELATED, 0.4),
        (HAS, PART_OF):     (RELATED, 0.5),
        (HAS, IS_A):        (RELATED, 0.5),
        (CAUSE, PART_OF):   (RELATED, 0.4),
        (CAUSE, IS_A):      (RELATED, 0.4),
        (CAUSE, HAS):       (RELATED, 0.5),
        # RELATED 与任何关系组合保持 RELATED
        (RELATED, IS_A):    (RELATED, 0.4),
        (RELATED, PART_OF): (RELATED, 0.4),
        (RELATED, HAS):     (RELATED, 0.3),
        (RELATED, CAUSE):   (RELATED, 0.3),
        (IS_A, RELATED):    (RELATED, 0.4),
        (PART_OF, RELATED): (RELATED, 0.4),
        (HAS, RELATED):     (RELATED, 0.3),
        (CAUSE, RELATED):   (RELATED, 0.3),
    }

    @classmethod
    def compose(cls, rel_a: str, rel_b: str) -> Tuple[Optional[str], float]:
        """
        组合两个关系。

        Returns:
            (result_relation, confidence_decay) 或 (None, 0) 如果不可组合
        """
        rel_a_clean = rel_a.replace('<-', '')
        rel_b_clean = rel_b.replace('<-', '')
        result = cls.COMPOSITION.get((rel_a_clean, rel_b_clean))
        if result:
            return result
        # 同一关系 → 传递 (已在 induce 中处理)
        if rel_a_clean == rel_b_clean and rel_a_clean in cls.TRANSITIVE:
            return (rel_a_clean, 0.7)
        return (None, 0.0)


# ============================================================================
# 三元组数据结构
# ============================================================================

@dataclass
class Triple:
    """一条知识三元组"""
    subject: str
    relation: str
    object: str
    confidence: float = 0.5       # 置信度 0.0-1.0
    source: str = "manual"        # manual | seed | extract | infer | web
    evidence_count: int = 1       # 证据来源数
    inferred_from: List[Tuple[str, str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.object}"

    @property
    def is_inferred(self) -> bool:
        return self.source == "infer"


# ============================================================================
# 中文模式 → 三元组提取引擎
# ============================================================================

# 中文关系提取模式 (概念A, 关系词, 概念B)
# 格式: (正则模式, 关系类型, 提取组顺序: subject→relation→object)
EXTRACTION_PATTERNS = [
    # PART_OF: "X由Y组成" / "X是由Y构成的"
    (
        re.compile(r'(.{1,4})(?:是)?由(.{1,8})(?:和(.{1,8}))?(?:共同)?(?:所)?组成'),
        Relation.PART_OF, "object_subject"
    ),
    (
        re.compile(r'(.{1,4})(?:是)?由(.{1,8})(?:和(.{1,8}))?(?:共同)?(?:所)?构成'),
        Relation.PART_OF, "object_subject"
    ),
    # PART_OF: "Y组成X" / "Y构成了X"
    (
        re.compile(r'(.{1,8})(?:和(.{1,8}))?(?:共同)?(?:所)?组成了?(.{1,4})'),
        Relation.PART_OF, "subject_object"
    ),
    # HAS: "X包含Y" / "X具有Y"
    (
        re.compile(r'(.{1,4})包含(.{1,8})'),
        Relation.HAS, "subject_object"
    ),
    (
        re.compile(r'(.{1,4})具[有备](.{1,8})'),
        Relation.HAS, "subject_object"
    ),
    (
        re.compile(r'(.{1,4})含[有](.{1,8})'),
        Relation.HAS, "subject_object"
    ),
    # IS_A: "X是Y的一种" / "X是一种Y" / "X属于Y"
    (
        re.compile(r'(.{1,4})是(.{1,4})的?一种'),
        Relation.IS_A, "subject_object"
    ),
    (
        re.compile(r'(.{1,4})属于(.{1,4})'),
        Relation.IS_A, "subject_object"
    ),
    (
        re.compile(r'(.{1,4})[是属为](.{1,6})'),
        Relation.RELATED, "subject_object"
    ),
    # CAUSE: "X导致Y" / "X引起Y"
    (
        re.compile(r'(.{1,6})(?:会|可[以能])?(?:导[致]|引起|产生|造成|引发)(.{1,6})'),
        Relation.CAUSE, "subject_object"
    ),
    # PART_OF: "X是Y的一部分"
    (
        re.compile(r'(.{1,4})是(.{1,4})的?(?:一部分|组成部分|分支|子集)'),
        Relation.PART_OF, "subject_object"
    ),
    # "X包括Y" 
    (
        re.compile(r'(.{1,4})包括(.{1,6})'),
        Relation.HAS, "subject_object"
    ),
]

# 噪声词过滤（非概念词）
NOISE_CONCEPTS = {
    '可以', '能够', '需要', '必须', '应该', '已经', '没有', '不是',
    '这个', '那个', '这些', '那些', '它们', '我们', '他们', '你们',
    '一种', '一个', '一些', '所有', '每个', '任何', '什么', '怎么',
    '因为', '所以', '但是', '而且', '或者', '如果', '虽然', '然而',
    '其中', '之间', '之后', '之前', '之上', '之下', '之外', '之内',
    '通过', '根据', '按照', '关于', '对于', '由于', '随着', '为了',
    '可能', '也许', '大约', '大概', '几乎', '完全', '非常', '十分',
    '一样', '同样', '不同', '相似', '相同', '相关', '相应', '相当',
}


def _clean_concept(text: Optional[str]) -> Optional[str]:
    """清洗概念名：去噪、去标点、去空白"""
    if text is None:
        return None
    # 去掉标点符号和空白
    text = re.sub(r'[，,。\.！!？?；;：:""''（）()【】\[\]《》\s]+', '', text)
    text = text.strip()
    # 过滤噪声词和过短/过长词
    if not text or len(text) < 1 or len(text) > 12:
        return None
    if text in NOISE_CONCEPTS:
        return None
    return text


def extract_triples_from_text(text: str, max_results: int = 30) -> List[Tuple[str, str, str]]:
    """
    从中文文本中提取 (概念, 关系, 概念) 三元组。

    纯正则匹配，不依赖任何 LLM。

    Args:
        text: 中文文本
        max_results: 最大返回数

    Returns:
        [(subject, relation, object), ...]
    """
    if not text or len(text) < 4:
        return []

    triples = []
    seen = set()

    for pattern, rel_type, direction in EXTRACTION_PATTERNS:
        for match in pattern.finditer(text):
            if len(triples) >= max_results:
                break

            groups = match.groups()
            if not groups:
                continue

            if direction == "object_subject":
                # "X由Y组成" → Y PART_OF X
                obj = _clean_concept(groups[0])  # Y
                subj = _clean_concept(groups[-1])  # X (最后一个组)
                if obj and subj and obj != subj:
                    key = f"{obj}|{rel_type}|{subj}"
                    if key not in seen:
                        seen.add(key)
                        triples.append((obj, rel_type, subj))
                # 如果有中间组 (Y和Z共同组成X)
                for g in groups[1:-1]:
                    if g is None:
                        continue
                    obj2 = _clean_concept(g)
                    if obj2 and subj and obj2 != subj:
                        key = f"{obj2}|{rel_type}|{subj}"
                        if key not in seen:
                            seen.add(key)
                            triples.append((obj2, rel_type, subj))
            elif direction == "subject_object":
                # "Y组成了X" → Y PART_OF X
                subj = _clean_concept(groups[0])  # Y
                obj = _clean_concept(groups[-1])  # X
                if subj and obj and subj != obj:
                    key = f"{subj}|{rel_type}|{obj}"
                    if key not in seen:
                        seen.add(key)
                        triples.append((subj, rel_type, obj))
                for g in groups[1:-1]:
                    if g is None:
                        continue
                    subj2 = _clean_concept(g)
                    if subj2 and obj and subj2 != obj:
                        key = f"{subj2}|{rel_type}|{obj}"
                        if key not in seen:
                            seen.add(key)
                            triples.append((subj2, rel_type, obj))

    return triples


# ============================================================================
# 概念图核心类
# ============================================================================

class ConceptGraph:
    """
    多关系概念图。

    节点嵌入 = 字场字符嵌入的平均（从字场组合而来）
    边权重 = 景观能量 + 三元组置信度（越低越可信）

    完全独立于 LLM：知识提取用正则，推理用 BFS + 能量评估。
    """

    def __init__(self, zichang, landscape=None):
        self.zichang = zichang
        self.landscape = landscape
        self.embed_dim = zichang.embed_dim if zichang else 768

        # ── 节点 ──
        self.nodes: Dict[str, torch.Tensor] = {}        # concept → primary embedding
        self.context_embeddings: Dict[str, List[torch.Tensor]] = defaultdict(list)  # concept → [context_vecs]
        self.node_aliases: Dict[str, str] = {}           # alias → canonical

        # ── 边 ──
        self.triples: Dict[str, Triple] = {}             # key → Triple
        self.forward_index: Dict[str, Dict[str, str]] = defaultdict(dict)  # A → {B: rel}
        self.reverse_index: Dict[str, Dict[str, str]] = defaultdict(dict)  # B → {A: rel}

        # ── 统计 ──
        self.total_triples = 0
        self.total_inferred = 0

        # ── 持久化路径 ──
        self._data_dir = None

        # ── 写入脏标记: 避免每轮无变化时仍写 257MB JSON ──
        self._dirty_since_last_save = False

        # ── 字符邻接索引: char → {neighbor_chars}，O(1)查询盲区关联 ──
        self._char_adjacency: Dict[str, set] = defaultdict(set)

    # ═══════════════════════════════════════════════════════════════════════
    # 节点管理
    # ═══════════════════════════════════════════════════════════════════════

    def add_node(self, concept: str, context_text: str = None) -> torch.Tensor:
        """
        添加概念节点，嵌入从字场组合而来。返回嵌入向量。

        Args:
            concept: 概念名
            context_text: 可选的上下文文本，用于语境消歧
                - 如果提供，会额外存储一个上下文相关的嵌入
                - 例: add_node("行", "我在银行取钱") → 金融语境
                       add_node("行", "五行金木水火") → 哲学语境
        """
        # 主要嵌入（字场组合）
        if concept not in self.nodes:
            chars = list(concept)
            valid_idxs = []
            for c in chars:
                idx = getattr(self.zichang, '_char_to_idx', {}).get(c)
                if idx is not None:
                    valid_idxs.append(idx)
            if not valid_idxs:
                vec = torch.zeros(self.embed_dim)
            else:
                vec = self.zichang.anchors[valid_idxs].mean(dim=0)
            self.nodes[concept] = vec

        # 上下文嵌入（如果有上下文，存入语境向量簇）
        if context_text:
            # 从上下文中提取所有汉字的字场锚点平均 → 语境向量
            ctx_chars = [c for c in context_text if c in getattr(self.zichang, '_char_to_idx', {})]
            if ctx_chars:
                ctx_indices = [getattr(self.zichang, '_char_to_idx', {})[c] for c in ctx_chars]
                ctx_vec = self.zichang.anchors[ctx_indices].mean(dim=0)
                # 混合: 70% 语境 + 30% 概念本身 → 语境化的概念嵌入
                contextualized = 0.7 * ctx_vec + 0.3 * self.nodes[concept]
                self.context_embeddings[concept].append(contextualized)
                # 限制每个概念最多存10个语境向量
                if len(self.context_embeddings[concept]) > 10:
                    self.context_embeddings[concept] = self.context_embeddings[concept][-10:]

        return self.nodes[concept]

    def get_embedding(self, concept: str, context_vec: torch.Tensor = None) -> Optional[torch.Tensor]:
        """
        获取概念嵌入。如果提供了上下文向量，返回最匹配的语境化嵌入。

        Args:
            concept: 概念名
            context_vec: 可选的上下文向量（用于消歧）
                - 如果提供且概念有语境向量簇，返回余弦最近的那个
                - 否则返回默认嵌入
        """
        if concept not in self.nodes:
            self.add_node(concept)

        # 无上下文或无语境向量簇 → 返回默认嵌入
        if context_vec is None or concept not in self.context_embeddings or not self.context_embeddings[concept]:
            return self.nodes.get(concept)

        # 在语境向量簇中找最近邻
        ctx_vecs = self.context_embeddings[concept]
        sims = torch.nn.functional.cosine_similarity(
            context_vec.unsqueeze(0),
            torch.stack(ctx_vecs),
            dim=1
        )
        best_idx = sims.argmax().item()
        return ctx_vecs[best_idx]

    def disambiguate(
        self,
        concept: str,
        context: str,
    ) -> int:
        """
        为概念添加语境并消歧。

        例: cg.disambiguate("行", "银行取款") → 添加金融语境的"行"嵌入
            cg.disambiguate("行", "行走在路上") → 添加运动语境的"行"嵌入
            之后查询时, get_embedding("行", context_vec) 会自动选最匹配的

        Returns:
            当前语境向量簇的大小
        """
        self.add_node(concept, context_text=context)
        return len(self.context_embeddings.get(concept, []))

    def canonical(self, concept: str) -> str:
        """返回概念的标准名称"""
        return self.node_aliases.get(concept, concept)

    # ═══════════════════════════════════════════════════════════════════════
    # 三元组管理
    # ═══════════════════════════════════════════════════════════════════════

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 0.5,
        source: str = "manual",
        evidence_count: int = 1,
    ) -> Optional[Triple]:
        """
        添加一条知识三元组。

        Args:
            subject: 主体概念
            relation: 关系类型 (IS_A/PART_OF/HAS/CAUSE/OPPOSITE/RELATED)
            obj: 客体概念
            confidence: 置信度 0.0-1.0
            source: 来源 (manual|seed|extract|infer|web)
            evidence_count: 证据来源数（多源验证可提升置信度）

        Returns:
            Triple 对象，如果已存在且置信度不更高则返回 None
        """
        # 规范化
        subject = self.canonical(subject)
        obj = self.canonical(obj)

        if subject == obj:
            return None  # 自指无意义

        # 确保节点存在
        self.add_node(subject)
        self.add_node(obj)

        # 检查是否已存在
        key = f"{subject}|{relation}|{obj}"
        if key in self.triples:
            existing = self.triples[key]
            # 合并证据
            existing.evidence_count += evidence_count
            existing.confidence = max(existing.confidence, confidence)
            return existing

        # 新建三元组
        triple = Triple(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=confidence,
            source=source,
            evidence_count=evidence_count,
        )
        self.triples[key] = triple

        # 正向索引
        self.forward_index[subject][obj] = relation
        # 反向索引
        self.reverse_index[obj][subject] = relation

        self.total_triples += 1
        self._dirty_since_last_save = True

        # ── 字符邻接索引: 单字概念 → 快速查找关联字 ──
        if len(subject) == 1 and len(obj) == 1:
            self._char_adjacency[subject].add(obj)
            self._char_adjacency[obj].add(subject)

        return triple

    def add_triples_batch(
        self,
        triples: List[Tuple[str, str, str]],
        source: str = "manual",
        confidence: float = 0.5,
    ) -> int:
        """
        批量添加三元组。

        Args:
            triples: [(subject, relation, object), ...]
            source: 来源标识
            confidence: 统一置信度

        Returns:
            成功添加数
        """
        count = 0
        for s, r, o in triples:
            if self.add_triple(s, r, o, confidence=confidence, source=source):
                count += 1
        return count

    def get_char_pairs(self, char: str, max_pairs: int = 200) -> list:
        """O(1) 查询与指定汉字关联的所有字符（用于盲区快速填坑）。
        
        Args:
            char: 单个汉字
            max_pairs: 返回上限
        
        Returns:
            [(ia, ib), ...] 字对索引列表
        """
        neighbors = self._char_adjacency.get(char, set())
        if not neighbors:
            return []
        
        pairs = []
        seen = set()
        for other in neighbors:
            if len(pairs) >= max_pairs:
                break
            if other == char:
                continue
            ia = self.zichang._char_to_idx.get(char)
            ib = self.zichang._char_to_idx.get(other)
            if ia is not None and ib is not None:
                key = (min(ia, ib), max(ia, ib))
                if key not in seen:
                    seen.add(key)
                    pairs.append((ia, ib))
        return pairs

    # ═══════════════════════════════════════════════════════════════════════
    # 知识提取 — 从搜索文本中挖掘三元组
    # ═══════════════════════════════════════════════════════════════════════

    def extract_triples(
        self,
        texts: Union[str, List[str]],
        min_confidence: float = 0.3,
    ) -> List[Triple]:
        """
        从文本中提取三元组并添加到概念图。

        纯正则匹配，不依赖任何 LLM。

        Args:
            texts: 中文文本（单个字符串或字符串列表）
            min_confidence: 最低置信度阈值

        Returns:
            新添加的 Triple 列表
        """
        if isinstance(texts, str):
            texts = [texts]

        all_raw = []
        for text in texts:
            if not text:
                continue
            raw = extract_triples_from_text(text)
            all_raw.extend(raw)

        # 去重并计算多源置信度
        key_counts: Dict[str, int] = defaultdict(int)
        for s, r, o in all_raw:
            key = f"{s}|{r}|{o}"
            key_counts[key] += 1

        added = []
        for key, count in key_counts.items():
            s, r, o = key.split("|")
            # 多源证据 → 更高置信度
            conf = min(0.9, 0.3 + count * 0.15)
            if conf >= min_confidence:
                triple = self.add_triple(
                    s, r, o,
                    confidence=conf,
                    source="extract",
                    evidence_count=count,
                )
                if triple:
                    added.append(triple)

        return added

    def extract_from_search_results(self, search_results: Any) -> List[Triple]:
        """
        从 WebSearcher 搜索结果中提取概念三元组。

        适配 WebSearcher 返回格式（SearchResponse.results → 每条有 .text 字段）。

        Args:
            search_results: WebSearcher 返回的 SearchResponse 对象或 dict 列表

        Returns:
            新添加的 Triple 列表
        """
        texts = []

        # 适配多种返回格式
        if hasattr(search_results, 'results'):
            for item in search_results.results:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, dict):
                    texts.append(item.get('text', '') or item.get('snippet', '') or item.get('content', ''))
        elif isinstance(search_results, list):
            for item in search_results:
                if isinstance(item, dict):
                    texts.append(item.get('text', '') or item.get('snippet', '') or item.get('content', ''))
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(search_results, str):
            texts = [search_results]

        return self.extract_triples(texts)

    # ═══════════════════════════════════════════════════════════════════════
    # 归纳推理 — 传递闭包
    # ═══════════════════════════════════════════════════════════════════════

    def induce(
        self,
        relation: str = None,
        max_chain: int = 4,
        min_confidence: float = 0.2,
        decay: float = 0.7,
    ) -> List[Triple]:
        """
        归纳推理：利用传递性推断新三元组。

        核心规则:
          - PART_OF 传递: A ⊂ B, B ⊂ C → 推断 A ⊂ C
          - IS_A 传递:   A ∈ B, B ∈ C → 推断 A ∈ C
          - 对称关系:    RELATED(A,B) → RELATED(B,A)   (如果反向不存在)

        Args:
            relation: 限定关系类型，None=所有可传递关系
            max_chain: 最大传递链长
            min_confidence: 最低置信度阈值（低于此不添加）
            decay: 每跳衰减因子 (confidence *= decay 每跳)

        Returns:
            本次推断出的新 Triple 列表
        """
        rels = [relation] if relation else list(Relation.TRANSITIVE)
        new_triples = []

        for rel in rels:
            if rel not in Relation.TRANSITIVE:
                continue

            # BFS 传递闭包
            for start in list(self.forward_index.keys()):
                visited = {start: (1.0, 0)}  # node → (confidence, hops)
                queue = deque([(start, 1.0, 0)])  # (node, conf, hops)

                while queue:
                    node, node_conf, hops = queue.popleft()
                    if hops >= max_chain:
                        continue

                    for neighbor, edge_rel in self.forward_index.get(node, {}).items():
                        if edge_rel != rel:
                            continue
                        if neighbor in visited:
                            continue

                        # 计算置信度衰减
                        edge_triple = self.triples.get(f"{node}|{rel}|{neighbor}")
                        edge_conf = edge_triple.confidence if edge_triple else 0.5
                        new_conf = node_conf * edge_conf * decay

                        if new_conf < min_confidence:
                            continue

                        visited[neighbor] = (new_conf, hops + 1)
                        queue.append((neighbor, new_conf, hops + 1))

                        # 推断: start → neighbor (经过 hops+1 跳)
                        if start != neighbor:
                            key = f"{start}|{rel}|{neighbor}"
                            if key not in self.triples:
                                triple = self.add_triple(
                                    start, rel, neighbor,
                                    confidence=new_conf,
                                    source="infer",
                                )
                                if triple:
                                    new_triples.append(triple)
                                    self.total_inferred += 1

        # 对称关系补全
        for rel in (Relation.SYMMETRIC & set(Relation.ALL)):
            for (subj, edges) in list(self.forward_index.items()):
                for obj, edge_rel in edges.items():
                    if edge_rel != rel:
                        continue
                    reverse_key = f"{obj}|{rel}|{subj}"
                    if reverse_key not in self.triples:
                        orig = self.triples.get(f"{subj}|{rel}|{obj}")
                        orig_conf = orig.confidence if orig else 0.5
                        triple = self.add_triple(
                            obj, rel, subj,
                            confidence=orig_conf * 0.9,
                            source="infer",
                        )
                        if triple:
                            new_triples.append(triple)
                            self.total_inferred += 1

        return new_triples

    # ═══════════════════════════════════════════════════════════════════════
    # 矛盾检测
    # ═══════════════════════════════════════════════════════════════════════

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        检测概念图中的矛盾。

        检测规则:
          1. 环形 PART_OF: A⊂B⊂A
          2. 环形 IS_A: A∈B∈A
          3. 同一对概念的冲突关系

        Returns:
            [{'type': 'circular_party'|'conflict', 'path': [...], 'triples': [...]}, ...]
        """
        conflicts = []

        # 1. 环形 PART_OF / IS_A
        for rel in [Relation.PART_OF, Relation.IS_A]:
            for start in self.forward_index:
                visited = {start: [start]}
                queue = deque([start])

                while queue:
                    node = queue.popleft()
                    for neighbor, edge_rel in self.forward_index.get(node, {}).items():
                        if edge_rel != rel:
                            continue
                        if neighbor == start:
                            # 发现环路
                            path = visited[node] + [neighbor]
                            conflicts.append({
                                'type': f'circular_{rel.lower()}',
                                'relation': rel,
                                'path': path,
                                'message': f"环形{Relation.ZH.get(rel, rel)}: {' → '.join(path)}",
                            })
                            continue
                        if neighbor not in visited:
                            visited[neighbor] = visited[node] + [neighbor]
                            queue.append(neighbor)

        # 2. 冲突关系检查（同一对概念有多个关系类型）
        checked_pairs = set()
        for subj, edges in self.forward_index.items():
            for obj in edges:
                pair = (subj, obj) if subj < obj else (obj, subj)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                rels_a = set()
                for n, r in self.forward_index.get(subj, {}).items():
                    if n == obj:
                        rels_a.add(r)
                for n, r in self.forward_index.get(obj, {}).items():
                    if n == subj:
                        rels_a.add(r)

                if len(rels_a) > 1:
                    conflicts.append({
                        'type': 'conflict',
                        'concepts': (subj, obj),
                        'relations': list(rels_a),
                        'message': f"概念对 ({subj},{obj}) 存在多个关系: {rels_a}",
                    })

        return conflicts

    # ═══════════════════════════════════════════════════════════════════════
    # 能量评估
    # ═══════════════════════════════════════════════════════════════════════

    def triple_energy(self, subject: str, obj: str) -> float:
        """
        评估两个概念之间的关联能量。

        两个概念的中点 → 能量景观评估。
        能量越低 → 越关联。
        """
        if self.landscape is None:
            return 999.0

        e_a = self.get_embedding(subject)
        e_b = self.get_embedding(obj)
        if e_a is None or e_b is None:
            return 999.0

        device = next(self.landscape.parameters()).device
        mid = ((e_a.to(device) + e_b.to(device)) / 2).unsqueeze(0)

        with torch.no_grad():
            return self.landscape(mid).item()

    def triple_score(self, subject: str, obj: str) -> float:
        """
        综合评分：能量 + 置信度 → 0.0-1.0 分数。
        越接近 1.0 越好。
        """
        energy = self.triple_energy(subject, obj)
        # 能量归一化 (假设范围 -50 到 50)
        energy_score = max(0.0, min(1.0, 1.0 - (energy + 50) / 100.0))

        # 直接查找所有可能的三元组键
        conf = 0.5
        for r in Relation.ALL:
            key_fwd = f"{subject}|{r}|{obj}"
            if key_fwd in self.triples:
                conf = max(conf, self.triples[key_fwd].confidence)
            key_rev = f"{obj}|{r}|{subject}"
            if key_rev in self.triples:
                conf = max(conf, self.triples[key_rev].confidence)

        return 0.4 * energy_score + 0.6 * conf

    def cross_relation_reason(
        self,
        start: str,
        relation: str,
        target_relation: str,
        max_hops: int = 3,
    ) -> List[Tuple[List[str], List[str], float]]:
        """
        跨关系推理：从起始概念沿 mixed 关系路径推理。

        例: "张三" -(任职于)→ "腾讯" -(位于)→ "深圳"
            关系组合: 任职于 ○ 位于 → RELATED (通过 compose 推断)

        Args:
            start: 起始概念
            relation: 起始关系
            target_relation: 目标关系
            max_hops: 最大跳数

        Returns:
            [(concept_path, relation_path, confidence), ...]
            例: (["张三","腾讯","深圳"], ["任职于","位于"], 0.4)
        """
        results = []

        # 第一跳：沿起始关系找邻居
        first_neighbors = self._get_neighbors(start, relation, "forward")
        for neighbor_a, rel_a in first_neighbors:
            # 第二跳及以后：沿目标关系找邻居
            second_neighbors = self._get_neighbors(neighbor_a, target_relation, "both")
            for neighbor_b, rel_b in second_neighbors:
                if neighbor_b == start:
                    continue

                # 组合关系
                rel_a_clean = rel_a.replace('<-', '')
                rel_b_clean = rel_b.replace('<-', '')
                composed_rel, decay = Relation.compose(rel_a_clean, rel_b_clean)

                if composed_rel is None:
                    continue

                # 置信度 = 两边的置信度乘积 × 组合衰减
                key_a = f"{start}|{rel_a_clean}|{neighbor_a}"
                key_b = f"{neighbor_a}|{rel_b_clean}|{neighbor_b}"
                conf_a = self.triples.get(key_a, Triple(start, rel_a_clean, neighbor_a, 0.5)).confidence
                conf_b = self.triples.get(key_b, Triple(neighbor_a, rel_b_clean, neighbor_b, 0.5)).confidence
                confidence = conf_a * conf_b * decay

                path = [start, neighbor_a, neighbor_b]
                rel_path = [rel_a_clean, rel_b_clean]
                results.append((path, rel_path, confidence))

        # 按置信度排序
        results.sort(key=lambda x: -x[2])
        return results[:20]

    # ═══════════════════════════════════════════════════════════════════════
    # 多跳推理
    # ═══════════════════════════════════════════════════════════════════════

    def reason(
        self,
        start: str,
        relation: str = None,
        max_hops: int = 3,
        direction: str = "forward",
        min_confidence: float = 0.1,
        related_penalty: float = 0.3,
    ) -> List[List[str]]:
        """
        多跳推理：从起点沿指定关系遍历。

        Args:
            start: 起始概念
            relation: 关系类型（None=任意关系）
            max_hops: 最大跳数
            direction: "forward" | "backward" | "both"
            min_confidence: 过滤低置信度边
            related_penalty: RELATED 边的能量惩罚系数 (0-1, 越高越排斥RELATED)

        Returns:
            多条路径，按加权能量排序
        """
        paths = [[start]]

        for hop in range(max_hops):
            new_paths = []

            for path in paths:
                current = path[-1]
                next_nodes = self._get_neighbors(current, relation, direction)

                for neighbor, rel in next_nodes:
                    if neighbor in path:
                        continue

                    key = f"{current}|{rel}|{neighbor}"
                    triple = self.triples.get(key)
                    if triple and triple.confidence < min_confidence:
                        continue

                    new_paths.append(path + [neighbor])

            if not new_paths:
                break
            paths = new_paths

        # 按加权路径能量排序（越低越可信）
        # RELATED 边被降权：乘以 (1 + related_penalty)
        if self.landscape and len(paths) > 0:
            paths.sort(key=lambda p: self._path_energy_weighted(p, related_penalty))

        return paths

    def _path_energy_weighted(self, path: List[str], related_penalty: float = 0.3) -> float:
        """评估一条路径的加权能量，RELATED 边受惩罚"""
        total = 0.0
        for i in range(len(path) - 1):
            e = self.triple_energy(path[i], path[i + 1])
            # 查找该边的关系类型
            subj, obj = path[i], path[i + 1]
            edge_rel = self.forward_index.get(subj, {}).get(obj) or                        self.reverse_index.get(subj, {}).get(obj) or ""
            # RELATED 边权重惩罚：能量越高(越差) + 额外惩罚
            if edge_rel == "RELATED" or edge_rel.startswith("<-RELATED"):
                e = e * (1.0 + related_penalty)
            total += e
        return total / (len(path) - 1) if len(path) > 1 else 0.0

    def _get_neighbors(
        self, node: str, relation: str, direction: str
    ) -> List[Tuple[str, str]]:
        """获取邻居节点及其关系"""
        neighbors = []

        if direction in ("forward", "both"):
            for neighbor, rel in self.forward_index.get(node, {}).items():
                if relation is None or rel == relation:
                    neighbors.append((neighbor, rel))

        if direction in ("backward", "both"):
            for neighbor, rel in self.reverse_index.get(node, {}).items():
                if relation is None or rel == relation:
                    neighbors.append((neighbor, f"<-{rel}"))

        return neighbors

    def _path_energy(self, path: List[str]) -> float:
        """评估一条路径的平均能量"""
        total = 0.0
        for i in range(len(path) - 1):
            total += self.triple_energy(path[i], path[i + 1])
        return total / (len(path) - 1) if len(path) > 1 else 0.0

    def query(
        self,
        concept: str,
        relation: str = None,
        max_results: int = 10,
        sort_by: str = "energy",
    ) -> List[Dict[str, Any]]:
        """
        查询一个概念的直接关联。

        Args:
            concept: 概念名
            relation: 关系类型过滤
            max_results: 最大返回数
            sort_by: "energy" | "confidence" | "score"

        Returns:
            [{'concept': ..., 'relation': ..., 'direction': ..., 'energy': ..., 'confidence': ...}, ...]
        """
        results = []

        # 正向
        for neighbor, rel in self.forward_index.get(concept, {}).items():
            if relation is None or rel == relation:
                key = f"{concept}|{rel}|{neighbor}"
                triple = self.triples.get(key)
                results.append({
                    'concept': neighbor,
                    'relation': rel,
                    'direction': 'forward',
                    'energy': self.triple_energy(concept, neighbor),
                    'confidence': triple.confidence if triple else 0.5,
                    'source': triple.source if triple else 'unknown',
                })

        # 反向
        for neighbor, rel in self.reverse_index.get(concept, {}).items():
            if relation is None or rel == relation:
                key = f"{neighbor}|{rel}|{concept}"
                triple = self.triples.get(key)
                results.append({
                    'concept': neighbor,
                    'relation': rel,
                    'direction': 'backward',
                    'energy': self.triple_energy(concept, neighbor),
                    'confidence': triple.confidence if triple else 0.5,
                    'source': triple.source if triple else 'unknown',
                })

        # 排序
        if sort_by == "confidence":
            results.sort(key=lambda x: -x['confidence'])
        elif sort_by == "score":
            results.sort(key=lambda x: -(0.4 * max(0, 1 - x['energy']/100) + 0.6 * x['confidence']))
        else:
            results.sort(key=lambda x: x['energy'])

        return results[:max_results]

    # ═══════════════════════════════════════════════════════════════════════
    # 种子知识注入 — 多学科
    # ═══════════════════════════════════════════════════════════════════════

    def seed_from_dict(self, knowledge: Dict[str, List[Dict]]):
        """从结构化字典注入种子知识"""
        for subject, relations in knowledge.items():
            for item in relations:
                obj = item['concept']
                rel = item.get('relation', Relation.RELATED)
                conf = item.get('confidence', 0.8)
                self.add_triple(subject, rel, obj, confidence=conf, source="seed")

    def seed_basic_science(self):
        """注入基础科学知识种子（物理/生物/天文/计算机）"""
        seeds = {
            "物质": [
                {"concept": "原子", "relation": "PART_OF"},
                {"concept": "分子", "relation": "PART_OF"},
            ],
            "原子": [
                {"concept": "电子", "relation": "PART_OF"},
                {"concept": "质子", "relation": "PART_OF"},
                {"concept": "中子", "relation": "PART_OF"},
                {"concept": "原子核", "relation": "HAS"},
                {"concept": "分子", "relation": "PART_OF"},
            ],
            "分子": [
                {"concept": "原子", "relation": "PART_OF"},
                {"concept": "化合物", "relation": "IS_A"},
            ],
            "细胞": [
                {"concept": "细胞核", "relation": "HAS"},
                {"concept": "细胞膜", "relation": "HAS"},
                {"concept": "线粒体", "relation": "HAS"},
                {"concept": "组织", "relation": "PART_OF"},
                {"concept": "器官", "relation": "PART_OF"},
            ],
            "DNA": [
                {"concept": "基因", "relation": "HAS"},
                {"concept": "染色体", "relation": "PART_OF"},
                {"concept": "遗传", "relation": "RELATED"},
            ],
            "地球": [
                {"concept": "太阳系", "relation": "PART_OF"},
                {"concept": "月球", "relation": "HAS"},
                {"concept": "大气层", "relation": "HAS"},
            ],
            "太阳": [
                {"concept": "太阳系", "relation": "PART_OF"},
                {"concept": "恒星", "relation": "IS_A"},
                {"concept": "光", "relation": "CAUSE"},
                {"concept": "热", "relation": "CAUSE"},
            ],
            "计算机": [
                {"concept": "CPU", "relation": "HAS"},
                {"concept": "内存", "relation": "HAS"},
                {"concept": "硬盘", "relation": "HAS"},
                {"concept": "电子设备", "relation": "IS_A"},
            ],
            "CPU": [
                {"concept": "晶体管", "relation": "HAS"},
                {"concept": "运算", "relation": "RELATED"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_mathematics(self):
        """注入数学知识种子"""
        seeds = {
            "数学": [
                {"concept": "代数", "relation": "PART_OF"},
                {"concept": "几何", "relation": "PART_OF"},
                {"concept": "微积分", "relation": "PART_OF"},
                {"concept": "概率论", "relation": "PART_OF"},
                {"concept": "数论", "relation": "PART_OF"},
            ],
            "数": [
                {"concept": "整数", "relation": "IS_A"},
                {"concept": "分数", "relation": "IS_A"},
                {"concept": "实数", "relation": "IS_A"},
                {"concept": "复数", "relation": "IS_A"},
            ],
            "几何": [
                {"concept": "三角形", "relation": "RELATED"},
                {"concept": "圆", "relation": "RELATED"},
                {"concept": "角度", "relation": "RELATED"},
            ],
            "整数": [
                {"concept": "质数", "relation": "HAS"},
                {"concept": "偶数", "relation": "HAS"},
                {"concept": "奇数", "relation": "HAS"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_chemistry(self):
        """注入化学知识种子"""
        seeds = {
            "化学": [
                {"concept": "元素", "relation": "RELATED"},
                {"concept": "化合物", "relation": "RELATED"},
                {"concept": "反应", "relation": "RELATED"},
            ],
            "元素": [
                {"concept": "氢", "relation": "HAS"},
                {"concept": "氧", "relation": "HAS"},
                {"concept": "碳", "relation": "HAS"},
                {"concept": "铁", "relation": "HAS"},
                {"concept": "金", "relation": "HAS"},
            ],
            "水": [
                {"concept": "氢", "relation": "PART_OF"},
                {"concept": "氧", "relation": "PART_OF"},
                {"concept": "液体", "relation": "IS_A"},
            ],
            "化合物": [
                {"concept": "无机物", "relation": "IS_A"},
                {"concept": "有机物", "relation": "IS_A"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_history(self):
        """注入中国历史知识种子"""
        seeds = {
            "中国历史": [
                {"concept": "秦朝", "relation": "PART_OF"},
                {"concept": "汉朝", "relation": "PART_OF"},
                {"concept": "唐朝", "relation": "PART_OF"},
                {"concept": "宋朝", "relation": "PART_OF"},
                {"concept": "明朝", "relation": "PART_OF"},
                {"concept": "清朝", "relation": "PART_OF"},
            ],
            "秦朝": [
                {"concept": "秦始皇", "relation": "RELATED"},
                {"concept": "长城", "relation": "RELATED"},
                {"concept": "统一", "relation": "RELATED"},
            ],
            "唐朝": [
                {"concept": "李白", "relation": "RELATED"},
                {"concept": "杜甫", "relation": "RELATED"},
                {"concept": "唐诗", "relation": "RELATED"},
            ],
            "宋朝": [
                {"concept": "苏轼", "relation": "RELATED"},
                {"concept": "宋词", "relation": "RELATED"},
                {"concept": "活字印刷", "relation": "RELATED"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_literature(self):
        """注入文学知识种子"""
        seeds = {
            "文学": [
                {"concept": "诗歌", "relation": "PART_OF"},
                {"concept": "小说", "relation": "PART_OF"},
                {"concept": "散文", "relation": "PART_OF"},
                {"concept": "戏剧", "relation": "PART_OF"},
            ],
            "四大名著": [
                {"concept": "红楼梦", "relation": "PART_OF"},
                {"concept": "西游记", "relation": "PART_OF"},
                {"concept": "三国演义", "relation": "PART_OF"},
                {"concept": "水浒传", "relation": "PART_OF"},
            ],
            "李白": [
                {"concept": "诗仙", "relation": "IS_A"},
                {"concept": "唐诗", "relation": "RELATED"},
            ],
            "杜甫": [
                {"concept": "诗圣", "relation": "IS_A"},
                {"concept": "唐诗", "relation": "RELATED"},
            ],
            "苏轼": [
                {"concept": "宋词", "relation": "RELATED"},
                {"concept": "豪放派", "relation": "IS_A"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_philosophy(self):
        """注入哲学知识种子"""
        seeds = {
            "哲学": [
                {"concept": "唯物主义", "relation": "PART_OF"},
                {"concept": "唯心主义", "relation": "PART_OF"},
                {"concept": "逻辑学", "relation": "PART_OF"},
                {"concept": "伦理学", "relation": "PART_OF"},
            ],
            "中国哲学": [
                {"concept": "儒家", "relation": "PART_OF"},
                {"concept": "道家", "relation": "PART_OF"},
                {"concept": "法家", "relation": "PART_OF"},
                {"concept": "墨家", "relation": "PART_OF"},
            ],
            "儒家": [
                {"concept": "孔子", "relation": "RELATED"},
                {"concept": "孟子", "relation": "RELATED"},
                {"concept": "仁义", "relation": "RELATED"},
            ],
            "道家": [
                {"concept": "老子", "relation": "RELATED"},
                {"concept": "庄子", "relation": "RELATED"},
                {"concept": "道德经", "relation": "RELATED"},
            ],
            "辩证法": [
                {"concept": "矛盾", "relation": "RELATED"},
                {"concept": "对立统一", "relation": "RELATED"},
            ],
        }
        self.seed_from_dict(seeds)
        return len(seeds)

    def seed_all_domains(self) -> Dict[str, int]:
        """注入全部学科种子，返回每领域注入数"""
        domains = {
            '基础科学': self.seed_basic_science,
            '数学':     self.seed_mathematics,
            '化学':     self.seed_chemistry,
            '历史':     self.seed_history,
            '文学':     self.seed_literature,
            '哲学':     self.seed_philosophy,
        }
        counts = {}
        for name, fn in domains.items():
            counts[name] = fn()
        return counts

    # ═══════════════════════════════════════════════════════════════════════
    # 自评估
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate(self) -> Dict[str, Any]:
        """
        自评估：衡量概念图的质量。

        指标:
          - coverage:      每个概念的平均连接数
          - consistency:   无矛盾的比例
          - connectivity:  可到达节点比例（从任意起点能通过多跳到达的比例）
          - inferred_ratio: 推断三元组的比例（越低图越"原生"）
          - avg_confidence: 平均置信度
        """
        stats = self.stats()

        # coverage: 平均度
        total_degree = sum(
            len(self.forward_index.get(n, {})) + len(self.reverse_index.get(n, {}))
            for n in self.nodes
        )
        avg_degree = total_degree / max(1, len(self.nodes))

        # consistency: 无矛盾比例
        conflicts = self.detect_contradictions()
        consistency = 1.0 if stats['triples'] == 0 else \
            1.0 - min(1.0, len(conflicts) / max(1, stats['triples']) * 10)

        # connectivity: BFS 可达率
        if len(self.nodes) <= 1:
            connectivity = 1.0
        else:
            start = next(iter(self.nodes))
            visited = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in self.forward_index.get(node, {}):
                    if neighbor not in visited:
                        queue.append(neighbor)
                for neighbor in self.reverse_index.get(node, {}):
                    if neighbor not in visited:
                        queue.append(neighbor)
            connectivity = len(visited) / len(self.nodes)

        # avg_confidence
        avg_confidence = (
            sum(t.confidence for t in self.triples.values()) / max(1, len(self.triples))
        )

        report = {
            'nodes': len(self.nodes),
            'triples': stats['triples'],
            'inferred': self.total_inferred,
            'inferred_ratio': self.total_inferred / max(1, stats['triples']),
            'avg_degree': round(avg_degree, 2),
            'consistency': round(consistency, 3),
            'connectivity': round(connectivity, 3),
            'avg_confidence': round(avg_confidence, 3),
            'conflicts_found': len(conflicts),
            'relations': stats['relations'],
        }
        return report

    def suggest_expansions(
        self,
        min_degree: int = 1,
        max_suggestions: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        建议需要扩展的概念。

        查找连接稀疏的概念（需要更多关联），以及孤立概念。

        Returns:
            [{'concept': ..., 'degree': ..., 'reason': ...}, ...]
        """
        suggestions = []

        for concept in self.nodes:
            out_deg = len(self.forward_index.get(concept, {}))
            in_deg = len(self.reverse_index.get(concept, {}))
            total = out_deg + in_deg

            if total <= min_degree:
                reason = "孤立概念" if total == 0 else f"仅{total}条关联"
                suggestions.append({
                    'concept': concept,
                    'degree': total,
                    'out_degree': out_deg,
                    'in_degree': in_deg,
                    'reason': reason,
                })

        # 按度排序，最少连接的优先
        suggestions.sort(key=lambda x: x['degree'])
        return suggestions[:max_suggestions]


    def prune(
        self,
        min_confidence: float = 0.15,
        min_evidence: int = 0,
        source_keep: set = None,
    ) -> int:
        """
        剪除低质量三元组。

        删除同时满足以下条件的三元组:
          - 置信度 < min_confidence
          - 证据数 <= min_evidence
          - 来源不在 source_keep 中

        Args:
            min_confidence: 最低置信度阈值
            min_evidence: 最低证据数
            source_keep: 受保护的来源集合 (默认: {"seed", "manual"})

        Returns:
            删除的三元组数量
        """
        if source_keep is None:
            source_keep = {"seed", "manual"}

        to_remove = []
        for key, triple in self.triples.items():
            if triple.source in source_keep:
                continue
            if triple.confidence < min_confidence and triple.evidence_count <= min_evidence:
                to_remove.append(key)

        for key in to_remove:
            triple = self.triples.pop(key)
            # 从正向索引移除
            if triple.object in self.forward_index.get(triple.subject, {}):
                del self.forward_index[triple.subject][triple.object]
            # 从反向索引移除
            if triple.subject in self.reverse_index.get(triple.object, {}):
                del self.reverse_index[triple.object][triple.subject]
            self.total_triples -= 1
            if triple.source == "infer":
                self.total_inferred -= 1

        return len(to_remove)

    def align_to_landscape(
        self,
        learner=None,
        min_confidence: float = 0.6,
        max_pairs: int = 200,
        learning_rate: float = 0.02,
    ) -> List[Tuple[int, int]]:
        """
        ★ 写入权归大脑: 不再直接调用 learner.learn_pairs_batch。
        改为返回高置信度三元组的字对列表, orchestrator 统一注入。

        Args:
            learner: (保留参数兼容, 不再使用)
            min_confidence: 最低置信度
            max_pairs: 最大字对数
            learning_rate: (保留参数兼容)

        Returns:
            [(ia, ib), ...] 字对索引列表
        """
        if self.landscape is None:
            return []

        pairs = []
        for triple in self.triples.values():
            if triple.confidence < min_confidence:
                continue
            chars_a = list(triple.subject)
            chars_b = list(triple.object)
            for ca in chars_a[:2]:
                for cb in chars_b[:2]:
                    ia = getattr(self.zichang, '_char_to_idx', {}).get(ca)
                    ib = getattr(self.zichang, '_char_to_idx', {}).get(cb)
                    if ia is not None and ib is not None:
                        pairs.append((ia, ib))

        return pairs[:max_pairs]

    # ═══════════════════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════════════════


    def stats(self) -> Dict:
        """获取当前统计信息"""
        rel_counts = {}
        for t in self.triples.values():
            rel_counts[t.relation] = rel_counts.get(t.relation, 0) + 1

        return {
            'nodes': len(self.nodes),
            'triples': self.total_triples,
            'relations': rel_counts,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════════════

    def save(self, path: str, save_embeds: bool = False):
        """
        保存概念图。

        生成:
          path.json       — 图结构（含置信度/来源等元数据）
          path_embeds.pt  — 节点嵌入向量（默认不保存, 可重算, 1.9GB）

        Args:
            save_embeds: 是否保存嵌入。默认False——嵌入在内存中从字场现场计算更快。
        """
        base = path.replace('.json', '').replace('.pt', '')
        os.makedirs(os.path.dirname(base) if os.path.dirname(base) else '.', exist_ok=True)

        # 嵌入向量 (默认跳过——可从字场重算)
        if save_embeds:
            embeds = {k: v for k, v in self.nodes.items()}
            torch.save(embeds, base + '_embeds.pt')

        # 保存图结构
        triples_data = []
        for t in self.triples.values():
            triples_data.append({
                's': t.subject,
                'r': t.relation,
                'o': t.object,
                'c': round(t.confidence, 4),
                'src': t.source,
                'ev': t.evidence_count,
            })

        graph_data = {
            'triples': triples_data,
            'total_triples': self.total_triples,
            'total_inferred': self.total_inferred,
            'aliases': self.node_aliases,
        }

        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        self._data_dir = os.path.dirname(base) if os.path.dirname(base) else '.'
        self._dirty_since_last_save = False
        print(f"概念图已保存: {base}.json" + (" + {base}_embeds.pt" if save_embeds else ""))

    def load(self, path: str):
        """
        加载概念图。

        Args:
            path: 可以是 .json 或 .pt 路径（自动找配对文件）
        """
        base = path.replace('.json', '').replace('.pt', '').replace('_embeds', '')

        # 加载嵌入
        embeds_path = base + '_embeds.pt'
        if os.path.exists(embeds_path):
            data = torch.load(embeds_path, map_location='cpu')
            self.nodes = {k: v.float() for k, v in data.items()}
        else:
            self.nodes = {}

        # 加载图结构
        json_path = base + '.json'
        if os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                graph_data = json.load(f)

            self.triples.clear()
            self.forward_index.clear()
            self.reverse_index.clear()
            self._char_adjacency.clear()
            self.total_triples = 0
            self.total_inferred = graph_data.get('total_inferred', 0)
            self.node_aliases = graph_data.get('aliases', {})

            for td in graph_data.get('triples', []):
                self.add_triple(
                    td['s'], td['r'], td['o'],
                    confidence=td.get('c', 0.5),
                    source=td.get('src', 'unknown'),
                    evidence_count=td.get('ev', 1),
                )

        self._data_dir = os.path.dirname(base) if os.path.dirname(base) else '.'
        self._dirty_since_last_save = False
        print(f"概念图已加载: {len(self.nodes)}节点 {self.total_triples}三元组")

    @classmethod
    def create_with_seeds(cls, field, landscape=None, all_domains: bool = True) -> "ConceptGraph":
        """
        工厂方法：创建概念图并注入全部种子知识。

        Args:
            field: HanziAnchorField 字场
            landscape: FreqEnergyLandscape 能量景观
            all_domains: True=全部学科, False=仅基础科学

        Returns:
            已注入种子的 ConceptGraph 实例
        """
        cg = cls(field, landscape)
        if all_domains:
            cg.seed_all_domains()
        else:
            cg.seed_basic_science()
        return cg


# ============================================================================
# 演示
# ============================================================================

def demo_concept_graph(zichang, landscape=None):
    """演示完整概念图能力"""
    cg = ConceptGraph(zichang, landscape)

    print("=" * 60)
    print("🧠 龙珠概念图 — 完整演示")
    print("=" * 60)

    # 1. 种子注入
    print("\n📥 [1/7] 注入多学科种子知识...")
    counts = cg.seed_all_domains()
    for domain, n in counts.items():
        print(f"   {domain}: {n} 个概念集")
    print(f"   总计: {cg.stats()['nodes']} 节点, {cg.stats()['triples']} 三元组")

    # 2. 知识提取
    print("\n🔍 [2/7] 从文本提取知识...")
    demo_texts = [
        "原子由质子和中子组成，电子围绕原子核运动。原子是物质的基本单位。",
        "水分子由两个氢原子和一个氧原子组成。水是一种化合物。",
        "CPU包含运算器和控制器，内存用于存储数据。计算机是一种电子设备。",
    ]
    all_new = []
    for text in demo_texts:
        new = cg.extract_triples(text)
        all_new.extend(new)
        if new:
            for t in new[:3]:
                print(f"   ✅ {t.subject} {t.relation} {t.object} (conf={t.confidence:.2f})")
    print(f"   提取了 {len(all_new)} 个新三元组")

    # 3. 归纳推理
    print("\n🧩 [3/7] 归纳推理（传递闭包）...")
    inferred = cg.induce(max_chain=3)
    for t in inferred[:5]:
        print(f"   💡 [推断] {t.subject} {t.relation} {t.object} (conf={t.confidence:.2f})")
    if len(inferred) > 5:
        print(f"   ... 共推断 {len(inferred)} 条")

    # 4. 矛盾检测
    print("\n⚠️  [4/7] 矛盾检测...")
    conflicts = cg.detect_contradictions()
    if conflicts:
        for c in conflicts[:3]:
            print(f"   {c['message']}")
    else:
        print("   ✅ 未发现矛盾")

    # 5. 多跳推理
    print("\n🔗 [5/7] 多跳推理: '电子' → 双向3跳")
    paths = cg.reason("电子", max_hops=3, direction="both")
    for p in paths[:5]:
        energy = cg._path_energy(p)
        print(f"   {' → '.join(p)} (能:{energy:.1f})")

    # 6. 查询
    print("\n📋 [6/7] 查询: '原子' 的所有关联")
    for r in cg.query("原子", max_results=8):
        arrow = "→" if r['direction'] == 'forward' else "←"
        print(f"   原子 {arrow} [{r['relation']}] {r['concept']} "
              f"(能:{r['energy']:.1f} 信:{r['confidence']:.2f})")

    # 7. 自评估
    print("\n📊 [7/7] 自评估报告")
    report = cg.evaluate()
    print(f"   节点: {report['nodes']}")
    print(f"   三元组: {report['triples']} (推断: {report['inferred']})")
    print(f"   平均度: {report['avg_degree']}")
    print(f"   一致性: {report['consistency']}")
    print(f"   连通性: {report['connectivity']}")
    print(f"   均值信: {report['avg_confidence']}")
    print(f"   关系分布: {report['relations']}")

    # 扩展建议
    suggestions = cg.suggest_expansions(min_degree=1, max_suggestions=5)
    if suggestions:
        print(f"\n💡 需扩展概念 (前5):")
        for s in suggestions:
            print(f"   {s['concept']} (度:{s['degree']}) — {s['reason']}")

    print(f"\n{'=' * 60}")
    print("✅ 概念图完整演示结束")
    print("=" * 60)

    return cg


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from loongpearl.core.zichang import HanziAnchorField

    PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    field_path = os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt')

    if os.path.exists(field_path):
        field = HanziAnchorField.load(field_path, freeze=True)
        demo_concept_graph(field)
    else:
        print(f"⚠️ 字场模型不存在: {field_path}")
        print("请先构建字场或运行: python scripts/download_models.sh")
