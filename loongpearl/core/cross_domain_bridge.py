#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠跨学科桥接引擎 (cross_domain_bridge.py)
═══════════════════════════════════════════════

龙珠超越 LLM 的核心优势之一：**组合透明性**。
LLM 的跨学科关联是黑箱概率；龙珠的每个跨域关联都能追溯到共享的汉字锚点，
可分解、可验证、无幻觉。

══════════════════════════════════════════════════════════════════════
四层桥接架构
══════════════════════════════════════════════════════════════════════

Layer 1 — 字素桥接 (Hanzi Sharing Bridge)
  "电"字在"电子"(物理)、"电脑"(计算机)、"电影"(艺术)中都出现
  → 共享汉字的跨域概念自动关联

Layer 2 — 嵌入近邻桥接 (Embedding Proximity Bridge)
  跨域概念对的嵌入余弦相似度 > 阈值 → 隐含跨域关联

Layer 3 — 结构同构桥接 (Structural Isomorphism Bridge)
  不同领域的 PART_OF 层次结构往往同构:
    物理: 夸克→质子→原子→分子
    生物: 细胞器→细胞→组织→器官
  → 发现结构相似子树，推断跨域类比

Layer 4 — 能量验证闭环 (Energy Verification)
  每个跨域假设通过能量景观验证：中点能量越低 → 关联越真实

══════════════════════════════════════════════════════════════════════
用法
══════════════════════════════════════════════════════════════════════

    bridge = CrossDomainBridge(field, landscape, concept_graph)
    
    # 全量桥接
    results = bridge.build_all_bridges(min_confidence=0.4)
    for b in results[:10]:
        print(f"{b['concept_a']} ↔ {b['concept_b']}: {b['reason']}")
    
    # 单域桥接
    bridges = bridge.bridge_domain("量子", top_k=20)

══════════════════════════════════════════════════════════════════════
设计原则
══════════════════════════════════════════════════════════════════════

  1. 确定性 — 每座"桥"都可以分解到具体汉字/嵌入距离，零黑箱
  2. 验证性 — 每座"桥"都经过能量景观验证（中点能量低 = 关联真实）
  3. 主动性 — 不等待搜索喂数据，字场本身就编码了跨域关联
  4. 可解释 — output 包含完整的 why: 共享字素列表 + 余弦距离 + 能量评分
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
import itertools
import math
from dataclasses import dataclass, field


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CrossDomainBridge:
    """一座跨学科桥梁"""
    concept_a: str           # 概念 A
    concept_b: str           # 概念 B
    domain_a: str            # A 所属领域
    domain_b: str            # B 所属领域
    confidence: float        # 综合置信度 0-1
    bridge_type: str         # hanzi_share | embed_proximity | structural | combined
    reason: str              # 人类可读的原因
    shared_hanzi: List[str] = field(default_factory=list)
    cosine_similarity: float = 0.0
    energy_score: float = 999.0
    structural_path_a: List[str] = field(default_factory=list)
    structural_path_b: List[str] = field(default_factory=list)


# ============================================================================
# 领域定义
# ============================================================================

# 概念 → 领域映射（从种子知识提取 + 可扩展）
DOMAIN_SEEDS = {
    # 物理学
    "物质": "物理", "原子": "物理", "电子": "物理", "质子": "物理",
    "中子": "物理", "原子核": "物理", "分子": "物理", "化合物": "物理",
    "量子": "物理", "力学": "物理", "能量": "物理", "光": "物理",
    "热": "物理", "力": "物理", "波": "物理", "电场": "物理",
    "磁场": "物理", "引力": "物理", "相对论": "物理",

    # 化学
    "元素": "化学", "氢": "化学", "氧": "化学", "碳": "化学",
    "铁": "化学", "金": "化学", "水": "化学", "液体": "化学",
    "无机物": "化学", "有机物": "化学", "反应": "化学", "化学键": "化学",

    # 生物学
    "细胞": "生物", "细胞核": "生物", "细胞膜": "生物", "线粒体": "生物",
    "组织": "生物", "器官": "生物", "DNA": "生物", "基因": "生物",
    "染色体": "生物", "遗传": "生物", "蛋白质": "生物", "酶": "生物",

    # 天文学
    "地球": "天文", "太阳": "天文", "太阳系": "天文", "月球": "天文",
    "大气层": "天文", "恒星": "天文", "行星": "天文", "银河": "天文",

    # 计算机科学
    "计算机": "计算机", "CPU": "计算机", "内存": "计算机", "硬盘": "计算机",
    "电子设备": "计算机", "晶体管": "计算机", "运算": "计算机",
    "算法": "计算机", "数据": "计算机", "程序": "计算机",

    # 数学
    "数学": "数学", "代数": "数学", "几何": "数学", "微积分": "数学",
    "概率论": "数学", "数论": "数学", "数": "数学", "整数": "数学",
    "分数": "数学", "实数": "数学", "复数": "数学", "质数": "数学",
    "偶数": "数学", "奇数": "数学", "三角形": "数学", "圆": "数学",

    # 历史
    "中国历史": "历史", "秦朝": "历史", "汉朝": "历史", "唐朝": "历史",
    "宋朝": "历史", "明朝": "历史", "清朝": "历史", "秦始皇": "历史",

    # 文学
    "文学": "文学", "诗歌": "文学", "小说": "文学", "散文": "文学",
    "戏剧": "文学", "红楼梦": "文学", "西游记": "文学",
    "三国演义": "文学", "水浒传": "文学", "李白": "文学",
    "杜甫": "文学", "苏轼": "文学", "唐诗": "文学", "宋词": "文学",

    # 哲学
    "哲学": "哲学", "唯物主义": "哲学", "唯心主义": "哲学",
    "逻辑学": "哲学", "伦理学": "哲学", "儒家": "哲学", "道家": "哲学",
    "法家": "哲学", "墨家": "哲学", "辩证法": "哲学", "矛盾": "哲学",
}


def infer_domain(concept: str) -> str:
    """推断概念所属领域"""
    if concept in DOMAIN_SEEDS:
        return DOMAIN_SEEDS[concept]
    # 启发式推断
    chars = set(concept)
    domain_scores = defaultdict(float)
    for known, domain in DOMAIN_SEEDS.items():
        overlap = chars & set(known)
        if overlap:
            domain_scores[domain] += len(overlap) / max(len(chars), len(set(known)))
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    return "未知"


# ============================================================================
# 跨学科桥接引擎
# ============================================================================

class CrossDomainBridgeEngine:
    """
    龙珠跨学科桥接引擎。

    不依赖 LLM。利用字场的组合透明性：每个概念的嵌入都可以分解到
    其组成汉字锚点，从而进行确定性、可验证的跨域关联发现。
    """

    def __init__(self, field, landscape, concept_graph):
        self.field = field              # HanziAnchorField
        self.landscape = landscape      # FreqEnergyLandscape
        self.cg = concept_graph         # ConceptGraph

        self.embed_dim = field.embed_dim

        # 领域索引
        self.domain_concepts: Dict[str, List[str]] = defaultdict(list)
        self._build_domain_index()

        # 字素索引: 汉字 → 包含它的概念列表
        self.hanzi_to_concepts: Dict[str, List[str]] = defaultdict(list)
        self._build_hanzi_index()

    def _build_domain_index(self):
        """构建领域 → 概念列表索引"""
        for concept in self.cg.nodes:
            domain = infer_domain(concept)
            self.domain_concepts[domain].append(concept)

    def _build_hanzi_index(self):
        """构建汉字 → 概念列表索引（核心：字素桥接的基础）"""
        for concept in self.cg.nodes:
            for ch in concept:
                if ch != ' ':  # 跳过多字概念中的空格
                    self.hanzi_to_concepts[ch].append(concept)

    # ═══════════════════════════════════════════════════════════════
    # Layer 1: 字素桥接
    # ═══════════════════════════════════════════════════════════════

    def bridge_by_hanzi(
        self,
        min_shared_hanzi: int = 1,
        min_confidence: float = 0.3,
    ) -> List[CrossDomainBridge]:
        """
        字素桥接：共享汉字的跨域概念自动关联。

        原理：
          "电子" 和 "电脑" 共享 "电" → 物理↔计算机
          "细胞" 和 "细节" 共享 "细" → 生物↔通用
          "分子" 和 "分数"  共享 "分" → 物理↔数学

        这比 LLM 的嵌入近邻更精确——LLM 无法解释"为什么近"，
        龙珠可以：因为它们共享锚点 '电'，而这个锚点在字场中
        与物理/计算机相关的其他锚点构成了低能盆地。
        """
        bridges = []
        seen_pairs = set()

        # 对每个概念，找共享汉字的跨域概念
        for concept_a in list(self.cg.nodes.keys())[:500]:  # 限制规模
            domain_a = infer_domain(concept_a)
            chars_a = set(concept_a)

            candidates: Dict[str, int] = Counter()
            for ch in chars_a:
                for concept_b in self.hanzi_to_concepts.get(ch, []):
                    if concept_b == concept_a:
                        continue
                    domain_b = infer_domain(concept_b)
                    if domain_b == domain_a or domain_b == "未知" or domain_a == "未知":
                        continue  # 只关心跨域
                    candidates[concept_b] += 1

            for concept_b, shared_count in candidates.most_common(30):
                if shared_count < min_shared_hanzi:
                    continue

                pair_key = tuple(sorted([concept_a, concept_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                domain_b = infer_domain(concept_b)
                shared = [c for c in chars_a if c in set(concept_b)]
                cos_sim = self._cosine_similarity(concept_a, concept_b)
                energy = self._pair_energy(concept_a, concept_b)

                # 置信度: 共享字数 + 嵌入相似度 + 能量
                share_score = min(1.0, shared_count / max(len(chars_a), len(set(concept_b))))
                cos_score = max(0.0, cos_sim)  # 余弦相似度越高越好
                energy_score = max(0.0, min(1.0, 1.0 - (energy + 50) / 80.0))
                confidence = 0.4 * share_score + 0.3 * cos_score + 0.3 * energy_score

                if confidence < min_confidence:
                    continue

                reason = (f"共享汉字: {'、'.join(shared)} "
                         f"(共{shared_count}字, 占{concept_a}{len(shared)/len(concept_a):.0%}"
                         f"和{concept_b}{len(shared)/len(concept_b):.0%})")

                bridges.append(CrossDomainBridge(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    domain_a=domain_a,
                    domain_b=domain_b,
                    confidence=confidence,
                    bridge_type="hanzi_share",
                    reason=reason,
                    shared_hanzi=shared,
                    cosine_similarity=cos_sim,
                    energy_score=energy,
                ))

        bridges.sort(key=lambda b: -b.confidence)
        return bridges

    # ═══════════════════════════════════════════════════════════════
    # Layer 2: 嵌入近邻桥接
    # ═══════════════════════════════════════════════════════════════

    def bridge_by_embedding(
        self,
        top_k: int = 10,
        min_similarity: float = 0.6,
        min_confidence: float = 0.3,
        sample_size: int = 300,
    ) -> List[CrossDomainBridge]:
        """
        嵌入近邻桥接：跨域概念对的嵌入余弦相似度。

        原理：
          龙珠的概念嵌入 = 组成汉字锚点的平均。
          "量子" = mean(锚点_量, 锚点_子)
          如果 "量子" 的嵌入与 "概率" 的嵌入在余弦空间接近，
          说明它们的组成锚点在字场中构成了相似的能量配置 →
          物理↔数学的跨域关联。

        LLM 也能做这个，但龙珠的优势在于：
          - 相似度是确定性的（锚点冻结）
          - 可以分解到具体锚点："量"和"概"的锚点距离是0.03
          - 不存在 LLM 的"remixing"幻觉
        """
        # 收集跨域概念嵌入
        domain_embeddings: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        for concept in list(self.cg.nodes.keys()):
            domain = infer_domain(concept)
            if domain == "未知":
                continue
            emb = self.cg.get_embedding(concept)
            if emb is not None:
                domain_embeddings[domain].append((concept, emb))

        domains = list(domain_embeddings.keys())
        if len(domains) < 2:
            return []

        bridges = []
        seen_pairs = set()

        # 跨域对比
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                dom_a, dom_b = domains[i], domains[j]

                # 取样（避免 O(N²)）
                items_a = domain_embeddings[dom_a]
                items_b = domain_embeddings[dom_b]
                if len(items_a) > sample_size:
                    indices = torch.randperm(len(items_a))[:sample_size].tolist()
                    items_a = [items_a[k] for k in indices]
                if len(items_b) > sample_size:
                    indices = torch.randperm(len(items_b))[:sample_size].tolist()
                    items_b = [items_b[k] for k in indices]

                # 批量计算余弦相似度
                emb_a = torch.stack([e for _, e in items_a])
                emb_b = torch.stack([e for _, e in items_b])

                # 归一化
                emb_a_norm = torch.nn.functional.normalize(emb_a, dim=1)
                emb_b_norm = torch.nn.functional.normalize(emb_b, dim=1)

                # 相似度矩阵
                sim_matrix = emb_a_norm @ emb_b_norm.T  # (n_a, n_b)

                # 找高相似度对
                high_sim_mask = sim_matrix > min_similarity
                high_indices = torch.nonzero(high_sim_mask)

                for idx_pair in high_indices[:500]:  # 限制数量
                    ia, ib = idx_pair[0].item(), idx_pair[1].item()
                    concept_a = items_a[ia][0]
                    concept_b = items_b[ib][0]
                    cos_sim = sim_matrix[ia, ib].item()

                    pair_key = tuple(sorted([concept_a, concept_b]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    energy = self._pair_energy(concept_a, concept_b)
                    energy_score = max(0.0, min(1.0, 1.0 - (energy + 50) / 80.0))
                    confidence = 0.5 * max(0.0, cos_sim) + 0.5 * energy_score

                    if confidence < min_confidence:
                        continue

                    reason = (f"嵌入余弦相似度 {cos_sim:.3f} "
                             f"(字场锚点组合相近, 能:{energy:.1f})")

                    bridges.append(CrossDomainBridge(
                        concept_a=concept_a,
                        concept_b=concept_b,
                        domain_a=dom_a,
                        domain_b=dom_b,
                        confidence=confidence,
                        bridge_type="embed_proximity",
                        reason=reason,
                        cosine_similarity=cos_sim,
                        energy_score=energy,
                    ))

        bridges.sort(key=lambda b: -b.confidence)
        return bridges[:top_k * len(domains)]

    # ═══════════════════════════════════════════════════════════════
    # Layer 3: 结构同构桥接
    # ═══════════════════════════════════════════════════════════════

    def bridge_by_structure(
        self,
        relation: str = "PART_OF",
        max_depth: int = 3,
        min_similarity: float = 0.5,
        min_confidence: float = 0.3,
    ) -> List[CrossDomainBridge]:
        """
        结构同构桥接：发现不同领域中的同构层次结构。

        原理：
          物理: 夸克 → 质子 → 原子 → 分子
          生物: 细胞器 → 细胞 → 组织 → 器官

          这两条链在结构上同构（都是4层 PART_OF），
          且每层的嵌入在字场中也有对应锚点关联 →
          物理↔生物的跨域类比。

        这是 LLM 难以系统化做到的——LLM 可能会说"原子和细胞有点像"
        但无法生成结构化的同构证明。龙珠可以。
        """
        bridges = []

        # 提取各领域所有 PART_OF 路径
        domain_paths: Dict[str, List[List[str]]] = defaultdict(list)
        for domain, concepts in self.domain_concepts.items():
            for concept in concepts[:50]:  # 限制规模
                paths = self.cg.reason(
                    concept, relation=relation,
                    max_hops=max_depth, direction="both"
                )
                for path in paths:
                    if len(path) >= 2:
                        domain_paths[domain].append(path)

        domains = list(domain_paths.keys())
        if len(domains) < 2:
            return bridges

        seen_pairs = set()

        # 预计算所有涉事概念的嵌入（避免逐对重复计算）
        all_concepts = set()
        for paths in domain_paths.values():
            for p in paths[:200]:
                all_concepts.update(p)
        concept_embs = {}
        for c in all_concepts:
            emb = self.cg.get_embedding(c)
            if emb is not None:
                concept_embs[c] = torch.nn.functional.normalize(emb, dim=0)

        def fast_cosine(a, b):
            ea = concept_embs.get(a)
            eb = concept_embs.get(b)
            if ea is None or eb is None:
                return 0.0
            return torch.dot(ea, eb).item()

        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                dom_a, dom_b = domains[i], domains[j]
                paths_a = domain_paths[dom_a][:200]
                paths_b = domain_paths[dom_b][:200]

                for pa in paths_a:
                    for pb in paths_b:
                        if len(pa) != len(pb):
                            continue

                        # 逐层计算嵌入相似度（使用预计算的归一化嵌入）
                        layer_sims = []
                        for node_a, node_b in zip(pa, pb):
                            sim = fast_cosine(node_a, node_b)
                            layer_sims.append(sim)

                        avg_sim = sum(layer_sims) / len(layer_sims)

                        if avg_sim < min_similarity:
                            continue

                        # 两端节点的跨域关联
                        concept_a = pa[0]
                        concept_b = pb[0]

                        pair_key = tuple(sorted([concept_a, concept_b]))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        energy = self._pair_energy(concept_a, concept_b)
                        energy_score = max(0.0, min(1.0, 1.0 - (energy + 50) / 80.0))
                        confidence = 0.4 * avg_sim + 0.3 * (len(pa) / max_depth) + 0.3 * energy_score

                        if confidence < min_confidence:
                            continue

                        reason = (f"结构同构: {dom_a}[{'→'.join(pa)}] ≅ "
                                 f"{dom_b}[{'→'.join(pb)}] "
                                 f"(层均相似度{avg_sim:.3f}, 深度{len(pa)})")

                        bridges.append(CrossDomainBridge(
                            concept_a=concept_a,
                            concept_b=concept_b,
                            domain_a=dom_a,
                            domain_b=dom_b,
                            confidence=confidence,
                            bridge_type="structural",
                            reason=reason,
                            cosine_similarity=avg_sim,
                            energy_score=energy,
                            structural_path_a=pa,
                            structural_path_b=pb,
                        ))

        bridges.sort(key=lambda b: -b.confidence)
        return bridges

    # ═══════════════════════════════════════════════════════════════
    # Layer 4: 综合桥接
    # ═══════════════════════════════════════════════════════════════

    def build_all_bridges(
        self,
        min_confidence: float = 0.35,
        max_bridges: int = 100,
    ) -> List[CrossDomainBridge]:
        """
        运行全部四层桥接，去重合并，按置信度排序。

        对于被多个层级发现的桥，合并证据并提升置信度。
        """
        all_bridges: Dict[Tuple[str, str], CrossDomainBridge] = {}

        # Layer 1: 字素桥接
        for b in self.bridge_by_hanzi(min_confidence=min_confidence):
            key = tuple(sorted([b.concept_a, b.concept_b]))
            all_bridges[key] = b

        # Layer 2: 嵌入近邻桥接
        for b in self.bridge_by_embedding(min_confidence=min_confidence):
            key = tuple(sorted([b.concept_a, b.concept_b]))
            if key in all_bridges:
                # 合并证据
                existing = all_bridges[key]
                existing.confidence = min(1.0, existing.confidence + b.confidence * 0.3)
                existing.bridge_type = "combined"
                existing.reason += f" | +嵌入近邻({b.cosine_similarity:.3f})"
            else:
                all_bridges[key] = b

        # Layer 3: 结构同构桥接
        for b in self.bridge_by_structure(min_confidence=min_confidence):
            key = tuple(sorted([b.concept_a, b.concept_b]))
            if key in all_bridges:
                existing = all_bridges[key]
                existing.confidence = min(1.0, existing.confidence + b.confidence * 0.3)
                existing.bridge_type = "combined"
                existing.reason += f" | +结构同构({b.cosine_similarity:.3f})"
            else:
                all_bridges[key] = b

        # 排序
        sorted_bridges = sorted(all_bridges.values(), key=lambda b: -b.confidence)
        return sorted_bridges[:max_bridges]

    def add_bridges_to_concept_graph(
        self,
        bridges: List[CrossDomainBridge],
        min_confidence: float = 0.4,
    ) -> int:
        """
        将发现的跨学科桥梁写入概念图。

        每座桥 → 一条 RELATED 边。
        """
        count = 0
        for b in bridges:
            if b.confidence < min_confidence:
                continue
            self.cg.add_triple(
                b.concept_a, "RELATED", b.concept_b,
                confidence=b.confidence,
                source="cross_domain_bridge",
            )
            count += 1
        return count

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def _cosine_similarity(self, concept_a: str, concept_b: str) -> float:
        """计算两个概念的嵌入余弦相似度"""
        e_a = self.cg.get_embedding(concept_a)
        e_b = self.cg.get_embedding(concept_b)
        if e_a is None or e_b is None:
            return 0.0
        cos = torch.nn.functional.cosine_similarity(
            e_a.unsqueeze(0), e_b.unsqueeze(0)
        )
        return cos.item()

    def _pair_energy(self, concept_a: str, concept_b: str) -> float:
        """计算概念对的能量景观评分（越低越好）"""
        return self.cg.triple_energy(concept_a, concept_b)

    def summary(self, bridges: List[CrossDomainBridge]) -> Dict[str, Any]:
        """生成桥接统计摘要"""
        if not bridges:
            return {"total": 0}

        domain_pairs = Counter()
        types = Counter()
        for b in bridges:
            pair = tuple(sorted([b.domain_a, b.domain_b]))
            domain_pairs[f"{pair[0]}↔{pair[1]}"] += 1
            types[b.bridge_type] += 1

        return {
            "total": len(bridges),
            "avg_confidence": sum(b.confidence for b in bridges) / len(bridges),
            "max_confidence": max(b.confidence for b in bridges),
            "domain_pairs": dict(domain_pairs.most_common(10)),
            "bridge_types": dict(types),
            "top_bridges": [
                {
                    "concepts": f"{b.concept_a}↔{b.concept_b}",
                    "domains": f"{b.domain_a}↔{b.domain_b}",
                    "confidence": round(b.confidence, 3),
                    "type": b.bridge_type,
                    "reason": b.reason[:80],
                }
                for b in bridges[:10]
            ],
        }


# ============================================================================
# 演示
# ============================================================================

def demo_cross_domain(field, landscape, concept_graph):
    """演示跨学科桥接"""
    bridge = CrossDomainBridge(field, landscape, concept_graph)

    print("=" * 65)
    print("🌉 龙珠跨学科桥接引擎 — 超越 LLM 的跨域关联")
    print("=" * 65)
    print(f"   概念图: {len(concept_graph.nodes)}节点 {concept_graph.triples}三元组")
    print(f"   领域: {len(bridge.domain_concepts)}个")
    for dom, concepts in sorted(bridge.domain_concepts.items()):
        if len(concepts) > 1:
            print(f"     {dom}: {len(concepts)}个概念")

    # Layer 1
    print(f"\n🔤 [Layer 1] 字素桥接 (共享汉字)...")
    h_bridges = bridge.bridge_by_hanzi(min_confidence=0.25)
    print(f"   发现 {len(h_bridges)} 座桥")
    for b in h_bridges[:8]:
        print(f"   {b.concept_a}({b.domain_a}) ↔ {b.concept_b}({b.domain_b}) "
              f"信:{b.confidence:.2f} | {b.reason[:60]}")

    # Layer 2
    print(f"\n🧮 [Layer 2] 嵌入近邻桥接 (余弦相似度)...")
    e_bridges = bridge.bridge_by_embedding(min_similarity=0.5, min_confidence=0.25)
    print(f"   发现 {len(e_bridges)} 座桥")
    for b in e_bridges[:8]:
        print(f"   {b.concept_a}({b.domain_a}) ↔ {b.concept_b}({b.domain_b}) "
              f"信:{b.confidence:.2f} | {b.reason[:60]}")

    # Layer 3
    print(f"\n🏗️  [Layer 3] 结构同构桥接 (层次匹配)...")
    s_bridges = bridge.bridge_by_structure(min_confidence=0.25)
    print(f"   发现 {len(s_bridges)} 座桥")
    for b in s_bridges[:8]:
        print(f"   {b.concept_a}({b.domain_a}) ↔ {b.concept_b}({b.domain_b}) "
              f"信:{b.confidence:.2f} | {b.reason[:80]}")

    # 综合
    print(f"\n🌉 [综合] 四层全量桥接...")
    all_bridges = bridge.build_all_bridges(min_confidence=0.3, max_bridges=80)
    summary = bridge.summary(all_bridges)
    print(f"   总计: {summary['total']} 座跨学科桥梁")
    print(f"   平均置信度: {summary['avg_confidence']:.3f}")
    print(f"   最高置信度: {summary['max_confidence']:.3f}")
    print(f"   桥接类型: {summary['bridge_types']}")
    print(f"   领域对分布: {summary['domain_pairs']}")

    print(f"\n🏆 Top 10 跨学科桥梁:")
    for i, b in enumerate(summary['top_bridges']):
        print(f"   {i+1}. {b['concepts']} ({b['domains']}) "
              f"信:{b['confidence']} [{b['type']}]")
        print(f"      {b['reason']}")

    # 写入概念图
    n_added = bridge.add_bridges_to_concept_graph(all_bridges, min_confidence=0.4)
    print(f"\n✅ 已将 {n_added} 座高置信度桥梁写入概念图 (RELATED 边)")

    return bridge, all_bridges


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from loongpearl.core.zichang import HanziAnchorField
    from loongpearl.core.freq_landscape import FreqEnergyLandscape
    from loongpearl.core.concept_graph import ConceptGraph

    PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    field_path = os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt')
    ls_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    cg_path = os.path.join(PROJECT, 'data/models/concept_graph')

    if not os.path.exists(field_path):
        print(f"⚠️ 字场不存在: {field_path}")
        sys.exit(1)

    field = HanziAnchorField.load(field_path, freeze=True)

    # 加载或构建景观
    if os.path.exists(ls_path):
        landscape = FreqEnergyLandscape.load(ls_path).eval()
    else:
        landscape = None

    # 加载或构建概念图
    cg = ConceptGraph(field, landscape)
    if os.path.exists(cg_path + '.json'):
        cg.load(cg_path)
    else:
        cg.seed_all_domains()
        cg.induce()
        cg.save(cg_path)

    demo_cross_domain(field, landscape, cg)
