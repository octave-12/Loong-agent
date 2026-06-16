#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠词级能量景观 (word_energy.py) — 从字对到词链
═══════════════════════════════════════════════════

在概念图 + 字场 + 能量景观之上，实现多字词语级别的序列评估和生成。

与 sequence_energy.py 的区别:
  sequence_energy.py: 字级 —— "画" → "龙" → "点" → "睛"
  本模块:            词级 —— "量子" → "力学" / "纠缠" / "计算"

核心原理:
  1. 词嵌入 = 组成汉字的锚点平均（与概念图一致）
  2. 词对能量 = landscape(mid(word_a_emb, word_b_emb))
  3. 候选词对来自概念图的 RELATED 边
  4. 束搜索在词级别进行，由概念图关系约束

══════════════════════════════════════════════════════════════════
用法
══════════════════════════════════════════════════════════════════

    we = WordEnergy(field, landscape, concept_graph)
    
    # 词级补全
    results = we.complete("量子")  
    # → [("量子力学", -35.2), ("量子计算", -33.1), ("量子纠缠", -31.8)]
    
    # 多跳词链
    chain = we.chain("电子", max_words=3)
    # → ["电子", "原子", "分子", "物质"]

    # 批量评估
    scores = we.rank(["量子力学", "量子计算", "量子纠缠", "量子场论"])
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class WordBeam:
    """词级束搜索的一个候选"""
    words: List[str]
    chars: List[str]
    energy: float
    score: float = 0.0


class WordEnergy:
    """
    词级能量景观评估器。

    不依赖 LLM。词嵌入来自字场锚点组合，词对能量来自
    能量景观评估，候选词对来自概念图 RELATED 边。
    """

    def __init__(self, field, landscape, concept_graph):
        self.field = field              # HanziAnchorField
        self.landscape = landscape      # FreqEnergyLandscape
        self.cg = concept_graph         # ConceptGraph
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.landscape.to(self.device).eval()
        self.embed_dim = field.embed_dim

        # ── 词级索引 ──
        # 前缀 → 匹配的概念列表（用于快速查找）
        self._prefix_index: Dict[str, List[str]] = defaultdict(list)
        # 概念 → 词嵌入缓存
        self._word_embeddings: Dict[str, torch.Tensor] = {}
        self._build_indexes()

    def _build_indexes(self):
        """从概念图构建词级索引"""
        for concept in self.cg.nodes:
            # 前缀索引（支持1-4字前缀）
            for i in range(1, min(5, len(concept) + 1)):
                prefix = concept[:i]
                self._prefix_index[prefix].append(concept)

            # 词嵌入缓存
            emb = self.cg.get_embedding(concept)
            if emb is not None:
                self._word_embeddings[concept] = emb.to(self.device)

        # 也从字场中找常用2-4字词（概念图可能不包含所有词）
        # 这部分通过搜索时动态扩展

        total_prefixes = len(self._prefix_index)
        total_words = len(self._word_embeddings)
        print(f"词级引擎: {total_words}个词, {total_prefixes}个前缀索引")

    # ═══════════════════════════════════════════════════════════════
    # 词嵌入
    # ═══════════════════════════════════════════════════════════════

    def word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """获取词的嵌入向量（从字场锚点组合）"""
        if word in self._word_embeddings:
            return self._word_embeddings[word]

        # 动态计算
        chars = list(word)
        valid_indices = []
        for c in chars:
            idx = getattr(self.field, '_char_to_idx', {}).get(c)
            if idx is not None:
                valid_indices.append(idx)

        if not valid_indices:
            return None

        emb = self.field.anchors[valid_indices].mean(dim=0).to(self.device)
        self._word_embeddings[word] = emb
        return emb

    # ═══════════════════════════════════════════════════════════════
    # 能量评估
    # ═══════════════════════════════════════════════════════════════

    def word_pair_energy(self, word_a: str, word_b: str) -> float:
        """两个词之间的转移能量（越低越连贯）"""
        e_a = self.word_embedding(word_a)
        e_b = self.word_embedding(word_b)
        if e_a is None or e_b is None:
            return 999.0

        mid = ((e_a + e_b) / 2).unsqueeze(0)
        with torch.no_grad():
            return self.landscape(mid).item()

    def word_chain_energy(self, words: List[str]) -> float:
        """词链的总能量"""
        if len(words) < 2:
            return 0.0
        total = 0.0
        for i in range(len(words) - 1):
            total += self.word_pair_energy(words[i], words[i + 1])
        return total

    # ═══════════════════════════════════════════════════════════════
    # 候选词查找
    # ═══════════════════════════════════════════════════════════════

    def next_word_candidates(
        self,
        prev_word: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        给定前一个词，找最连贯的下一个词候选。

        候选来源（按优先级）:
          1. 概念图中的 RELATED 边（最可靠）
          2. 共享前缀的概念（如 "量子" 后可能接 "量子力学"）
          3. 嵌入最近邻（兜底）
        """
        candidates = {}

        # 来源1: 概念图 RELATED 边
        for neighbor, rel in self.cg.forward_index.get(prev_word, {}).items():
            if rel == "RELATED":
                energy = self.word_pair_energy(prev_word, neighbor)
                candidates[neighbor] = energy
        for neighbor, rel in self.cg.reverse_index.get(prev_word, {}).items():
            if rel == "RELATED":
                if neighbor not in candidates:
                    energy = self.word_pair_energy(prev_word, neighbor)
                    candidates[neighbor] = energy

        # 来源2: 共享前缀（如 "量子" → "量子力学", "量子计算"）
        extend_prefix = prev_word  # 搜索以 prev_word 为前缀的更长的词
        for concept in self._prefix_index.get(extend_prefix, [])[:50]:
            if concept != prev_word and concept not in candidates:
                energy = self.word_pair_energy(prev_word, concept)
                candidates[concept] = energy

        # 来源3: 嵌入最近邻（取相似度最高的几个概念作为候选）
        if len(candidates) < 5 and prev_word in self._word_embeddings:
            emb = self._word_embeddings[prev_word]
            # 采样评估所有已缓存词嵌入
            all_words = list(self._word_embeddings.keys())
            if len(all_words) > 100:
                indices = torch.randperm(len(all_words))[:100]
                sample_words = [all_words[i] for i in indices.tolist()]
            else:
                sample_words = all_words

            for word in sample_words:
                if word == prev_word or word in candidates:
                    continue
                energy = self.word_pair_energy(prev_word, word)
                if energy < -20:  # 只保留低能量（高连贯）的
                    candidates[word] = energy

        # 按能量排序，取最低的
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
        return sorted_candidates[:top_k]

    # ═══════════════════════════════════════════════════════════════
    # 前缀匹配词
    # ═══════════════════════════════════════════════════════════════

    def find_words_by_prefix(self, prefix: str, max_results: int = 50) -> List[str]:
        """
        找所有以 prefix 开头的已知词。

        来源:
          1. 概念图节点
          2. 前缀索引
        """
        results = set()
        for concept in self._prefix_index.get(prefix, []):
            results.add(concept)
        # 也从概念图节点中筛选
        for concept in self.cg.nodes:
            if concept.startswith(prefix):
                results.add(concept)
        return list(results)[:max_results]

    # ═══════════════════════════════════════════════════════════════
    # 词级束搜索
    # ═══════════════════════════════════════════════════════════════

    def beam_search(
        self,
        prefix: str,
        beam_width: int = 5,
        max_words: int = 4,
        temperature: float = 0.3,
    ) -> List[WordBeam]:
        """
        词级束搜索：从给定前缀出发，逐步扩展为词链。

        每一步:
          1. 找当前最后一个词的候选下一个词
          2. 评估每个候选的转移能量
          3. 保留 beam_width 条最优路径

        Args:
            prefix: 起始词或字串（如 "量子"）
            beam_width: 束宽
            max_words: 最大词数（含起始词）
            temperature: 采样温度
        """
        prefix_chars = list(prefix)

        # 初始化束：找到所有以 prefix 开头的已知词
        matching_words = self.find_words_by_prefix(prefix)
        if not matching_words:
            # 如果找不到匹配词，将 prefix 本身作为起始词
            matching_words = [prefix]

        beams = []
        for word in matching_words[:beam_width * 2]:
            # 计算起始能量（与空字符串的能量差）
            energy = self.word_pair_energy("", word) if word != prefix else 0.0
            beams.append(WordBeam(
                words=[word],
                chars=list(word),
                energy=energy,
            ))

        if not beams:
            beams = [WordBeam(words=[prefix], chars=prefix_chars, energy=0.0)]

        beams = sorted(beams, key=lambda b: b.energy)[:beam_width]

        # 逐词扩展
        for step in range(1, max_words):
            candidates = []

            for beam in beams:
                last_word = beam.words[-1]
                next_candidates = self.next_word_candidates(last_word, top_k=beam_width * 2)

                for word, pair_energy in next_candidates:
                    # 避免重复（但不是绝对禁止——如"量子力学原理"可以）
                    if word == last_word:
                        continue

                    new_energy = beam.energy + pair_energy
                    new_words = beam.words + [word]
                    new_chars = beam.chars + list(word)

                    candidates.append(WordBeam(
                        words=new_words,
                        chars=new_chars,
                        energy=new_energy,
                    ))

            if not candidates:
                break

            candidates.sort(key=lambda b: b.energy)

            # 温度采样
            if temperature > 0 and len(candidates) > beam_width:
                energies = torch.tensor([c.energy for c in candidates[:beam_width * 2]])
                # 确保能量有限
                finite_mask = torch.isfinite(energies)
                if finite_mask.sum() >= beam_width:
                    energies = energies[finite_mask]
                    candidates = [candidates[i] for i in range(len(finite_mask)) if finite_mask[i]]
                if len(candidates) >= beam_width:
                    probs = F.softmax(-energies / (temperature + 1e-8), dim=0)
                    sampled_idx = torch.multinomial(probs, min(beam_width, len(probs)), replacement=False)
                    beams = [candidates[i.item()] for i in sampled_idx]
                else:
                    beams = candidates[:beam_width]
            else:
                beams = candidates[:beam_width]

        # 归一化分数
        if beams:
            max_e = max(b.energy for b in beams)
            min_e = min(b.energy for b in beams)
            if max_e > min_e:
                for b in beams:
                    b.score = 1.0 - (b.energy - min_e) / (max_e - min_e)
            else:
                for b in beams:
                    b.score = 1.0

        beams.sort(key=lambda b: b.energy)
        return beams

    # ═══════════════════════════════════════════════════════════════
    # 智能补全
    # ═══════════════════════════════════════════════════════════════

    def complete(
        self,
        prefix: str,
        top_n: int = 10,
        beam_width: int = 8,
    ) -> List[Tuple[str, float, str]]:
        """
        智能补全：给定前缀，返回最可能的完整词语/短语。

        策略:
          1. 先找以 prefix 开头的已知词（精确匹配优先）
          2. 再尝试词级束搜索（多词补全）
          3. 按能量排序

        Returns:
            [(完整文本, 能量, 来源), ...]
            来源: "exact" | "beam" | "extend"
        """
        results = []

        # 策略1: 以 prefix 开头的已知词（如 "量子" → "量子力学"）
        matching = self.find_words_by_prefix(prefix, max_results=top_n * 2)
        for word in matching:
            if word == prefix:
                continue
            energy = self.word_pair_energy(prefix, word)
            results.append(("".join(list(word)), energy, "exact"))

        # 策略2: 词级束搜索（生成多词补全，如 "量子" → "量子 力学 原理"）
        beams = self.beam_search(prefix, beam_width=beam_width, max_words=3)
        for b in beams[:top_n]:
            full_text = "".join(b.chars)
            # 避免与策略1重复
            if full_text not in {r[0] for r in results}:
                results.append((full_text, b.energy, "beam"))

        # 策略3: 字符级回退（如果前面都没结果）
        if not results:
            from loongpearl.core.sequence_energy import SequenceEnergy
            seq = SequenceEnergy(self.field, self.landscape)
            char_results = seq.complete(prefix, top_n=top_n)
            for text, energy in char_results:
                results.append((text, energy, "char_fallback"))

        # 按能量排序
        results.sort(key=lambda x: x[1])
        return results[:top_n]

    # ═══════════════════════════════════════════════════════════════
    # 词链推理
    # ═══════════════════════════════════════════════════════════════

    def chain(
        self,
        start_word: str,
        max_words: int = 5,
        relation_filter: str = None,
    ) -> List[List[str]]:
        """
        从起始词出发，沿概念图关系生成词链。

        结合概念图的语义关系 + 能量景观的连贯性评分。
        如果起始词不在概念图中，回退到字符级。

        Args:
            start_word: 起始词
            max_words: 最大词数
            relation_filter: 关系过滤器（如 "PART_OF"）

        Returns:
            多条词链，按能量排序
        """
        # 先用概念图推理找到路径
        if start_word in self.cg.nodes:
            if relation_filter:
                paths = self.cg.reason(start_word, relation=relation_filter,
                                       max_hops=max_words - 1, direction="both")
            else:
                paths = self.cg.reason(start_word, max_hops=max_words - 1, direction="both")

            scored = []
            for path in paths[:20]:
                if len(path) < 2:
                    continue
                energy = self.word_chain_energy(path)
                scored.append((path, energy))
            scored.sort(key=lambda x: x[1])
            if scored:
                return [p for p, _ in scored[:10]]

        # 回退：字符级束搜索生成词链
        from loongpearl.core.sequence_energy import SequenceEnergy
        seq = SequenceEnergy(self.field, self.landscape)
        beams = seq.beam_search(start_word, beam_width=5, max_len=len(start_word) + max_words * 2)
        fallback_chains = []
        for b in beams[:5]:
            chars = b.chars
            # 尝试将字符序列分组为词
            words = [start_word] if start_word else []
            remaining = ''.join(chars[len(start_word):]) if len(start_word) <= len(chars) else ''.join(chars)
            if remaining:
                words.append(remaining)
            if len(words) >= 2:
                energy = self.word_chain_energy(words)
                fallback_chains.append((words, energy))

        fallback_chains.sort(key=lambda x: x[1])
        return [p for p, _ in fallback_chains[:10]]

    def rank(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """对多个候选词语按能量排序"""
        scored = []
        for word in candidates:
            # 用字场中点的平均能量评估
            chars = list(word)
            if len(chars) < 2:
                scored.append((word, 0.0))
                continue

            # 词内字对能量
            total_e = 0.0
            for i in range(len(chars) - 1):
                ia = self.field._char_to_idx.get(chars[i])
                ib = self.field._char_to_idx.get(chars[i + 1])
                if ia is not None and ib is not None:
                    va = self.field.anchors[ia:ia + 1].to(self.device)
                    vb = self.field.anchors[ib:ib + 1].to(self.device)
                    mid = (va + vb) / 2
                    with torch.no_grad():
                        total_e += self.landscape(mid).item()

            avg_e = total_e / (len(chars) - 1) if len(chars) > 1 else 0.0
            scored.append((word, avg_e))

        scored.sort(key=lambda x: x[1])
        return scored


# ============================================================================
# 演示
# ============================================================================

def demo_word_energy(field, landscape, concept_graph):
    """演示词级能量景观"""
    we = WordEnergy(field, landscape, concept_graph)

    print("=" * 60)
    print("🔤 龙珠词级能量景观 — 演示")
    print("=" * 60)
    print(f"   词数: {len(we._word_embeddings)} | 前缀索引: {len(we._prefix_index)}")

    # 1. 前缀补全
    test_prefixes = ["量子", "电子", "计算", "中国", "细胞"]
    for prefix in test_prefixes:
        results = we.complete(prefix, top_n=5)
        if results:
            print(f"\n📝 '{prefix}' →")
            for text, energy, source in results:
                print(f"   {text} (能:{energy:.1f} [{source}])")
        else:
            print(f"\n📝 '{prefix}' → ⚠️ 无已知词匹配")

    # 2. 词链推理
    print(f"\n🔗 词链: '电子' → ...")
    chains = we.chain("电子", max_words=4)
    for c in chains[:5]:
        print(f"   {' → '.join(c)}")

    # 3. 候选排序
    test_words = ["量子力学", "量子计算", "量子纠缠", "量子场论", "量子信息"]
    print(f"\n📊 候选排序:")
    ranked = we.rank(test_words)
    for word, energy in ranked:
        print(f"   {word}: {energy:.1f}")

    return we


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

    field = HanziAnchorField.load(field_path, freeze=True)
    landscape = FreqEnergyLandscape.load(ls_path).eval() if os.path.exists(ls_path) else None

    cg = ConceptGraph(field, landscape)
    if os.path.exists(cg_path + '.json'):
        cg.load(cg_path)
    else:
        cg.seed_all_domains()
        cg.induce()

    demo_word_energy(field, landscape, cg)
