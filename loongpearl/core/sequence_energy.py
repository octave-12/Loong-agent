#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠序列能量 (sequence_energy.py) — 从字对到字链
═══════════════════════════════════════════════════
在字对能量景观之上，评估和生成汉字序列。

核心能力:
  1. sequence_energy(chars)  — 评估一串字的连贯性
  2. beam_search(start, n)   — 束搜索生成下一个字
  3. generate(prefix, len)   — 给定前缀，续写序列
  4. rank(candidates)        — 对多个候选序列打分排序

原理:
  序列能量 = Σ landscape(mid(chars[i], chars[i+1]))
  能量越低 → 字链越连贯 → 越可能是真实语言

这是龙珠"自组织语言"的最小可行起点——
不需要LLM，只需要能量景观 + 束搜索。
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BeamItem:
    """束搜索中的一个候选序列"""
    chars: List[str]
    indices: List[int]
    energy: float  # 累积能量（越低越好）
    score: float   # 归一化分数（越高越好）


class SequenceEnergy:
    """
    序列能量评估器。
    
    用法:
        seq = SequenceEnergy(zichang, landscape)
        
        # 评估
        energy = seq.energy(["画", "龙", "点", "睛"])
        
        # 生成
        result = seq.beam_search("画龙", beam_width=5, max_len=4)
        
        # 排序
        ranked = seq.rank(["画龙点睛", "画蛇添足", "画虎类犬"])
    """
    
    def __init__(self, zichang, landscape, idiom_file=None, device='cuda'):
        self.zichang = zichang
        self.landscape = landscape
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.landscape.to(self.device).eval()
        
        # 构建成语字对索引：每个字 → 它在成语中后接的字集合
        self._next_chars: Dict[str, set] = {}
        if idiom_file is None:
            import os as _os
            idiom_file = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                'data/dicts/idioms.json'
            )
        self._build_pair_index(idiom_file)
    
    def _build_pair_index(self, idiom_file):
        """从成语词典构建字对后继索引"""
        import json
        try:
            with open(idiom_file, encoding='utf-8') as f:
                idioms = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        
        for idiom in idioms:
            chars = list(idiom)
            for i in range(len(chars) - 1):
                a, b = chars[i], chars[i + 1]
                if a not in self._next_chars:
                    self._next_chars[a] = set()
                self._next_chars[a].add(b)
        
        total_edges = sum(len(v) for v in self._next_chars.values())
        print(f"序列引擎: {len(self._next_chars)}个首字, {total_edges}条后继边")
    
    # ── 能量评估 ──────────────────────────────────────────────
    
    def pair_energy(self, a: str, b: str) -> float:
        """两个字之间的能量"""
        ia = self.zichang._char_to_idx.get(a)
        ib = self.zichang._char_to_idx.get(b)
        if ia is None or ib is None:
            return 999.0
        
        va = self.zichang.anchors[ia:ia+1].to(self.device)
        vb = self.zichang.anchors[ib:ib+1].to(self.device)
        mid = (va + vb) / 2
        
        with torch.no_grad():
            return self.landscape(mid).item()
    
    def energy(self, chars: List[str]) -> float:
        """整个字符序列的能量和"""
        if len(chars) < 2:
            return 0.0
        total = 0.0
        for i in range(len(chars) - 1):
            total += self.pair_energy(chars[i], chars[i+1])
        return total
    
    def energy_normalized(self, chars: List[str]) -> float:
        """归一化能量（除以长度-1，得到每对平均）"""
        n = len(chars)
        if n < 2:
            return 0.0
        return self.energy(chars) / (n - 1)
    
    # ── 束搜索生成 ────────────────────────────────────────────
    
    def next_char_candidates(
        self, 
        prev_char: str, 
        top_k: int = 20,
        allow_any: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        给定前一个字，找能量最低的下一个字候选。
        
        默认只在成语字对索引中搜索（真实出现过的字对），
        allow_any=True 则搜索全部94K字（慢且不可靠）。
        """
        idx = self.zichang._char_to_idx.get(prev_char)
        if idx is None:
            return []
        
        # 获取允许的后继字
        allowed = self._next_chars.get(prev_char)
        if not allowed:
            if allow_any:
                return self._search_all(idx, top_k)
            return []
        
        # 只评估真实出现过的后继字
        allowed_indices = []
        for ch in allowed:
            ci = self.zichang._char_to_idx.get(ch)
            if ci is not None:
                allowed_indices.append(ci)
        
        if not allowed_indices:
            return []
        
        v_prev = self.zichang.anchors[idx:idx+1].to(self.device)
        v_next_all = self.zichang.anchors[allowed_indices].to(self.device)
        v_prev_expanded = v_prev.expand(len(allowed_indices), -1)
        mids = (v_prev_expanded + v_next_all) / 2
        
        with torch.no_grad():
            energies = self.landscape(mids).squeeze(-1)
        
        top_k = min(top_k, len(allowed_indices))
        top_values, top_idx = torch.topk(energies, top_k, largest=False)
        
        return [
            (self.zichang.hanzi_list[allowed_indices[i.item()]], top_values[j].item())
            for j, i in enumerate(top_idx)
        ]
    
    def _search_all(self, prev_idx: int, top_k: int) -> List[Tuple[str, float]]:
        """全量搜索94K字（备选，慢）"""
        v_prev = self.zichang.anchors[prev_idx:prev_idx+1].to(self.device)
        batch_size = 10000
        all_energies = []
        
        with torch.no_grad():
            for start in range(0, self.zichang.num_hanzi, batch_size):
                end = min(start + batch_size, self.zichang.num_hanzi)
                v_next = self.zichang.anchors[start:end].to(self.device)
                mids = (v_prev.expand(end - start, -1) + v_next) / 2
                energies = self.landscape(mids).squeeze(-1)
                all_energies.append(energies)
        
        all_energies = torch.cat(all_energies)
        top_indices = torch.topk(all_energies, top_k, largest=False)
        
        return [
            (self.zichang.hanzi_list[i.item()], all_energies[i].item())
            for i in top_indices.indices
        ]
    
    def beam_search(
        self,
        prefix: str,
        beam_width: int = 5,
        max_len: int = 8,
        temperature: float = 0.3,
    ) -> List[BeamItem]:
        """
        束搜索：从给定前缀出发，逐步扩展，保留beam_width条最优路径。
        
        Args:
            prefix: 起始字符串（如"画龙"）
            beam_width: 束宽
            max_len: 最大生成长度（含前缀）
            temperature: 采样温度（0=贪心，越大越随机）
        
        Returns:
            按能量排序的beam_width条候选序列
        """
        prefix_chars = list(prefix)
        
        # 初始化束
        beams = [
            BeamItem(
                chars=prefix_chars.copy(),
                indices=[self.zichang._char_to_idx.get(c, 0) for c in prefix_chars],
                energy=self.energy(prefix_chars),
                score=0.0,
            )
        ]
        
        for step in range(len(prefix_chars), max_len):
            candidates = []
            
            for beam in beams:
                last_char = beam.chars[-1]
                next_candidates = self.next_char_candidates(last_char, top_k=beam_width * 2)
                
                for char, pair_energy in next_candidates:
                    if char == last_char:  # 避免连续重复字
                        continue
                    
                    new_energy = beam.energy + pair_energy
                    new_chars = beam.chars + [char]
                    
                    candidates.append(BeamItem(
                        chars=new_chars,
                        indices=beam.indices + [self.zichang._char_to_idx.get(char, 0)],
                        energy=new_energy,
                        score=0.0,
                    ))
            
            if not candidates:
                break
            
            # 按能量排序，保留top beam_width
            candidates.sort(key=lambda x: x.energy)
            
            # 加入温度采样（避免总是选同一个）
            if temperature > 0 and len(candidates) > beam_width:
                energies = torch.tensor([c.energy for c in candidates[:beam_width * 2]])
                probs = F.softmax(-energies / temperature, dim=0)
                sampled_idx = torch.multinomial(probs, beam_width, replacement=False)
                beams = [candidates[i.item()] for i in sampled_idx]
            else:
                beams = candidates[:beam_width]
            
            # 归一化分数
            max_e = max(b.energy for b in beams)
            min_e = min(b.energy for b in beams)
            if max_e > min_e:
                for b in beams:
                    b.score = 1.0 - (b.energy - min_e) / (max_e - min_e)
            else:
                for b in beams:
                    b.score = 1.0
        
        beams.sort(key=lambda x: x.energy)
        return beams
    
    # ── 候选排序 ──────────────────────────────────────────────
    
    def rank(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """对多个候选字符串按能量排序（能量越低越好）"""
        scored = []
        for s in candidates:
            chars = list(s)
            e = self.energy_normalized(chars)
            scored.append((s, e))
        scored.sort(key=lambda x: x[1])
        return scored
    
    # ── 前缀匹配生成（基于成语词典）────────────────────────
    
    def complete_from_dict(
        self,
        prefix: str,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        从成语词典中匹配前缀，用能量景观排序。
        
        这才是龙珠"自组织语言"的正确姿势：
        不是从零造字，而是在已知模式中找最佳匹配。
        """
        import json
        
        # 加载成语（如果还没加载）
        if not hasattr(self, '_idiom_list'):
            idiom_file = None
            for attr in ['_idiom_file', 'idiom_file']:
                if hasattr(self, attr):
                    idiom_file = getattr(self, attr)
            if idiom_file is None:
                import os as _os
                idiom_file = _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                    'data/dicts/idioms.json'
                )
            with open(idiom_file, encoding='utf-8') as f:
                self._idiom_list = json.load(f)
        
        prefix_chars = list(prefix)
        plen = len(prefix_chars)
        
        # 匹配：成语前缀匹配 OR 成语包含此前缀
        matches = []
        for idiom in self._idiom_list:
            chars = list(idiom)
            if len(chars) < plen:
                continue
            if chars[:plen] == prefix_chars:
                matches.append((idiom, 'prefix'))
            elif prefix in idiom:
                matches.append((idiom, 'contains'))
        
        if not matches:
            return []
        
        # 用能量景观评分排序
        scored = []
        for idiom, match_type in matches:
            e = self.energy_normalized(list(idiom))
            scored.append((idiom, e, match_type))
        
        # 前缀匹配优先，能量低优先
        scored.sort(key=lambda x: (0 if x[2] == 'prefix' else 1, x[1]))
        
        return [(s[0], s[1]) for s in scored[:top_n]]
    
    def complete(
        self,
        prefix: str,
        top_n: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        智能补全：成语查表优先，查不到返回包含匹配。
        不再使用束搜索兜底（不可靠），宁愿返回少也不要返回错。
        
        Returns: [(完整字符串, 归一化能量), ...]
        """
        # 成语查表（前缀匹配优先 + 包含匹配兜底）
        from_dict = self.complete_from_dict(prefix, top_n=top_n)
        if from_dict:
            return from_dict
        
        # 没有任何匹配 → 诚实返回空
        return []


# ============================================================================
# 便捷函数
# ============================================================================

def demo_sequence_energy(zichang, landscape):
    """演示序列能量功能"""
    seq = SequenceEnergy(zichang, landscape)
    
    print("=" * 50)
    print("🐉 序列能量演示")
    print("=" * 50)
    
    # 1. 评估已知成语
    tests = ["画龙点睛", "龙飞凤舞", "乱七八糟", "天地玄黄"]
    print("\n📊 成语能量评估:")
    for s in tests:
        e = seq.energy_normalized(list(s))
        print(f"  {s}: {e:.2f}")
    
    # 2. 补全前缀
    print("\n🔍 前缀补全 — '画龙':")
    completions = seq.beam_search("画龙", beam_width=5, max_len=4)
    for b in completions:
        print(f"  {''.join(b.chars)} (能量: {b.energy:.1f})")
    
    # 3. 候选排序
    print("\n📋 候选排序:")
    ranked = seq.rank(["画龙点睛", "画蛇添足", "画虎类犬", "画饼充饥"])
    for s, e in ranked:
        print(f"  {s}: {e:.2f}")
    
    return seq


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from loongpearl.core.zichang import HanziAnchorField
    from loongpearl.core.freq_landscape import FreqEnergyLandscape
    
    PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)
    ls = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt'))
    
    demo_sequence_energy(field, ls)
