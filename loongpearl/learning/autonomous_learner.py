#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠自主学习引擎 (autonomous_learner.py)
=======================================
不依赖 Ollama、不依赖 Hermes Agent——龙珠自己的学习神经。

学习循环:
  1. 检测缺口: 字场查询 → 能量异常高 / 死路 → 标记"未知"
  2. 全网搜索: 调用 WebSearcher 搜索该字/词的知识
  3. 知识提取: 从搜索结果中提取字对关联 (a,b) → 频率
  4. Hebbian 注入: 用能量景观的 Hebbian 学习将新关联写入
  5. 验证: 重新查询 → 能量降低 → 确认学会

与现有模块的关系:
  - zichang.py: 字场锚点（不动）
  - energy_landscape.py: 能量景观（可学习）
  - learner.py: Hebbian 学习器（写入机制）
  - searcher.py: 全网搜索（发现新知识）
  - 本模块: 编排以上组件，实现自主闭环

原则:
  - 零 LLM 依赖: 学习过程不调 Ollama
  - 零 Agent 依赖: 搜索用自有 WebSearcher
  - 能量景观是唯一知识源: 不在 JSON 里存知识
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.web.searcher import WebSearcher


class AutonomousLearner:
    """
    龙珠自主学习引擎。
    
    不依赖任何外部 LLM——用自己的字场定位、能量景观评估、
    WebSearcher 发现、Hebbian 学习写入。
    """
    
    def __init__(
        self,
        zichang: HanziAnchorField,
        landscape: EnergyLandscape,
        learner: DragonBallLearner,
    ):
        self.zichang = zichang
        self.landscape = landscape
        self.learner = learner
        self.searcher = WebSearcher(timeout=8, cache_enabled=True)
        
        # 统计
        self.total_learned = 0
        self.total_searched = 0
    
    # ── 公开 API ──────────────────────────────────────────────
    
    def learn_if_unknown(
        self,
        query_vec: torch.Tensor,
        context: str = "",
        auto_search: bool = True,
    ) -> Dict:
        """
        检测 → 搜索 → 学习 → 验证 完整循环。
        
        Args:
            query_vec: 查询向量 (1024d)
            context: 查询文本（用于搜索关键词）
            auto_search: 是否自动联网搜索
        
        Returns:
            {'status': 'learned'|'already_known'|'search_failed'|'no_network',
             'energy_before': float, 'energy_after': float, ...}
        """
        # 步骤1: 检测——用自知无知判断是否需要学习
        check = self.learner.check_knowledge(query_vec)
        
        if check['is_known'] and check['confidence'] > 0.5:
            return {
                'status': 'already_known',
                'confidence': check['confidence'],
                'diagnosis': check.get('diagnosis', ''),
            }
        
        # 步骤2: 搜索——上网发现新知识
        energy_before = check.get('energy', 999)
        search_results = None
        
        if auto_search and context:
            search_results = self._search_knowledge(context)
            if not search_results:
                return {
                    'status': 'search_failed',
                    'energy_before': energy_before,
                    'message': '全网搜索未找到相关知识',
                }
        
        # 步骤3: 学习——将搜索到的关联注入能量景观
        if search_results:
            pairs = self._extract_pairs(search_results, context)
            if pairs:
                self._inject_pairs(pairs, energy_before)
                self.total_learned += len(pairs)
        
        # 步骤4: 验证——重新检测
        check_after = self.learner.check_knowledge(query_vec)
        
        return {
            'status': 'learned',
            'energy_before': energy_before,
            'energy_after': check_after.get('energy', 999),
            'pairs_learned': len(pairs) if search_results else 0,
            'sources': search_results.sources if search_results else [],
        }
    
    def learn_idiom_batch(
        self,
        idioms: List[str],
        batch_size: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        批量将成语注入能量景观。
        
        每个成语 "画龙点睛" → 注入三对: (画,龙), (龙,点), (点,睛)
        使用 Hebbian 学习降低这些字对之间的能量。
        
        Args:
            idioms: 成语列表
            batch_size: 每批注入多少对后统一优化
        
        Returns:
            {'total_idioms': N, 'total_pairs': M, 'time': seconds}
        """
        start = time.time()
        total_pairs = 0
        
        for i in range(0, len(idioms), batch_size):
            batch = idioms[i:i + batch_size]
            batch_pairs = []
            
            for idiom in batch:
                # 确保每个字都在字场中
                chars = list(idiom)
                if not all(ch in self.zichang._char_to_idx for ch in chars):
                    continue
                
                # 提取相邻字对
                for j in range(len(chars) - 1):
                    a, b = chars[j], chars[j + 1]
                    ia = self.zichang._char_to_idx[a]
                    ib = self.zichang._char_to_idx[b]
                    batch_pairs.append((ia, ib))
            
            if batch_pairs:
                # 批量 Hebbian 学习
                self.learner.learn_pairs_batch(batch_pairs, learning_rate=0.05)
                total_pairs += len(batch_pairs)
            
            if verbose and (i + batch_size) % 2000 == 0:
                elapsed = time.time() - start
                print(f"  已注入 {i + len(batch)}/{len(idioms)} 个成语 "
                      f"({total_pairs} 对, {elapsed:.0f}s)")
        
        elapsed = time.time() - start
        result = {
            'total_idioms': len(idioms),
            'total_pairs': total_pairs,
            'time': elapsed,
        }
        
        if verbose:
            print(f"\n✅ 成语注入完成: {len(idioms)} 成语 → "
                  f"{total_pairs} 字对, {elapsed:.1f}s")
        
        return result
    
    def detect_dead_ends(self) -> List[str]:
        """
        检测能量景观中的"死路"——尾字无低能出路的汉字。
        
        扫描所有汉字: 对每个字 ch，计算 (ch, X) 对所有 X 的最小能量。
        如果最小能量 > 阈值，标记为死路，需要学习。
        """
        dead_ends = []
        threshold = -5.0  # 能量高于此值视为"无通路"
        
        with torch.no_grad():
            for ch, idx in self.zichang._char_to_idx.items():
                # 取样一些目标字，计算最小能量
                sample_size = min(100, len(self.zichang._char_to_idx))
                sample_indices = torch.randint(
                    0, len(self.zichang.anchors), (sample_size,)
                )
                
                src_vec = self.zichang.anchors[idx].unsqueeze(0).expand(sample_size, -1)
                tgt_vecs = self.zichang.anchors[sample_indices]
                mid_vecs = (src_vec + tgt_vecs) / 2
                
                energies = self.landscape(mid_vecs)
                min_energy = energies.min().item()
                
                if min_energy > threshold:
                    dead_ends.append(ch)
        
        return dead_ends
    
    # ── 内部方法 ──────────────────────────────────────────────
    
    def _search_knowledge(self, context: str):
        """搜索知识，返回 SearchResponse"""
        self.total_searched += 1
        
        # 检测知识域
        if len(context) <= 2:
            # 单字或双字，加释义关键词
            query = f"{context} 是什么意思 释义"
        elif all('\u4e00' <= c <= '\u9fff' for c in context) and len(context) == 4:
            # 四字成语
            query = f"{context} 成语 释义 出处"
        else:
            query = context
        
        return self.searcher.search(query)
    
    def _extract_pairs(
        self,
        search_response,
        context: str,
    ) -> List[Tuple[int, int]]:
        """
        从搜索结果中提取字符关联对。
        
        策略:
          - 从搜索结果摘要中提取所有相邻中文字符对
          - 过滤: 两个字符都在字场中
          - 去重 + 计数频率
        """
        import re
        from collections import Counter
        
        pair_counter = Counter()
        
        # 合并所有搜索结果的摘要
        all_text = ' '.join(r.snippet for r in search_response.results[:5])
        all_text += ' ' + search_response.answer
        
        # 提取连续中文字符
        cn_chars = re.findall(r'[\u4e00-\u9fff]', all_text)
        
        for i in range(len(cn_chars) - 1):
            a, b = cn_chars[i], cn_chars[i + 1]
            if (a in self.zichang._char_to_idx and
                b in self.zichang._char_to_idx):
                pair_counter[(a, b)] += 1
        
        # 转换为索引对（去重，保留频率高的）
        pairs = []
        for (a, b), freq in pair_counter.most_common(50):
            ia = self.zichang._char_to_idx[a]
            ib = self.zichang._char_to_idx[b]
            pairs.append((ia, ib))
        
        return pairs
    
    def _inject_pairs(
        self,
        pairs: List[Tuple[int, int]],
        base_energy: float,
    ):
        """
        将字符关联对注入能量景观。
        
        使用 Hebbian 学习: 同时激活的两个锚点 → 降低它们中点能量。
        学习率根据当前能量自适应: 能量越高 → 学习率越大。
        """
        # 自适应学习率
        if base_energy > 100:
            lr = 0.1
        elif base_energy > 10:
            lr = 0.05
        elif base_energy > 0:
            lr = 0.02
        else:
            lr = 0.01
        
        self.learner.learn_pairs_batch(pairs, learning_rate=lr)


# ============================================================================
# 便捷函数
# ============================================================================

def inject_idioms_to_landscape(
    zichang: HanziAnchorField,
    landscape: EnergyLandscape,
    learner: DragonBallLearner,
    idiom_file: str = None,
    verbose: bool = True,
) -> Dict:
    """
    将成语词典注入能量景观（种子学习）。
    
    这是初始化学步骤——把 29K 成语的字对关联写入能量景观，
    让龙珠拥有基本的汉字组合知识。
    
    Args:
        idiom_file: idioms.json 路径，默认从 data/dicts/ 读取
    """
    import json
    from pathlib import Path
    
    if idiom_file is None:
        from loongpearl.data_config import DICT_DIR
        idiom_file = DICT_DIR / "idioms.json"
    
    with open(idiom_file, encoding='utf-8') as f:
        idioms = json.load(f)
    
    if verbose:
        print(f"🐉 种子注入: {len(idioms)} 个成语 → 能量景观")
    
    al = AutonomousLearner(zichang, landscape, learner)
    result = al.learn_idiom_batch(idioms, verbose=verbose)
    
    if verbose:
        # 检测死路减少量
        dead = al.detect_dead_ends()
        print(f"  死路字: {len(dead)} 个")
    
    return result


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🐉 龙珠自主学习引擎 — 自测")
    print("=" * 60)
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n加载字场...")
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
        freeze=True
    )
    
    print("加载能量景观...")
    ls = EnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    )
    
    print("初始化学习器...")
    learner = DragonBallLearner(field, ls)
    
    # 创建自主学习引擎
    al = AutonomousLearner(field, ls, learner)
    
    # 测试: 检测死路
    print("\n🔍 检测死路字...")
    dead = al.detect_dead_ends()
    print(f"  死路: {len(dead)} 个")
    if dead:
        print(f"  样本: {dead[:20]}")
    
    # 测试: 批量注入成语
    print("\n📖 注入成语到能量景观...")
    result = inject_idioms_to_landscape(field, ls, learner, verbose=True)
    
    # 注入后重新检测
    print("\n🔍 注入后重新检测死路...")
    dead_after = al.detect_dead_ends()
    print(f"  死路: {len(dead_after)} 个 (减少 {len(dead) - len(dead_after)})")
