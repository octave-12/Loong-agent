#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠自主学习引擎 (autonomous_learner.py) v2
===========================================
不依赖 Ollama、不依赖 Hermes Agent——龙珠自己的学习神经。

学习循环（五步闭环）:
  1. 检测缺口:  字场查询 → 能量异常高 → 标记"未知"
  2. 全网搜索:  WebSearcher 搜索该概念 → 获取释义/关联
  3. 知识提取:  从搜索结果中提取中文字符相邻对 (a,b)
  4. Hebbian注入: 批量将字对写入能量景观（降低中点能量）
  5. 验证:      重新查询 → 能量降低 → 确认学会

与现有模块的关系:
  - zichang.py:     字场锚点（永久冻结）
  - freq_landscape.py: 能量景观（可学习，含 infer/resolve）
  - learner.py:     Hebbian 学习器（写入机制）
  - web/searcher.py: 全网搜索（发现新知识）
  - 本模块:         编排以上组件，实现自主闭环

原则:
  - 零 LLM 依赖:    学习过程不调 Ollama
  - 能量景观是唯一知识源: 不在 JSON 里存新知识
  - 基于检测结果做决策:   先测后学，学完验证
"""

import re
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.web.searcher import WebSearcher, SearchResponse


# ============================================================================
# 自主学习引擎
# ============================================================================

class AutonomousLearner:
    """
    龙珠自主学习引擎。
    
    不依赖任何外部 LLM——用自己的字场定位、能量景观评估、
    WebSearcher 发现、Hebbian 学习写入。

    用法:
        al = AutonomousLearner(zichang, landscape, learner)
        
        # 单条学习
        result = al.learn_if_unknown("量子计算")
        
        # 批量注入
        al.learn_idiom_batch(idiom_list)
        
        # 检测知识盲区
        dead_ends = al.detect_dead_ends()
    """
    
    def __init__(
        self,
        zichang: HanziAnchorField,
        landscape: FreqEnergyLandscape,
        learner: DragonBallLearner,
    ):
        self.zichang = zichang
        self.landscape = landscape
        self.learner = learner
        self.searcher = WebSearcher(timeout=8, cache_enabled=True)
        
        # 统计
        self.total_learned = 0
        self.total_searched = 0
        self.total_injected = 0
    
    # ── 公开 API ──────────────────────────────────────────────
    
    def learn_if_unknown(
        self,
        query_text: str,
        query_vec: Optional[torch.Tensor] = None,
        auto_search: bool = True,
    ) -> Dict:
        """
        检测 → 搜索 → 学习 → 验证 完整闭环。
        
        Args:
            query_text: 查询文本（用于搜索关键词）
            query_vec: 查询向量 (1024d)，None则跳过无知检测
            auto_search: 是否自动联网搜索
        
        Returns:
            {'status': 'learned'|'already_known'|'search_failed'|'no_network',
             'pairs_learned': int, 'energy_before': float, 'energy_after': float, ...}
        """
        result = {
            'status': 'unknown',
            'pairs_learned': 0,
            'energy_before': None,
            'energy_after': None,
            'sources': [],
            'message': '',
        }
        
        # 步骤1: 检测 —— 判断是否需要学习
        if query_vec is not None:
            check = self.learner.check_knowledge(query_vec)
            if check.get('is_known') and check.get('confidence', 0) > 0.5:
                result['status'] = 'already_known'
                result['confidence'] = check['confidence']
                result['diagnosis'] = check.get('diagnosis', '')
                return result
            result['energy_before'] = check.get('energy', 999)
        
        # 步骤2: 搜索 —— 联网发现知识
        if not auto_search:
            result['status'] = 'no_network'
            result['message'] = '未启用自动搜索'
            return result
        
        self.total_searched += 1
        search_results = self._search_knowledge(query_text)
        
        if not search_results or not search_results.results:
            result['status'] = 'search_failed'
            result['message'] = '全网搜索未找到相关知识'
            return result
        
        result['sources'] = search_results.sources
        
        # 步骤3: 提取 —— 从搜索结果中提取字对关联
        pairs = self._extract_pairs(search_results, query_text)
        if not pairs:
            result['status'] = 'search_failed'
            result['message'] = '搜索到结果但无法提取字对关联'
            return result
        
        # 步骤4: 注入 —— Hebbian 学习写入能量景观
        energy_before = result.get('energy_before')
        n_injected = self._inject_pairs(pairs, energy_before or 999)
        self.total_injected += n_injected
        self.total_learned += 1
        
        # 步骤5: 验证 —— 重新检测
        energy_after = None
        if query_vec is not None:
            check_after = self.learner.check_knowledge(query_vec)
            energy_after = check_after.get('energy', 999)
        
        result.update({
            'status': 'learned',
            'pairs_learned': n_injected,
            'energy_after': energy_after,
            'energy_delta': (energy_before - energy_after) if (energy_before and energy_after) else None,
        })
        
        return result
    
    def learn_idiom_batch(
        self,
        idioms: List[str],
        batch_size: int = 500,
        verbose: bool = True,
    ) -> Dict:
        """
        批量将成语注入能量景观。
        
        每个成语 "画龙点睛" → 注入三对: (画,龙), (龙,点), (点,睛)
        
        Args:
            idioms: 成语列表
            batch_size: 每批注入多少
        
        Returns:
            {'total_idioms': N, 'total_pairs': M, 'time': seconds}
        """
        start = time.time()
        total_pairs = 0
        
        for i in range(0, len(idioms), batch_size):
            batch = idioms[i:i + batch_size]
            batch_pairs = []
            
            for idiom in batch:
                chars = list(idiom)
                if not all(ch in self.zichang._char_to_idx for ch in chars):
                    continue
                
                for j in range(len(chars) - 1):
                    a, b = chars[j], chars[j + 1]
                    ia = self.zichang._char_to_idx[a]
                    ib = self.zichang._char_to_idx[b]
                    batch_pairs.append((ia, ib))
            
            if batch_pairs:
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
    
    def detect_dead_ends(self, threshold: float = -5.0, sample_size: int = 200) -> List[str]:
        """
        检测能量景观中的"死路"——尾字无低能出路的汉字。
        
        扫描所有汉字，对每个字 ch，取样目标字计算 (ch, X) 的最小能量。
        如果最小能量 > threshold，标记为死路，需要学习。
        
        Args:
            threshold: 能量高于此值视为"无通路"
            sample_size: 每个字的采样目标数
        
        Returns:
            死路汉字列表
        """
        dead_ends = []
        num_anchors = len(self.zichang.anchors)
        device = next(self.landscape.parameters()).device
        
        with torch.no_grad():
            anchors_device = self.zichang.anchors.to(device)
            
            for ch, idx in self.zichang._char_to_idx.items():
                # 随机采样目标字
                sample_indices = torch.randint(0, num_anchors, (sample_size,), device=device)
                
                src_vec = anchors_device[idx].unsqueeze(0).expand(sample_size, -1)
                tgt_vecs = anchors_device[sample_indices]
                mid_vecs = (src_vec + tgt_vecs) / 2
                
                energies = self.landscape(mid_vecs)
                min_energy = energies.min().item()
                
                if min_energy > threshold:
                    dead_ends.append(ch)
        
        return dead_ends
    
    # ── 内部方法 ──────────────────────────────────────────────
    
    def _search_knowledge(self, context: str) -> Optional[SearchResponse]:
        """搜索知识，自动检测知识域"""
        # 检测知识域并构造查询
        if len(context) <= 2:
            query = f"{context} 是什么意思 释义"
        elif all('\u4e00' <= c <= '\u9fff' for c in context) and len(context) == 4:
            query = f"{context} 成语 释义 出处"
        else:
            query = f"{context} 解释 含义"
        
        try:
            return self.searcher.search(query)
        except Exception as e:
            print(f"  搜索异常: {e}")
            return None
    
    def _extract_pairs(
        self,
        search_response: SearchResponse,
        context: str,
    ) -> List[Tuple[int, int]]:
        """
        从搜索结果中提取字符关联对。
        
        策略:
          - 从搜索结果摘要中提取所有相邻中文字符对
          - 过滤: 两个字符都在字场中
          - 按频率排序，取高频对（高频对更可能是有效的语义关联）
          - 加入自关联 (ch, ch) 强化单字锚点
        
        Args:
            search_response: 搜索结果
            context: 原始查询文本
        
        Returns:
            [(idx_a, idx_b), ...] 去重的字对索引列表
        """
        pair_counter = Counter()
        
        # 合并所有搜索结果的摘要 + 综合回答
        all_text = ' '.join(r.snippet for r in search_response.results[:5] if r.snippet)
        all_text += ' ' + (search_response.answer or '')
        
        # 提取连续中文字符对
        cn_chars = re.findall(r'[\u4e00-\u9fff]', all_text)
        
        for i in range(len(cn_chars) - 1):
            a, b = cn_chars[i], cn_chars[i + 1]
            if (a in self.zichang._char_to_idx and
                b in self.zichang._char_to_idx):
                pair_counter[(a, b)] += 1
        
        # 也加入上下文本身拆解出的字对
        query_chars = re.findall(r'[\u4e00-\u9fff]', context)
        for i in range(len(query_chars) - 1):
            a, b = query_chars[i], query_chars[i + 1]
            if (a in self.zichang._char_to_idx and
                b in self.zichang._char_to_idx):
                pair_counter[(a, b)] += 2  # 上下文中的对权重加倍
        
        if not pair_counter:
            return []
        
        # 转换为索引对（取频率前50对，去重）
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
    ) -> int:
        """
        将字符关联对注入能量景观。
        
        使用 Hebbian 学习: 同时激活的两个锚点 → 降低它们中点能量。
        学习率根据当前能量自适应: 能量越高 → 学习率越大（知识越新，越用力学）。
        
        Returns:
            成功注入的对数
        """
        if not pairs:
            return 0
        
        # 自适应学习率
        if base_energy > 100:
            lr = 0.1
        elif base_energy > 10:
            lr = 0.05
        elif base_energy > 0:
            lr = 0.02
        else:
            lr = 0.01
        
        result = self.learner.learn_pairs_batch(pairs, learning_rate=lr)
        return result.get('pairs_learned', len(pairs))


# ============================================================================
# 便捷函数
# ============================================================================

def inject_idioms_to_landscape(
    zichang: HanziAnchorField,
    landscape: FreqEnergyLandscape,
    learner: DragonBallLearner,
    idiom_file: str = None,
    verbose: bool = True,
) -> Dict:
    """
    将成语词典注入能量景观（种子学习）。
    
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
        dead = al.detect_dead_ends()
        print(f"  死路字: {len(dead)} 个")
    
    return result


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🐉 龙珠自主学习引擎 v2 — 自测")
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
    ls = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    )
    ls.eval()
    
    print("初始化学习器...")
    learner = DragonBallLearner(ls, field, device='cpu')
    try:
        learner.calibrate()
    except Exception:
        pass
    
    al = AutonomousLearner(field, ls, learner)
    
    # 检测死路
    print("\n🔍 检测死路字...")
    dead = al.detect_dead_ends(threshold=-3.0, sample_size=100)
    print(f"  死路: {len(dead)} 个")
    if dead:
        print(f"  样本: {dead[:15]}")
    
    # 测试搜索+学习（不联网也会走缓存或报search_failed）
    print("\n📖 测试自主学习...")
    r = al.learn_if_unknown("境由心生", auto_search=True)
    print(f"  状态: {r['status']}")
    print(f"  注入对: {r.get('pairs_learned', 0)}")
    if r.get('energy_before') is not None:
        print(f"  能量: {r['energy_before']:.2f} → {r.get('energy_after', '?')}")
    
    print("\n✅ 自主学习引擎就绪")
