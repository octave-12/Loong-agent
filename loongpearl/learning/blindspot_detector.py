#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠多因子盲区检测器 (blindspot_detector.py)
═══════════════════════════════════════════════
7个独立因子 × 分区并行扫描 × 优先级队列 → 驱动自主学习

设计原则:
  1. 因子独立: 每个因子只读能量景观，写只写检测结果
  2. 分区并行: 94117汉字切分为N区，各区独立扫描
  3. 优先级合并: 多因子检测结果去重+排序，送入学习队列
  4. 增量更新: 学完一个盲区后，只重扫受影响的分区

检测因子:
  F1 统计因子: 尾字频高但首字频低 → 接龙盲区
  F2 能量因子: 中点能量异常高的字对 → 景观盲区  
  F3 覆盖因子: 某字在词典中关联数<3 → 知识稀疏
  F4 死路因子: 尾字在词典中无后续候选 → 硬死路
  F5 梯度因子: 锚点梯度异常偏离基准 → 景观畸变
  F6 语义群因子: 字簇间无跨簇连接 → 语义孤岛
  F7 新鲜度因子: 从未被激活过的冷门字 → 沉睡知识

用法:
    detector = MultiFactorDetector(field, landscape, idioms)
    gaps = detector.scan_all()  # 全因子全分区扫描
    # 或单因子:
    gaps = detector.scan_factor('statistical')
"""

import math
import time
import threading
import queue
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape


# ============================================================================
# 数据类
# ============================================================================

@dataclass(order=True)
class BlindSpot:
    """一个知识盲区"""
    priority: float = 0.0          # 优先级(越小越紧急, heap用)
    char: str = ""                 # 核心汉字
    factor: str = ""               # 发现因子名
    score: float = 0.0             # 盲区程度(越大越盲)
    evidence: dict = field(default_factory=dict)  # 详细证据
    timestamp: float = 0.0         # 发现时间
    
    def __repr__(self):
        return (f"BlindSpot({self.char} f={self.factor} "
                f"score={self.score:.1f})")


@dataclass
class ScanResult:
    """一次扫描的结果"""
    factor: str
    gaps: List[BlindSpot]
    time: float
    chars_scanned: int
    
    def __repr__(self):
        return (f"ScanResult({self.factor}: {len(self.gaps)} gaps, "
                f"{self.time:.1f}s, {self.chars_scanned} chars)")


# ============================================================================
# 分区扫描引擎
# ============================================================================

class PartitionedScanner:
    """
    将 N 个汉字分成 K 个分区，每个分区独立扫描。
    
    支持并行扫描(多线程)和增量重扫(只扫受影响的分区)。
    """
    
    def __init__(self, num_chars: int, num_partitions: int = 8):
        self.num_chars = num_chars
        self.num_partitions = num_partitions
        self._build_partitions()
    
    def _build_partitions(self):
        """将 [0, num_chars) 均匀分成 num_partitions 个区间"""
        self.partitions = []
        chunk = self.num_chars // self.num_partitions
        for i in range(self.num_partitions):
            start = i * chunk
            end = start + chunk if i < self.num_partitions - 1 else self.num_chars
            self.partitions.append((start, end))
    
    def partition_for(self, char_idx: int) -> int:
        """返回汉字所在的分区号"""
        for pid, (s, e) in enumerate(self.partitions):
            if s <= char_idx < e:
                return pid
        return 0
    
    def partitions_for_chars(self, char_indices: List[int]) -> Set[int]:
        """返回一组汉字涉及的分区号"""
        return {self.partition_for(i) for i in char_indices}


# ============================================================================
# 检测因子基类
# ============================================================================

class FactorDetector:
    """检测因子基类——所有因子继承此接口"""
    
    def __init__(self, name: str, field: HanziAnchorField,
                 landscape: FreqEnergyLandscape, idioms: list,
                 priority_weight: float = 1.0):
        self.name = name
        self.field = field
        self.landscape = landscape
        self.idioms = idioms
        self.priority_weight = priority_weight
        self.device = next(landscape.parameters()).device
        self.ci = field._char_to_idx
        self._last_scan_time = 0.0
    
    def scan(self, partition_range: Optional[Tuple[int, int]] = None) -> List[BlindSpot]:
        """
        扫描盲区。
        
        Args:
            partition_range: 可选的分区范围 (start_idx, end_idx),
                           为None则全量扫描
        
        Returns:
            检测到的盲区列表
        """
        raise NotImplementedError
    
    def chars_to_indices(self, chars: List[str]) -> List[int]:
        """汉字列表 → 索引列表(过滤掉不在字场的)"""
        return [self.ci[c] for c in chars if c in self.ci]


# ============================================================================
# F1: 统计因子 —— 尾字频高但首字频低
# ============================================================================

class StatisticalFactor(FactorDetector):
    """
    统计盲区: 某字作为尾字经常出现(龙珠常走到它)，
    但作为首字的成语很少(龙珠不知如何从它出发)。
    
    盲区度 = (尾字频率 - 首字频率) / 总频率
    值越大 → 越需要学习
    """
    
    def __init__(self, field, landscape, idioms):
        super().__init__('statistical', field, landscape, idioms, priority_weight=1.2)
        self._build_stats()
    
    def _build_stats(self):
        """建立汉字频率统计"""
        self.head_freq = Counter()
        self.tail_freq = Counter()
        self.total_freq = Counter()
        
        for w in self.idioms:
            if len(w) != 4: continue
            self.head_freq[w[0]] += 1
            self.tail_freq[w[-1]] += 1
            for c in w:
                self.total_freq[c] += 1
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        t0 = time.time()
        gaps = []
        
        chars_to_scan = list(self.total_freq.keys())
        # 按频率过滤: 只检测总频率>=3的字(太冷门的不急于学)
        
        for ch in chars_to_scan:
            freq = self.total_freq[ch]
            if freq < 3:
                continue
            head = self.head_freq.get(ch, 0)
            tail = self.tail_freq.get(ch, 0)
            
            # 盲区条件: 作为尾字≥2次 且 首字≤2次
            if tail >= 2 and head <= 2:
                # 盲区度 = 尾频/总频(越高越盲)
                gap_score = tail / max(freq, 1) * (1 + (tail - head))
                gaps.append(BlindSpot(
                    priority=-gap_score,  # 负数用于heap
                    char=ch,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'head_freq': head,
                        'tail_freq': tail,
                        'total_freq': freq,
                    }
                ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F2: 能量因子 —— 中点能量异常高的字对
# ============================================================================

class EnergyFactor(FactorDetector):
    """
    能量盲区: 对每个尾字取样的目标字，计算中点能量。
    如果所有目标字的能量都偏高 → 该尾字缺乏低能通路。
    
    使用GPU批量计算，每批处理多个尾字。
    """
    
    def __init__(self, field, landscape, idioms, sample_targets=200):
        super().__init__('energy', field, landscape, idioms, priority_weight=0.8)
        self.sample_targets = sample_targets
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        t0 = time.time()
        
        # 只扫描在成语词典中出现过的字
        active_chars = set()
        for w in self.idioms:
            if len(w) == 4:
                active_chars.add(w[-1])  # 尾字
        
        active_chars = [c for c in active_chars if c in self.ci]
        if partition_range is not None:
            # TODO: 分区过滤
            pass
        
        if not active_chars:
            return []
        
        gaps = []
        anchors = self.field.anchors.to(self.device)
        num_anchors = len(anchors)
        batch_size = 200  # 每批200个尾字
        
        with torch.no_grad():
            for i in range(0, len(active_chars), batch_size):
                batch_chars = active_chars[i:i + batch_size]
                src_indices = [self.ci[c] for c in batch_chars]
                src = anchors[torch.tensor(src_indices, device=self.device)]  # (B, 1024)
                
                # 随机采样目标字
                tgt_indices = torch.randint(0, num_anchors, 
                                           (self.sample_targets,), device=self.device)
                tgt = anchors[tgt_indices]  # (S, 1024)
                
                # 向量化: (B, S, 1024) → reshape → 批量前向
                B = len(batch_chars)
                S = self.sample_targets
                
                # 分批前向避免OOM
                chunk = 500
                all_energies = []
                for j in range(0, B * S, chunk):
                    end = min(j + chunk, B * S)
                    bi = torch.arange(j, end, device=self.device) // S
                    si = torch.arange(j, end, device=self.device) % S
                    mids = (src[bi] + tgt[si]) / 2
                    e = self.landscape(mids).squeeze(-1)
                    all_energies.append(e)
                
                energies = torch.cat(all_energies).reshape(B, S)
                min_energies = energies.min(dim=1).values  # 每个尾字的最小能量
                mean_energies = energies.mean(dim=1)       # 平均能量
                
                for j, ch in enumerate(batch_chars):
                    min_e = min_energies[j].item()
                    mean_e = mean_energies[j].item()
                    
                    # 盲区: 最小能量 > -5 (缺乏深盆地连接)
                    if min_e > -5.0:
                        gap_score = (min_e + 5.0) * 2 + (mean_e - min_e)
                        gaps.append(BlindSpot(
                            priority=-gap_score,
                            char=ch,
                            factor=self.name,
                            score=gap_score,
                            evidence={
                                'min_energy': min_e,
                                'mean_energy': mean_e,
                            }
                        ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F3: 覆盖因子 —— 某字在词典中关联数太少
# ============================================================================

class CoverageFactor(FactorDetector):
    """
    覆盖盲区: 汉字在成语词典中的出现次数太少或角色单一。
    
    如果一个字只作为首字出现、或只作为尾字出现，
    说明龙珠只知道它的一种用法——需要学习其他用法。
    """
    
    def __init__(self, field, landscape, idioms):
        super().__init__('coverage', field, landscape, idioms, priority_weight=0.9)
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        t0 = time.time()
        
        # 统计每个字的角色分布
        char_roles = defaultdict(lambda: {'head': 0, 'tail': 0, 'mid': 0, 'total': 0})
        
        for w in self.idioms:
            if len(w) != 4: continue
            for pos, c in enumerate(w):
                char_roles[c]['total'] += 1
                if pos == 0:
                    char_roles[c]['head'] += 1
                elif pos == 3:
                    char_roles[c]['tail'] += 1
                else:
                    char_roles[c]['mid'] += 1
        
        gaps = []
        
        for ch, roles in char_roles.items():
            total = roles['total']
            if total < 2:
                continue  # 出现次数太少，暂不处理
            
            # 计算角色多样性
            # 如果只有一个角色占比>80% → 单一用法
            max_role = max(roles['head'], roles['tail'], roles['mid'])
            ratio = max_role / max(total, 1)
            
            if ratio > 0.8 and total >= 3:
                dominant = 'head' if roles['head'] == max_role else \
                           'tail' if roles['tail'] == max_role else 'mid'
                gap_score = ratio * total  # 盲区度 = 单角色比例 × 总次数
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=ch,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'roles': dict(roles),
                        'dominant_role': dominant,
                        'ratio': ratio,
                    }
                ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F4: 死路因子 —— 尾字在词典中无后续候选
# ============================================================================

class DeadEndFactor(FactorDetector):
    """
    死路因子: 检查每个成语尾字 → 是否在词典中找到以它开头的成语。
    找不到 → 硬死路 → 需要学习该字开头的成语。
    """
    
    def __init__(self, field, landscape, idioms):
        super().__init__('dead_end', field, landscape, idioms, priority_weight=1.5)
        self._build_index()
    
    def _build_index(self):
        """构建首字→成语 和 尾字集合"""
        self.head_idx = defaultdict(list)
        self.all_tails = set()
        for w in self.idioms:
            if len(w) != 4: continue
            self.head_idx[w[0]].append(w)
            self.all_tails.add(w[-1])
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        t0 = time.time()
        gaps = []
        
        # 扫描所有尾字
        tail_freq = Counter()
        for w in self.idioms:
            if len(w) == 4:
                tail_freq[w[-1]] += 1
        
        for tail, count in tail_freq.most_common():
            if tail not in self.head_idx or len(self.head_idx[tail]) == 0:
                # 硬死路: 没有成语以此字开头
                gap_score = count * 3.0
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=tail,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'tail_count': count,
                        'head_count': 0,
                        'is_hard_dead_end': True,
                    }
                ))
            elif len(self.head_idx[tail]) == 1:
                # 软死路: 只有一个成语以此字开头 → 容易重复
                gap_score = count * 1.5
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=tail,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'tail_count': count,
                        'head_count': 1,
                        'only_option': self.head_idx[tail][0],
                        'is_soft_dead_end': True,
                    }
                ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F5: 梯度因子 —— 锚点梯度异常偏离基准
# ============================================================================

class GradientFactor(FactorDetector):
    """
    梯度盲区: 能量景观在锚点处的梯度异常。
    
    已训练锚点的梯度集中在 mean±σ 范围内。
    梯度显著偏离 → 该锚点区域的景观结构异常 → 需要更多学习来稳定。
    """
    
    def __init__(self, field, landscape, idioms):
        super().__init__('gradient', field, landscape, idioms, priority_weight=0.6)
        self.calibrated = False
        self.grad_mean = 0.0
        self.grad_std = 1.0
    
    def calibrate(self, n_samples=500):
        """校准: 在随机锚点上计算梯度分布"""
        indices = torch.randperm(self.field.num_hanzi)[:n_samples]
        grads = []
        
        self.landscape.train()
        for idx in indices:
            x = self.field.anchors[idx].clone().detach().to(self.device)
            x.requires_grad_(True)
            e = self.landscape(x.unsqueeze(0))
            e.backward()
            grads.append(x.grad.norm().item())
        
        arr = np.array(grads)
        self.grad_mean = float(arr.mean())
        self.grad_std = float(arr.std())
        self.calibrated = True
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        if not self.calibrated:
            self.calibrate()
        
        t0 = time.time()
        gaps = []
        
        # 只扫描在成语中出现的字(它们应该有良好训练的梯度)
        active_chars = set()
        for w in self.idioms:
            if len(w) == 4:
                for c in w:
                    active_chars.add(c)
        active_chars = [c for c in active_chars if c in self.ci]
        
        self.landscape.train()
        for ch in active_chars[:500]:  # 限制检测量
            idx = self.ci[ch]
            x = self.field.anchors[idx].clone().detach().to(self.device)
            x.requires_grad_(True)
            e = self.landscape(x.unsqueeze(0))
            e.backward()
            grad_norm = x.grad.norm().item()
            
            z = abs(grad_norm - self.grad_mean) / max(self.grad_std, 1e-6)
            if z > 4.0:  # 梯度异常偏离(>4σ)
                gap_score = z
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=ch,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'grad_norm': grad_norm,
                        'z_score': z,
                    }
                ))
        
        self.landscape.eval()
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F6: 语义群因子 —— 字簇间无跨簇连接
# ============================================================================

class SemanticGapFactor(FactorDetector):
    """
    语义孤岛: 基于字嵌入的余弦相似度聚类。
    
    如果两个高频字簇之间没有任何成语连接，说明存在语义孤岛。
    龙珠应该学习跨簇的关联。
    """
    
    def __init__(self, field, landscape, idioms):
        super().__init__('semantic', field, landscape, idioms, priority_weight=0.5)
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        # 简化版: 检测高频字之间的连接强度
        t0 = time.time()
        gaps = []
        
        # 取前200个高频字
        char_freq = Counter()
        for w in self.idioms:
            if len(w) == 4:
                for c in w:
                    char_freq[c] += 1
        
        top_chars = [c for c, _ in char_freq.most_common(200) if c in self.ci]
        
        # 检测哪些高频字对之间没有成语直接连接
        connected = set()
        for w in self.idioms:
            if len(w) == 4 and w[0] in self.ci and w[-1] in self.ci:
                connected.add((w[0], w[-1]))
        
        anchors = self.field.anchors.to(self.device)
        with torch.no_grad():
            for i, a in enumerate(top_chars):
                for b in top_chars[i+1:i+10]:  # 只检查相邻的
                    if (a, b) not in connected and (b, a) not in connected:
                        # 嵌入相似度高但没有连接 → 语义孤岛
                        va = anchors[self.ci[a]]
                        vb = anchors[self.ci[b]]
                        sim = torch.cosine_similarity(va.unsqueeze(0), 
                                                      vb.unsqueeze(0)).item()
                        if sim > 0.6:  # 高相似但无连接
                            gap_score = sim * 10
                            gaps.append(BlindSpot(
                                priority=-gap_score,
                                char=a,
                                factor=self.name,
                                score=gap_score,
                                evidence={
                                    'related_char': b,
                                    'similarity': sim,
                                }
                            ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# F7: 新鲜度因子 —— 从未被激活过的冷门字
# ============================================================================

class FreshnessFactor(FactorDetector):
    """
    新鲜度盲区: 检测高频汉字中哪些从未在能量景观中被"激活"过。
    
    "激活"定义: 该字的锚点嵌入作为查询向量被使用过，
    或该字出现在成语接龙链中。
    
    由于龙珠没有记录激活历史，这里用替代指标:
    字在中华常用字表中的排名 vs 在成语词典中的排名差异。
    """
    
    # 常用汉字表(前1000高频字, 来自现代汉语语料库)
    # 这里内置一个简化版
    COMMON_CHARS = set(
        '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情最何总通干光门社'
    )
    
    def __init__(self, field, landscape, idioms):
        super().__init__('freshness', field, landscape, idioms, priority_weight=0.7)
    
    def scan(self, partition_range=None) -> List[BlindSpot]:
        t0 = time.time()
        gaps = []
        
        # 统计成语中所有字
        idiom_chars = set()
        for w in self.idioms:
            if len(w) == 4:
                for c in w:
                    idiom_chars.add(c)
        
        # 常用字但不出现在任何成语中 → 知识盲区
        for ch in self.COMMON_CHARS:
            if ch not in idiom_chars and ch in self.ci:
                gap_score = 5.0  # 固定分数
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=ch,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'is_common': True,
                        'in_any_idiom': False,
                    }
                ))
        
        # 也检测: 在成语中出现但只出现在1-2个成语中的常用字
        char_count = Counter()
        for w in self.idioms:
            if len(w) == 4:
                for c in w:
                    char_count[c] += 1
        
        for ch in self.COMMON_CHARS:
            count = char_count.get(ch, 0)
            if 1 <= count <= 2 and ch in self.ci:
                gap_score = max(1.0, 5.0 - count)
                gaps.append(BlindSpot(
                    priority=-gap_score,
                    char=ch,
                    factor=self.name,
                    score=gap_score,
                    evidence={
                        'is_common': True,
                        'idiom_count': count,
                    }
                ))
        
        self._last_scan_time = time.time() - t0
        return gaps


# ============================================================================
# 多因子合并引擎
# ============================================================================

class MultiFactorDetector:
    """
    多因子盲区检测器 —— 7个因子并行/串行扫描，结果合并去重。
    
    使用方式:
        detector = MultiFactorDetector(field, landscape, idioms)
        
        # 全因子全分区扫描
        results = detector.scan_all()
        
        # 获取优先级最高的N个盲区
        top = detector.top_gaps(n=20)
        
        # 标记一个盲区为"已学习"(从队列中移除)
        detector.mark_learned(char)
    """
    
    def __init__(self, field: HanziAnchorField,
                 landscape: FreqEnergyLandscape,
                 idioms: list,
                 num_partitions: int = 8):
        self.field = field
        self.landscape = landscape
        self.idioms = idioms
        self.num_partitions = num_partitions
        
        # 初始化7个因子
        self.factors = [
            StatisticalFactor(field, landscape, idioms),
            EnergyFactor(field, landscape, idioms),
            CoverageFactor(field, landscape, idioms),
            DeadEndFactor(field, landscape, idioms),
            GradientFactor(field, landscape, idioms),
            SemanticGapFactor(field, landscape, idioms),
            FreshnessFactor(field, landscape, idioms),
        ]
        
        # 结果管理
        self._gap_queue: List[BlindSpot] = []  # 优先级堆
        self._gap_map: Dict[str, List[BlindSpot]] = defaultdict(list)  # char → gaps
        self._learned: Set[str] = set()  # 已学习的字
        self._last_scan_results: List[ScanResult] = []
        self._lock = threading.Lock()
        
        # 统计
        self.total_scans = 0
        self.total_gaps_found = 0
    
    def scan_all(self, parallel: bool = False, 
                 factors: List[str] = None) -> List[ScanResult]:
        """
        全因子扫描。
        
        Args:
            parallel: 是否并行扫描(多线程)
            factors: 指定因子名列表，None=全部
        
        Returns:
            各因子的扫描结果
        """
        t0 = time.time()
        self.total_scans += 1
        
        active_factors = [f for f in self.factors 
                         if factors is None or f.name in factors]
        
        results = []
        
        if parallel and len(active_factors) > 1:
            # 多线程并行扫描
            result_queue = queue.Queue()
            
            def _scan_worker(factor):
                try:
                    gaps = factor.scan()
                    result_queue.put(ScanResult(
                        factor=factor.name,
                        gaps=gaps,
                        time=factor._last_scan_time,
                        chars_scanned=len(gaps),
                    ))
                except Exception as e:
                    result_queue.put(ScanResult(
                        factor=factor.name,
                        gaps=[],
                        time=0,
                        chars_scanned=0,
                    ))
            
            threads = []
            for factor in active_factors:
                t = threading.Thread(target=_scan_worker, args=(factor,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            while not result_queue.empty():
                results.append(result_queue.get())
        else:
            # 串行扫描
            for factor in active_factors:
                try:
                    gaps = factor.scan()
                    results.append(ScanResult(
                        factor=factor.name,
                        gaps=gaps,
                        time=factor._last_scan_time,
                        chars_scanned=len(gaps),
                    ))
                except Exception as e:
                    print(f"  ⚠️ 因子 {factor.name} 扫描失败: {e}", flush=True)
                    results.append(ScanResult(
                        factor=factor.name,
                        gaps=[],
                        time=0,
                        chars_scanned=0,
                    ))
        
        # 合并结果
        with self._lock:
            self._merge_results(results)
        
        self._last_scan_results = results
        elapsed = time.time() - t0
        
        total_gaps = len(self._gap_map)
        print(f"  📊 扫描完成 ({elapsed:.1f}s): {total_gaps} 个盲区(去重)", flush=True)
        for r in results:
            if r.gaps:
                print(f"    {r.factor}: {len(r.gaps)} 个 ({r.time:.1f}s)", flush=True)
        
        return results
    
    def _merge_results(self, results: List[ScanResult]):
        """合并多个因子的扫描结果，去重，按优先级排序"""
        self._gap_map.clear()
        
        for result in results:
            for gap in result.gaps:
                if gap.char not in self._learned:
                    self._gap_map[gap.char].append(gap)
        
        # 重建优先级队列: 同一字符的多个因子取最高分
        import heapq
        self._gap_queue = []
        for char, gaps in self._gap_map.items():
            best = max(gaps, key=lambda g: g.score)
            # 多因子加分: 同一字符被多个因子发现 → 加分
            if len(gaps) >= 3:
                best.score *= 1.5  # 三因子共识 → 提权
                best.priority *= 1.5
            elif len(gaps) >= 2:
                best.score *= 1.2  # 两因子共识 → 小幅提权
                best.priority *= 1.2
            heapq.heappush(self._gap_queue, best)
        
        self.total_gaps_found = len(self._gap_map)
    
    def top_gaps(self, n: int = 20) -> List[BlindSpot]:
        """获取优先级最高的N个盲区"""
        import heapq
        with self._lock:
            # 复制堆(避免修改原堆)
            temp = list(self._gap_queue)
            result = heapq.nsmallest(n, temp)
        return result
    
    def pop_gap(self) -> Optional[BlindSpot]:
        """弹出一个优先级最高的盲区(用于学习)"""
        import heapq
        with self._lock:
            if not self._gap_queue:
                return None
            gap = heapq.heappop(self._gap_queue)
            return gap
    
    def mark_learned(self, char: str):
        """标记一个字符为'已学习'"""
        with self._lock:
            self._learned.add(char)
            if char in self._gap_map:
                del self._gap_map[char]
    
    def get_stats(self) -> Dict:
        """获取检测统计"""
        return {
            'total_scans': self.total_scans,
            'total_gaps_found': self.total_gaps_found,
            'pending_gaps': len(self._gap_queue),
            'learned_chars': len(self._learned),
            'factors': [
                {'name': f.name, 'last_scan': f._last_scan_time}
                for f in self.factors
            ],
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_detector(field, landscape, idioms, **kwargs) -> MultiFactorDetector:
    """快速创建多因子检测器"""
    return MultiFactorDetector(field, landscape, idioms, **kwargs)


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🐉 龙珠多因子盲区检测器 — 自测")
    print("=" * 60)
    
    import sys, os, json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n加载模型...")
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
        freeze=True
    )
    landscape = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    ).eval()
    
    with open(os.path.join(PROJECT, 'data/dicts/idioms.json'), encoding='utf-8') as f:
        idioms = json.load(f)
    
    print(f"字场:{field.num_hanzi} 成语:{len(idioms)}")
    
    # 创建检测器
    detector = MultiFactorDetector(field, landscape, idioms, num_partitions=4)
    
    # 全因子扫描
    print("\n🔍 全因子扫描...")
    results = detector.scan_all(parallel=False)
    
    # 展示结果
    print(f"\n📊 优先学习队列 (前20):")
    print(f"{'排名':<5} {'字':<4} {'因子':<12} {'分数':<8} {'证据'}")
    print("-" * 60)
    for i, gap in enumerate(detector.top_gaps(20)):
        evidence_str = str(gap.evidence)[:60]
        print(f"{i+1:<5} {gap.char:<4} {gap.factor:<12} {gap.score:<8.1f} {evidence_str}")
    
    print(f"\n✅ 检测器就绪 | 统计: {detector.get_stats()}")
