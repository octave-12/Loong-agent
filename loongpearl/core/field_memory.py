#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 三层记忆系统 — GPU热缓存 / RAM温层 / NVMe冷层
═══════════════════════════════════════════════════════

重心分散: GPU管速度, CPU管吞吐, SSD管容量。没有单个零件过载。

三层:
  GPU 热缓存: ~200K 模式, 最近激活的常驻显存 (400MB float16)
  RAM 温层:   ~600K 模式, OS mmap 页缓存自动热管理
  NVMe 冷层:  全量 1.2M+ 模式, 流式按需扫描

查询流程:
  热缓存 → 不够深? → 温层 → 还不够? → 冷层全扫
  最终收敛后, 激活的模式晋升到热缓存。
"""

import torch
import os
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .dragon_field import DragonField, FieldResult

log = logging.getLogger(__name__)


@dataclass
class TieredResult:
    """分层查询结果 — 比 FieldResult 多了层级信息"""
    field_result: FieldResult
    tier: str              # 'hot' | 'warm' | 'cold'
    elapsed_ms: float
    patterns_scanned: int


class FieldMemory:
    """
    三层记忆管理器。

    管理 DragonField 的存储分层, 提供统一的查询接口。
    CPU 和 GPU 并行打分 (分块), SSD 按需流式加载。
    """

    def __init__(
        self,
        field: DragonField,
        hot_size: int = 200_000,
        warm_size: int = 600_000,
        device: str = 'cuda',
    ):
        self.field = field
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.device = device

        # 热缓存索引 — GPU 上的模式子集
        self._hot_indices: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._hot_threshold: float = 0.85  # 热缓存信任阈值

        # 温层 mmap — 常驻系统内存
        self._warm_patterns: Optional[torch.Tensor] = None
        self._warm_indices: List[int] = []

        # 全量索引 — SSD 上
        self._full_path: Optional[str] = None

        # 主题→模式索引映射 (O(1) 精确查询)
        self._subject_index: Dict[str, List[int]] = {}
        self._build_subject_index()

        # 线程安全
        self._query_lock = threading.Lock()

        # 统计
        self._stats = {
            'hot_hits': 0,
            'warm_hits': 0,
            'cold_hits': 0,
            'total_queries': 0,
        }

        # 初始化分层
        self._init_tiers()

    def _build_subject_index(self):
        """构建 subject → 模式索引列表的映射"""
        for i, subj in enumerate(self.field._pattern_subjects):
            if subj not in self._subject_index:
                self._subject_index[subj] = []
            self._subject_index[subj].append(i)
        distinct = len(self._subject_index)
        log.info(f"主题索引: {distinct:,} 个唯一概念 → {self.field.num_patterns:,} 个模式")

    def _init_tiers(self):
        """根据模式总数分配三层"""
        N = self.field.num_patterns
        if N == 0:
            return

        # 热缓存: 前 hot_size 个 (后续按激活次数动态调整)
        hot_n = min(self.hot_size, N)
        self._hot_indices = torch.arange(hot_n, dtype=torch.long)

        # 温层: 接下来的 warm_size 个
        warm_start = hot_n
        warm_n = min(self.warm_size, N - hot_n)
        self._warm_indices = list(range(warm_start, warm_start + warm_n))

        # 剩下的就是冷层

        log.info(
            f"三层记忆: 热{hot_n} / 温{warm_n} / 冷{N - hot_n - warm_n} "
            f"(总数 {N})"
        )

    # ── 查询 ──────────────────────────────────────────────────

    def query(
        self,
        query_vec: torch.Tensor,
        hot_threshold: float = 0.85,
        max_steps: int = 30,
        project_to_sphere: bool = True,
    ) -> TieredResult:
        """
        分层查询: 热→温→冷逐级下沉。

        Args:
            query_vec: (D,) 查询向量 (应在目标设备上)
            hot_threshold: 热缓存信任阈值 (相似度 > 此值 → 信任热缓存)
            max_steps: 收敛步数
            project_to_sphere: 是否投影回球面

        Returns:
            TieredResult
        """
        self._stats['total_queries'] += 1
        t_start = time.time()

        # ── 第一级: GPU 热缓存 ──
        hot = self._get_hot_patterns()
        if hot is not None and hot.shape[0] > 0:
            result = self.field.converge(
                query_vec, patterns=hot,
                max_steps=max_steps,
                convergence_threshold=1e-4,
                project_to_sphere=project_to_sphere,
            )
            if result.top_similarities and result.top_similarities[0] >= hot_threshold:
                self._stats['hot_hits'] += 1
                self.field.bump_activation(result.top_pattern_indices[:3])
                return TieredResult(
                    field_result=result,
                    tier='hot',
                    elapsed_ms=(time.time() - t_start) * 1000,
                    patterns_scanned=hot.shape[0],
                )

        # ── 第二级: RAM 温层 (CPU 打分) ──
        warm = self._get_warm_patterns()
        if warm is not None and warm.shape[0] > 0:
            # CPU 上做相似度打分 — 并行于 GPU 其他任务
            q_cpu = query_vec.cpu().to(torch.float32)
            sim_cpu = (warm @ q_cpu)  # (warm_N,)

            # 合并热缓存的结果用于最终收敛
            if hot is not None and result is not None:
                # 合并热+温的 score, 在 GPU 上做最终收敛
                sim_hot = (hot @ query_vec).cpu()
                all_sim = torch.cat([sim_hot, sim_cpu])
                all_pat = torch.cat([
                    hot.cpu().to(torch.float32),
                    warm
                ], dim=0)
            else:
                all_sim = sim_cpu
                all_pat = warm

            result = self.field.converge(
                query_vec.cpu().to(torch.float32),
                patterns=all_pat,
                max_steps=max_steps,
                convergence_threshold=1e-4,
                project_to_sphere=project_to_sphere,
            )

            if result.top_similarities and result.top_similarities[0] >= 0.7:
                self._stats['warm_hits'] += 1
                self.field.bump_activation(result.top_pattern_indices[:5])
                self._promote_to_hot(result.top_pattern_indices[:5])
                return TieredResult(
                    field_result=result,
                    tier='warm',
                    elapsed_ms=(time.time() - t_start) * 1000,
                    patterns_scanned=all_pat.shape[0],
                )

        # ── 第三级: NVMe 全量扫描 (流式, 不分块到显存) ──
        cold_result = self._cold_scan(
            query_vec,
            max_steps=max_steps,
            project_to_sphere=project_to_sphere,
        )
        self._stats['cold_hits'] += 1
        if cold_result.top_similarities:
            self.field.bump_activation(cold_result.top_pattern_indices[:10])
            self._promote_to_hot(cold_result.top_pattern_indices[:10])

        return TieredResult(
            field_result=cold_result,
            tier='cold',
            elapsed_ms=(time.time() - t_start) * 1000,
            patterns_scanned=self.field.num_patterns,
        )

    def _get_hot_patterns(self) -> Optional[torch.Tensor]:
        """获取GPU热缓存模式"""
        if self.field.num_patterns == 0:
            return None
        if self.field.num_patterns <= self.hot_size:
            return self.field.patterns.to(
                device=self.device, dtype=torch.float32
            )
        idx = self._hot_indices[:self.hot_size]
        return self.field.patterns[idx].to(
            device=self.device, dtype=torch.float32
        )

    def _get_warm_patterns(self) -> Optional[torch.Tensor]:
        """获取温层模式 (CPU上, OS mmap自动管理)"""
        if not self._warm_indices:
            return None
        idx = torch.tensor(self._warm_indices, dtype=torch.long)
        return self.field.patterns[idx].to(dtype=torch.float32)  # 保持 CPU

    def _cold_scan(
        self,
        query_vec: torch.Tensor,
        max_steps: int = 30,
        batch_size: int = 300_000,
        project_to_sphere: bool = True,
    ) -> FieldResult:
        """
        NVMe 全量扫描 — 流式分块，不占显存。
        CPU 逐块打分，GPU 只做最终收敛。
        """
        N = self.field.num_patterns
        q_cpu = query_vec.cpu().to(torch.float32)
        patterns_cpu = self.field.patterns.to(dtype=torch.float32)

        best_sim = -float('inf')
        best_idx = -1
        all_top_scores = []

        # 分块扫全量
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = patterns_cpu[start:end]  # mmap → 按需读 SSD

            sim = batch @ q_cpu  # (batch,)
            batch_top = torch.topk(sim, k=min(5, end - start))

            if batch_top.values[0] > best_sim:
                best_sim = batch_top.values[0].item()
                best_idx = start + batch_top.indices[0].item()

            all_top_scores.extend(
                list(zip(
                    (start + batch_top.indices).tolist(),
                    batch_top.values.tolist()
                ))
            )

        # 取全局 top-10
        all_top_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in all_top_scores[:10]]

        # 用 top 候选在 GPU 上精确收敛
        top_pat = patterns_cpu[torch.tensor(top_indices)].to(
            device=self.device, dtype=torch.float32
        )
        q_gpu = query_vec.to(device=self.device, dtype=torch.float32)

        return self.field.converge(
            q_gpu, patterns=top_pat,
            max_steps=max_steps,
            convergence_threshold=1e-5,
            project_to_sphere=project_to_sphere,
        )

    # ── 热缓存管理 ──────────────────────────────────────────

    def _promote_to_hot(self, pattern_indices: List[int]):
        """将激活的模式晋升到热缓存"""
        if self.hot_size <= 0:
            return
        current_hot = set(self._hot_indices.tolist())
        for idx in pattern_indices:
            if idx is None or idx >= self.field.num_patterns:
                continue
            if idx not in current_hot:
                while len(current_hot) >= self.hot_size:
                    evict = self.field._activation_count[
                        self._hot_indices
                    ].argmin().item()
                    to_remove = self._hot_indices[evict].item()
                    if to_remove in current_hot:
                        current_hot.remove(to_remove)
                    else:
                        # 如果不在集合中, 随便弹一个
                        current_hot.pop()
                current_hot.add(idx)
        self._hot_indices = torch.tensor(
            sorted(current_hot), dtype=torch.long
        )

    def refresh_hot_cache(self):
        """定期刷新热缓存: 按激活次数排序"""
        if self.field._activation_count.numel() == 0:
            return
        _, top_k = torch.topk(
            self.field._activation_count.float(),
            k=min(self.hot_size, self.field.num_patterns)
        )
        self._hot_indices = top_k.sort().values

    # ── 统计 ──────────────────────────────────────────────────

    def query_by_concept(
        self,
        concept: str,
        query_vec: torch.Tensor,
        max_candidates: int = 5000,
        max_steps: int = 20,
    ) -> TieredResult:
        """
        按概念主题精确查询。

        先按 subject 精确匹配筛选候选模式,
        再与 BGE 近邻合并, 在小范围内收敛。

        Args:
            concept: 查询概念词 (如 "龙", "原子")
            query_vec: BGE 编码的查询向量
            max_candidates: 候选模式数上限
            max_steps: 收敛步数

        Returns:
            TieredResult
        """
        import time
        t_start = time.time()
        self._stats['total_queries'] += 1

        # 1. 精确主题匹配: O(1) 从索引查
        subject_indices = self._subject_index.get(concept, [])
        if len(subject_indices) > max_candidates:
            subject_indices = subject_indices[:max_candidates]

        # 2. BGE 近邻: CPU 全量打分, 找 top-k
        patterns_cpu = self.field.patterns.to(dtype=torch.float32)
        q_cpu = query_vec.cpu().to(torch.float32)
        sim_all = patterns_cpu @ q_cpu  # (N,)
        _, bge_top = torch.topk(sim_all, k=min(max_candidates, self.field.num_patterns))
        bge_indices = bge_top.tolist()

        # 3. 合并候选集 (去重)
        candidate_set = set(subject_indices) | set(bge_indices[:max_candidates//2])
        candidates = list(candidate_set)[:max_candidates]

        if not candidates:
            # 无主题匹配, 回退到分层查询
            return self.query(query_vec)

        # 4. 在候选子集上收敛
        cand_tensor = torch.tensor(candidates, dtype=torch.long)
        subset = patterns_cpu[cand_tensor].to(
            device=self.device, dtype=torch.float32
        )
        q_dev = query_vec.to(device=self.device, dtype=torch.float32)

        result = self.field.converge(
            q_dev, patterns=subset,
            max_steps=max_steps,
            convergence_threshold=1e-5,
        )

        # 将子集索引映射回全局索引
        if result.top_pattern_indices:
            result.top_pattern_indices = [
                candidates[i] for i in result.top_pattern_indices
            ]

        # 记录
        self.field.bump_activation(result.top_pattern_indices[:10])
        self._promote_to_hot(result.top_pattern_indices[:10])

        elapsed = (time.time() - t_start) * 1000
        return TieredResult(
            field_result=result,
            tier='concept' if subject_indices else 'bge',
            elapsed_ms=elapsed,
            patterns_scanned=len(candidates),
        )

    @property
    def stats(self) -> Dict:
        return {
            **self._stats,
            'hot_ratio': (
                self._stats['hot_hits'] / max(self._stats['total_queries'], 1)
            ),
            'memory_usage': {
                'hot_gpu': (
                    len(self._hot_indices) * self.field.embed_dim * 2
                    / 1024**2
                ),
                'warm_ram': (
                    len(self._warm_indices) * self.field.embed_dim * 2
                    / 1024**2
                ),
                'cold_ssd': self.field.num_patterns * self.field.embed_dim * 2
                             / 1024**2,
            },
        }

    def print_stats(self):
        s = self.stats
        log.info(
            f"三层记忆: {s['total_queries']}查询 "
            f"(热{s['hot_ratio']:.0%} 温{s.get('warm_ratio',0):.0%}) "
            f"GPU:{s['memory_usage']['hot_gpu']:.0f}MB "
            f"RAM:{s['memory_usage']['warm_ram']:.0f}MB "
            f"SSD:{s['memory_usage']['cold_ssd']:.0f}MB"
        )
