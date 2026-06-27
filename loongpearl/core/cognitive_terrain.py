#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3: 认知地形景观 (Cognitive Terrain Landscape)

将能量景观从"打分器"升级为"导航器"。
输出不是"这个概念对不对"，而是"哪里需要学"。

核心功能:
  1. 景观评分: 给概念及其邻居打分
  2. 地形分类: mastered / fuzzy / blind_spot
  3. 盲区导航: 找最近的 mastered 概念 → 给出学习路径
  4. 对接 KnowledgePipeline: 替代当前的粗糙盲区检测

使用:
  terrain = CognitiveTerrain(landscape_path, field_path, db_path)
  region = terrain.survey("量子")  # 量子周围的认知地形
  blind = terrain.top_blind_spots(n=20)  # 最该学的20个盲区
"""

import os
import sqlite3
import logging
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TerrainPoint:
    """地形上的一个点"""
    concept: str
    energy: float
    zone: str                # 'mastered' | 'fuzzy' | 'blind_spot'
    neighbors: List[str] = field(default_factory=list)
    nearest_mastered: Optional[str] = None
    nearest_distance: float = 999.0


class CognitiveTerrain:
    """
    认知地形 — 能量景观的导航层。

    能量解释:
      低能量 (接近 0)   = 锚点/吸引子 = 已掌握的概念
      中等能量 (0.1~0.5) = 过渡区      = 有线索但不完整的模糊区
      高能量 (>0.5)     = 壁垒区      = 未知/盲区

    地形不是固定的——每次学新概念后，相关的能量低谷会扩展。
    """

    def __init__(
        self,
        landscape_path: str = None,
        field_path: str = None,
        db_path: str = None,
        device: str = "cpu",
    ):
        # 路径解析
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        models_dir = os.path.join(project_root, "data", "models")

        if landscape_path is None:
            landscape_path = os.path.join(models_dir, "energy_landscape_1024d.pt")
        if field_path is None:
            field_path = os.path.join(models_dir, "zichang_94117_1024d.pt")
        if db_path is None:
            db_path = os.path.join(project_root, "data", "models", "concept_graph.db")

        self.landscape_path = landscape_path
        self.field_path = field_path
        self.db_path = db_path
        self.device = device

        # 懒加载
        self._landscape = None
        self._field = None
        self._conn: Optional[sqlite3.Connection] = None
        self._concept_vectors: Dict[str, np.ndarray] = {}
        self._loaded = False

    # ── 加载 ──

    def load(self) -> 'CognitiveTerrain':
        """加载能量景观和字场"""
        if self._loaded:
            return self

        # 加载景观
        try:
            from loongpearl.core.freq_landscape import FreqEnergyLandscape
            self._landscape = FreqEnergyLandscape()
            state = torch.load(self.landscape_path, map_location=self.device)
            if isinstance(state, dict):
                # 提取嵌套的 model_state_dict
                weights = state.get("model_state_dict", state)
                self._landscape.load_state_dict(weights, strict=False)
            else:
                self._landscape = state
            self._landscape.to(self.device)
            self._landscape.eval()
            log.info(f"景观加载: {self.landscape_path}")
        except Exception as e:
            log.error(f"景观加载失败: {e}")
            self._landscape = None

        # 加载字场
        try:
            from loongpearl.core.zichang import HanziAnchorField
            self._field = HanziAnchorField.load(self.field_path, freeze=True)
            log.info(f"字场加载: {self.field_path} ({len(self._field.hanzi_list)} 汉字)")
        except Exception as e:
            log.error(f"字场加载失败: {e}")
            self._field = None

        self._loaded = True
        return self

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # ── 向量获取 ──

    def _get_vector(self, concept: str) -> Optional[np.ndarray]:
        """获取概念的向量表示（优先字场，回退到概念图嵌入）"""
        if concept in self._concept_vectors:
            return self._concept_vectors[concept]

        # 单字 → 字场
        if len(concept) == 1 and self._field:
            try:
                idx = self._field._char_to_idx.get(concept)
                if idx is not None:
                    vec = self._field.anchors[idx].cpu().numpy()
                    self._concept_vectors[concept] = vec
                    return vec
            except Exception:
                pass

        # 多字 → 字场平均
        if self._field:
            try:
                vecs = []
                for ch in concept:
                    idx = self._field._char_to_idx.get(ch)
                    if idx is not None:
                        vecs.append(self._field.anchors[idx].cpu().numpy())
                if vecs:
                    vec = np.mean(vecs, axis=0)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    self._concept_vectors[concept] = vec
                    return vec
            except Exception:
                pass

        return None

    # ── 地形勘探 ──

    def score(self, concept: str) -> float:
        """给单个概念打能量分（越低越掌握）"""
        vec = self._get_vector(concept)
        if vec is None or self._landscape is None:
            return 999.0

        with torch.no_grad():
            t = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            energy = self._landscape(t).item()
        return energy

    def classify(self, energy: float) -> str:
        """能量值 → 地形分类（基于实际景观分布校准）

        景观特征: 锚点 < 0, 非锚点 > 0, 无向量 = 999
        校准策略: 负能量→掌握, 近零→模糊, 正能量→盲区
        """
        if energy >= 999.0:
            return "blind_spot"
        elif energy < -10.0:
            return "mastered"
        elif energy < 0.0:
            return "fuzzy"
        else:
            return "blind_spot"

    def survey(self, concept: str, radius: int = 30) -> Dict:
        """
        勘探某个概念周围的认知地形。

        步骤:
          1. 从概念图找到邻居概念
          2. 对每个邻居评分
          3. 按地形分类
          4. 对盲区找最近 mastered 邻居

        Returns:
            {
                "center": TerrainPoint,
                "neighbors": [TerrainPoint, ...],
                "blind_count": int,
                "fuzzy_count": int,
                "mastered_count": int,
                "learning_hints": [(blind, nearest_mastered), ...],
            }
        """
        if not self._loaded:
            self.load()

        center_vec = self._get_vector(concept)
        center_energy = self.score(concept)
        center = TerrainPoint(
            concept=concept,
            energy=center_energy,
            zone=self.classify(center_energy),
        )

        # 找邻居: concept_graph.db 中以 concept 为主语或宾语的三元组
        neighbors_set = set()
        rows = self.conn.execute(
            "SELECT DISTINCT o FROM triples WHERE s=? AND r IN ('IS_A','RELATED','COOCCURS_WITH','DEFINED_AS') LIMIT ?",
            (concept, radius)
        ).fetchall()
        for (o,) in rows:
            if o and o != concept:
                neighbors_set.add(o)

        rows = self.conn.execute(
            "SELECT DISTINCT s FROM triples WHERE o=? AND r IN ('IS_A','RELATED','COOCCURS_WITH') LIMIT ?",
            (concept, radius)
        ).fetchall()
        for (s,) in rows:
            if s and s != concept:
                neighbors_set.add(s)

        # 对邻居评分
        neighbors = []
        mastered_points = []

        for neighbor in list(neighbors_set)[:radius]:
            energy = self.score(neighbor)
            zone = self.classify(energy)
            tp = TerrainPoint(concept=neighbor, energy=energy, zone=zone)
            neighbors.append(tp)
            if zone == "mastered":
                mastered_points.append(tp)

        # 排序
        neighbors.sort(key=lambda x: x.energy)

        # 为盲区找最近 mastered
        learning_hints = []
        for tp in neighbors:
            if tp.zone == "blind_spot" and mastered_points:
                # 简单策略: 找能量最低的 mastered（最近的低谷）
                best = min(mastered_points, key=lambda m: m.energy)
                tp.nearest_mastered = best.concept
                tp.nearest_distance = tp.energy - best.energy
                learning_hints.append((tp, best))

        # 统计
        zones = {"mastered": 0, "fuzzy": 0, "blind_spot": 0}
        for tp in neighbors:
            zones[tp.zone] += 1

        return {
            "center": center,
            "neighbors": neighbors,
            "blind_count": zones["blind_spot"],
            "fuzzy_count": zones["fuzzy"],
            "mastered_count": zones["mastered"],
            "learning_hints": [(h[0].concept, h[1].concept, h[0].energy)
                             for h in learning_hints[:10]],
        }

    def top_blind_spots(self, n: int = 20, sample_size: int = 1000) -> List[TerrainPoint]:
        """
        全局搜索：找能量最高的概念（最该学的盲区）。

        从概念图中随机采样概念，按能量排序。
        """
        if not self._loaded:
            self.load()

        # 从 concept_graph 随机采样概念
        rows = self.conn.execute(
            "SELECT DISTINCT s FROM triples ORDER BY RANDOM() LIMIT ?",
            (sample_size,)
        ).fetchall()

        points = []
        for (concept,) in rows:
            if not concept:
                continue
            energy = self.score(concept)
            zone = self.classify(energy)
            points.append(TerrainPoint(concept=concept, energy=energy, zone=zone))

        points.sort(key=lambda x: x.energy, reverse=True)
        return points[:n]

    def terrain_for_pipeline(self, max_demands: int = 10) -> list:
        """
        为 KnowledgePipeline 提供高质量的需求检测。

        替代当前的 _detect_blind_spots (粗糙的随机采样盲区字符)。
        """
        blind = self.top_blind_spots(n=max_demands * 2)

        demands = []
        for tp in blind:
            # 为每个盲区找最近的 mastered 邻居 → 学习锚点
            survey = self.survey(tp.concept, radius=20)
            nearest = None
            for h in survey.get("learning_hints", []):
                if h[0] == tp.concept:
                    nearest = h[1]
                    break

            demand = {
                "type": "BLIND_SPOT",
                "target": tp.concept,
                "priority": min(1.0, tp.energy / 10.0),  # 能量越高越优先
                "context": {
                    "energy": tp.energy,
                    "zone": tp.zone,
                    "nearest_mastered": nearest,
                },
            }
            demands.append(demand)

        # 按优先级排序
        demands.sort(key=lambda d: d["priority"], reverse=True)
        return demands[:max_demands]

    def stats(self) -> Dict:
        """地形统计"""
        if not self._loaded:
            self.load()

        # 采样统计
        rows = self.conn.execute(
            "SELECT DISTINCT s FROM triples ORDER BY RANDOM() LIMIT 500"
        ).fetchall()

        zones = {"mastered": 0, "fuzzy": 0, "blind_spot": 0}
        energies = []

        for (concept,) in rows:
            energy = self.score(concept)
            energies.append(energy)
            zones[self.classify(energy)] += 1

        return {
            "total_sampled": len(rows),
            "mastered_pct": zones["mastered"] / len(rows) * 100,
            "fuzzy_pct": zones["fuzzy"] / len(rows) * 100,
            "blind_pct": zones["blind_spot"] / len(rows) * 100,
            "avg_energy": np.mean(energies),
            "median_energy": np.median(energies),
            "min_energy": np.min(energies),
            "max_energy": np.max(energies),
        }


# ── CLI ──
if __name__ == "__main__":
    import sys

    terrain = CognitiveTerrain()
    terrain.load()

    print("=== 认知地形统计 ===")
    for k, v in terrain.stats().items():
        print(f"  {k}: {v}")

    concept = sys.argv[1] if len(sys.argv) > 1 else "量子"
    print(f"\n=== 勘探 [{concept}] ===")
    result = terrain.survey(concept, radius=30)

    center = result["center"]
    print(f"  中心: {center.concept} (能量={center.energy:.3f}, {center.zone})")
    print(f"  掌握: {result['mastered_count']}  模糊: {result['fuzzy_count']}  盲区: {result['blind_count']}")
    print(f"  邻居 Top-10:")
    for tp in result["neighbors"][:10]:
        marker = "←" if tp.zone == "mastered" else " " if tp.zone == "fuzzy" else "⚠"
        print(f"    {marker} {tp.concept:20s} 能量={tp.energy:7.3f}  {tp.zone}")

    if result["learning_hints"]:
        print(f"\n  学习提示 (盲区 → 最近掌握概念):")
        for blind, nearest, energy in result["learning_hints"][:5]:
            print(f"    ⚠ {blind:20s} → {nearest:20s} (盲区能量={energy:.3f})")

    print(f"\n=== 全局 Top-10 盲区 ===")
    for tp in terrain.top_blind_spots(10):
        print(f"  ⚠ {tp.concept:20s} 能量={tp.energy:.3f}")
