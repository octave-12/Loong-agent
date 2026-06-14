#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量渲染：复用已有 UMAP 降维结果，只追加更多梯度下降轨迹。"""

import sys, os, time, pickle, numpy as np, torch
import plotly.graph_objects as go

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape
from visualize_landscape_3d import LandscapeVisualizer3D

# ── 全量轨迹列表 ──────────────────────────────────────────────
ALL_TRAJECTORIES = [
    # 原有
    "龙", "宇宙", "知识",
    # 自然
    "火", "水", "山", "光", "雷", "风",
    # 人文
    "爱", "心", "梦", "道", "禅", "诗",
    # 抽象
    "时间", "生命", "自由", "真理", "无限",
    # 龙珠相关
    "悟", "空", "战", "界", "神",
]

CACHE_DIR = os.path.join(PROJECT_DIR, ".vis_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

UMAP_CACHE = os.path.join(CACHE_DIR, "umap_3d.pkl")
GRID_CACHE = os.path.join(CACHE_DIR, "grid_3d.npz")

def main():
    print("=" * 60)
    print("龙珠能量景观 · 全量轨迹增量渲染")
    print("=" * 60)

    # 1. 加载字场 + 能量景观
    vis = LandscapeVisualizer3D()
    vis.load()

    # 2. UMAP（检查缓存）
    if os.path.exists(UMAP_CACHE) and os.path.exists(GRID_CACHE):
        print("[跳过] UMAP + 网格已有缓存，直接加载...")
        with open(UMAP_CACHE, "rb") as f:
            cache = pickle.load(f)
        vis.umap_model = cache["umap"]
        vis.anchors_3d = cache["anchors_3d"]
        vis._anchor_min = cache["anchor_min"]
        vis._anchor_max = cache["anchor_max"]
        vis.anchors_3d_norm = cache["anchors_3d_norm"]

        grid = np.load(GRID_CACHE)
        vis.grid_energies = grid["energies"]
        vis.grid_x = grid["xx"]; vis.grid_y = grid["yy"]; vis.grid_z = grid["zz"]
    else:
        print("[2/3] UMAP 降维 (1024d → 3d)...")
        vis.reduce_dimensions(n_fit=15000)

        # 保存 UMAP 缓存
        with open(UMAP_CACHE, "wb") as f:
            pickle.dump({
                "umap": vis.umap_model,
                "anchors_3d": vis.anchors_3d,
                "anchor_min": vis._anchor_min,
                "anchor_max": vis._anchor_max,
                "anchors_3d_norm": vis.anchors_3d_norm,
            }, f)
        print(f"      UMAP 缓存已保存至 {UMAP_CACHE}")

        print("[3/3] 计算能量网格 (25³)...")
        vis.compute_energy_grid(resolution=25)

        np.savez_compressed(GRID_CACHE,
                            energies=vis.grid_energies,
                            xx=vis.grid_x, yy=vis.grid_y, zz=vis.grid_z)
        print(f"      网格缓存已保存至 {GRID_CACHE}")

    # 3. 渲染（全量轨迹）
    output = os.path.join(PROJECT_DIR, "landscape_3d.html")
    vis.render(n_anchors=500, trajectories=ALL_TRAJECTORIES, output_path=output)

    print(f"\n✅ 全量 {len(ALL_TRAJECTORIES)} 条推理轨迹已植入")
    print(f"   用浏览器打开: {output}")

if __name__ == "__main__":
    main()
