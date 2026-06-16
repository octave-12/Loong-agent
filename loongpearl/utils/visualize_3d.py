#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠能量景观三维可视化（visualize_landscape_3d.py）
=====================================================
将 94117 汉字锚点从 1024 维降到 3 维，用 Plotly 绘制交互式能量景观。

可视化层次:
  图层1: 能量等值面（isosurface）—— 深蓝=低谷(已知), 浅色=山峰(未知)
  图层2: 汉字锚点散点 —— 红色圆点，hover 显示汉字和能量值
  图层3: 梯度下降轨迹 —— 彩线从随机起点到收敛锚点

依赖: torch, umap-learn, plotly, numpy
输出: landscape_3d.html (可交互HTML，浏览器打开即可旋转/缩放)

用法:
  python visualize_landscape_3d.py
  python visualize_landscape_3d.py --n-anchors 800   # 显示更多锚点
  python visualize_landscape_3d.py --resolution 25    # 更细的等值面网格

作者: Hermes + 李泽坤
版本: 1.0.0
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR


# ============================================================================
# 配置
# ============================================================================

# UMAP 采样量（全量 94117 太慢，取子集做降维，再用变换映射其余点）
UMAP_FIT_SAMPLES = 15000

# 默认参数
DEFAULT_N_ANCHORS = 500       # 散点图层显示的锚点数
DEFAULT_GRID_RES = 30         # 等值面网格分辨率 (30^3 = 27,000 采样点)
DEFAULT_TRAJECTORIES = [      # 推理演示: 每个元素为 (文字, 起始偏移方向)
    ("龙", None),
    ("宇宙", None),
    ("知识", None),
]

# 输出路径
OUTPUT_HTML = os.path.join(PROJECT_DIR, "landscape_3d.html")


# ============================================================================
# 核心可视化器
# ============================================================================

class LandscapeVisualizer3D:
    """
    龙珠能量景观 3D 可视化器。
    
    流程:
      1. 加载字场 + 能量景观
      2. UMAP 降维 (1024d → 3d)
      3. 构建 3D 网格 + 计算能量值
      4. Plotly 渲染 → HTML
    """
    
    def __init__(
        self,
        zichang_path: str = None,
        landscape_path: str = None,
        device: str = "cpu",
    ):
        """
        初始化可视化器。
        
        Args:
            zichang_path: 字场文件路径
            landscape_path: 能量景观文件路径
            device: 计算设备
        """
        self.zichang_path = zichang_path or os.path.join(
            PROJECT_DIR, "data/models/zichang_94117_1024d.pt"
        )
        self.landscape_path = landscape_path or os.path.join(
            PROJECT_DIR, "data/models/energy_landscape_1024d.pt"
        )
        self.device = device
        
        # 加载后填充
        self.zichang = None
        self.landscape = None
        self.anchors = None          # 全量锚点 (N, 1024)
        self.anchors_3d = None       # 降维后的锚点 (N, 3)
        self.umap_model = None       # UMAP 模型（用于变换新点）
        self.grid_energies = None    # 3D 网格能量值
        self.loaded = False
    
    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    
    def load(self, verbose: bool = True):
        """加载字场和能量景观"""
        print("[1/5] 加载字场...")
        self.zichang = HanziAnchorField.load(self.zichang_path)
        self.anchors = self.zichang.anchors.float()
        print(f"      字场: {self.zichang.num_hanzi} 汉字 × {self.zichang.embed_dim}维")
        
        print("[2/5] 加载能量景观...")
        self.landscape = EnergyLandscape.load(self.landscape_path)
        self.landscape.eval()
        self.landscape.to(self.device)
        n_params = sum(p.numel() for p in self.landscape.parameters())
        print(f"      能量景观: {n_params:,} 参数")
        
        self.loaded = True
    
    # ------------------------------------------------------------------
    # UMAP 降维
    # ------------------------------------------------------------------
    
    def reduce_dimensions(self, n_fit: int = UMAP_FIT_SAMPLES, verbose: bool = True):
        """
        用 UMAP 将 1024 维锚点降到 3 维。
        
        策略: 在全量锚点上做 UMAP 太慢 (>10分钟)。
              取 n_fit 个锚点训练 UMAP，然后用 transform() 映射其余点。
        
        Args:
            n_fit: 用于训练 UMAP 的锚点数量
        """
        if not self.loaded:
            raise RuntimeError("请先调用 load()")
        
        print(f"[3/5] UMAP 降维 (1024d → 3d, 训练集={n_fit})...")
        t0 = time.time()
        
        # 采样训练集
        n_total = self.anchors.shape[0]
        n_fit = min(n_fit, n_total)
        
        rng = np.random.default_rng(42)
        fit_indices = rng.choice(n_total, n_fit, replace=False)
        fit_data = self.anchors[fit_indices].numpy()
        
        # 训练 UMAP
        self.umap_model = umap.UMAP(
            n_components=3,
            metric='cosine',
            n_neighbors=30,
            min_dist=0.1,
            n_jobs=-1,          # 用满全部 CPU 核心
            verbose=verbose,
        )
        fit_3d = self.umap_model.fit_transform(fit_data)
        
        # 变换全量锚点
        all_data = self.anchors.numpy()
        self.anchors_3d = self.umap_model.transform(all_data)
        
        elapsed = time.time() - t0
        print(f"      完成: {self.anchors_3d.shape} ({elapsed:.1f}s)")
        
        # 归一化到 [0, 1] 范围（方便后续网格定义）
        self._anchor_min = self.anchors_3d.min(axis=0)
        self._anchor_max = self.anchors_3d.max(axis=0)
        self.anchors_3d_norm = (self.anchors_3d - self._anchor_min) / (
            self._anchor_max - self._anchor_min + 1e-8
        )
    
    # ------------------------------------------------------------------
    # 能量网格
    # ------------------------------------------------------------------
    
    def compute_energy_grid(self, resolution: int = DEFAULT_GRID_RES):
        """
        在 3D UMAP 空间中构建网格，计算每个格点的能量值。
        
        做法:
          1. 在归一化空间中定义 resolution^3 个格点
          2. 用 UMAP inverse_transform 映射回 1024 维
          3. 在能量景观中计算能量
        
        注意: UMAP inverse_transform 是近似的，但足以用于可视化。
        
        Args:
            resolution: 每维网格分辨率 (resolution^3 个采样点)
        """
        if self.umap_model is None:
            raise RuntimeError("请先调用 reduce_dimensions()")
        
        n_points = resolution ** 3
        print(f"[4/5] 计算能量网格 ({resolution}³ = {n_points:,} 格点)...")
        t0 = time.time()
        
        # 在归一化空间 [0,1]³ 中生成均匀网格
        lin = np.linspace(0.05, 0.95, resolution)  # 留 5% 边距
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
        grid_3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # 反归一化
        grid_3d_unnorm = grid_3d * (self._anchor_max - self._anchor_min) + self._anchor_min
        
        # UMAP 逆变换: 3D → 1024D
        grid_1024d = self.umap_model.inverse_transform(grid_3d_unnorm)
        
        # 归一化（与锚点保持在同一球面上）
        grid_tensor = torch.from_numpy(grid_1024d).float()
        grid_tensor = torch.nn.functional.normalize(grid_tensor, dim=1)
        
        # 批量计算能量
        batch_size = 2048
        energies = []
        self.landscape.eval()
        
        with torch.no_grad():
            for i in range(0, len(grid_tensor), batch_size):
                batch = grid_tensor[i:i+batch_size].to(self.device)
                e = self.landscape.energy(batch).cpu().numpy()
                energies.append(e)
        
        self.grid_energies = np.concatenate(energies).reshape(resolution, resolution, resolution)
        self.grid_x = xx.reshape(resolution, resolution, resolution)
        self.grid_y = yy.reshape(resolution, resolution, resolution)
        self.grid_z = zz.reshape(resolution, resolution, resolution)
        
        elapsed = time.time() - t0
        print(f"      完成: 能量范围 [{self.grid_energies.min():.2f}, {self.grid_energies.max():.2f}] ({elapsed:.1f}s)")
    
    # ------------------------------------------------------------------
    # 梯度下降轨迹
    # ------------------------------------------------------------------
    
    def compute_trajectory(
        self,
        text: str,
        steps: int = 40,
        lr: float = 0.02,
    ) -> dict:
        """
        从一个随机偏移点出发，沿能量梯度下降到最近吸引子，记录轨迹。
        
        Args:
            text: 目标概念文字
            steps: 梯度下降步数
            lr: 学习率
        
        Returns:
            dict: {
                'text': 概念文字,
                'trajectory_3d': (steps+1, 3) UMAP空间中的轨迹,
                'start_char': 起始最近汉字,
                'end_char': 收敛汉字,
                'energy_start': 起始能量,
                'energy_end': 最终能量,
            }
        """
        # 编码文本 → 找到最近锚点作为目标
        target_vec = self.zichang.encode_text(text)
        if target_vec.shape[0] == 0:
            return None
        
        target_vec = target_vec.mean(dim=0)  # 多字取均值
        target_vec = torch.nn.functional.normalize(target_vec, dim=-1)
        
        # 添加随机扰动作为起始点（模拟"不确定的查询"）
        noise = torch.randn(1024) * 0.3
        start_vec = target_vec + noise
        start_vec = torch.nn.functional.normalize(start_vec, dim=-1)
        
        # 梯度下降
        x = start_vec.clone().detach().to(self.device)
        x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        
        trajectory_1024d = [x.detach().cpu().numpy().copy()]
        energies = []
        
        self.landscape.train()
        for _ in range(steps):
            optimizer.zero_grad()
            e = self.landscape.energy(x.unsqueeze(0))
            e.backward()
            optimizer.step()
            
            with torch.no_grad():
                x.data = torch.nn.functional.normalize(x.data, dim=-1)
            
            trajectory_1024d.append(x.detach().cpu().numpy().copy())
            energies.append(e.item())
        
        self.landscape.eval()
        
        # 映射到 UMAP 3D 空间
        traj_1024d = np.array(trajectory_1024d)
        traj_3d = self.umap_model.transform(traj_1024d)
        
        # 找到起点和终点最近的汉字
        start_char = self._nearest_char(torch.from_numpy(traj_1024d[0]).float())
        end_char = self._nearest_char(torch.from_numpy(traj_1024d[-1]).float())
        
        return {
            'text': text,
            'trajectory_3d': traj_3d,
            'start_char': start_char,
            'end_char': end_char,
            'energy_start': energies[0] if energies else 0,
            'energy_end': energies[-1] if energies else 0,
        }
    
    def _nearest_char(self, vec: torch.Tensor) -> str:
        """找到最近的汉字"""
        _, chars, _ = self.zichang.find_nearest(vec, k=1)
        return chars[0] if chars else "?"
    
    # ------------------------------------------------------------------
    # Plotly 渲染
    # ------------------------------------------------------------------
    
    def render(
        self,
        n_anchors: int = DEFAULT_N_ANCHORS,
        trajectories: list = None,
        output_path: str = None,
    ):
        """
        渲染完整的 Plotly 3D 可视化并保存为 HTML。
        
        Args:
            n_anchors: 散点图层显示的锚点数量
            trajectories: 推理演示文字列表
            output_path: 输出 HTML 路径
        """
        if self.grid_energies is None:
            raise RuntimeError("请先调用 compute_energy_grid()")
        
        output_path = output_path or OUTPUT_HTML
        trajectories = trajectories or DEFAULT_TRAJECTORIES
        
        print(f"[5/5] 渲染 Plotly 3D 可视化...")
        t0 = time.time()
        
        fig = go.Figure()
        
        # ==================================================================
        # 图层1: 能量等值面（isosurface）
        # ==================================================================
        
        # 选择 3-4 个能量层级绘制等值面
        e_min = float(self.grid_energies.min())
        e_max = float(self.grid_energies.max())
        
        # 低能量层（锚点盆地，深蓝色）
        iso_low = e_min + (e_max - e_min) * 0.15
        # 中等能量层（过渡区，中蓝色）
        iso_mid = e_min + (e_max - e_min) * 0.35
        # 高能量层（未知地带，浅色）
        iso_high = e_min + (e_max - e_min) * 0.60
        
        isosurface_layers = [
            (iso_low,  'rgba(20, 40, 120, 0.25)',  '低能量盆地（已知知识区）'),
            (iso_mid,  'rgba(60, 100, 200, 0.18)',  '过渡区'),
            (iso_high, 'rgba(180, 200, 240, 0.12)', '高能量山脊（未知区域）'),
        ]
        
        for iso_val, color, name in isosurface_layers:
            fig.add_trace(go.Isosurface(
                x=self.grid_x.ravel(),
                y=self.grid_y.ravel(),
                z=self.grid_z.ravel(),
                value=self.grid_energies.ravel(),
                isomin=iso_val,
                isomax=iso_val,
                surface_count=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=name,
                hoverinfo='name',
                opacity=0.3,
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))
        
        # ==================================================================
        # 图层2: 汉字锚点散点
        # ==================================================================
        
        n_total = self.anchors_3d.shape[0]
        n_show = min(n_anchors, n_total)
        rng = np.random.default_rng(123)
        scatter_indices = rng.choice(n_total, n_show, replace=False)
        
        scatter_3d = self.anchors_3d[scatter_indices]
        scatter_chars = [self.zichang.hanzi_list[i] for i in scatter_indices]
        
        # 计算这些锚点的能量值
        with torch.no_grad():
            scatter_vecs = self.anchors[scatter_indices].to(self.device)
            scatter_energies = self.landscape.energy(scatter_vecs).cpu().numpy()
        
        # 构建 hover 文本
        hover_texts = [
            f"<b>{ch}</b><br>能量: {e:.3f}<br>索引: {idx}"
            for ch, e, idx in zip(scatter_chars, scatter_energies, scatter_indices)
        ]
        
        fig.add_trace(go.Scatter3d(
            x=scatter_3d[:, 0],
            y=scatter_3d[:, 1],
            z=scatter_3d[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,
                color=scatter_energies,
                colorscale='RdBu_r',
                colorbar=dict(
                    title='能量值',
                    x=1.02,
                    len=0.5,
                ),
                cmin=e_min,
                cmax=e_max,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
            ),
            text=hover_texts,
            hoverinfo='text',
            name=f'汉字锚点 ({n_show}个)',
        ))
        
        # ==================================================================
        # 图层3: 梯度下降轨迹
        # ==================================================================
        
        trajectory_colors = ['#FF6B35', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
        
        for idx, traj_info in enumerate(trajectories):
            if isinstance(traj_info, str):
                text = traj_info
            else:
                text = traj_info[0]
            
            print(f"      计算轨迹: {text}...")
            traj = self.compute_trajectory(text, steps=40)
            
            if traj is None:
                continue
            
            color = trajectory_colors[idx % len(trajectory_colors)]
            t3d = traj['trajectory_3d']
            
            # 轨迹线
            fig.add_trace(go.Scatter3d(
                x=t3d[:, 0],
                y=t3d[:, 1],
                z=t3d[:, 2],
                mode='lines',
                line=dict(color=color, width=3),
                name=f'推理: 「{text}」→「{traj["end_char"]}」',
                hovertext=[
                    f'<b>{text}</b> 推理第{i}步<br>能量: {traj["energy_start"] - (traj["energy_start"]-traj["energy_end"])*i/len(t3d):.3f}'
                    for i in range(len(t3d))
                ],
                hoverinfo='text',
            ))
            
            # 起点（小球）
            fig.add_trace(go.Scatter3d(
                x=[t3d[0, 0]],
                y=[t3d[0, 1]],
                z=[t3d[0, 2]],
                mode='markers',
                marker=dict(size=6, color=color, symbol='circle'),
                name=f'起点 ({traj["start_char"]})',
                hovertext=f'<b>起点</b><br>字: {traj["start_char"]}<br>能量: {traj["energy_start"]:.3f}',
                hoverinfo='text',
            ))
            
            # 终点（大星）
            fig.add_trace(go.Scatter3d(
                x=[t3d[-1, 0]],
                y=[t3d[-1, 1]],
                z=[t3d[-1, 2]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='diamond', line=dict(width=2, color='white')),
                name=f'收敛 「{traj["end_char"]}」',
                hovertext=f'<b>收敛</b><br>字: {traj["end_char"]}<br>能量: {traj["energy_end"]:.3f}',
                hoverinfo='text',
            ))
        
        # ==================================================================
        # 布局
        # ==================================================================
        
        fig.update_layout(
            title=dict(
                text='🐉 龙珠能量景观 · 三维可视化',
                font=dict(size=24),
                x=0.5,
            ),
            scene=dict(
                xaxis_title='UMAP 维度 1',
                yaxis_title='UMAP 维度 2',
                zaxis_title='UMAP 维度 3',
                xaxis=dict(showgrid=True, gridcolor='rgba(180,180,180,0.15)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(180,180,180,0.15)'),
                zaxis=dict(showgrid=True, gridcolor='rgba(180,180,180,0.15)'),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            paper_bgcolor='rgba(245,245,250,1)',
            plot_bgcolor='rgba(245,245,250,1)',
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=11),
            ),
            hovermode='closest',
        )
        
        # 添加说明注释
        fig.add_annotation(
            text=(
                "🔵 深蓝等值面 = 低能量盆地（已知知识区）<br>"
                "⚪ 浅色等值面 = 高能量山脊（未知区域）<br>"
                "🔴 散点 = 汉字锚点（颜色=能量值，蓝=低能量）<br>"
                "🌈 彩线 = 梯度下降推理轨迹（→ 收敛到最近锚点）"
            ),
            xref='paper', yref='paper',
            x=0.01, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=12, color='#333'),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=8,
        )
        
        # 保存
        fig.write_html(output_path)
        elapsed = time.time() - t0
        print(f"      完成! 保存至: {output_path} ({elapsed:.1f}s)")
        print(f"      文件大小: {os.path.getsize(output_path)/1024/1024:.1f} MB")
    
    # ------------------------------------------------------------------
    # 一键运行
    # ------------------------------------------------------------------
    
    def run(
        self,
        n_anchors: int = DEFAULT_N_ANCHORS,
        grid_resolution: int = DEFAULT_GRID_RES,
        trajectories: list = None,
        output_path: str = None,
    ):
        """一键运行完整流程"""
        self.load()
        self.reduce_dimensions()
        self.compute_energy_grid(resolution=grid_resolution)
        self.render(
            n_anchors=n_anchors,
            trajectories=trajectories,
            output_path=output_path,
        )
        print(f"\n✅ 可视化完成！用浏览器打开: {output_path or OUTPUT_HTML}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="龙珠能量景观 3D 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_landscape_3d.py
  python visualize_landscape_3d.py --n-anchors 1000 --resolution 35
  python visualize_landscape_3d.py --output my_landscape.html
        """,
    )
    
    parser.add_argument('--n-anchors', '-n', type=int, default=DEFAULT_N_ANCHORS,
                        help=f'散点图层锚点数 (默认: {DEFAULT_N_ANCHORS})')
    parser.add_argument('--resolution', '-r', type=int, default=DEFAULT_GRID_RES,
                        help=f'等值面网格分辨率 (默认: {DEFAULT_GRID_RES})')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_HTML,
                        help=f'输出HTML路径 (默认: {OUTPUT_HTML})')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        help='计算设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    visualizer = LandscapeVisualizer3D(device=args.device)
    visualizer.run(
        n_anchors=args.n_anchors,
        grid_resolution=args.resolution,
        trajectories=DEFAULT_TRAJECTORIES,
        output_path=args.output,
    )
