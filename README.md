# 🐉 龙珠 LoongPearl

**以汉字为锚点的确定性知识内核**

龙珠是一个以 94,117 个汉字嵌入为锚点的可微分知识系统。它由字场（嵌入基底）、能量景观（吸引子网络）和学习机制（Hebbian 学习 + 自知无知）三部分组成，支持知识查询、推理、学习和"知道自己不知道什么"的元认知能力。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 核心架构

```
查询文本 → BAAI/bge-large-zh 编码(1024d)
         → 自知无知检测（梯度+能量+距离三信号）
              ├─ 已知 → 能量景观梯度下降 → 收敛到吸引子 → 映射回汉字
              └─ 未知 → DeepSeek-R1 学习 → Hebbian 注入 → 重试
```

| 模块 | 文件 | 说明 |
|------|------|------|
| **字场** | `zichang.py` | 94,117 汉字 × 1024 维锚点矩阵（永久冻结） |
| **能量景观** | `energy_landscape.py` | 4 层 MLP (1024→3072→3072→1536→1)，17.3M 参数 |
| **学习器** | `loongpearl_learner.py` | Hebbian 学习 + 自知无知检测 + 衰减调度 |
| **播种器** | `loongpearl_seeder.py` | 用 DeepSeek-R1 批量生成汉字语义关联 |
| **龙珠主类** | `loongpearl.py` | 整合以上模块的端到端查询-推理-学习循环 |
| **3D 可视化** | `visualize_landscape_3d.py` | 交互式能量景观 3D 可视化（等值面+锚点+轨迹） |
| **增量渲染** | `render_trajectories.py` | 快速追加推理轨迹，复用 UMAP/网格缓存 |

---

## ⚡ 快速开始

### 环境要求

- Python 3.11+
- PyTorch 2.x
- umap-learn, plotly, numpy
- Ollama（可选，用于知识播种和学习）
- 8GB+ 内存

### 安装

```bash
git clone https://github.com/octave-12/loong-pearl.git
cd loong-pearl
pip install torch sentence-transformers requests umap-learn plotly

# 下载预训练模型
bash download_models.sh          # 从 HuggingFace 下载
# 或自行构建（需要 Ollama + BAAI/bge-large-zh）
```

### 一键启动

```python
from loongpearl import quick_start

loongpearl = quick_start()                         # 初始化
result = loongpearl.query("人工智能")                # 查询
# QueryResult(✅已知 conf=75.7% energy=-2.3 steps=45)
# 「智」是知识网络中最相关的概念。（相关：能、算、机）

chars = loongpearl.find_nearest_chars("算法", k=5)  # 快速检索
info = loongpearl.reason_between("火", "水")         # 汉字间推理
```

### 3D 能量景观可视化

```bash
# 生成交互式 3D 可视化（等值面 + 锚点散点 + 梯度下降轨迹）
python visualize_landscape_3d.py
# 用浏览器打开 landscape_3d.html

# 追加更多推理轨迹（跳过 UMAP 降维，秒级完成）
python render_trajectories.py
```

> 🖱️ 在 HTML 中可用鼠标旋转/缩放/平移，hover 锚点查看汉字名称和能量值

### 命令行

```bash
# 交互式查询
python loongpearl.py

# 端到端测试
python test_loongpearl.py
python test_loongpearl.py --quick          # 冒烟测试

# 知识播种（用 DeepSeek-R1 生成语义关联）
python run_seed.py --chars 500             # 播种 500 字
python run_seed.py --dry-run 10            # 干运行预览
python run_seed.py --chars 3755            # 全量 GB2312 一级汉字
```

---

## 📊 实验数据

| 指标 | 数值 |
|------|------|
| 字场规模 | 94,117 汉字 |
| 嵌入维度 | 1,024 |
| **能量分离度** | **2.4x**（已知区域 vs 随机区域） |
| **推理准确率 (Top-1)** | **51.2%** |
| **推理准确率 (Top-3)** | **54.0%** |
| **自知无知 F1** | **100%** ⭐ |
| 自知无知精确率 | 100% |
| 自知无知召回率 | 100% |

> 🎯 **自知无知检测零误差**：系统能完美区分已知锚点和未知随机点，实现"知道自己的边界"。

---

## 🔬 原理简述

### 字场（Hanzi Anchor Field）
94,117 个汉字通过 BAAI/bge-large-zh 编码为 1024 维嵌入向量，形成永久冻结的语义基底。查询时，任意文本先映射到这个空间，再通过能量景观进行推理。

### 能量景观（Energy Landscape）
一个可微分的吸引子网络。训练后，已知汉字锚点处于能量极小值（深谷），未知区域处于能量高值（山脊）。推理即从查询点出发沿梯度下降到最近吸引子。

### 学习机制
- **用进（Hebbian 学习）**：确认的查询-答案关联降低路径能量
- **废退（权重衰减）**：不活跃的关联逐渐弱化
- **自知无知（元认知）**：通过梯度模长、能量值、锚点距离三信号综合判断

### 3D 可视化
使用 UMAP 将 94,117 个 1024 维锚点降到 3 维，Plotly 渲染交互式能量景观：
- **图层1** 能量等值面 — 深蓝低能盆地（已知知识区）→ 浅色高能山脊（未知区域）
- **图层2** 汉字锚点散点 — 红色圆点，hover 显示汉字名和能量值
- **图层3** 梯度下降轨迹 — 彩线从随机起点到收敛锚点

---

## 📁 项目结构

```
loong-pearl/
├── loongpearl.py              # 龙珠主类（查询/推理/学习/持久化）
├── zichang.py                 # 字场模块（嵌入生成/检索）
├── energy_landscape.py        # 能量景观（吸引子网络/推理引擎）
├── loongpearl_learner.py      # 学习机制（Hebbian/自知无知/衰减）
├── loongpearl_seeder.py       # 播种器（Ollama 批量语义关联）
├── visualize_landscape_3d.py  # 3D 能量景观可视化
├── render_trajectories.py     # 轨迹增量渲染器
├── run_seed.py                # 播种启动脚本
├── test_loongpearl.py         # 端到端测试
├── download_models.sh         # 模型下载脚本
├── hanzi_top3500.txt          # 3500 高频字表
├── hanzi_list.txt             # 全量 94117 汉字列表
└── doc/                       # 设计文档
```

---

## 📄 论文

`doc/loongpearl_paper.tex` — 完整论文 LaTeX 源码

---

## 📝 License

MIT © 2025 李泽坤 (octave-12)
