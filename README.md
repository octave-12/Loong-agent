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

| 模块 | 位置 | 说明 |
|------|------|------|
| **字场** | `loongpearl/core/zichang.py` | 94,117 汉字 × 1024 维锚点矩阵（永久冻结） |
| **能量景观** | `loongpearl/core/energy_landscape.py` | 4 层 MLP (1024→3072→3072→1536→1)，17.3M 参数 |
| **频率感知景观** | `loongpearl/core/freq_landscape.py` | 基于字频的能量调节，区分高频/低频知识通道 |
| **学习器** | `loongpearl/learning/learner.py` | Hebbian 学习 + 自知无知检测 + 衰减调度 |
| **播种器** | `loongpearl/learning/seeder.py` | 用 DeepSeek-R1 批量生成汉字语义关联 |
| **婴儿课程** | `loongpearl/learning/curriculum.py` | 从单字→组词→成语的渐进式学习路径 |
| **增量学习** | `loongpearl/learning/incremental_learn.py` | 基于部件的增量学习，不破坏已有知识 |
| **龙珠主引擎** | `loongpearl/interaction/engine.py` | 整合以上模块的端到端查询-推理-学习循环 |
| **原生回答** | `loongpearl/interaction/native_answer.py` | 零 LLM 原生答案生成 |
| **自演化对话** | `loongpearl/interaction/self_evolving.py` | 用户提问 → 字场检索 → 能量推理 → 多源校验 |
| **嗓音合成** | `loongpearl/voice/baby_voice.py` | 口哨合成引擎，龙珠用自己的嗓音发声 |
| **联网检索** | `loongpearl/web/lookup.py` | 字典检索与网络知识获取 |
| **3D 可视化** | `loongpearl/utils/visualize_3d.py` | 交互式能量景观 3D 可视化（等值面+锚点+轨迹） |
| **轨迹渲染** | `loongpearl/utils/render_trajectories.py` | 快速追加推理轨迹，复用 UMAP/网格缓存 |

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
bash scripts/download_models.sh      # 从 HuggingFace 下载
# 或自行构建（需要 Ollama + BAAI/bge-large-zh）
```

### 一键查询

```python
from loongpearl import HanziAnchorField, EnergyLandscape
from loongpearl.interaction.engine import LoongPearl

lp = LoongPearl()
result = lp.query("人工智能")
# QueryResult(✅已知 conf=75.7% energy=-2.3 steps=45)
# 「智」是知识网络中最相关的概念。（相关：能、算、机）

chars = lp.find_nearest_chars("算法", k=5)    # 快速检索
info = lp.reason_between("火", "水")            # 汉字间推理
```

### 3D 能量景观可视化

```bash
# 生成交互式 3D 可视化（等值面 + 锚点散点 + 梯度下降轨迹）
python loongpearl/utils/visualize_3d.py
# 用浏览器打开 artifacts/landscape_3d.html

# 追加更多推理轨迹（跳过 UMAP 降维，秒级完成）
python loongpearl/utils/render_trajectories.py
```

> 🖱️ 在 HTML 中可用鼠标旋转/缩放/平移，hover 锚点查看汉字名称和能量值

### 命令行

```bash
# 交互式查询
python loongpearl/interaction/engine.py

# 端到端测试
python tests/test_loongpearl.py
python tests/test_loongpearl.py --quick       # 冒烟测试

# 知识播种（用 DeepSeek-R1 生成语义关联）
python scripts/run_seed.py --chars 500         # 播种 500 字
python scripts/run_seed.py --dry-run 10        # 干运行预览
python scripts/run_seed.py --chars 3755        # 全量 GB2312 一级汉字
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
- **频率感知**：高频字对（如"中国"）形成高速通道，低频字对（如"龘龖"）保持高墙

### 3D 可视化
使用 UMAP 将 94,117 个 1024 维锚点降到 3 维，Plotly 渲染交互式能量景观：
- **图层1** 能量等值面 — 深蓝低能盆地（已知知识区）→ 浅色高能山脊（未知区域）
- **图层2** 汉字锚点散点 — 红色圆点，hover 显示汉字名和能量值
- **图层3** 梯度下降轨迹 — 彩线从随机起点到收敛锚点

---

## 📁 项目结构

```
loong-pearl/
├── loongpearl/                    # 🧠 核心 Python 包
│   ├── __init__.py                #     统一入口（懒加载导出）
│   ├── data_config.py             #     数据路径解析
│   ├── core/                      #     基础层
│   │   ├── zichang.py             #       字场 94K 锚点
│   │   ├── energy_landscape.py    #       能量景观
│   │   └── freq_landscape.py      #       频率感知版
│   ├── learning/                  #     学习层
│   │   ├── learner.py             #       增量学习引擎
│   │   ├── seeder.py              #       知识播种器
│   │   ├── curriculum.py          #       婴儿课程表
│   │   ├── imprint_words.py       #       词语烙印
│   │   └── incremental_learn.py   #       部件增量学习
│   ├── voice/                     #     嗓音层
│   │   ├── baby_voice.py          #       口哨合成引擎
│   │   └── learn_voice.py         #       嗓音学习
│   ├── web/                       #     联网层
│   │   ├── lookup.py              #       通用检索
│   │   └── knowledge_web.py       #       联网知识播种
│   ├── interaction/               #     对话层
│   │   ├── engine.py              #       龙珠主引擎
│   │   ├── native_answer.py       #       原生答案生成
│   │   └── self_evolving.py       #       自演化对话
│   └── utils/                     #     工具层
│       ├── monitor.py             #       训练监控
│       ├── visualize_3d.py        #       3D 可视化
│       ├── render_trajectories.py #       轨迹渲染
│       └── download_dicts.py      #       字典下载
├── scripts/                       # 🔧 训练/播种脚本
│   ├── run_seed.py
│   ├── seed_knowledge_v2.py
│   ├── train_v5_final.py
│   ├── download_models.sh
│   └── seed/                      #     辅助播种脚本
├── tests/                         # 🧪 测试
├── data/                          # 📦 数据
│   ├── models/     (*.pt)         #     模型文件（不入 git）
│   ├── dicts/      (*.json)       #     字典数据
│   ├── wordlists/  (*.txt)        #     字表词表
│   └── runtime/    (*.json)       #     运行时缓存（不入 git）
├── artifacts/                     # 🖼️ 产出物
│   └── landscape_3d.html          #     3D 能量景观可视化
├── doc/                           # 📄 论文/设计文档
└── logs/                          # 📋 日志
```

---

## 📄 论文

`doc/loongpearl_paper.tex` — 完整论文 LaTeX 源码

---

## 📝 License

MIT © 2025 李泽坤 (octave-12)
