# 🐉 龙珠 LoongPearl

**以汉字为锚点的自演化知识内核**

龙珠是一个以 94,117 个 Unicode 汉字嵌入为锚点的可微分知识系统。它由**字场**（锚点基底）、**能量景观**（吸引子网络）、**七因子盲区检测器**和**自主学习引擎**四部分组成，能主动发现自己的知识盲区、联网搜索学习、并通过 Hebbian 机制将新知写入能量景观——全过程零 LLM 依赖。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203060-green.svg)]()

---

## 🧠 总体架构

```
                              ┌──────────────────────────────────┐
  用户输入 ─────────────────→ │       🐉 龙珠主引擎              │
                              │   interaction/engine.py          │
                              │   query → 自知无知 → 推理/学习   │
                              └──────────┬───────────┬──────────┘
                                         │           │
                    ┌────────────────────┼───┐  ┌────┴──────────────────┐
                    │   查询/回答层       │   │  │   自演化对话层          │
                    ├────────────────────┤   │  ├───────────────────────┤
                    │ NativeAnswerEngine │   │  │ SelfEvolvingLoongPearl│
                    │ 字典释义+能量→答案  │   │  │ 多源校验+反噬回退      │
                    └────────────────────┘   │  └───────────────────────┘
                                             │
    ┌────────────────────────────────────────┼──────────────────────────────┐
    │              🧠 学习层 (learning/)     │                              │
    ├──────────────┬──────────────┬──────────┼───────┬──────────┬──────────┤
    │ learner.py   │autonomous_   │blindspot_│curric │incremen │ seeder.py│
    │ Hebbian+自知 │ learner.py  │detector  │ulum.py│tal_learn│ 语义播种 │
    │ 无知检测     │ 零LLM搜索学 │7因子扫描 │八阶段 │ 增量路径 │ (Ollama) │
    └──────────────┴──────────────┴──────────┴───────┴──────────┴──────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                    ⚛️ 核心层 (core/)                                 │
    ├──────────────────────────────────┬───────────────────────────────────┤
    │  zichang.py (字场)              │  freq_landscape.py (能量景观 v2)   │
    │  94,117 汉字 × 1024 维         │  1024→2048→2048→1024→512→1         │
    │  BAAI/bge-large-zh 编码        │  双通路: 基础能量 + 频率偏移        │
    │  永久冻结锚点矩阵               │  infer() 梯度下降 + resolve() 映射  │
    └──────────────────────────────────┴───────────────────────────────────┘

    ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────────┐
    │ 🎤 嗓音层    │  │ 🌐 联网层    │  │ 🛠️ 工具层                     │
    ├──────────────┤  ├──────────────┤  ├───────────────────────────────┤
    │ baby_voice   │  │ searcher.py  │  │ compute_sandbox.py 计算沙盒    │
    │ 10次谐波+F3  │  │ 百度/维基/DDG│  │ visualize_3d.py 交互式3D      │
    │ 共振峰合成   │  │ knowledge_web│  │ download_dicts.py 字典下载    │
    │ 婴儿发声     │  │ lookup.py    │  │                               │
    └──────────────┘  └──────────────┘  └───────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                    📜 脚本层 (scripts/)                              │
    ├───────────────┬───────────────┬───────────────┬─────────────────────┤
    │ auto_learn.py │idiom_inject   │train_v5_final │ seed/ (知识播种)     │
    │ 7×24自主学习  │ _gpu 成语注入 │ 频率感知训练  │ parallel_seed 等     │
    └───────────────┴───────────────┴───────────────┴─────────────────────┘
```

---

## 🔬 核心原理

### 1. 字场 — 89K 汉字锚点矩阵

94,117 个汉字通过 BAAI/bge-large-zh 编码为 1024 维嵌入，形成永久冻结的语义基底。覆盖 Unicode CJK 全部区间（含繁体、异体、日韩汉字）。

### 2. 能量景观 — 可微分吸引子网络

一个双通路 MLP 将任意 1024 维向量映射为标量能量值：
- **主通路**: 判断向量是否在"已知知识区域"（深谷 ~-15）还是"未知区域"（山脊 ~+15）
- **频率通路**: 高频字对（如"中国"）形成高速通道，低频字对保持高墙
- **推理**: 从查询点出发沿梯度下降到最近吸引子（汉字锚点）

### 3. 七因子盲区检测器 — 龙珠的"自知无知"

7 个独立因子并行扫描 94K 汉字空间，发现知识盲区：

| 因子 | 检测内容 | 示例 |
|------|---------|------|
| F1 统计 | 尾字频率 vs 首字频率失衡 | "之"出现 1186 次但仅 5 次作首字 |
| F2 能量 | GPU 批量扫描中点能量异常 | 某字的所有关联能量偏高 |
| F3 覆盖 | 角色单一性（仅作中位/仅作尾位）| "而" 438/440 次在中位 |
| F4 死路 | 尾字在词典中无后续候选 | "迹" 57 次作尾字，0 个首字成语 |
| F5 梯度 | 锚点梯度 z-score 异常偏离 | 景观畸变区域检测 |
| F6 语义 | 嵌入相似但无连接的字对 | 语义孤岛 |
| F7 新鲜 | 常用字未出现在成语中 | 沉睡知识 |

> 多因子共识加权：同一汉字被 ≥2 个因子发现 → 优先级 ×1.2，≥3 个 → ×1.5

### 4. 自主学习引擎 — 零 LLM 的知识获取

```
发现盲区 → 百度搜索 → 提取字对关联 → Hebbian 注入能量景观 → 验证 → 保存
   ↑                                                                      ↓
   └──────────────────── 下一轮（间隔 120s）──────────────────────────────┘
```

全过程不依赖任何外部 LLM。龙珠作为独立 Agent，自行决定学什么、怎么学。

### 5. Hebbian 学习 — 用进废退

- **用进**: 确认的关联 → 降低路径能量（通道加深）
- **废退**: 长期未用 → 权重衰减（通道变浅）
- **自知无知**: 能量 + 梯度 + 距离三信号综合判断

---

## ⚡ 快速开始

### 环境

```bash
Python 3.11+, PyTorch 2.x, CUDA 12.x（可选）
RTX 3060 12GB（推荐）/ CPU 可用
Ollama（可选，传统播种模式用）
```

### 安装

```bash
git clone https://github.com/octave-12/loong-pearl.git
cd loong-pearl
pip install torch sentence-transformers requests numpy
```

### 7×24 自主学习守护进程

```bash
# 启动持续自主学习（每 120s 扫描一轮）
python scripts/auto_learn.py --daemon

# 自定义间隔和每轮学习量
python scripts/auto_learn.py --daemon -i 300 -m 5

# 只用特定因子
python scripts/auto_learn.py --daemon -f statistical,dead_end,coverage

# 只扫描不学习
python scripts/auto_learn.py --scan-only
```

### 成语接龙（纯能量景观，不查表）

```bash
python tests/test_energy_chain.py
```

### 端到端查询

```python
from loongpearl import HanziAnchorField, FreqEnergyLandscape
from loongpearl.interaction.engine import LoongPearl

lp = LoongPearl()
lp.initialize()
result = lp.query("人工智能")
# → QueryResult(✅已知, 「智」是最相关的概念)
```

---

## 📊 关键指标

| 指标 | 数值 |
|------|------|
| 字场规模 | 94,117 汉字 |
| 嵌入维度 | 1,024 (BAAI/bge-large-zh) |
| 能量景观参数 | ~8.9M (FreqEnergyLandscape v2) |
| 成语词典 | 29,514 条 (手工精选 + 新华成语大词典) |
| 注入字对 | 88,542 对 (成语连续字对) |
| 盲区检测因子 | 7 个独立因子 |
| 扫描速度 | ~2.7s/全因子全量扫描 |
| 学习速度 | ~25s/盲区 (搜索+提取+注入) |
| 能量分离度 | 锚点 -16 vs 随机 +15 (30x) |

---

## 📁 项目结构

```
loong-pearl/
├── loongpearl/                      # 🧠 核心包
│   ├── core/                        #   基础层
│   │   ├── zichang.py               #     字场 94K 锚点（永久冻结）
│   │   ├── energy_landscape.py      #     能量景观（基础版 512d）
│   │   └── freq_landscape.py        #     频率门控景观 v2（1024d, 当前主力）
│   ├── learning/                    #   学习层
│   │   ├── learner.py               #     Hebbian 学习 + 自知无知检测
│   │   ├── autonomous_learner.py    #     零 LLM 自主学习引擎
│   │   ├── blindspot_detector.py    #     7 因子盲区检测器
│   │   ├── curriculum.py            #     婴儿八阶段学语课程
│   │   ├── incremental_learn.py     #     部件增量学习
│   │   ├── imprint_words.py         #     词语能量烙印
│   │   └── seeder.py               #     Ollama 语义知识播种
│   ├── voice/                       #   嗓音层
│   │   ├── baby_voice.py            #     10 谐波 + F3 共振峰合成
│   │   └── learn_voice.py           #     嗓音学习
│   ├── web/                         #   联网层
│   │   ├── searcher.py              #     多引擎搜索 (百度/DuckDuckGo/维基)
│   │   ├── knowledge_web.py         #     联网知识播种
│   │   └── lookup.py                #     汉字联网查询 + 本地缓存
│   ├── interaction/                 #   对话层
│   │   ├── engine.py                #     龙珠主引擎 (LoongPearl)
│   │   ├── native_answer.py         #     零 LLM 原生回答
│   │   └── self_evolving.py         #     自演化对话系统
│   └── utils/                       #   工具层
│       ├── compute_sandbox.py       #     安全计算沙盒 (AST 白名单)
│       ├── visualize_3d.py          #     3D 能量景观可视化
│       └── download_dicts.py        #     字典下载
├── scripts/                         # 🔧 脚本
│   ├── auto_learn.py                #     7×24 自主学习守护进程
│   ├── idiom_inject_gpu.py          #     GPU 批量成语注入
│   ├── train_v5_final.py            #     频率感知训练流水线
│   └── seed/                        #     知识播种脚本集
├── tests/                           # 🧪 测试
│   ├── test_energy_chain.py         #     纯能量景观成语接龙
│   ├── test_autonomous_learn.py     #     自主学习端到端测试
│   └── test_compute.py              #     计算沙盒测试 (25/25)
├── data/                            # 📦 数据
│   ├── models/                      #     模型文件 (zichang ~369MB, landscape ~34MB)
│   ├── dicts/                       #     字典 (idioms.json, cedict, unihan...)
│   └── runtime/                     #     运行时缓存
├── artifacts/                       # 🖼️ 产出物 (landscape_3d.html)
├── doc/                             # 📄 论文 LaTeX
└── logs/                            # 📋 学习日志 + 盲区记录
```

---

## 🔄 数据流

```
用户输入
  │
  ├─ 数学题? → ComputeSandbox → 直接计算返回
  │
  ├─ 文本编码 (BAAI/bge-large-zh → 1024d)
  │   │
  │   ├─ 自知无知检测 (梯度+能量+距离)
  │   │   ├─ 已知 → 能量景观梯度下降 → resolve() → 返回最近汉字
  │   │   └─ 未知 → AutonomousLearner.learn_if_unknown()
  │   │         ├─ 联网搜索 (百度/DuckDuckGo/维基)
  │   │         ├─ 提取字对关联
  │   │         ├─ Hebbian 批量注入
  │   │         └─ 验证 → 返回
  │   │
  │   └─ NativeAnswerEngine
  │       ├─ 复合词最长匹配
  │       ├─ 字典释义查询
  │       └─ 能量近邻补全 → 拼装答案
  │
  └─ 7×24 后台: 盲区检测 → 主动学习 → 能量景观自演化
```

---

## 🐉 设计哲学

| 原则 | 含义 |
|------|------|
| **基于检测做决策** | 不推测、不假设，一切学习以实测数据驱动 |
| **零 LLM 主唱** | 龙珠用自己的能量景观推理，不依赖外部 LLM 生成答案 |
| **能量景观是唯一知识源** | 不在 JSON 里存新知识，所有学习写入能量景观参数 |
| **用进废退** | 高频通路自动强化，冷门知识自然衰减 |
| **婴儿式成长** | 先学单字读音，再组词、成语，最后成句成文 |
| **自主进化** | 自己发现盲区 → 自己去学 → 自己消化，无需人工喂数据 |

---

## 📝 License

MIT © 2025 李泽坤 (octave-12)
