# 🐉 龙珠 LoongPearl

**以汉字为锚点的自演化知识内核**

龙珠是一个以 94,117 个 Unicode 汉字嵌入为锚点的可微分知识系统。它由**字场**（锚点基底）、**能量景观**（吸引子网络）、**七因子盲区检测器**、**自主学习引擎**、**SQLite 概念图加速**和**混合化能器**六部分组成，能主动发现自己的知识盲区、联网搜索学习、并通过 Hebbian 机制将新知写入能量景观——全过程零 LLM 依赖。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203060-green.svg)]()
[![Version](https://img.shields.io/badge/version-v2.5-orange.svg)]()

---

## 🧠 总体架构

```
                              ┌──────────────────────────────────┐
  用户输入 ─────────────────→ │       🐉 龙珠主引擎              │
                              │   interaction/engine.py          │
                              │   query → 查询路由 → 推理/学习    │
                              └──────────┬───────────┬──────────┘
                                         │           │
                    ┌────────────────────┼───┐  ┌────┴──────────────────┐
                    │   查询/回答层       │   │  │   自演化对话层          │
                    ├────────────────────┤   │  ├───────────────────────┤
                    │ HybridDecoder      │   │  │ SelfEvolvingLoongPearl│
                    │ 模板优先+LLM润色    │   │  │ 多源校验+反噬回退      │
                    │ DualExtractor      │   │  └───────────────────────┘
                    │ 正则+LLM双重提取    │   │
                    └────────────────────┘   │
                                             │
    ┌────────────────────────────────────────┼──────────────────────────────┐
    │              🧠 学习层 (learning/)     │                              │
    ├──────────────┬──────────────┬──────────┼───────┬──────────┬──────────┤
    │ learner.py   │autonomous_   │blindspot_│curric │incremen │ seeder.py│
    │ Hebbian+自知 │ learner.py  │detector  │ulum.py│tal_learn│ 语义播种 │
    │ 无知 + EWC   │ 零LLM搜索学 │7因子扫描 │八阶段 │ 增量路径 │ (Ollama) │
    │ + 序列臂有向 │ 习引擎      │          │       │          │          │
    └──────────────┴──────────────┴──────────┴───────┴──────────┴──────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                    ⚛️ 核心层 (core/)                                 │
    ├──────────────────────────────────┬───────────────────────────────────┤
    │  zichang.py (字场)              │  freq_landscape.py (能量景观 v2)   │
    │  94,117 汉字 × 1024 维         │  1024→2048→2048→1024→512→1         │
    │  BAAI/bge-large-zh 编码        │  双通路: 基础能量 + 频率偏移        │
    │  永久冻结锚点矩阵               │  infer() 梯度下降 + resolve() 映射  │
    │  encode_sequence() 位置感知     │                                    │
    ├──────────────────────────────────┼───────────────────────────────────┤
    │  orchestrator.py (调度器 v3)    │  concept_graph.py (概念图)         │
    │  查询路由 _route_query()        │  193万三元组 + 字符邻接索引        │
    │  守护 v2 统一信号驱动循环       │  JSON 持久化 + 脏标记增量保存      │
    ├──────────────────────────────────┼───────────────────────────────────┤
    │  concept_graph_sqlite.py        │  hybrid_decoder.py (混合化能器)    │
    │  1.93M 三元组 O(log N) 查询     │  简单→模板 / 复杂→骨架+LLM润色    │
    │  UNION 双向索引 + 字对提取      │  LLM 不可用自动回退               │
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
    │ loong_main.py │ idiom_inject  │ train_v5      │ seed/ (知识播种)     │
    │ 龙珠主入口     │ _gpu 成语注入 │ 频率感知训练  │ parallel_seed 等     │
    │ 守护/对话/验证 │               │               │                     │
    └───────────────┴───────────────┴───────────────┴─────────────────────┘
```

---

## 🔬 核心原理

### 1. 字场 — 94K 汉字锚点矩阵

94,117 个汉字通过 BAAI/bge-large-zh 编码为 1024 维嵌入，形成永久冻结的语义基底。覆盖 Unicode CJK 全部区间（含繁体、异体、日韩汉字）。

**位置感知编码** (v2.2): `encode_sequence()` 使用 3^x 指数加权保留语序——"量子纠缠"≠"纠缠量子"（cos 从 1.0→0.987）。

### 2. 能量景观 — 可微分吸引子网络

一个双通路 MLP 将任意 1024 维向量映射为标量能量值：
- **主通路**: 判断向量是否在"已知知识区域"（深谷 ~-60）还是"未知区域"（山脊 ~+15）
- **频率通路**: 高频字对形成高速通道，低频字对保持高墙
- **推理**: 从查询点出发沿梯度下降到最近吸引子（汉字锚点）
- **EWC 弹性权重巩固** (v2.2): 每 50 轮 Fisher 采样锚定重要参数，防止灾难性遗忘

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
发现盲区 → SQLite 1.93M 查询 → 不足则百度搜索 → DualExtractor 提取 → Hebbian 注入 → EWC 正则 → 保存
   ↑                        ↓ 每5轮: 序列臂注入有向字对方向性 + 概念图对齐2000对  ↓
   └────────────────── 下一轮（间隔 120s）─────────────────────────────────────┘
```

**三层数据源漏斗** (v2.2):
1. SQLite O(log N) 索引查询 1.93M 三元组 → 毫秒级返回字对
2. 概念图邻接索引 O(1) 补充
3. 网络搜索兜底（仅 1 查询/字，DualExtractor 正则提取）

全过程不依赖任何外部 LLM。龙珠作为独立 Agent，自行决定学什么、怎么学。

### 5. Hebbian 学习 — 用进废退

- **语义臂** (每轮): 对比学习修复能量景观分离度，确认的关联 → 降低路径能量
- **序列臂** (每5轮): 从 POETIC_NEXT 提取的 14,800 对有向字对，非对称 Hebbian 注入建立方向性
- **EWC 锚定** (每50轮): Fisher 信息矩阵采样 200 个锚点，λ=0.05 拉回重要参数
- **废退**: 长期未用 → 权重衰减
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
python loong_main.py --daemon

# 自定义间隔和每轮学习量
python loong_main.py --daemon --interval 300 --max-learn 5

# 交互对话模式
python loong_main.py --chat
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
| 概念图三元组 | 1,931,225 条 (JSON 268MB) |
| SQLite 加速索引 | 228MB, WAL 模式, 4 索引 (s/sr/r/o) |
| SQLite 查询速度 | ~2ms/字 (vs JSON >30s) |
| 成语词典 | 29,514 条 |
| 有向字对 (序列臂) | 14,800 对 (POETIC_NEXT) |
| 盲区检测因子 | 7 个独立因子 |
| 扫描速度 | ~30s/全因子全量扫描 (parallel=True) |
| 守护学习速度 | **~63s/轮** (GPU加速+批量SGD, 原 90s) |
| 序列臂训练 | 409 有向关联/每5轮 (~14s) |
| 知识对齐 | 2,000 对概念/每5轮 |
| EWC 锚定 | 每50轮 Fisher 采样 200 锚点 |
| 能量分离度 | 锚点 ~-250 vs 随机 ~-150, 分离度 4.2 |

---

## 📁 项目结构

```
loong-pearl/
├── loongpearl/                      # 🧠 核心包
│   ├── core/                        #   基础层
│   │   ├── zichang.py               #     字场 94K 锚点（永久冻结 + 位置感知编码）
│   │   ├── energy_landscape.py      #     能量景观（基础版 512d）
│   │   ├── freq_landscape.py        #     频率门控景观 v2（1024d, 当前主力）
│   │   ├── orchestrator.py          #     龙珠调度器 v3: 查询路由 + 守护v2 + 序列臂
│   │   ├── concept_graph.py         #     概念图 (193万三元组, 字符邻接索引)
│   │   ├── concept_graph_sqlite.py  #     SQLite 加速层 (O(log N) 查询, UNION 双向索引)
│   │   ├── hybrid_decoder.py        #     混合化能器 (模板优先 + LLM 润色)
│   │   ├── fuzzy_graph.py           #     模糊格 (不确定性推理)
│   │   └── contra_resolver.py       #     矛盾消解器 (D-S 证据理论)
│   ├── learning/                    #   学习层
│   │   ├── learner.py               #     Hebbian 学习 + EWC 弹性权重巩固
│   │   ├── autonomous_learner.py    #     零 LLM 自主学习引擎
│   │   ├── blindspot_detector.py    #     7 因子盲区检测器
│   │   ├── dual_extractor.py        #     双重知识提取器 (正则 + LLM 兜底)
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
│   ├── loong_main.py                #     龙珠主入口 (守护/对话/验证)
│   ├── inject_concept_graph.py      #     概念图批量注入 (锁协调)
│   ├── idiom_inject_gpu.py          #     GPU 批量成语注入
│   ├── train_v5_final.py            #     频率感知训练流水线
│   └── seed/                        #     知识播种脚本集
├── tests/                           # 🧪 测试
│   ├── test_energy_chain.py         #     纯能量景观成语接龙
│   ├── test_autonomous_learn.py     #     自主学习端到端测试
│   └── test_compute.py              #     计算沙盒测试 (25/25)
├── data/                            # 📦 数据
│   ├── models/                      #     模型文件 (zichang ~369MB, landscape ~34MB, directed_pairs ~1MB)
│   ├── dicts/                       #     字典 (idioms.json, cedict, unihan...)
│   └── runtime/                     #     运行时缓存 (pending_queries.json)
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

### 架构合规状态 (v2.2)

| 原则 | 状态 | 实施 |
|------|:--:|------|
| 基于检测做决策 | ✅ | 7因子盲区扫描 + z-score校准自知无知 |
| 零 LLM 主唱 | ✅ | 自主学习引擎 + 能量推理 |
| 能量景观是唯一知识源 | ✅ | from_landscape=True, 盲区检测纯从景观拓扑 |
| 查询路由 | ✅ | _route_query() 4路分发 (事实/关系/序列/模糊) |
| 位置感知编码 | ✅ | encode_sequence() 3^x 指数加权保留语序 |
| 用进废退 | ✅ | 守护每轮 + 引擎每50查询自动衰减 |
| EWC 稳定性 | ✅ | 每轮正则 + 每50轮 Fisher 锚定 200 参数 |
| 有向序列学习 | ✅ | 序列臂每5轮注入 14,800 有向字对 |
| 概念图加速 | ✅ | SQLite 1.93M 三元组 O(log N) 查询 |
| 双重知识提取 | ✅ | 正则优先(conf=0.7) + LLM兜底(conf=0.5) |
| 混合化能器 | ✅ | 简单→模板 / 复杂→骨架+LLM润色 |
| 婴儿式成长 | ✅ | BabyCurriculum 接入引擎+守护 |
| 自主进化 | ✅ | 7×24守护 + 扫描→SQLite→注入→EWC→衰减闭环 |

---

## 📋 变更日志

### v2.5 (2026-06-19) — 三引擎创造架构实现

- **三引擎全部实现为运行代码** (非设计文档):
  - `perturbation_engine.py`: 对抗扰动 — 2000锚点子集→低相似远距对→参数噪声→D-S验证→负Hebbian修正 (0.3s)
  - `ds_generator.py`: D-S假设生成器 — 扰动+弱边+高相似三源→Dempster组合→注入 (0.1s)
  - `gradient_reverse.py`: 梯度反推 — 20K球面采样→P75/P90鞍点→负梯度追踪→锚点注入 (5.2s)
- **四层知识漏斗**: L1本地SQLite→L2 Wikipedia Dump→L3 Bing+Baidu双引擎并发→L4本地词典
- **搜索架构重写**: 删除百度百科/Google HTML抓取, Bing CN实测1.1s/75%置信度
- **熔断/限速/UA轮转**: `rate_limiter.py` 引擎级熔断器+请求抖动
- **策应器规则模板**: `policy_query.py` 7因子→查询模板映射
- **守护集成**: 每5轮扰动+D-S, 每20轮梯度反推
- **消除 cosine_similarity(unsqueeze)**: 全链路 `norm @ matmul` 避免 OOM (修复3处95GB→100MB)
- **守护速度**: 40-56s/轮 (vs 原80s, -35%)

### v2.3 (2026-06-18) — 三引擎创造架构 + 全链路 GPU 加速

- **三引擎设计**: 对抗扰动引擎 → D-S 假设生成器 → 梯度反推引擎，三引擎串联自主知识发现流水线
  - 对抗扰动: 200 远距对参数注入 → 检测脆性信号 → D-S 验证修正 (`docs/perturbation_engine.md`)
  - D-S 假设生成器: 三源汇聚 (扰动候选/弱概念边/高相似无连) → Dempster 组合 (`docs/ds_hypothesis_generator.md`)
  - 梯度反推: 20000 球面采样 → 鞍点搜索 → 负梯度追踪 → 已知性检测 (`docs/gradient_reverse_engine.md`)
- **七因子全部启用**: `from_landscape=True` 不再限制，7/7 因子有产出
  - EnergyFactor: 硬编码 `-5.0` → P25 百分位数据驱动
  - CoverageFactor: 景观模式用能量统计替代 idioms 词典
  - GradientFactor/SemanticGapFactor/FreshnessFactor: 去掉 `idioms is None` 守卫
- **Hebbian 批量 SGD**: 逐对循环 → 批量矩阵运算，107 对一次性处理
- **扫描并行化**: `parallel=True` 启用，全量扫描 ~30s (vs 原 45s)
- **所有硬编码阈值 → 数据驱动百分位**: 三引擎全部采用运行时 P90/P10/P5 自适应
- **GPU 加速**: 源3 5000×5000 全矩阵单次 kernel launch ~2ms + 重载路径补 `.to(DEVICE)`
- **守护速度**: 每轮 89.5s → 63.4s (-29%); 盲区扫描 6175~7566 去重; 分离度 33.3→4.2

### v2.2 (2026-06-18) — 架构补全 + 工程优化

- **P0 架构补全**: 查询路由 `_route_query()` 4路分发; 位置感知编码 `encode_sequence()` 3^x 加权
- **P1 稳定性加固**: EWC 弹性权重巩固 (每轮正则 + 每50轮 Fisher 锚定); 并发安全锁
- **P2 工程优化**: SQLite 概念图加速 (1.93M 三元组, O(log N), UNION 双向索引); DualExtractor 双重提取; HybridDecoder 混合化能器
- **性能**: 守护每轮 123s → 75s (-39%); 分离度从震荡 2↔230 稳定到 60±15
- **守护集成**: SQLite 三层漏斗 (SQLite→邻接索引→网络搜索); LLM 请求仅聊天模式

### v2.1 (2026-06-17) — 序列臂 + 概念图

- 序列臂: 14,800 对有向字对，非对称 Hebbian 注入方向性
- 概念图脏标记增量保存，消除 257MB JSON 每轮 I/O
- 守护文件锁协调 (inject.lock)

---

## 📝 License

MIT © 2025 李泽坤 (octave-12)
