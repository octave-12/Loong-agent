# D-S 证据理论假设生成器 — 设计文档

> **项目**: Loong-agent (龙珠智能体)  
> **模块**: `loongpearl.core.ds_hypothesis_generator`  
> **依赖**: `fuzzy_graph.py` (D-S引擎), `concept_graph.py` / `concept_graph_sqlite.py` (概念图), `zichang.py` (字场嵌入)  
> **版本**: v1.0  
> **作者**: Loong-agent 架构组  

---

## 一、设计动机

### 1.1 现状

Loong-agent 已具备完善的 D-S 证据理论基础设施：

| 模块 | 文件 | 功能 |
|------|------|------|
| D-S 引擎 | `loongpearl/core/fuzzy_graph.py` | `Evidence` / `BPA` / `DempsterShafer` / `FuzzyGraph` — 基本概率分配、Dempster组合、信念/似然查询 |
| D-S 裁决 | `loongpearl/core/orchestrator.py:860-930` | 候选排名时用 `_compute_candidate_belief()` 做 D-S 聚合裁决 |
| D-S 回写 | `loongpearl/core/orchestrator.py:1761-1776` | `_run_fuzzy_feedback_v2()` 定期将中等置信度三元组写入模糊格并回写概念图 |
| 矛盾消解 | `loongpearl/core/orchestrator.py:1733-1759` | `_run_contra_safe_v2()` 用证据量判断是否安全清除冲突 |
| 概念图 | `loongpearl/core/concept_graph.py` | 193万+ 三元组，6种关系类型，forward_index / char_adjacency 索引 |
| SQLite 加速 | `loongpearl/core/concept_graph_sqlite.py` | O(log N) 查询，`query_char_pairs()` 双向检索 |
| 闭环验证 | `loongpearl/learning/verify_loop.py` | 推断三元组 → 搜索验证 → 置信度修正 |
| 注入管道 | `scripts/inject_concept_graph.py` | 概念图 → 能量景观批量对齐 |

### 1.2 缺口

当前系统缺少一个**主动假设生成**机制。D-S 引擎目前以**被动模式**运行——只在已有三元组上添加证据、融合置信度。它不会主动发现"可能为真但尚未被确认"的潜在知识。

**D-S 假设生成器**填补这一缺口：从多种信号源自动生成候选假设（Hypothesis），为每个假设分配 D-S mass 函数，通过 Dempster 组合融合多源证据，对超过阈值的假设写入概念图（作为推断三元组），最终通过注入管道进入能量景观。

### 1.3 核心思路

```
┌─────────────────────────────────────────────────────────────┐
│                 D-S 假设生成器 流程图                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① 扰动结果     ② 弱边(0.3-0.5)   ③ 高相似无连接对          │
│  (perturbation)  (weak edges)      (high-sim no-edge)       │
│       │               │                   │                 │
│       ▼               ▼                   ▼                 │
│  ┌────────┐    ┌────────────┐    ┌────────────────┐        │
│  │ 源1 mass │    │ 源2 mass   │    │ 源3 mass       │        │
│  │ 函数设计 │    │ 函数设计    │    │ 函数设计       │        │
│  └────┬───┘    └─────┬──────┘    └───────┬────────┘        │
│       │               │                   │                 │
│       └───────────────┼───────────────────┘                 │
│                       ▼                                     │
│              ┌────────────────┐                             │
│              │ Dempster 组合   │                             │
│              │ m₁⊕m₂⊕m₃(A)   │                             │
│              └───────┬────────┘                             │
│                      ▼                                      │
│              ┌────────────────┐                             │
│              │ 置信度 > 0.7?  │                             │
│              └───┬───────┬────┘                             │
│               YES│       │NO                               │
│                  ▼       ▼                                  │
│          ┌──────────┐ ┌──────────┐                         │
│          │ 注入概念图 │ │ 标记待验证│                         │
│          │ + D-S回写 │ │ (pending) │                         │
│          └──────────┘ └──────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、假设生成源（3大来源）

### 2.1 源1：扰动结果（Perturbation Results）

**定义**: 对概念图的节点/边施加微小扰动后，观察能量景观的变化，将"扰动后能量显著下降"的概念对作为候选假设。

**实现机制**:

```
扰动分析流程:
1. 选取种子概念集合 S = {前端查询热词, 低度节点, 盲区概念}
2. 对每个 s ∈ S:
   a. 在字场 anchors[s] 上添加高斯噪声 ε ~ N(0, σ²)
   b. 前向传播: 扰动向量 → 能量景观 E(·)
   c. 与原始能量比较: ΔE = E(v + ε) - E(v)
3. 收集负 ΔE 概念对 (能量下降 > 阈值 = 扰动"发现"了更优路径)
4. 对这些对查询概念图: 如果图中无直接边, 则生成候选假设
```

**伪代码**:
```python
def source_perturbation(field, landscape, cg, sigma=0.05, n_perturb=50):
    """
    扰动源假设生成 — 二阶段过滤: 先收集所有ΔE→计算百分位阈值→再过滤
    
    Returns: List[Hypothesis]
    """
    hypotheses = []
    hanzi_list = field.hanzi_list[:8000]
    seeds = random.sample(hanzi_list, n_perturb)
    device = next(landscape.parameters()).device
    
    # ═══ 阶段1: 收集所有扰动结果 ═══
    all_results = []  # [(seed, near_char, delta_E, cos_sim), ...]
    
    for seed in seeds:
        idx = field._char_to_idx.get(seed)
        if idx is None:
            continue
        anchor_vec = field.anchors[idx].to(device)
        
        for _ in range(5):
            epsilon = torch.randn(1024, device=device) * sigma
            perturbed = F.normalize(anchor_vec + epsilon, dim=-1)
            
            with torch.no_grad():
                E_orig = landscape(anchor_vec.unsqueeze(0)).item()
                E_pert = landscape(perturbed.unsqueeze(0)).item()
            delta_E = E_pert - E_orig
            
            if delta_E >= 0:
                continue  # 能量未下降, 跳过
            
            # 找最近锚点
            sims = F.cosine_similarity(perturbed.unsqueeze(0),
                                       field.anchors[:8000].to(device))
            top5 = sims.topk(5)
            for near_idx, sim_val in zip(top5.indices, top5.values):
                near_char = field.hanzi_list[near_idx.item()]
                if near_char == seed:
                    continue
                all_results.append((seed, near_char, delta_E, sim_val.item()))
    
    if not all_results:
        return hypotheses
    
    # ═══ 阶段2: 数据驱动阈值 → 过滤 + 生成假设 ═══
    all_deltas = [abs(r[2]) for r in all_results]
    delta_threshold = float(np.percentile(all_deltas, 90))  # P90: 仅保留10%最强信号
    
    for seed, near_char, delta_E, sim in all_results:
        if abs(delta_E) < delta_threshold:
            continue
        
        # ★ 修复: 用 forward_index 双向检查（不用不存在 .has_edge()）
        has_edge = (
            cg.forward_index.get(seed, {}).get(near_char) is not None or
            cg.forward_index.get(near_char, {}).get(seed) is not None
        )
        if has_edge:
            continue
        
        # ★ 修复: mass 归一化用 delta_threshold 替代硬编码 0.3
        abs_de_norm = min(abs(delta_E) / delta_threshold, 1.0)
        mass = 0.3 + 0.4 * abs_de_norm + 0.2 * sim
        
        hypotheses.append(Hypothesis(
            subject=seed,
            relation="RELATED",
            object=near_char,
            source="perturbation",
            source_confidence=mass,
            metadata={
                'delta_E': delta_E,
                'cosine_sim': sim,
                'sigma': sigma,
            }
        ))
    
    return hypotheses
```

**mass 函数设计（源1）**:

```
m₁(A) = 0.3 + 0.4 × min(|ΔE|/δ_threshold, 1.0)  +  0.2 × cosine_similarity

其中:
  - ΔE: 扰动能量下降量（负值，取绝对值）
  - δ_threshold: 所有扰动对 |ΔE| 的 P90 百分位（运行时计算，自适应能量尺度）
  - |ΔE|/δ_threshold: 归一化到 [0, 1]，δ_threshold 对应满权重
  - cosine_similarity: 扰动点与最近锚点的余弦相似度
  
  mass 范围: 0.3 ~ 0.9
  base=0.3: 扰动信号本身的中等置信度
  +0.4×(|ΔE|/δ_threshold): 能量下降归一化贡献（自动适配能量景观绝对尺度）
  +0.2×(cos_sim): 嵌入空间近邻贡献

⚠️ 为何不用硬编码 0.3 ceiling？
  能量景观实际输出范围 -250~-150，扰动 |ΔE| 可达 1~20。
  min(|ΔE|, 0.3)/0.3 恒为 1.0，无区分度。
  改用 δ_threshold (P90) 除 → 只有 |ΔE| 达到全量前10%
  的扰动对才获满权重，其余按比例阶梯递减。
```

**设计原理**: 扰动分析本质上是"在嵌入空间中探索潜在的低能量路径"。如果随机扰动能发现一条比原始锚点更优的能量配置，且该配置与另一个概念（锚点）的嵌入高度相似，则暗示这两个概念可能存在尚未被概念图捕获的关联。

---

### 2.2 源2：弱概念图边（Weak Edges, 0.3 ≤ confidence ≤ 0.5）

**定义**: 概念图中已存在但置信度介于 0.3 到 0.5 之间的三元组。这些边"有点可能为真但证据不足"——是假设生成的天然候选。

**实现机制**:

```python
def source_weak_edges(cg, min_conf=0.3, max_conf=0.5, max_candidates=200):
    """
    弱边源假设生成。
    从概念图 SQLite 中查询中等置信度的三元组。
    
    Returns: List[Hypothesis]
    """
    hypotheses = []
    
    # 方法1: 从概念图内存索引 (forward_index)
    for s, edges in cg.forward_index.items():
        for obj, rel in edges.items():
            key = f"{s}|{rel}|{obj}"
            triple = cg.triples.get(key)
            if triple and min_conf <= triple.confidence <= max_conf:
                hypotheses.append(Hypothesis(
                    subject=s,
                    relation=rel,
                    object=obj,
                    source="weak_edge",
                    source_confidence=triple.confidence,
                    metadata={
                        'original_source': triple.source,
                        'evidence_count': triple.evidence_count,
                        'inferred_from': triple.inferred_from,
                    }
                ))
            if len(hypotheses) >= max_candidates:
                break
        if len(hypotheses) >= max_candidates:
            break
    
    # 方法2: SQLite 加速（备选，大数据集时使用）
    # SELECT s, r, o, c, src FROM triples WHERE c BETWEEN 0.3 AND 0.5
    
    return hypotheses
```

**mass 函数设计（源2）**:

```
m₂(A) = base_conf × relation_weight × evidence_bonus

其中:
  - base_conf = triple.confidence  (原始置信度, 0.3~0.5)
  - relation_weight:
      IS_A:      1.0   (分类关系较可信)
      PART_OF:   1.0   (组成关系较可信)
      HAS:       0.95
      CAUSE:     0.85  (因果关系本就不确定)
      RELATED:   0.8   (一般相关最弱)
      OPPOSITE:  0.9
  - evidence_bonus = 1.0 + min(evidence_count, 3) * 0.05
    (每条额外证据 +5%, 最多 +15%)
  
  mass 范围: 0.24 ~ 0.575
```

**设计原理**: 弱边本身已经通过了概念图的添加门槛（置信度 ≥ 0.3），但由于证据不足未能达到高置信区间。D-S 假设生成器的职责是从其他信息源（扰动、嵌入相似度）收集额外证据，看能否将它们提升到可接受阈值以上。

---

### 2.3 源3：高相似度无连接对（High-Similarity No-Edge Pairs）

**定义**: 在字场嵌入空间（1024维）中余弦相似度高于阈值（如 ≥ 0.75），但概念图中没有任何直接边的概念对。

**实现机制**:

```python
def source_high_similarity_no_edge(field, cg, sim_threshold=0.75, 
                                     max_pairs=300):
    """
    高相似无连接对源假设生成。
    
    ★ GPU 单次全矩阵计算: N×N 余弦相似度 ≈ 100MB/2ms。
    无需分块，GPU tensor core 一次 kernel launch 完成全部计算。
    
    Returns: List[Hypothesis]
    """
    hypotheses = []
    anchors = field.anchors
    hanzi_list = field.hanzi_list
    device = anchors.device
    
    # 搜索范围: 5000 字（全矩阵 100MB，RTX 3060 12GB 完全够）
    search_range = min(5000, len(hanzi_list))
    sub_anchors = anchors[:search_range].to(device)  # (N, 1024)
    
    # ═══ GPU 单次全矩阵余弦相似度 ═══
    # (N, D) → (N, N): 一个 kernel launch, ~2ms
    sim_matrix = F.cosine_similarity(
        sub_anchors.unsqueeze(1),   # (N, 1, D)
        sub_anchors.unsqueeze(0),   # (1, N, D)
        dim=2
    )  # (N, N), dtype=float32, ~100MB
    
    # 构建已有边快速查询集合（对称化）
    existing_edges = set()
    for s, edges in cg.forward_index.items():
        for obj in edges:
            existing_edges.add((s, obj))
            existing_edges.add((obj, s))  # 双向: 任何方向有边即跳过
    
    # 提取候选: 上三角 + 高相似 + 排除已有边
    mask_upper = torch.triu(
        torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1
    )
    mask_sim = sim_matrix > sim_threshold
    candidates = (mask_upper & mask_sim).nonzero(as_tuple=False)
    
    for idx_pair in candidates:
        qi, si = idx_pair[0].item(), idx_pair[1].item()
        char_a = hanzi_list[qi]
        char_b = hanzi_list[si]
        
        if (char_a, char_b) in existing_edges:
            continue
        
        sim_val = sim_matrix[qi, si].item()
        rel = infer_relation_type(char_a, char_b, field)
        
        hypotheses.append(Hypothesis(
            subject=char_a,
            relation=rel,
            object=char_b,
            source="high_similarity_no_edge",
            source_confidence=sim_val,
            metadata={
                'cosine_sim': sim_val,
                'embedding_norm_a': sub_anchors[qi].norm().item(),
                'embedding_norm_b': sub_anchors[si].norm().item(),
            }
        ))
        
        if len(hypotheses) >= max_pairs:
            break
    
    return hypotheses
```

def infer_relation_type(char_a, char_b, field):
    """
    根据嵌入空间的几何关系推测最可能的关系类型。
    
    启发式:
      - 如果 a 的 norm 显著小于 b: IS_A (上位词)
      - 如果余弦相似度极高 (>0.85): RELATED 或 PART_OF
      - 如果方向性投影差距大: CAUSE
      - 默认: RELATED
    """
    idx_a = field._char_to_idx.get(char_a)
    idx_b = field._char_to_idx.get(char_b)
    if idx_a is None or idx_b is None:
        return "RELATED"
    
    vec_a = field.anchors[idx_a]
    vec_b = field.anchors[idx_b]
    
    norm_a = vec_a.norm().item()
    norm_b = vec_b.norm().item()
    sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
    
    if sim > 0.88:
        return "PART_OF"      # 极高相似 → 组成关系
    elif norm_a < norm_b * 0.85:
        return "IS_A"         # A 在语义上"包含于"B
    elif norm_b < norm_a * 0.85:
        return "HAS"          # A "拥有" B 的特征
    elif 0.75 <= sim < 0.85:
        return "RELATED"
    else:
        return "RELATED"
```

**mass 函数设计（源3）**:

```
m₃(A) = 0.2 + 0.6 × (cosine_sim - 0.75) / 0.25  +  0.1 × relation_novelty

其中:
  - cosine_sim: 余弦相似度 (0.75~1.0)
  - (cosine_sim - 0.75)/0.25: 归一化到 [0, 1]
  - relation_novelty: 
      RELATED    → 1.0  (最安全, 基础奖励)
      PART_OF    → 0.9
      IS_A       → 0.8
      HAS        → 0.7
      CAUSE      → 0.5  (因果关系需要更强的语义证据)
  
  mass 范围: 0.25 ~ 0.87
```

**设计原理**: 嵌入空间中的高余弦相似度暗示两个概念在语义上相关。如果概念图中没有对应的边，这是一个"知识缺口"信号。以 RELATED 作为最保守的关系猜测，辅以启发式规则推断更具体的关系类型。该源对 PART_OF/IS_A/HAS 等关系更可信，对 CAUSE 保持保守。

---

## 三、D-S Mass 函数总览

| 证据源 | mass 范围 | 核心设计 | 特点 |
|--------|-----------|----------|------|
| 源1：扰动结果 | 0.30 ~ 0.90 | `0.3 + 0.4×|ΔE|_norm + 0.2×cos_sim` | 捕获嵌入空间中隐藏的低能量关联 |
| 源2：弱边 | 0.24 ~ 0.58 | `conf × rel_weight × evidence_bonus` | 利用已有信息的"残值"，需多源证据支撑 |
| 源3：高相似无连接 | 0.25 ~ 0.87 | `0.2 + 0.6×sim_norm + 0.1×rel_novelty` | 发现知识缺口，语义相似度驱动 |

### 3.1 mass 函数约束

所有 mass 函数满足 D-S 理论基本约束：

```
∀ source_i:  0 ≤ mᵢ(A) ≤ 1
              mᵢ(Ω) = 1 - mᵢ(A)  (mass 分配给全集 = 不确定性)
              mᵢ(∅) = 0          (空集 mass = 0)
```

**全集分配** `mᵢ(Ω) = 1 - mᵢ(A)` 表示"证据不否认命题但不完全支持"的剩余概率质量，使 Dempster 组合能正确处理部分证据。

### 3.2 多证据源整合时的处理

当同一个命题（如 `("量子", "RELATED", "物理")`）从多个源获得证据时：

```python
# 从源1 (扰动) 获得: m₁("量子 RELATED 物理") = 0.45
# 从源2 (弱边) 获得: m₂("量子 RELATED 物理") = 0.35
# 从源3 (高相似) 获得: m₃("量子 RELATED 物理") = 0.52

# 三个独立证据通过 Dempster 组合融合:
# m = m₁ ⊕ m₂ ⊕ m₃
```

---

## 四、Dempster 组合规则

### 4.1 标准公式

Dempster 组合规则用于融合来自多个独立证据源的 mass 函数：

```
m₁⊕m₂(A) = ∑_{B∩C=A} m₁(B)·m₂(C) / (1 - K)

其中:
  K = ∑_{B∩C=∅} m₁(B)·m₂(C)   (冲突度量)
  1 - K 为归一化因子
```

### 4.2 本模块中的实现

直接复用 `fuzzy_graph.py` 中已验证的 `BPA.combine()` 方法：

```python
# loongpearl/core/fuzzy_graph.py BPA.combine() (已存在)
# 迭代逐对组合:
#   combined = {"命题": m₁, "Ω": 1 - m₁}
#   for each mᵢ in remaining:
#       K = 冲突量
#       new_combined = 归一化后的交集质量
#   return combined["命题"]
```

假设生成器中的调用方式：

```python
from loongpearl.core.fuzzy_graph import BPA, Evidence

def combine_sources(hypothesis, source_evidences):
    """
    对同一假设的多源证据进行 Dempster 组合。
    
    Args:
        hypothesis: Hypothesis 对象
        source_evidences: List[Evidence], 每个证据来自不同源
    
    Returns:
        combined_belief: float  (融合后的信念质量)
        conflict_K: float       (冲突度量)
    """
    prop_str = f"{hypothesis.subject} {hypothesis.relation} {hypothesis.object}"
    bpa = BPA(proposition=prop_str, evidences=source_evidences)
    combined = bpa.combine()
    return combined, bpa.combined_mass
```

### 4.3 冲突处理

当多个证据源互相矛盾时（如源1支持而源3否定），冲突量 K 会增大：

```python
# 示例: m₁=0.6, m₂=0.3  (低冲突)
#   m₁⊕m₂ = 0.6×0.3 + 0.6×0.7 + 0.4×0.3 = 0.18 + 0.42 + 0.12 = 0.72
#   K = 0 (同向证据无冲突)

# 高冲突场景: m₁(A)=0.7, m₂(¬A)=0.6
#   K = 0.7 × 0.6 = 0.42 (高冲突, 但本系统不直接处理否定命题)
```

本系统的设计避免了"直接矛盾"——所有三个源都生成 **支持性** 证据（mᵢ(A) > 0），因此冲突主要来自"证据量的差异"而非"方向性矛盾"。

### 4.4 冲突排查

当冲突量 K > 0.3 时，记录告警日志，标记假设为 `CONFLICTING`，不自动注入概念图：

```python
def resolve_conflicting_hypothesis(hypothesis, K):
    if K > 0.5:
        # 高冲突: 丢弃或人工审核
        hypothesis.status = "REJECTED_CONFLICT"
        hypothesis.rejection_reason = f"高冲突 K={K:.2f}"
    elif K > 0.3:
        # 中等冲突: 标记待验证
        hypothesis.status = "PENDING_RESOLUTION"
        hypothesis.metadata['conflict_K'] = K
    else:
        # 低冲突: 继续
        pass
```

---

## 五、假设接受阈值

### 5.1 阈值设计

经过 Dempster 组合后的最终信念质量 `combined_belief`，使用三级阈值：

| 阈值区间 | 决策 | 动作 |
|----------|------|------|
| **bel ≥ 0.70** | ✅ **接受** (ACCEPT) | 作为推断三元组注入概念图 → 写入模糊格 → 可选对齐景观 |
| **0.40 ≤ bel < 0.70** | ⏳ **待验证** (PENDING) | 加入 pending_hypotheses 队列，等待闭环验证 |
| **bel < 0.40** | ❌ **拒绝** (REJECT) | 丢弃或归档到 `rejected_hypotheses` 日志 |

**阈值 0.7 的理由**:

1. **与 D-S 回写一致**: `_run_fuzzy_feedback_v2()` 中，只有 `conf > 0.2` 且 `< 0.7` 的三元组才被写入模糊格；≥0.7 的被认为是"已知高置信知识"，不再重新验证。因此 0.7 是"高置信"的自然分界。

2. **与矛盾消解一致**: `_run_contra_safe_v2()` 中以 `conf > 0.7` 作为"不可清除"的高置信阈值。

3. **D-S 语义**: `bel ≥ 0.7` 意味着"多个独立证据源一致支持该命题，综合信念超过七成"。在知识图谱语境中，这是"合理接受"的水平。

4. **误差容忍**: 0.7 给了足够的"不确定性缓冲"(m(Ω) ≤ 0.3)，避免过度自信。

### 5.2 自适应阈值（可选扩展）

```python
def adaptive_threshold(hypothesis, global_stats):
    """
    根据全局统计动态调整接受阈值。
    - 当已接受假设数过多时，提高阈值
    - 当关系类型为 CAUSE 时，额外 +0.05（因果关系更难验证）
    """
    base = 0.70
    
    # 关系类型调整
    if hypothesis.relation == "CAUSE":
        base += 0.05
    elif hypothesis.relation == "RELATED":
        base -= 0.03  # RELATED 是最安全的推测
    
    # 证据源数量调整：只有1个源支持时要求更高
    n_sources = len(set(e.source for e in hypothesis.evidences))
    if n_sources == 1:
        base += 0.08  # 单源需更强证据
    
    # 全局饱和度调整
    recent_accept_rate = global_stats.get('accept_rate_24h', 0.3)
    if recent_accept_rate > 0.5:
        base += 0.05  # 接受太多，收紧阈值
    
    return min(0.85, max(0.60, base))
```

### 5.3 接受后的置信度赋值

```python
def assign_confidence(belief, n_sources):
    """
    将 D-S 融合后的信念质量映射为概念图置信度。
    
    bel ∈ [0.7, 1.0] → conf ∈ [0.55, 0.85]
    
    注意: 即使 D-S bel 高达 0.95，概念图 conf 也封顶在 0.85，
          因为假设生成仍属于"推断"而非"确认"知识。
          后续闭环验证可以进一步提升到 0.95+。
    """
    if n_sources >= 2:
        # 多源交叉验证: 更可信
        conf = 0.55 + (belief - 0.7) * 1.2
    else:
        # 单源: 更保守
        conf = 0.50 + (belief - 0.7) * 0.8
    
    return min(0.85, max(0.50, conf))
```

---

## 六、与概念图注入管道的集成

### 6.1 集成架构

```
┌──────────────────────────────────────────────────────────────┐
│                   Orchestrator.daemon_tick_v2()               │
│                                                              │
│  每 5 轮:                                                     │
│    self._run_ds_hypothesis_generation_v2()  ← 新增           │
│      │                                                       │
│      ├─ 1. 源1: 扰动分析 (n=50 种子)                         │
│      ├─ 2. 源2: 弱边查询 (n=200 候选)                        │
│      ├─ 3. 源3: 高相似无连接探测 (n=300 候选)                 │
│      ├─ 4. 去重 + 合并同命题的多源证据                         │
│      ├─ 5. D-S Dempster 组合每个假设                          │
│      ├─ 6. 阈值过滤 (bel ≥ 0.7)                              │
│      │                                                       │
│      └─ 7. 注入概念图:                                       │
│            cg.add_triple(s, r, o, conf, source="ds_hypothesis")│
│            fuzzy.add_evidence(s, r, o, source="ds_gen", mass) │
│                                                              │
│  已存在的 _run_fuzzy_feedback_v2() 随后会同步 D-S 回写         │
│  已存在的 _run_prune_and_align() 会挑高置信度边对齐景观         │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 核心类接口

```python
# 文件: loongpearl/core/ds_hypothesis_generator.py

@dataclass
class Hypothesis:
    """一条 D-S 假设"""
    subject: str
    relation: str
    object: str
    source: str                    # "perturbation" | "weak_edge" | "high_similarity"
    source_confidence: float
    evidences: List[Evidence] = field(default_factory=list)
    combined_belief: float = 0.0
    conflict_K: float = 0.0
    status: str = "PENDING"        # PENDING | ACCEPTED | REJECTED
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class DSHypothesisGenerator:
    """D-S 证据理论假设生成器"""
    
    def __init__(self, field, landscape, concept_graph, fuzzy_graph=None):
        self.field = field          # HanziAnchorField
        self.landscape = landscape  # FreqEnergyLandscape
        self.cg = concept_graph     # ConceptGraph
        self.fuzzy = fuzzy_graph    # FuzzyGraph (可选)
        self.stats = {...}
    
    def generate_all(self, 
                     perturbation_count=50,
                     weak_edge_limit=200,
                     similarity_limit=300,
                     similarity_threshold=0.75,
                     acceptance_threshold=0.70) -> Dict:
        """
        主入口: 从三个源生成假设、融合、过滤。
        
        Returns:
            {
                'generated': int,         # 生成的候选假设总数
                'accepted': int,          # 接受的假设数
                'pending': int,           # 待验证数
                'rejected': int,          # 拒绝数
                'injected_into_cg': int,  # 注入概念图的数量
                'accepted_hypotheses': [...],
                'stats': {...},
            }
        """
    
    def inject_to_concept_graph(self, hypothesis) -> bool:
        """将接受的假设注入概念图和模糊格"""
    
    def inject_to_landscape(self, hypotheses, batch_size=1000):
        """将接受的假设批量注入能量景观（Hebbian 学习）"""
```

### 6.3 调度器集成

在 `orchestrator.py` 的 `daemon_tick_v2()` 中添加调用（每5轮执行）：

```python
# 在 orchestrator.py 的 daemon_tick_v2 中 (~line 1530 附近):
if round_num % 5 == 0:
    # ... 现有代码 ...
    
    # ★ 新增: D-S 假设生成
    try:
        self._run_ds_hypothesis_generation_v2()
    except Exception as e:
        log.debug(f"  D-S假设生成异常: {e}")

# 对应的新方法:
def _run_ds_hypothesis_generation_v2(self) -> Dict:
    """D-S 假设生成 + 注入"""
    if not hasattr(self, '_ds_gen'):
        from loongpearl.core.ds_hypothesis_generator import DSHypothesisGenerator
        self._ds_gen = DSHypothesisGenerator(
            self.field, self.landscape, self.cg, self.fuzzy
        )
    
    result = self._ds_gen.generate_all(
        perturbation_count=50,
        weak_edge_limit=200,
        similarity_limit=300,
    )
    
    if result['accepted'] > 0:
        for hyp in result['accepted_hypotheses']:
            # 写入概念图（作为 infer 源以避免被闭环验证跳过）
            self.cg.add_triple(
                hyp.subject, hyp.relation, hyp.object,
                confidence=hyp.concept_confidence,
                source="ds_hypothesis_gen"
            )
            # 写入模糊格
            self.fuzzy.add_evidence(
                hyp.subject, hyp.relation, hyp.object,
                source=f"ds_gen_{hyp.source}",
                mass=hyp.combined_belief
            )
        
        log.info(f"  🧬 D-S假设生成: 接受{result['accepted']}条, "
                f"注入概念图{result['injected_into_cg']}条")
    
    return result
```

### 6.4 与注入管道的关系

假设生成器产出的三元组经过两条路径进入系统：

```
路径A: 概念图注入 (即时生效)
  D-S假设 → cg.add_triple(conf=0.50-0.85, source="ds_hypothesis_gen")
          → _sync_fuzzy_to_cg() 回写模糊格融合置信度
          → forward_index 索引更新
          → 可被后续查询检索

路径B: 能量景观注入 (批量对齐)
  每5轮 _run_prune_and_align() 自动将概念图中 conf ≥ 0.5 的三元组
  提取字对 → Hebbian 学习注入能量景观
  → 已有管道自动覆盖 D-S 生成的假设
```

### 6.5 性能考量与调度

**频率控制**: 假设生成涉及扰动分析（需前向传播）和高相似度计算（O(N²) 矩阵运算），成本较高。因此不是每轮运行：

| 操作 | 频率 | 成本 |
|------|------|------|
| 源1：扰动分析 | 每 5 轮 | 高（50×5×前向传播） |
| 源2：弱边查询 | 每 5 轮 | 低（SQLite 索引查询） |
| 源3：高相似检测 | 每 20 轮 | 低（GPU 单次全矩阵 5000×5000, ~2ms, 100MB） |
| D-S 组合 + 注入 | 每 5 轮 | 低 |

**去重机制**: 同一命题可能从不同源获得证据，通过 `(subject, relation, object)` 三元组 key 去重合并。

**增量模式**: 高相似度检测维护一个 `_processed_pairs` 缓存，避免重复计算已处理的概念对。

---

## 七、数据结构与持久化

### 7.1 假设日志

```python
# data/runtime/ds_hypothesis_log.jsonl
# 每行一条 JSON，记录假设的完整生命周期

{
    "id": "ds_hyp_20260618_0001",
    "proposition": "量子 RELATED 物理",
    "sources": ["perturbation", "weak_edge"],
    "mass_values": {"perturbation": 0.45, "weak_edge": 0.35},
    "combined_belief": 0.68,
    "conflict_K": 0.05,
    "status": "ACCEPTED",
    "injected_confidence": 0.58,
    "timestamp": 1718672400.0
}
```

### 7.2 统计指标

```python
# 假设生成器 stats
{
    'total_generated': 15230,          # 累计生成假设
    'total_accepted': 1847,            # 累计接受
    'total_rejected': 11238,           # 累计拒绝
    'total_pending': 2145,             # 当前待验证
    'accept_rate': 0.121,              # 接受率
    'by_source': {
        'perturbation': {'gen': 4520, 'acc': 412},
        'weak_edge': {'gen': 6800, 'acc': 890},
        'high_similarity': {'gen': 3910, 'acc': 545},
    },
    'avg_combined_belief_accepted': 0.76,
    'avg_conflict_K': 0.08,
    'top_relations_accepted': {
        'RELATED': 1021,
        'PART_OF': 345,
        'IS_A': 289,
        'HAS': 152,
        'CAUSE': 40,
    },
}
```

---

## 八、安全性考量

### 8.1 防止知识污染

1. **源分离标记**: 每个假设记录所有证据源，注入概念图时 `source="ds_hypothesis_gen"`，与人工知识 (`manual`)、搜索提取 (`extract`)、归纳推理 (`infer`) 明确区分。

2. **置信度封顶**: 即使 D-S 信念高达 0.95，概念图置信度也封顶在 0.85。此限制确保假设生成永不超越"人工确认"或"强搜索证据"的知识。

3. **闭环验证兜底**: 被接受的假设置信度在 0.50-0.85 之间。`VerifyLoop` 可以随后对它们进行搜索验证 → 确认则提升，矛盾则降低。

### 8.2 防止级联放大

1. **禁止自举**: 假设生成器生成的边不会被用作后续假设生成的证据（避免"以假修真"）。

2. **源1扰动独立性**: 扰动分析只依赖字场嵌入和能量景观，不引用概念图内容（除检查已有边外）。

3. **周期性重置**: 建议每 100 轮清空 `_processed_pairs` 缓存，允许对"新获得的信息"重新评估。

### 8.3 可审计性

- 每条注入概念图的假设都可通过 `triple.source == "ds_hypothesis_gen"` 追溯
- `ds_hypothesis_log.jsonl` 保存完整证据链：哪些源、各源 mass、融合后信念
- FuzzyGraph 中对应的 BPA 保存每条独立证据的详细来源

---

## 九、迁移路线图

### Phase 1: 核心实现 (1-2周)

- [ ] `loongpearl/core/ds_hypothesis_generator.py` — 核心类实现
- [ ] 单元测试: 各源的 mass 函数输出在 [0,1] 范围内
- [ ] 单元测试: Dempster 组合结果与手算一致
- [ ] 集成测试: 在空概念图 + 模拟数据上运行

### Phase 2: 调度器集成 (1周)

- [ ] `orchestrator.py` 添加 `_run_ds_hypothesis_generation_v2()`
- [ ] 频率控制: 源3 改为每20轮执行
- [ ] 统计指标暴露到 `status_report()`

### Phase 3: 调优与评估 (1周)

- [ ] 阈值校准: 在 1.93M 三元组的真实概念图上运行，观察接受率
- [ ] A/B 测试: 开启/关闭假设生成，评估概念图覆盖率的改善
- [ ] 人工抽检: 随机抽取 100 条接受的假设做人工质量评估

### Phase 4: 高级特性 (可选)

- [ ] 自适应阈值
- [ ] 假设排名: 按 `combined_belief × source_diversity` 排序优先注入
- [ ] 与 `CrossDomainBridge` 联动: 跨域桥接生成的新边作为假设生成器的附加源

---

## 十、附录：关键公式速查

### Dempster 组合（迭代形式）

```python
# 输入: m_list = [m₁, m₂, ..., mₙ]  (每个 mᵢ ∈ [0,1])
# 命题: A = "s relation o"

combined = {"A": m_list[0], "Ω": 1 - m_list[0]}
for m in m_list[1:]:
    new = {}
    K = 0
    for key1, val1 in combined.items():
        for key2, val2 in {"A": m, "Ω": 1-m}.items():
            if key1 == "Ω":  inter = key2
            elif key2 == "Ω": inter = key1
            elif key1 == key2: inter = key1
            else: inter = None  # 冲突
            if inter is None: K += val1 * val2
            else: new[inter] = new.get(inter, 0) + val1 * val2
    for k in new: new[k] /= (1 - K) if K < 1 else 1
    combined = new
belief = combined.get("A", 0.0)
```

### 三源 Mass 函数

```
源1 (扰动):       m₁ = clamp(0.3 + 0.4·|ΔE|_norm + 0.2·cos_sim, 0, 1)
源2 (弱边):       m₂ = conf × rel_weight × (1 + min(ev_count,3)×0.05)
源3 (高相似):     m₃ = clamp(0.2 + 0.6·sim_norm + 0.1·rel_novelty, 0, 1)

其中:
  |ΔE|_norm = min(|ΔE|, 0.3) / 0.3
  sim_norm = (cosine_sim - 0.75) / 0.25
```

### 概念图置信度映射

```
conf = 0.55 + (bel - 0.7) × 1.2   (多源)
conf = 0.50 + (bel - 0.7) × 0.8   (单源)
封顶: 0.85
```

---

*文档版本: v1.0 | 最后更新: 2026-06-18 | 关联模块: `fuzzy_graph.py`, `concept_graph.py`, `orchestrator.py`, `concept_graph_sqlite.py`, `inject_concept_graph.py`*
