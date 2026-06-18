# 梯度反推引擎（Gradient Reverse Prediction Engine）设计文档

> **版本**: v1.0  
> **日期**: 2026-06-18  
> **所属项目**: Loong-agent / 龙珠知识系统  
> **依赖**: `FreqEnergyLandscape` (freq_landscape.py), `HanziAnchorField` (zichang.py), `ConceptGraph` (concept_graph.py), `ConceptGraphSQLite` (concept_graph_sqlite.py)

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [能量景观核心回顾](#2-能量景观核心回顾)
3. [鞍点搜索算法](#3-鞍点搜索算法)
4. [负梯度追踪到锚点](#4-负梯度追踪到锚点)
5. [已知性检测（概念图 + SQLite）](#5-已知性检测概念图--sqlite)
6. [候选过滤标准](#6-候选过滤标准)
7. [概念图注入集成](#7-概念图注入集成)
8. [实现架构与数据结构](#8-实现架构与数据结构)
9. [使用流程](#9-使用流程)
10. [性能估算与调优参数](#10-性能估算与调优参数)

---

## 1. 背景与动机

### 1.1 当前系统的盲点

龙珠当前的信号驱动推理管道（`Orchestrator.query()`）是被动反应式的：

```
用户查询 → 能量景观梯度下降 → 信号发射 → 手脚响应
```

只有当用户**主动询问**某个未知概念时，系统才会检测到 `blind_spot` 信号，然后触发双臂搜索。这意味着：

- **未知知识的发现完全依赖用户输入**，无法主动探索知识边界
- **大量潜在的概念关联**沉睡在能量景观的山脊/鞍部，从未被发掘
- **概念图的增长是被动的**，无法自主扩展

### 1.2 梯度反推引擎的核心思想

**逆向思考**：既然梯度下降是 "从查询走到最近吸引子"（正向），那反过来——"从能量景观的山脊/鞍部出发，沿负梯度跟踪到锚点"——就能主动发现**尚未形成盆地的概念关联**。

```
正向（现有）:  查询向量 x₀ ──∇E↓──→ 吸引子盆地 → 汉字锚点
反向（新增）:  鞍点 s ──∇E↓──→ 最近锚点 a → 候选配对 (s, a)
```

### 1.3 引擎目标

| 目标 | 说明 |
|------|------|
| **主动发现** | 在能量景观中找到高能量、高梯度范数的鞍点区域 |
| **反向追踪** | 从鞍点沿负梯度下降，找到最近的汉字锚点 |
| **去重过滤** | 检查 (源概念, 锚点) 配对是否已存在于概念图 |
| **智能筛选** | 按能量差、梯度范数、锚点距离等指标排序候选 |
| **闭环注入** | 将高质量候选自动注入概念图，驱动能量景观重训练 |

---

## 2. 能量景观核心回顾

### 2.1 FreqEnergyLandscape 关键方法

```python
# freq_landscape.py — FreqEnergyLandscape 类

class FreqEnergyLandscape(nn.Module):
    """
    双通路架构:
      主通路: 嵌入 → MLP(2048→2048→1024→512→1) → 基础能量
      频率通路: freq → MLP(32→1) → 能量偏移
      最终能量 = 基础能量 + 频率偏移
    """

    def forward(self, x, freq=None):
        """x: (N, 1024), freq: (N,) or None → (N, 1) 能量值"""
        base = self.net(x)  # 6层 MLP
        if freq is not None:
            shift = self.freq_shift(freq.unsqueeze(-1))
            return base + shift
        return base

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """计算标量能量值。x: (N,1024) → (N,)"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.forward(x).squeeze(-1)

    def infer(self, query_vec, steps=50, lr=0.02, zichang=None) -> Dict:
        """
        梯度下降推理:
          1. x.requires_grad_(True)
          2. optimizer = Adam([x], lr=lr)
          3. for step in range(steps):
               e = self.energy(x); e.backward()
               optimizer.step()
               project_to_sphere(x)
          4. return {state, energy, signal, top_candidates, ...}
        """
```

### 2.2 梯度计算方式

```python
# 获取某点的能量和梯度：
x = some_point.clone().detach()
x.requires_grad_(True)
e = landscape.energy(x)    # 标量能量
e.backward()               # 计算梯度
grad = x.grad              # (embed_dim,) 梯度向量
grad_norm = grad.norm().item()  # 梯度范数
```

### 2.3 HanziAnchorField 关键方法

```python
# zichang.py — HanziAnchorField 类

class HanziAnchorField:
    anchors: torch.Tensor      # (94117, 1024) 冻结的锚点矩阵
    _char_to_idx: Dict[str, int]  # 汉字 → 锚点索引

    def find_nearest(self, query_vec, k=5):
        """
        余弦相似度检索:
          similarities = cosine_similarity(query_vec, anchors, dim=1)
          top_k = torch.topk(similarities, k)
        Returns: (indices, chars, similarities)
        """
```

### 2.4 能量景观的几何直觉

```
能量 E ↑
    │
    │   ╱╲        ╱╲          ← 山脊（高能量，概念间过渡区）
    │  ╱  ╲  ╱╲  ╱  ╲
    │ ╱    ╲╱  ╲╱    ╲        ← 鞍点（高能量 + 高梯度范数）
    │╱                  ╲
    └──────────────────────→ 嵌入空间
       盆地A   鞍部   盆地B    ← 盆地（低能量，已知概念锚点）
```

- **盆地 (Basin)**：锚点附近，能量低，梯度小 → `infer()` 的收敛目标
- **山脊 (Ridge)**：概念间过渡区，能量高，梯度中等
- **鞍点 (Saddle)**：两个盆地之间，能量高，梯度范数高 → **梯度反推引擎的搜索目标**

---

## 3. 鞍点搜索算法

### 3.1 鞍点的数学定义

在能量景观 `E: ℝ¹⁰²⁴ → ℝ` 中，鞍点满足：

```
E(s) > E_threshold           (高能量 — 说明不在已知盆地中)
‖∇E(s)‖ > grad_threshold     (高梯度范数 — 说明处于两盆地之间的"分水岭")
```

### 3.2 搜索策略：球面随机采样 + 梯度验证

由于嵌入空间维度极高（1024维），穷举搜索不可行。采用**两步筛选法**：

#### 第一步：球面均匀采样

```python
def sample_sphere_points(n_samples: int = 10000, embed_dim: int = 1024) -> Tensor:
    """
    在单位球面上均匀采样。
    字场锚点均经过 L2 归一化，采样点也必须在同一流形上。
    """
    points = torch.randn(n_samples, embed_dim)       # 标准正态采样
    points = F.normalize(points, p=2, dim=1)          # 投影到单位球面
    return points
```

#### 第二步：批量能量+梯度评估

```python
def evaluate_landscape_batch(
    landscape: FreqEnergyLandscape,
    points: Tensor,          # (N, 1024)
    batch_size: int = 256,
) -> Tuple[Tensor, Tensor]:
    """
    批量计算每个点的能量值和梯度范数。
    Returns: (energies: (N,), grad_norms: (N,))
    """
    all_energies = []
    all_grad_norms = []

    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size].clone().detach()
        batch.requires_grad_(True)

        e = landscape.energy(batch)  # (B,)

        # 计算每个样本独立的梯度范数
        grad_norms = []
        for j in range(len(batch)):
            if batch.grad is not None:
                batch.grad.zero_()
            e[j].backward(retain_graph=True)
            gn = batch.grad[j].norm().item()
            grad_norms.append(gn)

        all_energies.append(e.detach().cpu())
        all_grad_norms.append(torch.tensor(grad_norms))

    return (
        torch.cat(all_energies),      # (N,)
        torch.cat(all_grad_norms),    # (N,)
    )
```

> **优化技巧**：对于真正的批量梯度计算，可用 `e.sum().backward()` 一次性得到所有点的梯度，然后通过 `batch.grad.norm(dim=1)` 批量计算范数。

```python
# 高效批量版本
def evaluate_landscape_batch_fast(
    landscape: FreqEnergyLandscape,
    points: Tensor,          # (N, 1024)
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor]:
    """高效批量评估：用 sum().backward() 一次计算全部梯度"""
    energies_list = []
    grad_norms_list = []

    landscape.eval()  # 不需要训练模式来获取梯度

    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size].clone().detach()
        batch.requires_grad_(True)

        e = landscape.energy(batch)          # (B,)
        e_sum = e.sum()                       # 标量，backward 得到每个分量的梯度
        e_sum.backward()

        gn = batch.grad.norm(dim=1).detach().cpu()  # (B,) 每点梯度范数

        energies_list.append(e.detach().cpu())
        grad_norms_list.append(gn)

        landscape.zero_grad()

    return torch.cat(energies_list), torch.cat(grad_norms_list)
```

#### 第三步：鞍点筛选

```python
def find_saddle_points(
    landscape: FreqEnergyLandscape,
    n_samples: int = 20000,
    energy_pctl: float = 75.0,     # 数据驱动：取能量P75
    grad_pctl: float = 90.0,       # 数据驱动：取梯度P90
    top_k: int = 100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    在能量景观中寻找鞍点。使用数据驱动百分位阈值。
    
    能量阈值 = P75（能量最高的25%的点）
    梯度阈值 = P90（梯度最大的10%的点）
    两者交集 = 同时高能量+高梯度的鞍部区域
    """
    import numpy as np
    
    # 1. 球面采样
    samples = sample_sphere_points(n_samples)
    
    # 2. 批量评估
    energies, grad_norms = evaluate_landscape_batch_fast(landscape, samples)
    
    # 3. 数据驱动阈值
    en = energies.numpy()
    gn = grad_norms.numpy()
    e_thresh = float(np.percentile(en, energy_pctl))
    g_thresh = float(np.percentile(gn, grad_pctl))
    
    # 4. 筛选：高能量 + 高梯度范数 = 鞍点
    mask = (energies > e_thresh) & (grad_norms > g_thresh)
    candidates = samples[mask]
    cand_energies = energies[mask]
    cand_grads = grad_norms[mask]
    
    # 5. 按质量排序
    quality = cand_energies * cand_grads
    _, top_indices = torch.topk(quality, min(top_k, len(quality)))
    
    return (
        candidates[top_indices],
        cand_energies[top_indices],
        cand_grads[top_indices],
    )
```

### 3.3 进阶：基于梯度的主动搜索（可选）

除纯随机采样外，还可以**从已知锚点出发，沿梯度上升方向探索**到鞍点：

```python
def ascend_to_saddle(
    landscape: FreqEnergyLandscape,
    start_anchor: Tensor,    # (1024,) 某个锚点向量
    steps: int = 30,
    lr: float = 0.01,
    noise_std: float = 0.1,
) -> Tensor:
    """
    从一个已知锚点出发，沿梯度上升方向走到鞍部。
    梯度上升 = 远离盆地 → 进入山脊/鞍部区域。

    加噪声防止精确走到另一个盆地。
    """
    x = start_anchor.clone().detach()
    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        e = landscape.energy(x)
        (-e).backward()  # 梯度上升：最大化能量
        optimizer.step()

        # 投影回球面
        with torch.no_grad():
            x.data = F.normalize(x.data, p=2, dim=-1)

        # 加入微小噪声防止掉入另一盆地
        with torch.no_grad():
            x.data += noise_std * torch.randn_like(x.data)
            x.data = F.normalize(x.data, p=2, dim=-1)

    return x.detach()
```

---

## 4. 负梯度追踪到锚点

### 4.1 核心思路

从鞍点 `s` 出发，沿**负梯度方向** `-∇E(s)` 下降，直到收敛到某个盆地（锚点）。这等价于对鞍点执行一次 `infer()`：

```
鞍点 s ──梯度下降──→ 收敛点 c ──余弦最近邻──→ 锚点 a
```

但这里我们关心的是 **(来源鞍点区域, 目标锚点)** 的配对关系，而非收敛结果本身。

### 4.2 追踪算法

```python
def trace_to_anchor(
    landscape: FreqEnergyLandscape,
    zichang: HanziAnchorField,
    saddle_point: Tensor,     # (1024,)
    steps: int = 80,
    lr: float = 0.03,
    project_to_sphere: bool = True,
    return_trajectory: bool = True,
) -> Dict:
    """
    从鞍点沿负梯度追踪到最近锚点。

    与 infer() 的区别：
      - 追踪步数更多（80 vs 50），因为鞍点离盆地更远
      - 学习率稍高（0.03 vs 0.02），加速跨越山脊
      - 必须返回轨迹，用于分析鞍点→锚点的映射关系

    Returns:
        {
            'saddle': Tensor,           # 原始鞍点坐标
            'converged': Tensor,        # 收敛点坐标
            'anchor_char': str,         # 最近汉字字符
            'anchor_idx': int,          # 锚点索引
            'anchor_similarity': float, # 收敛点与锚点的余弦相似度
            'energy_start': float,      # 鞍点能量
            'energy_end': float,        # 收敛能量
            'energy_drop': float,       # 能量降幅
            'grad_norm_start': float,   # 鞍点梯度范数
            'trajectory': List[Tensor], # 下行轨迹
        }
    """
    device = next(landscape.parameters()).device
    x = saddle_point.clone().detach().to(device)
    if project_to_sphere:
        x = F.normalize(x, p=2, dim=-1)
    x.requires_grad_(True)

    # 记录起始信息
    with torch.no_grad():
        energy_start = landscape.energy(x).item()
    grad_start = _compute_grad_norm(landscape, x.detach().clone())

    optimizer = torch.optim.Adam([x], lr=lr)
    trajectory = [x.detach().cpu().clone()]

    prev_energy = energy_start
    no_improvement = 0

    for step in range(steps):
        optimizer.zero_grad()
        e = landscape.energy(x)
        e.backward()
        optimizer.step()

        if project_to_sphere:
            with torch.no_grad():
                x.data = F.normalize(x.data, p=2, dim=-1)

        trajectory.append(x.detach().cpu().clone())

        current_energy = e.item()
        delta = abs(current_energy - prev_energy)

        if current_energy >= prev_energy:
            no_improvement += 1
        else:
            no_improvement = 0

        if delta < 1e-5:
            break
        if no_improvement >= 5:
            break

        prev_energy = current_energy

    # 收敛后查找最近锚点
    converged = x.detach().cpu()
    with torch.no_grad():
        energy_end = landscape.energy(x).item()

    indices, chars, sims = zichang.find_nearest(converged, k=1)

    return {
        'saddle': saddle_point.cpu(),
        'converged': converged,
        'anchor_char': chars[0],
        'anchor_idx': indices[0].item(),
        'anchor_similarity': sims[0].item(),
        'energy_start': energy_start,
        'energy_end': energy_end,
        'energy_drop': energy_start - energy_end,
        'grad_norm_start': grad_start,
        'trajectory': trajectory,
    }


def _compute_grad_norm(landscape, x: Tensor) -> float:
    """计算某点的梯度范数（不修改原向量）"""
    x = x.clone().detach().requires_grad_(True)
    e = landscape.energy(x)
    e.backward()
    gn = x.grad.norm().item()
    landscape.zero_grad()
    return gn
```

### 4.3 批量追踪

对多个鞍点并行追踪以加速：

```python
def trace_to_anchors_batch(
    landscape: FreqEnergyLandscape,
    zichang: HanziAnchorField,
    saddle_points: Tensor,   # (K, 1024)
    steps: int = 80,
    lr: float = 0.03,
) -> List[Dict]:
    """
    批量追踪多个鞍点到各自最近锚点。

    由于每个鞍点可能收敛到完全不同的盆地，无法完全并行化梯度下降
    （每个样本需要独立的梯度路径）。但对于小批量（K ≤ 20），
    可以逐个处理，利用 GPU 加速前向/反向传播。
    """
    results = []
    for i in range(len(saddle_points)):
        # 使用 infer() 的内置信号功能，自动获取收敛质量和候选
        infer_result = landscape.infer(
            saddle_points[i],
            steps=steps,
            lr=lr,
            zichang=zichang,
            return_trajectory=True,
        )
        results.append({
            'saddle_idx': i,
            'saddle': saddle_points[i].cpu(),
            'converged': infer_result['state'],
            'anchor_char': infer_result['top_candidates'][0] if infer_result['top_candidates'] else '?',
            'anchor_similarity': infer_result['top_similarities'][0] if infer_result['top_similarities'] else 0.0,
            'energy_start': None,  # 需单独记录
            'energy_end': infer_result['energy'],
            'signal': infer_result['signal'],
            'gradient_norm': infer_result['gradient_norm'],
        })
    return results
```

### 4.4 鞍点→锚点 配对质量分数

```python
def score_pairing(trace_result: Dict) -> float:
    """
    计算 (鞍点, 锚点) 配对的质量分数。

    高质量配对的特征：
      - 能量降幅大（说明从"不知道"到"知道"的变化显著）
      - 鞍点梯度范数高（说明处于明确的决策边界）
      - 锚点相似度高（说明收敛明确，无歧义）
      - 收敛信号为 'certain' 或 'low_confidence'（而非 blind_spot）
    """
    energy_drop = trace_result.get('energy_drop', 0)
    grad_norm = trace_result.get('grad_norm_start', 0)
    anchor_sim = trace_result.get('anchor_similarity', 0)

    # 归一化各分量
    energy_score = min(energy_drop / 20.0, 1.0)   # 能量降幅≤20视为满分
    grad_score = min(grad_norm / 5.0, 1.0)         # 梯度范数≤5视为满分
    sim_score = anchor_sim                           # 相似度直接作为分数

    # 信号惩罚：如果鞍点收敛后仍是 blind_spot，说明该区域确实无知
    signal = trace_result.get('signal', 'certain')
    signal_penalty = {
        'certain': 1.0,
        'low_confidence': 0.8,
        'conflict': 0.5,
        'blind_spot': 0.1,
    }.get(signal, 0.5)

    # 综合分数
    quality = (0.3 * energy_score + 0.3 * grad_score + 0.4 * sim_score) * signal_penalty
    return quality
```

---

## 5. 已知性检测（概念图 + SQLite）

### 5.1 检测目标

在将 (来源鞍点, 目标锚点) 注入概念图之前，必须检查：

1. **直接关系已存在**：`(锚点字符, ANY_RELATION, 源概念) → 已存在`
2. **间接关系已存在**：通过传递闭包可推导 → 已存在
3. **相反关系已存在**：`(源概念, ANY_RELATION, 锚点字符) → 已存在`
4. **同义/别名关系**：锚点的别名已被关联 → 已存在

### 5.2 通过概念图检测

```python
def is_known_in_concept_graph(
    cg: ConceptGraph,
    source_concept: str,      # 来源概念（由鞍点附近区域表征）
    target_anchor: str,       # 目标锚点字符
) -> Tuple[bool, str]:
    """
    检查 (source_concept, *, target_anchor) 是否已存在于概念图。

    检测层级：
      1. 正向边: source_concept → target_anchor
      2. 反向边: target_anchor → source_concept
      3. 字符邻接: _char_adjacency[source_concept] ∩ {target_anchor}
      4. 传递闭包: 多跳 BFS（可选，较昂贵）

    Returns:
        (is_known, evidence)
          - is_known=True:  已存在，不需要注入
          - is_known=False: 不存在，可以注入
          - evidence: 描述已知关系的字符串
    """
    # 层级1: 正向索引
    if source_concept in cg.forward_index:
        if target_anchor in cg.forward_index[source_concept]:
            rel = cg.forward_index[source_concept][target_anchor]
            return True, f"正向边: {source_concept} -{rel}→ {target_anchor}"

    # 层级2: 反向索引
    if target_anchor in cg.forward_index:
        if source_concept in cg.forward_index[target_anchor]:
            rel = cg.forward_index[target_anchor][source_concept]
            return True, f"反向边: {target_anchor} -{rel}→ {source_concept}"

    # 层级3: 字符邻接索引（O(1) 快速检测）
    if hasattr(cg, '_char_adjacency'):
        if len(source_concept) == 1 and len(target_anchor) == 1:
            neighbors = cg._char_adjacency.get(source_concept, set())
            if target_anchor in neighbors:
                return True, f"字符邻接: {source_concept} ↔ {target_anchor}"

    # 层级4: 别名/规范化等效
    canonical_s = cg.canonical(source_concept)
    canonical_t = cg.canonical(target_anchor)
    if canonical_s != source_concept or canonical_t != target_anchor:
        if canonical_s in cg.forward_index and canonical_t in cg.forward_index[canonical_s]:
            return True, f"别名等效: {canonical_s} → {canonical_t}"

    return False, ""
```

### 5.3 通过 SQLite 快速检测

对于大规模概念图（数十万三元组），走 JSON 内存索引可能较慢。SQLite 提供 O(log N) 检测：

```python
def is_known_in_sqlite(
    db: ConceptGraphSQLite,
    source_concept: str,
    target_anchor: str,
) -> Tuple[bool, str]:
    """
    通过 SQLite 索引检测 (source, *, target) 是否存在。

    使用双 UNION 查询（参考 concept_graph_sqlite.py 的 query_char_pairs 设计）:
      1. SELECT WHERE s=? AND o=? → 正向
      2. SELECT WHERE s=? AND o=? → 反向（交换）
    """
    conn = db.conn

    # 正向: source → target
    rows = conn.execute(
        "SELECT r, c FROM triples WHERE s=? AND o=? LIMIT 1",
        (source_concept, target_anchor)
    ).fetchall()
    if rows:
        return True, f"SQLite正向: {source_concept} -{rows[0][0]}→ {target_anchor}"

    # 反向: target → source
    rows = conn.execute(
        "SELECT r, c FROM triples WHERE s=? AND o=? LIMIT 1",
        (target_anchor, source_concept)
    ).fetchall()
    if rows:
        return True, f"SQLite反向: {target_anchor} -{rows[0][0]}→ {source_concept}"

    return False, ""
```

### 5.4 统一检测入口

```python
def check_known(
    cg: ConceptGraph,
    db: Optional[ConceptGraphSQLite],
    source_concept: str,
    target_anchor: str,
    use_transitive: bool = False,
) -> Tuple[bool, str]:
    """
    统一检测入口：同步检测概念图内存 + SQLite。

    Args:
        cg: 概念图实例
        db: SQLite加速层（可选）
        source_concept: 来源概念（通常是单个汉字或多个字组成的词）
        target_anchor: 目标锚点字符
        use_transitive: 是否启用传递闭包检测（开销较大）

    Returns:
        (is_known, evidence_string)
    """
    # 1. 概念图内存检测（O(1) ~ O(d)）
    known, evidence = is_known_in_concept_graph(cg, source_concept, target_anchor)
    if known:
        return True, f"[CG] {evidence}"

    # 2. SQLite 检测（O(log N)）
    if db is not None:
        known, evidence = is_known_in_sqlite(db, source_concept, target_anchor)
        if known:
            return True, f"[SQLite] {evidence}"

    # 3. 传递闭包检测（O(V+E)，仅按需开启）
    if use_transitive:
        paths = cg.reason(
            source_concept,
            max_hops=3,
            direction="both",
            min_confidence=0.2,
        )
        for path, relations, conf in paths:
            if path[-1] == target_anchor:
                return True, f"[传递] {source_concept} → ... → {target_anchor} (hops={len(path)-1}, conf={conf:.2f})"

    return False, ""
```

---

## 6. 候选过滤标准

### 6.1 过滤管道

```
全部追踪结果 → [1.信号过滤] → [2.已知性过滤] → [3.质量过滤] → [4.多样性过滤] → 注入候选
```

### 6.2 逐层过滤

```python
@dataclass
class ReverseCandidate:
    """梯度反推候选配对"""
    saddle_point: Tensor          # 鞍点坐标 (1024,)
    saddle_energy: float          # 鞍点能量
    saddle_grad_norm: float       # 鞍点梯度范数
    anchor_char: str              # 目标锚点字符
    anchor_idx: int               # 锚点索引
    anchor_similarity: float      # 收敛点与锚点的余弦相似度
    energy_drop: float            # 能量降幅
    quality_score: float          # 综合质量分数
    is_novel: bool = True         # 是否为新发现
    concept_relation: str = ""    # 发现的关系类型（概念图上下文）
    source_concept: str = ""      # 来源概念名


class CandidateFilter:
    """候选过滤器"""

    def __init__(
        self,
        energy_drop_pctl: float = 75.0,        # 数据驱动: 能量降幅 P75
        min_anchor_similarity: float = 0.6,    # 最小锚点相似度
        min_quality_score: float = 0.3,        # 最小质量分数
        max_candidates_per_anchor: int = 5,    # 每锚点最大候选数
        max_total_candidates: int = 100,       # 总候选上限
        exclude_signals: Tuple[str, ...] = ('blind_spot',),
    ):
        self.energy_drop_pctl = energy_drop_pctl
        self.min_anchor_similarity = min_anchor_similarity
        self.min_quality_score = min_quality_score
        self.max_candidates_per_anchor = max_candidates_per_anchor
        self.max_total_candidates = max_total_candidates
        self.exclude_signals = exclude_signals

    def filter(
        self,
        candidates: List[ReverseCandidate],
        cg: ConceptGraph,
        db: Optional[ConceptGraphSQLite] = None,
    ) -> List[ReverseCandidate]:
        """执行完整过滤管道。使用数据驱动阈值。"""
        import numpy as np
        
        # ── 第1层: 信号过滤 ──
        candidates = [
            c for c in candidates
            if c.concept_relation not in self.exclude_signals
        ]
        if not candidates:
            return []

        # ── 第2层: 数据驱动硬指标过滤 ──
        drops = np.array([c.energy_drop for c in candidates])
        drop_threshold = float(np.percentile(drops, self.energy_drop_pctl))
        
        candidates = [
            c for c in candidates
            if (c.energy_drop >= drop_threshold
                and c.anchor_similarity >= self.min_anchor_similarity)
        ]

        # ── 第3层: 已知性过滤 ──
        novel_candidates = []
        for c in candidates:
            known, evidence = check_known(
                cg, db,
                source_concept=c.source_concept or c.anchor_char,
                target_anchor=c.anchor_char,
            )
            if known:
                c.is_novel = False
            else:
                c.is_novel = True
                novel_candidates.append(c)

        # ── 第4层: 质量排序 ──
        novel_candidates.sort(key=lambda c: c.quality_score, reverse=True)

        # ── 第5层: 多样性过滤（每锚点最多N个候选） ──
        anchor_counts: Dict[str, int] = {}
        diverse = []
        for c in novel_candidates:
            count = anchor_counts.get(c.anchor_char, 0)
            if count < self.max_candidates_per_anchor:
                diverse.append(c)
                anchor_counts[c.anchor_char] = count + 1
            if len(diverse) >= self.max_total_candidates:
                break

        return diverse
```

### 6.3 过滤标准汇总

| 过滤层 | 参数 | 默认值 | 说明 |
|--------|------|--------|------|
| 信号过滤 | `exclude_signals` | `('blind_spot',)` | 鞍点收敛到盲区说明该区域确实无知识 |
| 能量降幅 | `min_energy_drop` | 3.0 | 从鞍点到锚点的能量下降至少3.0 |
| 梯度范数 | `min_grad_norm` | 0.5 | 鞍点梯度范数至少0.5 |
| 锚点相似度 | `min_anchor_similarity` | 0.6 | 收敛点与锚点的余弦相似度至少0.6 |
| 质量分数 | `min_quality_score` | 0.3 | 综合质量分数阈值 |
| 锚点多样性 | `max_candidates_per_anchor` | 5 | 同一锚点最多关联5个不同来源 |
| 总量控制 | `max_total_candidates` | 100 | 单次运行最多产生100个注入候选 |

---

## 7. 概念图注入集成

### 7.1 注入策略

梯度反推引擎发现的候选配对需要注入概念图，方式有两种：

#### 策略A：直接三元组注入

```python
def inject_reverse_candidates(
    cg: ConceptGraph,
    db: Optional[ConceptGraphSQLite],
    candidates: List[ReverseCandidate],
    learner=None,          # 可选：同时更新能量景观
    landscape=None,         # 可选：能量景观引用
) -> Dict:
    """
    将梯度反推候选注入概念图。

    注入动作：
      1. 概念图添加三元组: (source_concept, RELATED, anchor_char)
      2. 同步写入 SQLite
      3. 可选：调用 learner 进行 Hebbian 强化
      4. 可选：转交 orchestrator._inject_knowledge()

    Returns:
        {'injected': int, 'skipped': int, 'details': [...]}
    """
    stats = {'injected': 0, 'skipped': 0, 'details': []}

    for c in candidates:
        if not c.is_novel:
            stats['skipped'] += 1
            continue

        source = c.source_concept or f"saddle_{c.anchor_char}"
        relation = c.concept_relation or "RELATED"
        confidence = min(0.7, c.quality_score * 0.8)  # 质量分 → 置信度

        # 1. 概念图注入
        triple = cg.add_triple(
            subject=source,
            relation=relation,
            obj=c.anchor_char,
            confidence=confidence,
            source="gradient_reverse",  # 标记来源
        )

        # 2. SQLite 同步（如果启用）
        if db is not None and triple is not None:
            try:
                db.conn.execute(
                    "INSERT OR REPLACE INTO triples(s, r, o, c, src, ev) VALUES(?,?,?,?,?,?)",
                    (source, relation, c.anchor_char, confidence, "gradient_reverse", "")
                )
                db.conn.commit()
            except Exception:
                pass

        if triple:
            stats['injected'] += 1
            stats['details'].append({
                'source': source,
                'anchor': c.anchor_char,
                'confidence': confidence,
                'quality': c.quality_score,
            })

    return stats
```

#### 策略B：通过能量景观 Hebbian 学习注入

```python
def inject_via_hebbian(
    landscape: FreqEnergyLandscape,
    zichang: HanziAnchorField,
    learner,                   # HebbianLearner 实例
    candidates: List[ReverseCandidate],
) -> Dict:
    """
    通过 Hebbian 学习将候选配对注入能量景观本身。

    这是更深层的注入：不是添加到概念图，而是直接修改
    能量景观的权重，使从鞍点区域到目标锚点的路径变得更"低能"。

    原理（与 orchestrator._inject_knowledge 一致）：
      Hebbian.update(saddle_point, anchor_vector, feedback=+0.5)
      → 降低 (saddle + anchor)/2 中点处的能量
      → 使该区域形成新的盆地

    Returns:
        {'hebbian_updates': int}
    """
    device = next(landscape.parameters()).device
    count = 0

    for c in candidates:
        if c.anchor_idx is None:
            continue

        anchor_vec = zichang.anchors[c.anchor_idx].to(device)
        saddle_vec = c.saddle_point.to(device)

        try:
            learner.hebbian.update(
                saddle_vec,
                anchor_vec,
                feedback=min(0.8, c.quality_score * 0.8),
            )
            count += 1
        except Exception as e:
            logger.warning(f"Hebbian注入失败 ({c.anchor_char}): {e}")

    return {'hebbian_updates': count}
```

### 7.2 注入时机

| 时机 | 触发方式 | 说明 |
|------|----------|------|
| **守护模式定期扫描** | `GradientReverseEngine.run()` 被 cron/守护进程调用 | 每N小时自动运行一次 |
| **空闲时主动探索** | 交互模式无查询时后台运行 | 利用系统空闲资源 |
| **训练后验证** | 能量景观 `fit()` 或 `learn_pairs_batch()` 后触发 | 学习后立即探测新形成的鞍部 |
| **手动触发** | API/CLI 命令 `reverse_explore` | 开发者手动触发 |

### 7.3 与 orchestrator 信号系统的协作

梯度反推引擎的发现可以作为**预注入**：

```
梯度反推发现 → 注入概念图 → 当用户后续查询相关概念时，
能量景观已包含此盆地 → infer() 直接返回 'certain' → 无需双臂搜索
```

即：梯度反推引擎**将被动信号处理转化为主动预学习**。

---

## 8. 实现架构与数据结构

### 8.1 模块位置

```
loongpearl/core/gradient_reverse_engine.py   ← 新建
```

### 8.2 核心类设计

```python
"""
梯度反推引擎 — 主动发现能量景观中的未知概念关联

loongpearl/core/gradient_reverse_engine.py
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ReverseCandidate:
    """梯度反推候选配对"""
    saddle_point: torch.Tensor          # 鞍点坐标 (1024,)
    saddle_energy: float                # 鞍点能量
    saddle_grad_norm: float             # 鞍点梯度范数
    anchor_char: str                    # 目标锚点字符
    anchor_idx: int                     # 锚点在字场中的索引
    anchor_similarity: float            # 收敛点与锚点余弦相似度
    energy_drop: float                  # 能量降幅
    quality_score: float                # 综合质量分数 [0, 1]
    signal: str = 'certain'             # 收敛信号类型
    is_novel: bool = True               # 是否为新发现
    source_concept: str = ""            # 来源概念名
    concept_relation: str = "RELATED"   # 拟注入关系类型


class GradientReverseEngine:
    """
    梯度反推引擎 — 主动探索能量景观的未知区域。

    核心流程:
      1. sample_sphere()         → 球面采样候选点
      2. find_saddle_points()    → 筛选高能量+高梯度范数鞍点
      3. trace_to_anchors()      → 负梯度追踪到最近锚点
      4. check_novelty()         → 概念图/SQLite 已知性检测
      5. filter_candidates()     → 多层过滤排序
      6. inject()                → 注入概念图 + Hebbian 学习

    使用示例:
        engine = GradientReverseEngine(landscape, zichang, cg, learner)
        engine.run(n_samples=20000, top_k=50)
    """

    def __init__(
        self,
        landscape: 'FreqEnergyLandscape',
        zichang: 'HanziAnchorField',
        concept_graph: 'ConceptGraph',
        learner=None,
        sqlite_db: Optional['ConceptGraphSQLite'] = None,
        config: Optional[Dict] = None,
    ):
        self.landscape = landscape
        self.zichang = zichang
        self.cg = concept_graph
        self.learner = learner
        self.db = sqlite_db

        # 默认配置
        self.config = {
            # 采样
            'n_samples': 20000,
            'sample_batch_size': 512,

            # 鞍点检测（数据驱动百分位 — 不再硬编码）
            'energy_pctl': 75.0,      # 取能量最高的 P75 作为鞍点候选
            'grad_pctl': 90.0,        # 取梯度范数最高的 P90
            'max_saddle_points': 200,

            # 追踪
            'trace_steps': 80,
            'trace_lr': 0.03,
            'trace_batch_size': 20,

            # 过滤（数据驱动 — 基于实际追踪结果分布）
            'energy_drop_pctl': 75.0,       # 取能量降幅最大的 P75
            'min_anchor_similarity': 0.6,
            'min_quality_score': 0.3,
            'max_candidates_per_anchor': 5,
            'max_total_candidates': 100,

            # 注入
            'inject_to_cg': True,
            'inject_hebbian': False,    # 默认不触发 Hebbian（需显式开启）
            'default_confidence': 0.5,
            'default_relation': 'RELATED',
        }
        if config:
            self.config.update(config)

        # 运行统计
        self.stats: Dict = {}

    def run(self, **overrides) -> Dict:
        """
        执行完整的梯度反推-注入管道。

        Args:
            **overrides: 覆盖默认配置的参数

        Returns:
            {
                'saddle_points_found': int,
                'traced_to_anchors': int,
                'novel_candidates': int,
                'injected': int,
                'candidates': List[ReverseCandidate],
                'stats': Dict,
            }
        """
        config = {**self.config, **overrides}

        # ── 阶段1: 鞍点搜索 ──
        logger.info("[梯度反推] 阶段1: 球面采样 + 鞍点搜索")
        saddle_points, saddle_energies, saddle_grads = self._find_saddle_points(config)

        if len(saddle_points) == 0:
            logger.warning("[梯度反推] 未找到任何鞍点，终止")
            return {'saddle_points_found': 0, 'candidates': [], 'injected': 0}

        logger.info(f"[梯度反推] 发现 {len(saddle_points)} 个鞍点候选")

        # ── 阶段2: 追踪到锚点 ──
        logger.info("[梯度反推] 阶段2: 负梯度追踪 → 最近锚点")
        candidates = self._trace_to_anchors(saddle_points, saddle_energies, saddle_grads, config)

        logger.info(f"[梯度反推] 追踪完成，{len(candidates)} 个配对")

        # ── 阶段3: 已知性检测 + 过滤 ──
        logger.info("[梯度反推] 阶段3: 已知性检测 + 多层过滤")
        novel = self._filter_and_check_novelty(candidates, config)

        logger.info(f"[梯度反推] 过滤后 {len(novel)} 个新候选")

        # ── 阶段4: 注入 ──
        injected = 0
        if novel and config.get('inject_to_cg', True):
            logger.info("[梯度反推] 阶段4: 概念图注入")
            inject_stats = self._inject_candidates(novel, config)
            injected = inject_stats.get('injected', 0)

        # 更新统计
        self.stats = {
            'saddle_points_found': len(saddle_points),
            'traced_to_anchors': len(candidates),
            'novel_candidates': len(novel),
            'injected': injected,
            'timestamp': __import__('time').time(),
        }

        return {
            'saddle_points_found': len(saddle_points),
            'traced_to_anchors': len(candidates),
            'novel_candidates': len(novel),
            'injected': injected,
            'candidates': novel,
            'stats': self.stats,
        }

    # ── 内部方法（占位，完整实现见前文各算法） ──

    def _find_saddle_points(self, config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """球面采样 + 鞍点筛选（见第3节）"""
        samples = self._sample_sphere(config['n_samples'])
        energies, grad_norms = self._evaluate_batch(
            samples, batch_size=config['sample_batch_size']
        )
        mask = (energies > config['energy_threshold']) & (grad_norms > config['grad_threshold'])
        candidates = samples[mask]
        cand_e = energies[mask]
        cand_g = grad_norms[mask]
        quality = cand_e * cand_g
        _, idx = torch.topk(quality, min(config['max_saddle_points'], len(quality)))
        return candidates[idx], cand_e[idx], cand_g[idx]

    def _sample_sphere(self, n: int) -> torch.Tensor:
        """单位球面均匀采样"""
        points = torch.randn(n, self.zichang.embed_dim)
        return F.normalize(points, p=2, dim=1)

    def _evaluate_batch(self, points, batch_size=512):
        """批量评估能量+梯度范数"""
        # 实现见第3.2节 evaluate_landscape_batch_fast()
        pass

    def _trace_to_anchors(self, saddles, energies, grads, config):
        """批量追踪（见第4节）"""
        # 实现见第4.2节 trace_to_anchor()
        pass

    def _filter_and_check_novelty(self, candidates, config):
        """过滤 + 已知性检测（见第5-6节）"""
        # 实现见第6.2节 CandidateFilter
        pass

    def _inject_candidates(self, novel, config):
        """注入概念图（见第7节）"""
        # 实现见第7.1节
        pass
```

### 8.3 与现有模块的集成点

```
┌──────────────────────────────────────────────────────┐
│                  Orchestrator                         │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐                  │
│  │  query()    │    │ _handle_     │                  │
│  │  五步管道    │◄───│  signal()    │                  │
│  └─────────────┘    └──────┬───────┘                  │
│                            │                          │
│          ┌─────────────────┼──────────────────┐       │
│          │                 │                  │       │
│   ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐ │
│   │ 双臂搜索     │  │ 身体裁决    │  │ 双脚验证     │ │
│   │ blind_spot  │  │ conflict    │  │ low_conf     │ │
│   └─────────────┘  └─────────────┘  └──────────────┘ │
│                                                       │
│   ┌─────────────────────────────────────────────┐     │
│   │  ★ GradientReverseEngine (新增)              │     │
│   │   - 守护模式定期扫描                         │     │
│   │   - 空闲时主动探索                           │     │
│   │   - 训练后验证                               │     │
│   │  输出 → 注入 cg.add_triple()                 │     │
│   │       → Hebbian learner.hebbian.update()     │     │
│   └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## 9. 使用流程

### 9.1 守护模式集成（推荐）

```python
# 在 loong_main.py 的守护循环中添加

from loongpearl.core.gradient_reverse_engine import GradientReverseEngine

def daemon_loop(orchestrator):
    """守护模式主循环"""
    engine = GradientReverseEngine(
        landscape=orchestrator.landscape,
        zichang=orchestrator.field,
        concept_graph=orchestrator.cg,
        learner=orchestrator.learner,
        sqlite_db=getattr(orchestrator, '_cgdb', None),
    )

    while True:
        # 1. 处理待决队列（现有逻辑）
        process_pending_queries(orchestrator)

        # 2. 调整学习率等（现有逻辑）
        adjust_learning(orchestrator)

        # ★ 3. 梯度反推扫描（新增）
        if should_run_reverse_scan():
            logger.info("🔍 [守护] 启动梯度反推扫描...")
            result = engine.run(n_samples=10000, top_k=50)
            logger.info(f"🔍 [守护] 扫描完成: 发现{result['saddle_points_found']}鞍点, "
                       f"注入{result['injected']}条新关联")

        # 4. 常规自评估（现有逻辑）
        run_self_evaluation(orchestrator)

        # 5. 持久化（现有逻辑）
        save_progress(orchestrator)

        time.sleep(SCAN_INTERVAL)  # 如 3600 秒
```

### 9.2 手动触发

```python
# CLI 或 API 调用
engine = GradientReverseEngine(landscape, zichang, cg, learner)
result = engine.run(
    n_samples=50000,     # 更密集的扫描
    top_k=200,           # 更多候选
    inject_to_cg=True,   # 直接注入
)
print(f"发现 {result['injected']} 条新知识关联")
for c in result['candidates'][:10]:
    print(f"  {c.source_concept} → {c.anchor_char} "
          f"(质量={c.quality_score:.2f}, 能量降={c.energy_drop:.1f})")
```

### 9.3 训练后触发

```python
# 在 learner.learn_pairs_batch() 之后
def on_training_complete(landscape, zichang, cg, learner):
    """能量景观更新后，立即探测新形成的知识边界"""
    engine = GradientReverseEngine(landscape, zichang, cg, learner)
    # 训练后能量景观已变化 → 可能出现新的鞍部
    result = engine.run(
        n_samples=5000,    # 快速扫描
        top_k=20,
        inject_hebbian=False,  # 仅探测，不注入
    )
    logger.info(f"[训练后扫描] 发现 {len(result['candidates'])} 个潜在新关联")
    return result['candidates']
```

---

## 10. 性能估算与调优参数

### 10.1 时间复杂度

| 阶段 | 操作 | 时间复杂度 | 估算耗时 (CPU) |
|------|------|-----------|---------------|
| 球面采样 | `torch.randn(20000, 1024)` | O(N·D) | < 1s |
| 批量评估 | 前向+反向传播，batch_size=512 | O(N·D·L) | ~30s |
| 鞍点追踪 | 每个鞍点 80 步梯度下降 | O(K·T·L) | ~20s (K=100) |
| 已知性检测 | 概念图 O(1) + SQLite O(log N) | O(K) | < 1s |
| 注入 | 概念图 add_triple × K | O(K) | < 1s |
| **总计** | | | **~60s** |

> 注：L = 网络层数 (~6), D = 嵌入维度 (1024), N = 采样数, K = 鞍点数, T = 追踪步数

### 10.2 内存估算

| 数据结构 | 大小 |
|----------|------|
| 球面采样点 (20000 × 1024 × 4B) | ~82 MB |
| 能量值 (20000 × 4B) | ~80 KB |
| 梯度范数 (20000 × 4B) | ~80 KB |
| 鞍点 (200 × 1024 × 4B) | ~0.8 MB |
| 追踪结果 (200 × ~100KB) | ~20 MB |
| **峰值** | **~100 MB** |

### 10.3 调优参数速查

```python
# 探索性扫描（守护模式，广度优先）
CONFIG_EXPLORE = {
    'n_samples': 10000,
    'energy_pctl': 75.0,       # 数据驱动
    'grad_pctl': 90.0,         # 数据驱动
    'max_saddle_points': 100,
    'min_quality_score': 0.3,
    'max_total_candidates': 50,
}

# 深度扫描（手动触发，质量优先）
CONFIG_DEEP = {
    'n_samples': 50000,
    'energy_pctl': 85.0,       # 更严格的能量阈值
    'grad_pctl': 95.0,         # 更严格的梯度阈值
    'max_saddle_points': 500,
    'min_quality_score': 0.5,
    'max_total_candidates': 200,
}

# 快速扫描（训练后验证）
CONFIG_QUICK = {
    'n_samples': 5000,
    'energy_pctl': 70.0,
    'grad_pctl': 85.0,
    'max_saddle_points': 50,
    'min_quality_score': 0.2,
    'max_total_candidates': 20,
}
```

---

## 附录A：鞍点搜索的梯度计算优化详解

```python
def evaluate_landscape_batch_optimized(
    landscape: FreqEnergyLandscape,
    points: Tensor,          # (N, D)
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor]:
    """
    优化版批量评估：核心技巧。

    关键：`e.sum().backward()` 一次性对 batch 中所有样本计算梯度。
    因为 sum() 的梯度是每个分量独立传播的，等价于逐样本 backward。
    这比逐样本循环快 10-100 倍。

    注意：
      - 需要在 backward 前将模型设为 train() 模式
      - backward 后需要 zero_grad() 清理
      - gradient norm 用 dim=1 批量计算
    """
    was_training = landscape.training
    landscape.train()  # 确保梯度可计算
    device = next(landscape.parameters()).device

    all_energies = []
    all_grad_norms = []

    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size].to(device).detach()
        batch.requires_grad_(True)

        e = landscape.energy(batch)          # (B,)
        e_sum = e.sum()                       # 标量
        e_sum.backward()

        gn = batch.grad.norm(dim=1).detach().cpu()  # (B,)
        en = e.detach().cpu()

        all_energies.append(en)
        all_grad_norms.append(gn)

        landscape.zero_grad()

    landscape.train(was_training)
    return torch.cat(all_energies), torch.cat(all_grad_norms)
```

## 附录B：鞍点来源概念命名策略

鞍点本身没有语义（纯几何点），需要将其关联到可读的概念名：

```python
def name_saddle_source(
    saddle_point: Tensor,
    zichang: HanziAnchorField,
    cg: ConceptGraph,
    top_k: int = 5,
) -> str:
    """
    为鞍点命名：找到周围最近的几个锚点，组合成概念名。

    策略：
      1. 找鞍点最近的前 top_k 个锚点
      2. 如果某两个字在概念图中已有关联 → 用已知概念名
      3. 否则用最近两个字的拼接
    """
    _, chars, sims = zichang.find_nearest(saddle_point, k=top_k)

    # 检查概念图中是否存在由这些字组成的已知概念
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            candidate = chars[i] + chars[j]
            if candidate in cg.nodes:
                return candidate
            candidate = chars[j] + chars[i]
            if candidate in cg.nodes:
                return candidate

    # 回退: 最近两个字拼接
    return chars[0] + chars[1] if len(chars) >= 2 else chars[0]
```

---

> **文档结束** — 梯度反推引擎（Gradient Reverse Prediction Engine）设计 v1.0
