# 对抗扰动引擎（Perturbation Engine）设计文档

> **版本**: v1.0  
> **日期**: 2026-06-18  
> **状态**: 设计阶段  
> **目标**: 在 daemon_tick_v2 学习循环中插入自对抗鲁棒性检测，防止能量景观过度泛化产生虚假盆地

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [精确插入点](#2-精确插入点)
3. [核心算法: perturb_and_observe()](#3-核心算法-perturb_and_observe)
4. [远距字对采样策略](#4-远距字对采样策略)
5. [能量偏移测量](#5-能量偏移测量)
6. [候选过滤与 D-S 验证提交](#6-候选过滤与-d-s-验证提交)
7. [代码量与性能评估](#7-代码量与性能评估)
8. [实现计划与风险](#8-实现计划与风险)

---

## 1. 背景与动机

### 1.1 当前学习循环的问题

`daemon_tick_v2` 的当前流程（orchestrator.py:1458）：

```
0. 优先插队 → 1. 大脑扫描 → 2. 双臂搜索 + learn_pairs_batch → 3. decay_step
```

其中 `learn_pairs_batch`（learner.py:856）使用**对比学习**：
- **正样本**（已知字对）：降低中点能量 → 形成盆地
- **负样本**（随机字对）：保持/升高能量 → 维持势垒

训练目标：
```python
margin = 3.0        # 已知必须比随机至少低 3.0
target_low = -15.0  # 已知目标能级
loss = rank_loss + 0.3 * push_loss
```

随后 batch Hebbian（≤200 对时）在插值路径上进一步压低能量。

### 1.2 潜在风险：过度泛化

对比学习的负样本是**全局随机采样**的，存在两个盲区：

| 风险 | 描述 |
|------|------|
| **虚假盆地** | 某些语义无关的字对，其中点恰好落在被拉低的能量区域，形成不应存在的低能盆地 |
| **脆性记忆** | 景观对某些特定方向过度敏感，轻微参数扰动即导致能量剧烈变化，表明过拟合 |
| **负样本盲区** | 全局随机负样本难以覆盖所有"应该不相关"的字对组合，特别是近义词但语义冲突的字对 |

### 1.3 对抗扰动引擎的设计哲学

> **"以扰动测脆性，以脆性定位虚假关联"**

类比神经网络对抗训练中的 FGSM（Fast Gradient Sign Method）：
- 向景观参数注入受控噪声（对抗扰动）
- 测量"远距字对"（应不相关的锚点对）在扰动前后的能量变化
- 能量异常下降 → 该区域存在脆性虚假盆地 → 标记为可疑候选
- 提交 D-S 证据理论验证器进行裁决，必要时执行负 Hebbian 修正

---

## 2. 精确插入点

### 2.1 在 daemon_tick_v2 中的位置

扰动引擎插入在 **步骤 2（双臂搜索 + 学习注入）之后、步骤 3（衰减）之前**：

```
daemon_tick_v2 流程（修改后）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0. 优先插队          process_pending_queries()
1. 大脑扫描          _daemon_scan()
2. 双臂搜索+学习     _arms_search_batch() → learn_pairs_batch() → ewc_regularize()
   ★ 2.5. 对抗扰动   _perturb_and_observe()   ← 【新插入】
3. 定期调度           decay_step()
   每5轮:  序列臂 + 矛盾解 + D-S回写 + 剪枝对齐
   每10轮: 闭环验证 + 策应器规划
   每20轮: 万象格 + 金字塔
   每50轮: EWC Fisher 更新
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2.2 具体的代码插入位置

文件: `loongpearl/core/orchestrator.py`  
方法: `daemon_tick_v2`（第 1458 行）

**插入位置**: 在第 1515 行（EWC regularize 的 except 块结束）之后，第 1519 行（`# ── 3. 定期调度 ──`）之前：

```python
# 现有代码 (line 1511-1517):
                    # ★ EWC 正则: 拉回锚定参数
                    if hasattr(self.learner, 'ewc_regularize'):
                        try:
                            self.learner.ewc_regularize()
                        except Exception:
                            pass
            except Exception as e:
                log.warning(f"  双臂搜索/注入异常: {e}")

        # ★★★ 对抗扰动引擎插入点 ★★★
        # 每 5 轮执行一次扰动鲁棒性检测
        if round_num % 5 == 0:
            try:
                perturb_report = self._perturb_and_observe()
                tick_report['perturbation'] = perturb_report
            except Exception as e:
                log.debug(f"  对抗扰动异常: {e}")

        # ── 3. 定期调度 ──
        # 每轮: 衰减
        try:
```

### 2.3 为什么选这个位置？

| 优势 | 说明 |
|------|------|
| **学习后立即检测** | learn_pairs_batch 刚修改了景观参数，在衰减抹平之前检测新鲜引入的脆性 |
| **EWC 之后** | EWC 正则已拉回锚定参数，此时检测更能反映"净变化" |
| **衰减之前** | 衰减会微弱改变参数范数，混淆扰动测量的归因 |
| **每 5 轮** | 与序列臂、矛盾解同频，计算开销可控；每轮扰动过于频繁 |

---

## 3. 核心算法: perturb_and_observe()

### 3.1 算法伪代码

```
算法: perturb_and_observe()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:
    landscape:        FreqEnergyLandscape   (6层MLP, 1024→1)
    anchors:          HanziAnchorField      (94117字 × 1024维)
    learner:          DragonBallLearner
    orchestrator:     Orchestrator          (用于D-S裁决)
    
超参数:
    N_DISTANT:        200      # 远距字对数量
    SIMILARITY_PCTL:  10.0     # 相似度百分位阈值 (取最低10%)
    PERTURB_STD:      0.01     # 扰动噪声标准差 (相对参数范数)
    ENERGY_PCTL:      10.0     # 能量百分位阈值 (取最低P10，≈-165)
    ENERGY_DROP_PCTL: 5.0      # 能量下降百分位阈值 (取ΔE的P5)
    MAX_CANDIDATES:   20       # 最多提交D-S的候选数
    MIN_DS_CONF:      0.2      # D-S裁决阈值

注意: 阈值全部使用数据驱动百分位，运行时根据实际能量/ΔE分布动态计算，
    而非硬编码绝对值。模型训练会改变能量分布，百分位自适应跟随。

输出:
    report: {
        'distant_pairs_sampled': int,
        'candidates_flagged': int,
        'd_s_verified': int,
        'corrected': int,
        'avg_energy_shift': float,
        'fragility_score': float,     # 全局脆性评分
    }

步骤:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SAMPLE_DISTANT_PAIRS:
   ├── 随机抽取 2000 个锚点索引 (用于大范围扫描)
   ├── 计算锚点子集内部的余弦相似度矩阵 (2000×2000)
   ├── 取相似度最低的 200 对 → distant_pairs
   └── 计算每对的中点向量: mid = normalize((anchor_a + anchor_b) / 2)

2. BASELINE_ENERGY:
   ├── with torch.no_grad():
   │   └── E_before = landscape.energy(midpoints)  # (N_DISTANT,)
   └── 记录每个中点的基础能量

3. PARAMETER_PERTURBATION:
   ├── 保存当前参数副本: params_backup = {name: p.clone()}
   ├── for name, param in landscape.named_parameters():
   │   ├── noise_scale = PERTURB_STD * param.norm() / sqrt(param.numel())
   │   ├── noise = torch.randn_like(param) * noise_scale
   │   └── param.data.add_(noise)          # 注入高斯噪声
   └── 扰动完成

4. POST_PERTURB_ENERGY:
   ├── with torch.no_grad():
   │   └── E_after = landscape.energy(midpoints)  # (N_DISTANT,)
   └── 计算能量偏移: ΔE = E_after - E_before  # 负值=能量下降

5. RESTORE_PARAMETERS:
   ├── for name, param in landscape.named_parameters():
   │   └── param.data.copy_(params_backup[name])
   └── 恢复原始参数 (扰动仅为检测，不实际修改景观)

6. FILTER_CANDIDATES (数据驱动阈值):
   ├── 计算数据驱动的阈值:
   │   ├── energy_low_thresh = E_before 的 P[ENERGY_PCTL] (运行时动态计算)
   │   │   例: P10 ≈ -165 (10%的远距对中点能量低于此值)
   │   └── energy_drop_thresh = ΔE 的 P[ENERGY_DROP_PCTL] (仅负值方向)
   │       例: P5 ≈ 扰动后自然波动的下界
   ├── 条件A (能量异常低): E_before[i] < energy_low_thresh
   │   含义: 远距字对在中点已有异常低能盆地 → 疑似虚假关联
   ├── 条件B (脆性下降):  ΔE[i] < energy_drop_thresh
   │   含义: 轻微扰动即导致异常大幅能量下降 → 参数脆性区域
   ├── 综合评分: score[i] = -E_before[i] - ΔE[i] * 2.0
   │   (能量越低、下降越多 → 分数越高 → 越可疑)
   ├── 快速退出: if 无候选 → return {'candidates': 0, 'fragility_score': ...}
   ├── 按综合评分降序排列，取前 MAX_CANDIDATES
   └── flagged_candidates = [(ia, ib, score, E_before, ΔE), ...]

7. D_S_VERIFY_AND_CORRECT:
   ├── for each flagged candidate:
   │   ├── char_a, char_b = anchors.hanzi_list[ia], anchors.hanzi_list[ib]
   │   ├── 查询 D-S 模糊格: 这对字是否有证据支持关联?
   │   │   ├── 概念图邻接查询 (SQLite / 概念图)
   │   │   └── 模糊格信念质量计算
   │   ├── if D-S 置信度 < 阈值:
   │   │   └── 确认为虚假关联 → 执行负 Hebbian 修正
   │   │       learner.unlearn_chars(char_a, char_b, strength=0.3)
   │   │       (在路径中点抬高能量，削弱虚假盆地)
   │   └── else:
   │       └── D-S 确认存在证据 → 保留 (可能是合法但罕见的知识)
   └── 统计修正数量

8. COMPUTE_FRAGILITY_SCORE:
   ├── fragility = mean(|ΔE|) / std(|ΔE|)  归一化脆性
   ├── 高脆性 → 景观可能过拟合，建议增加正则化
   └── 低脆性 → 景观鲁棒性好

9. RETURN_REPORT
```

### 3.2 关键设计决策解释

| 决策 | 理由 |
|------|------|
| **扰动后恢复参数** | 扰动仅用于检测，不实际修改景观。实际修正通过负 Hebbian 精确执行 |
| **噪声缩放因子** | `PERTURB_STD * param.norm() / sqrt(param.numel())` — 保证每参数的扰动幅度与其自身范数成比例，避免对小参数过度扰动 |
| **综合评分公式** | `-E_before[i] - ΔE[i] * 2.0` — 兼顾"已经是低能"和"扰动后进一步下降"两种脆性来源 |
| **每 5 轮执行** | 与序列臂、矛盾解同步，避免每轮开销过大 |

---

## 4. 远距字对采样策略

### 4.1 采样目标

从 94117 个汉字锚点中找出**语义距离远、不应有关联**的字对。这些字对的中点应处于高能区，如果检测到低能，说明景观可能被错误泛化。

### 4.2 采样算法

```python
def _sample_distant_pairs(
    anchors: torch.Tensor,        # (94117, 1024)
    n_sample: int = 2000,         # 随机子集大小
    n_pairs: int = 200,           # 最终远距对数量
    similarity_percentile: float = 10.0,  # 取最低相似度的百分位
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样远距（低相似度）锚点对。
    
    Returns:
        pair_indices:  (n_pairs, 2)  锚点索引对
        midpoints:     (n_pairs, 1024) 中点向量
        similarities:  (n_pairs,)     余弦相似度
    """
    n_total = anchors.shape[0]
    
    # Step 1: 随机采样子集 (避免全矩阵 O(N²))
    subset_idx = torch.randperm(n_total)[:n_sample]
    subset = anchors[subset_idx]  # (2000, 1024)
    
    # Step 2: 计算子集内余弦相似度矩阵 (2000×2000 = 4M entries)
    subset_norm = F.normalize(subset, p=2, dim=1)
    sim_matrix = subset_norm @ subset_norm.T  # (2000, 2000)
    
    # Step 3: 取上三角 (排除自相似和对角)
    triu_idx = torch.triu_indices(n_sample, n_sample, offset=1)
    triu_sims = sim_matrix[triu_idx[0], triu_idx[1]]
    
    # Step 4: 按相似度排序，取最低的 n_pairs
    threshold = torch.quantile(triu_sims, similarity_percentile / 100.0)
    distant_mask = triu_sims <= threshold
    distant_sims = triu_sims[distant_mask]
    distant_pairs = triu_idx[:, distant_mask]  # (2, K)
    
    # Step 5: 随机采样 n_pairs (如果 K > n_pairs)
    if distant_pairs.shape[1] > n_pairs:
        sample_idx = torch.randperm(distant_pairs.shape[1])[:n_pairs]
        distant_pairs = distant_pairs[:, sample_idx]
        distant_sims = distant_sims[sample_idx]
    
    # Step 6: 映射回全局索引 + 计算中点
    global_a = subset_idx[distant_pairs[0]]  # (n_pairs,)
    global_b = subset_idx[distant_pairs[1]]
    pair_indices = torch.stack([global_a, global_b], dim=1)
    
    midpoints = F.normalize(
        (anchors[global_a] + anchors[global_b]) / 2,
        p=2, dim=1
    )
    
    return pair_indices, midpoints, distant_sims
```

### 4.3 性能分析

| 步骤 | 计算量 | 说明 |
|------|--------|------|
| 子集采样 2000 锚点 | O(2000) | 极轻量 |
| 相似度矩阵 2000×2000 | O(2000² × 1024) ≈ 4G FLOPs | 矩阵乘法，GPU 上 ~5ms |
| 取上三角 + 排序 | O(2M log 2M) | 可忽略 |
| **总计** | **~5-10ms (GPU)** | 完全可接受 |

### 4.4 替代采样策略（可选增强）

```python
# 策略B: 基于概念图反例采样
# 对每对锚点,查询概念图/模糊格中是否存在反证据
# 如 "火-冰" 有大量关系证据则排除, "火-凳" 无证据则保留

# 策略C: 基于能量景观自身
# 对随机中点采样,取当前能量最低的那些中点对应的锚点对
# 这些是"已经形成了不应存在的盆地"的候选
```

---

## 5. 能量偏移测量

### 5.1 扰动机制

```python
def _inject_perturbation(
    landscape: nn.Module,
    perturbation_std: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    向景观参数注入高斯扰动噪声。
    
    噪声幅度 = perturbation_std * ||param|| / sqrt(numel)
    这个缩放确保每个参数的噪声相对幅度一致。
    
    Returns:
        param_backup: 扰动前的参数副本（用于恢复）
    """
    param_backup = {}
    
    for name, param in landscape.named_parameters():
        if not param.requires_grad:
            continue
        
        # 保存副本
        param_backup[name] = param.data.clone()
        
        # 计算噪声幅度
        param_norm = param.data.norm().item()
        noise_scale = perturbation_std * param_norm / (param.numel() ** 0.5 + 1e-8)
        
        # 注入高斯噪声
        noise = torch.randn_like(param.data) * noise_scale
        param.data.add_(noise)
    
    return param_backup


def _restore_parameters(
    landscape: nn.Module,
    param_backup: Dict[str, torch.Tensor],
) -> None:
    """恢复景观参数到扰动前的状态"""
    for name, param in landscape.named_parameters():
        if name in param_backup:
            param.data.copy_(param_backup[name])
```

### 5.2 能量测量

```python
def _measure_energy_shifts(
    landscape: FreqEnergyLandscape,
    midpoints: torch.Tensor,        # (N, 1024)
    perturbation_std: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    测量扰动前后的能量偏移。
    
    Returns:
        E_before:  扰动前能量 (N,)
        E_after:   扰动后能量 (N,)
        delta_E:   能量偏移 = E_after - E_before (N,)
    """
    device = next(landscape.parameters()).device
    midpoints = midpoints.to(device)
    
    # 1. 基准能量
    landscape.eval()
    with torch.no_grad():
        E_before = landscape.energy(midpoints).detach().cpu()
    
    # 2. 注入扰动
    backup = _inject_perturbation(landscape, perturbation_std)
    
    # 3. 扰动后能量
    with torch.no_grad():
        E_after = landscape.energy(midpoints).detach().cpu()
    
    # 4. 恢复参数
    _restore_parameters(landscape, backup)
    
    delta_E = E_after - E_before
    
    return E_before, E_after, delta_E
```

### 5.3 能量偏移解读

| ΔE 符号 | 含义 | 解读 |
|---------|------|------|
| ΔE ≈ 0 | 稳定 | 该中点周围景观平坦、鲁棒 → 无虚假盆地 |
| ΔE > 0 | 能量上升 | 扰动破坏了原盆地的参数配置 → 可能是合法盆地 |
| ΔE < 0 (小) | 轻微下降 | 正常扰动响应 |
| **ΔE < -2.0 (大)** | **大幅下降** | ⚠️ 脆性信号 → 该区域对参数极其敏感，可能是虚假盆地 |

### 5.4 批量计算优化

利用景观的批处理能力，200 个中点的能量计算可以一次完成：

```python
# landscape.energy() 天然支持批量输入 (batch_size, 1024) → (batch_size,)
E_before = landscape.energy(midpoints)  # 一次前向传播，200 个点
# 计算成本: 200个样本的批量推理 ≈ 1个样本的推理 × 1.2 (GPU批处理效率)
```

---

## 6. 候选过滤与 D-S 验证提交

### 6.1 过滤流水线

```
远距字对 (200个)
    │
    ├── 基准能量测量 E_before
    ├── 扰动后能量测量 E_after
    ├── 计算 ΔE = E_after - E_before
    │
    ▼
过滤阶段
    ├── 条件A: E_before < -5.0       → "已有虚假盆地"
    ├── 条件B: ΔE < -2.0            → "脆性区域"
    ├── 综合评分: score = -E_before - ΔE * 2.0
    └── Top-K 按 score 降序 (K ≤ 20)
    │
    ▼
D-S 验证阶段
    ├── 概念图查询: 该字对是否有关系？
    ├── 模糊格 D-S 融合: 综合多源证据
    └── 裁决: 确认真实关联 vs 虚假关联
    │
    ▼
修正阶段
    ├── D-S 确认虚假 → unlearn_chars() [负Hebbian修正]
    └── D-S 确认真实 → 保留 (可能是罕见但合法的知识)
```

### 6.2 过滤实现

```python
def _filter_candidates(
    E_before: torch.Tensor,
    delta_E: torch.Tensor,
    pair_indices: torch.Tensor,
    energy_pctl: float = 10.0,
    energy_drop_pctl: float = 5.0,
    max_candidates: int = 20,
) -> Tuple[List, float, float]:
    """
    过滤可疑的远距字对。使用数据驱动百分位阈值。
    
    Returns:
        候选列表: [(ia, ib, score, E_before, delta_E), ...]
        energy_thresh: 实际使用的能量阈值
        drop_thresh: 实际使用的下降阈值
    """
    import numpy as np
    
    # 数据驱动阈值计算
    eb_np = E_before.numpy()
    de_np = delta_E.numpy()
    energy_thresh = float(np.percentile(eb_np, energy_pctl))
    drop_thresh = float(np.percentile(de_np, energy_drop_pctl))
    
    # 快速退出
    candidates = []
    for i in range(len(E_before)):
        eb = E_before[i].item()
        de = delta_E[i].item()
        
        is_low = eb < energy_thresh
        is_fragile = de < drop_thresh
        
        if is_low or is_fragile:
            score = -eb - de * 2.0
            ia = pair_indices[i, 0].item()
            ib = pair_indices[i, 1].item()
            candidates.append((ia, ib, score, eb, de))
    
    if not candidates:
        return [], energy_thresh, drop_thresh
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:max_candidates], energy_thresh, drop_thresh
```

### 6.3 D-S 验证与修正

```python
def _d_s_verify_and_correct(
    self,  # Orchestrator
    candidates: List[Tuple[int, int, float, float, float]],
    min_ds_confidence: float = 0.2,  # D-S置信度低于此值→确认为虚假
) -> Dict:
    """
    提交候选字对到 D-S 验证器，对确认为虚假关联的进行负 Hebbian 修正。
    
    利用已有的概念图查询和模糊格 D-S 融合基础设施:
    - _get_evidence_for(char)  → 概念图证据查询
    - _compute_candidate_belief(char) → D-S 信念质量
    """
    verified = 0
    corrected = 0
    
    for ia, ib, score, eb, de in candidates:
        char_a = self.field.hanzi_list[ia]
        char_b = self.field.hanzi_list[ib]
        
        # 1. 查询双方在概念图中的证据
        evidence_a = self._get_evidence_for(char_a)
        evidence_b = self._get_evidence_for(char_b)
        
        # 2. 检查这对字之间是否有直接的关联证据
        has_direct_evidence = False
        for rel, obj, conf, src in evidence_a:
            if obj == char_b and conf > 0.3:
                has_direct_evidence = True
                break
        for rel, obj, conf, src in evidence_b:
            if obj == char_a and conf > 0.3:
                has_direct_evidence = True
                break
        
        # 3. 查询 SQLite 三元组
        if not has_direct_evidence and hasattr(self, '_cgdb') and self._cgdb:
            try:
                sqlite_conf = self._cgdb.query_pair_confidence(char_a, char_b)
                if sqlite_conf and sqlite_conf > 0.3:
                    has_direct_evidence = True
            except Exception:
                pass
        
        # 4. D-S 信念质量
        ds_belief_a = self._compute_candidate_belief(char_a)
        ds_belief_b = self._compute_candidate_belief(char_b)
        avg_belief = (ds_belief_a + ds_belief_b) / 2 if (ds_belief_a > 0 or ds_belief_b > 0) else 0
        
        verified += 1
        
        # 5. 裁决: 无证据 + 低信念 → 虚假关联
        if not has_direct_evidence and avg_belief < min_ds_confidence:
            # 执行负 Hebbian 修正
            if self.learner:
                try:
                    self.learner.unlearn_chars(char_a, char_b, strength=0.3)
                    corrected += 1
                    log.info(f"  🔧 扰动修正: '{char_a}'-'{char_b}' "
                            f"(EB={eb:.1f}, ΔE={de:+.1f}, score={score:.1f})")
                except Exception as e:
                    log.debug(f"  修正失败({char_a}/{char_b}): {e}")
        else:
            log.debug(f"  ✅ D-S保留: '{char_a}'-'{char_b}' "
                     f"(有证据={has_direct_evidence}, belief={avg_belief:.3f})")
    
    return {
        'verified': verified,
        'corrected': corrected,
        'retained': verified - corrected,
    }
```

### 6.4 D-S 验证器的复用

扰动引擎**不引入新的验证器**，完全复用现有基础设施：

| 组件 | 文件 | 方法 | 用途 |
|------|------|------|------|
| 概念图证据 | concept_graph.py | `get_char_pairs()` | 查询两字是否有直接关联 |
| SQLite 加速 | concept_graph_sqlite.py | `query_pair_confidence()` | O(log N) 关联置信度查询 |
| 模糊格 D-S | fuzzy_graph.py | 经由 `_compute_candidate_belief()` | 综合多源证据的信念质量 |
| 负 Hebbian | learner.py:1107 | `unlearn_chars()` | 在路径中点抬高能量，削弱虚假盆地 |

---

## 7. 代码量与性能评估

### 7.1 预估代码量

| 组件 | 文件 | 新增行数 | 说明 |
|------|------|----------|------|
| `_sample_distant_pairs()` | orchestrator.py | ~50 | 远距字对采样 |
| `_inject_perturbation()` | orchestrator.py | ~25 | 参数噪声注入 |
| `_restore_parameters()` | orchestrator.py | ~10 | 参数恢复 |
| `_measure_energy_shifts()` | orchestrator.py | ~30 | 能量偏移测量 |
| `_filter_candidates()` | orchestrator.py | ~25 | 候选过滤 |
| `_d_s_verify_and_correct()` | orchestrator.py | ~60 | D-S验证+修正 |
| `_perturb_and_observe()` | orchestrator.py | ~70 | 主入口 |
| daemon_tick_v2 修改 | orchestrator.py | ~10 | 插入调度代码 |
| **合计** | | **~280 行** | |

### 7.2 性能评估

| 步骤 | 计算 | 估计耗时 (GPU) | 估计耗时 (CPU) |
|------|------|----------------|----------------|
| 远距对采样 (2000×2000 相似度) | 矩阵乘法 | ~5 ms | ~200 ms |
| 基准能量 (200 中点) | 1 次前向传播 | ~3 ms | ~10 ms |
| 参数扰动注入 | O(params) ≈ 20M | ~1 ms | ~5 ms |
| 扰动后能量 (200 中点) | 1 次前向传播 | ~3 ms | ~10 ms |
| 参数恢复 | O(params) | ~1 ms | ~3 ms |
| 过滤 (200 对) | 纯 Python | < 1 ms | < 1 ms |
| D-S 验证 (≤20 候选) | DB查询 | ~50 ms | ~50 ms |
| 负 Hebbian (≤20 修正) | SGD 各5步 | ~200 ms | ~1000 ms |
| **总计（GPU）** | | **~260 ms** | |
| **总计（CPU）** | | | **~1.3 s** |

### 7.3 对守护循环的影响

```
daemon_tick_v2 当前耗时 (典型):
  scan:         ~500 ms
  arms_search:  ~2000 ms (含网络请求)
  learn_batch:  ~150 ms
  decay:        ~5 ms
  定期调度:     ~500-2000 ms
  ─────────────────────
  总计/轮:      ~3-5 秒

添加扰动引擎后 (每5轮执行1次):
  perturb:      ~260 ms (GPU) / ~1.3 s (CPU)
  ─────────────────────
  额外开销:     每轮摊销 ~50 ms (GPU) / ~260 ms (CPU)
  相对增幅:     < 5%
```

**结论**: 扰动引擎对守护循环的性能影响极小，每5轮执行一次，GPU 场景下单次约 260 ms，远小于双臂搜索的网络请求耗时。

### 7.4 内存影响

- 扰动期间需备份全部景观参数（~34 MB × 2 = 68 MB 临时分配）
- 2000×2000 相似度矩阵 ≈ 16 MB（float32）
- 总计内存峰值增加 ≈ 84 MB，在合理范围内

---

## 8. 实现计划与风险

### 8.1 实现分阶段

| 阶段 | 任务 | 时间估计 |
|------|------|----------|
| Phase 1 | 采样 + 扰动 + 测量（纯检测，不修正） | 2-3 小时 |
| Phase 2 | 候选过滤 + D-S 验证 + 负 Hebbian 修正 | 2-3 小时 |
| Phase 3 | 集成到 daemon_tick_v2 + 日志 + 测试 | 1-2 小时 |
| Phase 4 | 超参数调优（per 守护轮统计观察） | 运行时调节 |

### 8.2 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 远距对采样触发的负 Hebbian 过于激进 | 中 | 抹除合法但罕见的关联 | D-S 置信度阈值有保守默认值 (0.2)；先 Phase 1 纯检测观察 |
| CPU 场景下耗时过高 | 低 | 守护轮变慢 | 每 5 轮执行 + CPU 场景减小 `N_DISTANT` |
| 扰动幅度选择不当 | 中 | 检测信号噪声过大/过小 | 提供 `PERTURB_STD` 超参数配置 + 自适应缩放 |
| 概念图/模糊格不可用时降级 | 低 | D-S 验证无法执行 | 降级为纯启发式裁决，仅依赖能量信号 |

### 8.3 超参数配置表

```python
# 扰动引擎默认配置 (可在 orchestrator.__init__ 中覆写)
PERTURBATION_CONFIG = {
    'enabled': True,              # 是否启用
    'interval_rounds': 5,         # 执行间隔（每N轮）
    'n_distant_pairs': 200,       # 远距对数量
    'sample_subset_size': 2000,   # 相似度计算子集大小
    'similarity_percentile': 10.0,# 低相似度百分位
    'perturbation_std': 0.01,     # 扰动噪声标准差
    'energy_pctl': 10.0,          # 能量低阈值百分位 (P10 ≈ -165)
    'energy_drop_pctl': 5.0,      # 能量下降阈值百分位 (P5)
    'max_candidates': 20,         # 最多D-S候选数
    'min_ds_confidence': 0.2,     # D-S裁决阈值
    'correction_strength': 0.3,   # 负Hebbian强度
}
# 注意: energy_pctl 和 energy_drop_pctl 为数据驱动百分位，
#       具体数值随模型训练动态变化，此处为默认百分位参数。
```

---

## 附录 A: daemon_tick_v2 完整修改对比

### 修改前 (line 1458-1591)

```python
def daemon_tick_v2(self, round_num: int) -> Dict[str, Any]:
    tick_report = {'scanned': 0, 'learned': 0, 'pairs_injected': 0, 'pending_resolved': 0}

    # 0. 优先插队
    ...
    # 1. 大脑扫描
    ...
    # 2. 双臂搜索 + 脑当场吸收
    ...
    # 3. 定期调度
    # 每轮: 衰减
    try:
        if self.learner:
            decay_result = self.learner.decay_step()
            ...
    except Exception:
        pass

    # 每5轮: 序列臂 + 桥接 + 矛盾解 + D-S回写 + 概念图→景观对齐
    if round_num % 5 == 0:
        ...
    ...
```

### 修改后

```python
def daemon_tick_v2(self, round_num: int) -> Dict[str, Any]:
    tick_report = {'scanned': 0, 'learned': 0, 'pairs_injected': 0, 'pending_resolved': 0}

    # 0. 优先插队
    ...
    # 1. 大脑扫描
    ...
    # 2. 双臂搜索 + 脑当场吸收
    ...
    # ★ 2.5 对抗扰动引擎: 检测学习后的虚假盆地（每5轮）
    if round_num % 5 == 0:
        try:
            perturb_report = self._perturb_and_observe()
            tick_report['perturbation'] = perturb_report
        except Exception as e:
            log.debug(f"  对抗扰动异常: {e}")

    # 3. 定期调度
    # 每轮: 衰减
    try:
        if self.learner:
            decay_result = self.learner.decay_step()
            ...
    except Exception:
        pass

    # 每5轮: 序列臂 + 桥接 + 矛盾解 + D-S回写 + 概念图→景观对齐
    if round_num % 5 == 0:
        ...
    ...
```

---

## 附录 B: 数据流图

```
                     daemon_tick_v2
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    _daemon_scan    _arms_search    learn_pairs_batch
         │                │                │
         │         all_pairs              │
         │                │                │
         │                └──────┬─────────┘
         │                       ▼
         │              【对比学习 + Hebbian】
         │              已知对能量 ↓  随机对能量 ↑
         │                       │
         │                       ▼
         │              ╔═══════════════════╗
         │              ║ 对抗扰动引擎      ║  ← ★ 新增
         │              ║                   ║
         │              ║ 1. 采样远距字对   ║
         │              ║ 2. 基准能量测量   ║
         │              ║ 3. 参数扰动注入   ║
         │              ║ 4. 扰动后能量测量 ║
         │              ║ 5. 参数恢复      ║
         │              ║ 6. 候选过滤      ║
         │              ║ 7. D-S验证+修正   ║
         │              ╚═══════╤═══════════╝
         │                      │
         │           ┌──────────┴──────────┐
         │           ▼                     ▼
         │     虚假盆地修正           合法关联保留
         │     (unlearn_chars)       (无操作)
         │           │                     │
         └───────────┴─────────────────────┘
                     │
                     ▼
               decay_step()
                     │
                     ▼
              [定期调度...]
```

---

## 附录 C: 关键常量与依赖

| 符号 | 值 | 来源 |
|------|-----|------|
| 锚点总数 | 94117 | HanziAnchorField.num_hanzi |
| 嵌入维度 | 1024 | HanziAnchorField.embed_dim |
| 景观参数量 | ~34 MB | FreqEnergyLandscape (6层MLP) |
| 景观模型文件 | freq_landscape.py | `loongpearl/core/` |
| 锚点模型文件 | zichang.py | `loongpearl/core/` |
| 学习器 | learner.py | `loongpearl/learning/` |
| D-S 模糊格 | fuzzy_graph.py | `loongpearl/core/` |
| 概念图 SQLite | concept_graph_sqlite.py | `loongpearl/core/` |
