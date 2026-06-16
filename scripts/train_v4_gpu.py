#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
龙珠完整训练 v4 — GPU加速 + 多目标 + 实时监控。

三目标联合优化：
  T1. 锚点盆地：3725个已学汉字 → 能量=-15（深谷）
  T2. 知识通道：已知字对（词语/部件）→ 能量=-10（通道）
  T3. 认知边界：随机字对 → 能量=0（墙）
  
  频率加权：高频词对通道更宽（能量更低）

资源：RTX 3060 12GB GPU
监控：实时输出 + Ctrl+C 安全停止 + 自动保存最优
"""
import torch, torch.nn as nn, numpy as np
import sys, os, json, random, time, signal

PROJECT = "/mnt/d/soso/projects/Loong-agent/Loong-pearl"
sys.path.insert(0, PROJECT)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape

# ===== 配置 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16384          # GPU大batch
EPOCHS = 50               # 更多轮数，充分收敛
LR = 0.002
TARGET_ANCHOR = -15.0
TARGET_POS    = -10.0
TARGET_NEG    =   2.0    # 正值，明确的知识边界
SAVE_PATH = os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt")
LOG_INTERVAL = 1           # 每N批输出一次

# ===== 全局状态（用于Ctrl+C安全停止）=====
STOP_REQUESTED = False
BEST_STATE = None
BEST_GAP = 0.0

def signal_handler(sig, frame):
    global STOP_REQUESTED
    print("\n⚠️  收到停止信号，完成当前轮后保存...", flush=True)
    STOP_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log(msg):
    print(msg, flush=True)

# ============================================================
log("=" * 60)
log(f"🐉 龙珠完整训练 v4 (GPU加速)")
log(f"   设备: {DEVICE}  |  GPU内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if DEVICE.type == "cuda" else f"   设备: CPU")
log("=" * 60)

log(f"\n训练参数: bs={BATCH_SIZE} epochs={EPOCHS} lr={LR}")
log(f"目标: 锚点={TARGET_ANCHOR}  通道={TARGET_POS}  墙={TARGET_NEG}")
log(f"Ctrl+C 随时安全停止\n")

# ============================================================
# 1. 加载数据
# ============================================================
log("📦 加载数据...")
t = time.time()

# 字场
field = HanziAnchorField.load(
    os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"), freeze=True)
anchors_all = field.anchors.to(DEVICE)  # 放到GPU
log(f"  字场: {anchors_all.shape} → GPU ({time.time()-t:.1f}s)")

# 从零初始化能量景观
log("  能量景观: 随机初始化 → GPU")
landscape = EnergyLandscape(embed_dim=field.embed_dim).to(DEVICE)
landscape.train()

# 词典 → 正样本
t = time.time()
with open(os.path.join(PROJECT, "data/dicts/cedict_parsed.json"), encoding="utf-8") as f:
    cedict = json.load(f)

positive_pairs = set()
pair_freq = {}  # 词频统计
for word in cedict:
    if not all('\u4e00' <= c <= '\u9fff' for c in word):
        continue
    for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        positive_pairs.add(pair)
        pair_freq[pair] = pair_freq.get(pair, 0) + 1

log(f"  词典: {len(cedict)} 词条 → {len(positive_pairs)} 字对 ({time.time()-t:.1f}s)")

# 过滤 → GPU tensor + 频率权重
t = time.time()
pos_indices = []
pos_weights = []  # 高频词对权重更高
for a, b in positive_pairs:
    ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
    if ia is not None and ib is not None:
        pos_indices.append((ia, ib))
        freq = pair_freq.get((a, b), 1)
        pos_weights.append(min(np.log1p(freq), 5.0))  # log频率，上限5

pos_tensor = torch.tensor(pos_indices, dtype=torch.long, device=DEVICE)
pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=DEVICE)
log(f"  正样本: {len(pos_indices)} 对 → GPU ({time.time()-t:.1f}s)")

# 负样本
t = time.time()
all_idx = list(range(field.num_hanzi))
neg_set = set()
while len(neg_set) < len(pos_indices):
    ia, ib = random.sample(all_idx, 2)
    a_ch, b_ch = field.hanzi_list[ia], field.hanzi_list[ib]
    if (a_ch, b_ch) in positive_pairs or (b_ch, a_ch) in positive_pairs:
        continue
    neg_set.add((ia, ib))
neg_indices = list(neg_set)
neg_tensor = torch.tensor(neg_indices, dtype=torch.long, device=DEVICE)
log(f"  负样本: {len(neg_indices)} 对 → GPU ({time.time()-t:.1f}s)")

total = len(pos_indices)

# 已学汉字锚点代理（取3725个中的2000个采样，加速）
# 如果有baby_curriculum，用已知字；否则用前3725个
anchor_proxy_n = 2000
anchor_proxy_idx = torch.tensor(
    random.sample(range(min(3725, field.num_hanzi)), anchor_proxy_n),
    dtype=torch.long, device=DEVICE
)

# ============================================================
# 2. 训练
# ============================================================
optimizer = torch.optim.AdamW(landscape.parameters(), lr=LR, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5, verbose=False)

batches_per_epoch = (total + BATCH_SIZE - 1) // BATCH_SIZE

log(f"\n{'='*70}")
log(f"🚀 开始训练: {total}对 × {EPOCHS}轮 = {batches_per_epoch}批/轮")
log(f"{'='*70}")
log(f"{'Epoch':>5} {'Batch':>6} {'正E':>8} {'负E':>8} {'间隔':>8} {'锚E':>8} {'Loss':>8} {'LR':>9} {'耗时'}")
log(f"{'─'*70}")

global_step = 0
for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()
    
    # 打乱
    pos_perm = torch.randperm(total, device=DEVICE)
    neg_perm = torch.randperm(total, device=DEVICE)
    
    sum_pos, sum_neg, sum_anchor, sum_loss = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    
    for start in range(0, total, BATCH_SIZE):
        if STOP_REQUESTED:
            break
            
        end = min(start + BATCH_SIZE, total)
        
        # ---- 正样本中点 ----
        p_idx = pos_perm[start:end]
        p_rows = pos_tensor[p_idx]
        p_mids = (anchors_all[p_rows[:, 0]] + anchors_all[p_rows[:, 1]]) / 2
        
        # ---- 负样本中点 ----
        n_idx = neg_perm[start:end]
        n_rows = neg_tensor[n_idx]
        n_mids = (anchors_all[n_rows[:, 0]] + anchors_all[n_rows[:, 1]]) / 2
        
        # ---- 锚点代理 ----
        # 每100批换一批锚点代理
        if n_batches % 100 == 0:
            anchor_proxy_idx = torch.tensor(
                random.sample(range(min(3725, field.num_hanzi)), anchor_proxy_n),
                dtype=torch.long, device=DEVICE
            )
        anchor_vecs = anchors_all[anchor_proxy_idx]
        
        # ---- 前向 ----
        e_pos = landscape(p_mids).squeeze(-1)
        e_neg = landscape(n_mids).squeeze(-1)
        e_anchor = landscape(anchor_vecs).squeeze(-1)
        
        # ---- 损失（频率加权） ----
        w = pos_weights[p_idx]  # 高频对权重
        # 单侧惩罚：只罚过低，不罚过高（允许负样本能量>目标）
        pos_loss = ((e_pos - TARGET_POS) ** 2 * w).mean()
        neg_loss = (torch.clamp(TARGET_NEG - e_neg, min=0) ** 2).mean()
        anchor_loss = ((e_anchor - TARGET_ANCHOR) ** 2).mean()
        
        loss = pos_loss * 0.5 + neg_loss * 2.0 + anchor_loss * 3.0
        
        # ---- 反向传播 ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(landscape.parameters(), 1.0)
        optimizer.step()
        
        sum_pos += e_pos.mean().item()
        sum_neg += e_neg.mean().item()
        sum_anchor += e_anchor.mean().item()
        sum_loss += loss.item()
        n_batches += 1
        global_step += 1
        
        # 实时输出（每LOG_INTERVAL批）
        if n_batches % LOG_INTERVAL == 0 and n_batches > 0:
            elapsed = time.time() - ep_start
            pct = (start + BATCH_SIZE) / total * 100
            log(f"{epoch:>5} {n_batches:>4}/{batches_per_epoch} {e_pos.mean().item():>8.2f} "
                f"{e_neg.mean().item():>8.2f} {e_neg.mean().item()-e_pos.mean().item():>8.2f} "
                f"{e_anchor.mean().item():>8.2f} {loss.item():>8.2f} "
                f"{scheduler.get_last_lr()[0]:>8.6f} {elapsed:>5.1f}s ({pct:.0f}%)")
    
    if STOP_REQUESTED:
        log("\n⏸️  训练中断，保存当前模型...")
        break
    
    # ---- 轮结束统计 ----
    avg_pos = sum_pos / n_batches
    avg_neg = sum_neg / n_batches
    avg_anchor = sum_anchor / n_batches
    gap = avg_neg - avg_pos
    dur = time.time() - ep_start
    
    scheduler.step(sum_loss / n_batches)
    
    log(f"{'─'*70}")
    log(f"  ✅ Epoch {epoch} 完成: 正E={avg_pos:.2f} 负E={avg_neg:.2f} "
        f"间隔={gap:.2f} 锚E={avg_anchor:.2f} 耗时={dur:.1f}s")
    
    # 保存最优
    if gap > BEST_GAP:
        BEST_GAP = gap
        BEST_STATE = {k: v.clone() for k, v in landscape.state_dict().items()}
        log(f"  🏆 新最优! 间隔={gap:.2f}")
    
    log("")

# ============================================================
# 3. 保存 + 验证
# ============================================================
if BEST_STATE:
    landscape.load_state_dict(BEST_STATE)

landscape.eval()
landscape.cpu()  # 保存前移回CPU

log(f"\n{'='*60}")
log("💾 保存模型...")

# 最终验证
with torch.no_grad():
    anchors_cpu = field.anchors
    n_val = 5000
    vp = pos_tensor.cpu()[torch.randperm(total)[:n_val]]
    vn = neg_tensor.cpu()[torch.randperm(total)[:n_val]]
    
    final_anchor = landscape(anchors_cpu).mean().item()
    final_pos = landscape((anchors_cpu[vp[:,0]] + anchors_cpu[vp[:,1]]) / 2).mean().item()
    final_neg = landscape((anchors_cpu[vn[:,0]] + anchors_cpu[vn[:,1]]) / 2).mean().item()

log(f"📊 最终结果:")
log(f"  锚点: {final_anchor:.2f} (目标 {TARGET_ANCHOR})")
log(f"  通道: {final_pos:.2f} (目标 {TARGET_POS})")
log(f"  墙:   {final_neg:.2f} (目标 {TARGET_NEG})")
log(f"  分离: {final_neg - final_pos:.2f}")

landscape.save(SAVE_PATH)
log(f"✅ 已保存: {SAVE_PATH}")

# ============================================================
# 4. 快速测试
# ============================================================
log(f"\n{'='*60}")
log("🧪 知识通道 vs 认知边界")

tests = [
    ('中','国','知'), ('龙','珠','知'), ('学','习','知'),
    ('心','想','知'), ('天','地','知'), ('人','生','知'),
    ('中','龘','随'), ('龙','𰻝','随'), ('学','𪟝','随'),
]
with torch.no_grad():
    for a, b, tag in tests:
        ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
        if ia is not None and ib is not None:
            mid = (anchors_cpu[ia] + anchors_cpu[ib]) / 2
            e = landscape(mid.unsqueeze(0)).item()
            label = "✓通道" if e < -5 else ("△" if e < 0 else "✗墙")
            log(f"  [{tag}] {a}↔{b}: {e:+.2f} {label}")
