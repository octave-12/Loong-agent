#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
对比重训能量景观 v3 —— 从零训练三目标景观。

三目标（同时优化）：
  锚点→ -15  （已学汉字是能量盆地）
  正样本→ -10 （已知字对之间有知识通道）
  负样本→  0  （随机字对之间是认知边界/墙）
"""
import torch, torch.nn as nn, numpy as np
import sys, os, json, random, time

PROJECT = "/mnt/d/soso/projects/Loong-agent/Loong-pearl"
sys.path.insert(0, PROJECT)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape

DEVICE = "cpu"
BATCH_SIZE = 4096
EPOCHS = 20
LR = 0.001
TARGET_ANCHOR = -15.0   # 锚点目标能量
TARGET_POS    = -10.0   # 已知字对目标能量
TARGET_NEG    =   0.0   # 随机字对目标能量
SAVE_PATH = os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt")
ANCHOR_PROXY = 2000      # 用随机2000个锚点替代全量94117个，加速

def log(msg):
    print(msg, flush=True)

# ============================================================
# 1. 加载
# ============================================================
log("=" * 60)
log("🐉 对比重训能量景观 v3 (从零训练)")
log("=" * 60)

log("\n加载字场...")
t0 = time.time()
field = HanziAnchorField.load(
    os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"), freeze=True)
log(f"  ({time.time()-t0:.1f}s)")

# 从零初始化能量景观（不用旧权重）
log("初始化新能量景观（随机权重）...")
landscape = EnergyLandscape(embed_dim=field.embed_dim)
landscape.train()

# 随机选择锚点代理（每轮换一批，避免过拟合到特定锚点）
all_anchor_idx = list(range(field.num_hanzi))
random.shuffle(all_anchor_idx)

# ============================================================
# 2. 构建样本索引
# ============================================================
log("\n构建训练样本...")

log("  加载 CC-CEDICT...")
with open(os.path.join(PROJECT, "data/dicts/cedict_parsed.json"), encoding="utf-8") as f:
    cedict = json.load(f)

positive_pairs = set()
for word in cedict:
    if not all('\u4e00' <= c <= '\u9fff' for c in word):
        continue
    for i in range(len(word) - 1):
        positive_pairs.add((word[i], word[i+1]))
log(f"  字对: {len(positive_pairs)}")

# 过滤
pos_indices = []
for a, b in positive_pairs:
    ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
    if ia is not None and ib is not None:
        pos_indices.append((ia, ib))

random.shuffle(pos_indices)
log(f"  有效正样本: {len(pos_indices)}")

# 负样本
log(f"  生成负样本...")
neg_pool = set()
pairs_set = positive_pairs  # for O(1) lookup
while len(neg_pool) < len(pos_indices):
    ia, ib = random.sample(all_anchor_idx, 2)
    a_char, b_char = field.hanzi_list[ia], field.hanzi_list[ib]
    if (a_char, b_char) in pairs_set or (b_char, a_char) in pairs_set:
        continue
    neg_pool.add((ia, ib))

neg_indices = list(neg_pool)
random.shuffle(neg_indices)
log(f"  负样本: {len(neg_indices)}")

pos_tensor = torch.tensor(pos_indices, dtype=torch.long)
neg_tensor = torch.tensor(neg_indices, dtype=torch.long)
total = len(pos_indices)

# ============================================================
# 3. 训练
# ============================================================
optimizer = torch.optim.Adam(landscape.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

log(f"\n训练: {total} 对 × {EPOCHS} 轮 | bs={BATCH_SIZE} | lr={LR}")
log(f"  目标: 锚点={TARGET_ANCHOR}  正样本={TARGET_POS}  负样本={TARGET_NEG}")
log(f"{'─'*70}")
log(f"{'Epoch':<6} {'正E':>8} {'负E':>8} {'间隔':>8} {'锚E':>8} {'损失':>8} {'LR':>8} {'耗时':>6}")
log(f"{'─'*70}")

for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()
    
    # 每轮换锚点代理
    anchor_proxy_idx = random.sample(all_anchor_idx, ANCHOR_PROXY)
    anchor_proxy = field.anchors[anchor_proxy_idx]
    
    pos_perm = torch.randperm(total)
    neg_perm = torch.randperm(total)
    
    sum_pos, sum_neg, sum_anchor, sum_loss = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        
        p_idx = pos_perm[start:end]
        n_idx = neg_perm[start:end]
        
        # === 正样本中点 ===
        p_rows = pos_tensor[p_idx]
        p_mids = (field.anchors[p_rows[:, 0]] + field.anchors[p_rows[:, 1]]) / 2
        
        # === 负样本中点 ===
        n_rows = neg_tensor[n_idx]
        n_mids = (field.anchors[n_rows[:, 0]] + field.anchors[n_rows[:, 1]]) / 2
        
        # === 前向 ===
        e_pos = landscape(p_mids).squeeze(-1)
        e_neg = landscape(n_mids).squeeze(-1)
        e_anchor = landscape(anchor_proxy).squeeze(-1)
        
        # === 三目标 MSE 损失 ===
        pos_loss = ((e_pos - TARGET_POS) ** 2).mean()
        neg_loss = ((e_neg - TARGET_NEG) ** 2).mean()
        anchor_loss = ((e_anchor - TARGET_ANCHOR) ** 2).mean()
        
        # 权重：通道最重要（区分已知/未知），锚点其次，随机墙再次
        loss = pos_loss * 1.0 + neg_loss * 0.5 + anchor_loss * 0.3
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(landscape.parameters(), 1.0)
        optimizer.step()
        
        sum_pos += e_pos.mean().item()
        sum_neg += e_neg.mean().item()
        sum_anchor += e_anchor.mean().item()
        sum_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    avg_pos = sum_pos / n_batches
    avg_neg = sum_neg / n_batches
    avg_anchor = sum_anchor / n_batches
    avg_loss = sum_loss / n_batches
    gap = avg_neg - avg_pos
    dur = time.time() - ep_start
    
    log(f"{epoch:<6} {avg_pos:>8.2f} {avg_neg:>8.2f} {gap:>8.2f} {avg_anchor:>8.2f} {avg_loss:>8.2f} {scheduler.get_last_lr()[0]:>8.6f} {dur:>5.1f}s")

# ============================================================
# 4. 验证 + 保存
# ============================================================
landscape.eval()
log(f"\n{'─'*70}")

with torch.no_grad():
    final_anchor = landscape(field.anchors).mean().item()
    
    # 抽样5000个验证
    n_val = 5000
    vp = pos_tensor[torch.randperm(total)[:n_val]]
    vn = neg_tensor[torch.randperm(total)[:n_val]]
    final_pos = landscape((field.anchors[vp[:,0]] + field.anchors[vp[:,1]]) / 2).mean().item()
    final_neg = landscape((field.anchors[vn[:,0]] + field.anchors[vn[:,1]]) / 2).mean().item()

log(f"📊 最终结果:")
log(f"  锚点: {final_anchor:.2f} (目标 {TARGET_ANCHOR})")
log(f"  正样本: {final_pos:.2f} (目标 {TARGET_POS})")
log(f"  负样本: {final_neg:.2f} (目标 {TARGET_NEG})")
log(f"  分离度: {final_neg - final_pos:.2f} ✓")

# 保存
landscape.save(SAVE_PATH)
log(f"\n✅ 已保存: {SAVE_PATH}")

# ============================================================
# 5. 实际测试
# ============================================================
log(f"\n{'='*60}")
log("🧪 真正词对 vs 随机字对")

tests = [
    ('中','国'), ('龙','珠'), ('画','龙'), ('点','睛'),
    ('学','习'), ('知','识'), ('心','想'), ('事','成'),
    ('飞','舞'), ('天','地'), ('一','心'), ('人','生'),
    ('中','龘'), ('龙','𰻝'), ('画','鿫'), ('点','𪟝'),
    ('学','𬉼'), ('知','㑳'), ('心','𨭎'), ('事','鿫'),
]

with torch.no_grad():
    for a, b in tests:
        ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
        if ia is not None and ib is not None:
            mid = (field.anchors[ia] + field.anchors[ib]) / 2
            e = landscape(mid.unsqueeze(0)).item()
            label = "✓通道" if e < -5 else ("△边界" if e < 0 else "✗墙")
            print(f"  {a}↔{b}: {e:+.2f} ({label})", flush=True)
