#!/usr/bin/env python3
"""
知识播种 v2 — CPU并行注入部件/部首/语义知识。

数据源：
  dict_decompose.json → 部件关联（明↔日, 明↔月）
  dict_unihan.json    → 部首关联（同部首字）+ 语义关联（近义字）

输出：直接更新 energy_landscape_1024d.pt
"""
import torch, json, math, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

PROJECT = os.path.dirname(os.path.abspath(__file__))
LANDSCAPE_PATH = os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt")
DECOMP_PATH = os.path.join(PROJECT, "data/dicts/dict_decompose.json")
UNIHAN_PATH = os.path.join(PROJECT, "data/dicts/dict_unihan.json")

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("🧩 知识播种 v2 — 部件/部首/语义注入")
log("=" * 60)

# ===== 加载 =====
log("\n📦 加载...")
t0 = time.time()
field = HanziAnchorField.load(os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"), freeze=True)
ls = FreqEnergyLandscape.load(LANDSCAPE_PATH)
ls.train()
log(f"  ({time.time()-t0:.1f}s)")

# ===== 1. 部件关联（dict_decompose） =====
log("\n🔧 提取部件关联...")
with open(DECOMP_PATH, encoding='utf-8') as f:
    decomp = json.load(f)

# 统计部件共现
component_pairs = set()
for char, info in decomp.items():
    components = info.get('components', [])
    if not components:
        continue
    for comp in components:
        if len(comp) == 1 and '\u4e00' <= comp <= '\u9fff':
            component_pairs.add((char, comp))  # 字符↔部件
            component_pairs.add((comp, char))  # 双向

# 过滤有效对
valid_component = []
for a, b in component_pairs:
    ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
    if ia is not None and ib is not None:
        valid_component.append((ia, ib))

log(f"  部件关联: {len(valid_component)} 对")

# ===== 2. 部首关联（dict_unihan） =====
log("\n🔤 提取部首/语义关联...")
with open(UNIHAN_PATH, encoding='utf-8') as f:
    unihan = json.load(f)

# 按部首分组
radical_groups = {}
for char, info in unihan.items():
    if len(char) != 1 or not ('\u4e00' <= char <= '\u9fff'):
        continue
    # 尝试从definition中提取部首信息
    # Unihan数据较简单，用共现的definition关键词作语义关联
    pass

# 简单策略：取有definition的字，分组（按首字聚类）
semantic_pairs = set()
chars_with_def = []
for char, info in unihan.items():
    if len(char) == 1 and '\u4e00' <= char <= '\u9fff':
        defn = info.get('definition', '')
        if defn and len(defn) > 2:
            chars_with_def.append((char, defn))

# 基于definition关键词做简单聚类
# 取第一批5000个字的definition做关联
import random
random.shuffle(chars_with_def)
sample = chars_with_def[:5000]

# 提取关键词（取definition的前几个字/词）
from collections import Counter
keyword_index = {}
for char, defn in sample:
    # 提取中文关键词
    for word in defn.split():
        word = word.strip('(),;.')
        if word and all('\u4e00' <= c <= '\u9fff' for c in word):
            keyword_index.setdefault(word[:2], []).append(char)

# 同关键词的字符互相关联
for kw, chars in keyword_index.items():
    if len(chars) < 2 or len(chars) > 50:  # 太少没意义，太多太泛
        continue
    for i in range(len(chars)):
        for j in range(i+1, min(i+5, len(chars))):  # 只取最近5个
            semantic_pairs.add((chars[i], chars[j]))
            semantic_pairs.add((chars[j], chars[i]))

valid_semantic = []
for a, b in semantic_pairs:
    ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
    if ia is not None and ib is not None:
        valid_semantic.append((ia, ib))

log(f"  语义关联: {len(valid_semantic)} 对")

# ===== 合并所有新知识对 =====
all_new = list(set(valid_component + valid_semantic))
random.shuffle(all_new)
log(f"\n📊 合计新知识: {len(all_new)} 对 (部件{len(valid_component)} + 语义{len(valid_semantic)})")

# ===== 注入：极小学习率，只补缺不破坏已有知识 =====
log(f"\n💉 增量注入 (CPU, lr=0.00001, 保守模式)...")
optimizer = torch.optim.SGD(ls.parameters(), lr=0.00001, momentum=0.9)

# 为每个部件对设置合理的频率（部件关联=低频知识，freq≈1）
# 目标：浅通道 -11（不干扰已有的深通道）
TARGET_BASE = -11.0
BATCH = 512
STEPS = 100  # 只做100步，够建立浅通道即可

all_tensor = torch.tensor(all_new, dtype=torch.long)
total = len(all_new)
freq_val = math.log1p(1)  # 部件关联频率=1

for step in range(STEPS):
    # 随机取一批
    idx = torch.randint(0, total, (BATCH,))
    rows = all_tensor[idx]
    
    mids = (field.anchors[rows[:,0]] + field.anchors[rows[:,1]]) / 2
    freq = torch.full((BATCH,), freq_val)
    
    e = ls(mids, freq).squeeze(-1)
    # 只惩罚能量高于目标的（即通道太浅），不惩罚已经够深的
    loss = torch.clamp(e - TARGET_BASE, min=0).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ls.parameters(), 0.1)  # 极紧梯度裁剪
    optimizer.step()
    
    if (step+1) % 20 == 0:
        log(f"  Step {step+1}/{STEPS}: loss={loss.item():.4f} avg_e={e.mean().item():.2f}")

# ===== 验证 + 保存 =====
ls.eval()
log(f"\n💾 保存...")
ls.save(LANDSCAPE_PATH)

# 快速验证
log(f"\n🧪 验证新知识:")
tests = [
    ('明','日','部件'), ('明','月','部件'), ('休','木','部件'),
    ('中','国','对照'), ('大','学','对照'), ('中','龘','随机'),
]
with torch.no_grad():
    for a,b,tag in tests:
        ia, ib = field._char_to_idx.get(a), field._char_to_idx.get(b)
        if ia is not None and ib is not None:
            mid = (field.anchors[ia] + field.anchors[ib]) / 2
            f = torch.tensor([math.log1p(1)], dtype=torch.float32)
            e = ls(mid.unsqueeze(0), f).item()
            icon = "🔵" if e<-5 else ("🟡" if e<0 else "🔴")
            log(f"  [{tag}] {a}↔{b}: {e:+.1f} {icon}")

log(f"\n✅ 知识注入完成! ({time.time()-t0:.0f}s)")
