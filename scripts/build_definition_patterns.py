#!/usr/bin/env python3
"""构建首段序列模式 — Wikipedia 首段 (定义部分) → 3-gram → BGE → 缓存"""
import sys, os, time, sqlite3, torch
import numpy as np
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8"
)
DB = "data/wikipedia/zhwiki.db"
OUT = "data/models/dragon_field_seq_def.pt"
MAX_ARTICLES = 50000
MAX_PATTERNS = 200000

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL, device="cuda")

print(f"📖 读取首段 (上限{MAX_ARTICLES}篇)...")
conn = sqlite3.connect(DB)
cursor = conn.execute(
    "SELECT title, text FROM articles WHERE char_count > 200 ORDER BY char_count DESC LIMIT ?",
    (MAX_ARTICLES,)
)

all_windows = []
for i, (title, text) in enumerate(cursor):
    # 取首段: 第一个句号/换行之前, 或前200字
    para = text.split('\n')[0].split('。')[0][:200]
    chars = [c for c in para if '\u4e00' <= c <= '\u9fff']
    for j in range(len(chars) - 2):
        all_windows.append(''.join(chars[j:j+3]))
    if (i+1) % 5000 == 0:
        print(f"  {i+1} 篇, {len(all_windows):,} 窗口")

conn.close()
print(f"完成: {i+1} 篇 → {len(all_windows):,} 窗口")

# 去重+过滤
print(f"🔍 去重+过滤...")
counts = Counter(all_windows)
# 过滤出现在 >2% 文章中的碎片
max_freq = max(counts.values())
threshold = max(3, max_freq * 0.02)

filtered = []
for w, c in counts.most_common(MAX_PATTERNS * 3):
    if c > threshold and c > 50:
        continue
    filtered.append(w)
    if len(filtered) >= MAX_PATTERNS:
        break
print(f"唯一: {len(counts):,} → 过滤后: {len(filtered):,}")

# BGE 编码
print(f"🧠 BGE 编码 {len(filtered)} 条...")
batch_size = 512
vecs_list = []
for i in range(0, len(filtered), batch_size):
    batch = filtered[i:i+batch_size]
    with torch.no_grad():
        v = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
    vecs_list.append(v.cpu().to(torch.float16))
    if i % 10240 == 0 and i > 0:
        print(f"  {i}/{len(filtered)}")

vecs = torch.cat(vecs_list, dim=0)
print(f"完成: {vecs.shape}")

# 保存
types = ['sequence_3'] * len(filtered)
ids = list(range(len(filtered)))
torch.save({'vectors': vecs, 'ids': ids, 'subjects': filtered, 'pattern_types': types}, OUT)
print(f"✅ 首段模式: {len(filtered):,} → {OUT} ({vecs.element_size()*vecs.numel()/1024**2:.1f} MB)")
