#!/usr/bin/env python3
"""构建 DragonField 模式缓存 — 从 SQLite 提取高置信三元组, BGE 编码, 存为 .pt"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sqlite3
from sentence_transformers import SentenceTransformer

DB = "data/models/concept_graph.db"
OUT = "data/models/dragon_field_patterns.pt"
LIMIT = 200000  # 20万高置信三元组 (BGE编码~10min)

# 使用本地缓存避免网络请求
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8")
print("📖 加载 BGE 编码器 (本地缓存)...")
model = SentenceTransformer(MODEL_PATH, device="cuda")
print(f"   设备: {model.device}")

print("📊 从 DB 读取高置信三元组...")
conn = sqlite3.connect(DB)
rows = conn.execute(
    "SELECT s, r, o, id FROM triples WHERE c >= 0.7 ORDER BY c DESC LIMIT ?",
    (LIMIT,)
).fetchall()
conn.close()
print(f"   读取: {len(rows)} 条")

# 批量编码 (s + r + o 拼接)
texts = [f"{s} {r} {o}" for s, r, o, _ in rows]
ids = [row[3] for row in rows]
subjects = [row[0] for row in rows]

print(f"🧠 BGE 编码中 ({len(texts)} 条)...")
t0 = time.time()
batch_size = 512
vectors_list = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    with torch.no_grad():
        vecs = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
    vectors_list.append(vecs.cpu().to(torch.float16))
    if (i // batch_size) % 10 == 0:
        print(f"  {i}/{len(texts)} ({i/len(texts)*100:.0f}%)", end='\r')

vectors = torch.cat(vectors_list, dim=0)
elapsed = time.time() - t0
print(f"\n   完成: {vectors.shape} | {elapsed:.1f}s | {vectors.element_size()*vectors.numel()/1024**2:.1f} MB")

print(f"💾 保存: {OUT}")
torch.save({'vectors': vectors, 'ids': ids, 'subjects': subjects}, OUT)
print(f"✅ DragonField 模式缓存就绪 ({len(ids)} 模式)")
