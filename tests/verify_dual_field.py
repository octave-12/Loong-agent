#!/usr/bin/env python3
"""双场架构端到端验证: 语义场(Hopfield) → 序列场(Markov)"""
import sys, os, re, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loongpearl.core.dragon_field import DragonField
from loongpearl.core.sequence_field import SequenceField
from sentence_transformers import SentenceTransformer

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8"
)

# 加载语义场
print("📥 加载 DragonField (语义场)...")
df = DragonField(embed_dim=1024, beta=8.0)
data = torch.load(os.path.join(PROJECT, "data/models/dragon_field_patterns.pt"), map_location='cpu')
df.store_patterns(data['vectors'], data['ids'], data['subjects'])
print(f"   {df.num_patterns} 概念模式")

# 加载序列场
print("📥 加载 SequenceField (序列场)...")
sf = SequenceField.load(os.path.join(PROJECT, "data/models/sequence_field.json"))
stats = sf.global_stats()
print(f"   {stats['num_basins']} 盆, {stats['total_bigrams']} bigrams")

# 加载 BGE 编码器
print("📥 加载 BGE 编码器...")
model = SentenceTransformer(MODEL_PATH, device="cuda")

# 测试查询
queries = [
    "龙是什么",
    "水的三种形态",
    "人工智能是什么",
    "地球有多大",
    "什么是光合作用",
]

print("\n" + "═" * 60)
for q in queries:
    print(f"\n🔍 查询: {q}")
    query_chars = re.findall(r'[\u4e00-\u9fff]', q)

    # Step 1: BGE 编码查询
    with torch.no_grad():
        q_vec = model.encode(q, convert_to_tensor=True, normalize_embeddings=True)

    # Step 2: 语义场 converge → 找盆
    fr = df.converge(q_vec, max_steps=20, convergence_threshold=1e-4)

    # 提取 top basin subjects
    top_subjects = []
    for idx in fr.top_pattern_indices[:5]:
        if 0 <= idx < len(df._pattern_subjects):
            subj = df._pattern_subjects[idx]
            if subj:
                top_subjects.append(subj)

    basin = top_subjects[0] if top_subjects else "?"
    print(f"   语义场盆地: {basin} (sim={fr.top_similarities[0]:.3f})")
    print(f"   Top-5: {top_subjects[:5]}")

    # Step 3: 序列场 walk
    if basin in sf._basin_forward:
        output = sf.walk_bidirectional(
            basin_subject=basin,
            seed_chars=query_chars,
            length=20,
            temperature=0.7,
            fallback_basins=top_subjects[1:3] if len(top_subjects) > 1 else None,
        )
        print(f"   序列场: [{basin}盆] {output}")
    else:
        # 尝试 fallback
        found = False
        for fb in top_subjects[1:5]:
            if fb in sf._basin_forward:
                output = sf.walk_bidirectional(
                    basin_subject=fb,
                    seed_chars=query_chars, length=20, temperature=0.7,
                )
                print(f"   序列场: [{fb}盆] {output}")
                found = True
                break
        if not found:
            print(f"   ⚠️ 盆地未命中序列场 (top subjects 都无 bigram 数据)")

print("\n" + "═" * 60)
print("完成")
