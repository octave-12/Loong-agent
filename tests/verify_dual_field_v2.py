#!/usr/bin/env python3
"""双场架构端到端验证 v2 — 查询关键概念 → 序列场盆匹配"""
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

# 构建盆名倒排索引: 字 → 盆名列表
print("📥 构建盆名倒排索引...")
basin_index = {}
for basin_name in sf._basin_forward.keys():
    for ch in basin_name:
        if '\u4e00' <= ch <= '\u9fff':
            if ch not in basin_index:
                basin_index[ch] = []
            basin_index[ch].append(basin_name)
# 去重每个字的盆列表
for ch in basin_index:
    basin_index[ch] = list(set(basin_index[ch]))
print(f"   倒排索引: {len(basin_index)} 字 → 盆映射")

# 加载 BGE
print("📥 加载 BGE 编码器...")
model = SentenceTransformer(MODEL_PATH, device="cuda")


def find_basin(query_chars, sf, basin_index):
    """从查询字找到最佳匹配的序列场盆"""
    # 策略: 找包含最多查询字的盆, 且盆本身不能太长(优先精确概念)
    candidates = {}
    for ch in query_chars:
        if ch in basin_index:
            for basin_name in basin_index[ch]:
                # 计算盆名和查询的重叠字数
                overlap = sum(1 for c in query_chars if c in basin_name)
                if overlap > 0:
                    if basin_name not in candidates:
                        candidates[basin_name] = {'overlap': 0, 'len': len(basin_name)}
                    candidates[basin_name]['overlap'] += overlap

    if not candidates:
        return None, []

    # 排序: overlap 高优先, 盆名短优先(精确概念)
    ranked = sorted(
        candidates.items(),
        key=lambda x: (-x[1]['overlap'], x[1]['len'])
    )
    return ranked[0][0], [b for b, _ in ranked[:5]]


queries = [
    "龙是什么",
    "水的三种形态",
    "人工智能是什么",
    "地球有多大",
    "什么是光合作用",
    "中国历史",
    "计算机科学",
]

print("\n" + "═" * 60)
for q in queries:
    print(f"\n🔍 查询: {q}")
    query_chars = re.findall(r'[\u4e00-\u9fff]', q)

    # 语义场 converge (验证)
    with torch.no_grad():
        q_vec = model.encode(q, convert_to_tensor=True, normalize_embeddings=True)
    fr = df.converge(q_vec, max_steps=20, convergence_threshold=1e-4)
    df_top = [df._pattern_subjects[i] for i in fr.top_pattern_indices[:3]
              if 0 <= i < len(df._pattern_subjects)]
    print(f"   语义场: top-3={df_top} (sim={fr.top_similarities[0]:.3f})")

    # 序列场盆地匹配
    basin, alt_basins = find_basin(query_chars, sf, basin_index)
    if basin:
        bs = sf.basin_stats(basin)
        output = sf.walk_bidirectional(
            basin_subject=basin,
            seed_chars=query_chars,
            length=20,
            temperature=0.7,
            fallback_basins=alt_basins[:2],
        )
        print(f"   序列场: [{basin[:30]}] bigrams={bs['total_bigrams']}")
        print(f"   输出:   {output}")
    else:
        print(f"   ⚠️ 无匹配盆")

print("\n" + "═" * 60)
print("完成")
