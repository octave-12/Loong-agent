#!/usr/bin/env python3
"""
批量词库导入 — CC-CEDICT 12万词 → 概念图节点
==============================================
将 cedict_parsed.json 中的中文词批量加入概念图，
自动建立基于共享字素和前缀的关系边。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.concept_graph import ConceptGraph

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

print(f"📥 批量词库导入 — CC-CEDICT → 概念图")
print(f"   设备: {DEVICE}")

# 加载字场
print("加载字场...")
field = HanziAnchorField.load(
    os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'), freeze=True)

# 加载/创建概念图
cg_path = os.path.join(PROJECT, 'data/models/concept_graph')
cg = ConceptGraph(field)
if os.path.exists(cg_path + '.json'):
    cg.load(cg_path)
else:
    cg.seed_all_domains()

start_nodes = len(cg.nodes)
print(f"   当前: {start_nodes}节点 {cg.total_triples}三元组")

# 加载 CC-CEDICT
print("加载 CC-CEDICT...")
with open(os.path.join(PROJECT, 'data/dicts/cedict_parsed.json'), 'r', encoding='utf-8') as f:
    cedict = json.load(f)

total_words = len(cedict)
print(f"   词条总数: {total_words}")

# 过滤：只取纯中文词（2-6字），且在字场中
valid_words = []
for word in cedict:
    # 跳过含数字、英文、标点的
    if not all('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u9fff' for c in word):
        continue
    if len(word) < 2 or len(word) > 6:
        continue
    # 必须在字场中
    if not all(c in field._char_to_idx for c in word):
        continue
    valid_words.append(word)

print(f"   有效中文词: {len(valid_words)}")

# 批量添加节点
print("\n批量添加节点...")
t0 = time.time()
added = 0
skipped = 0
for word in valid_words:
    if word in cg.nodes:
        skipped += 1
        continue
    cg.add_node(word)
    added += 1
    if added % 10000 == 0:
        elapsed = time.time() - t0
        print(f"   {added}/{len(valid_words)} ({added/max(1,elapsed):.0f}词/s)")

elapsed = time.time() - t0
print(f"   新增: {added} 跳过: {skipped} 耗时: {elapsed:.1f}s")

# 自动建立关系边（基于共享字素）
print("\n自动建立 RELATED 边（共享字素）...")
t0 = time.time()
edges_added = 0
# 使用前缀索引加速
prefix_index = {}
for word in cg.nodes:
    for i in range(1, min(4, len(word) + 1)):
        prefix = word[:i]
        if prefix not in prefix_index:
            prefix_index[prefix] = []
        prefix_index[prefix].append(word)

for word in valid_words:
    if word not in cg.nodes:
        continue
    # 找共享前缀的词
    related = set()
    for i in range(1, min(4, len(word) + 1)):
        prefix = word[:i]
        for neighbor in prefix_index.get(prefix, [])[:20]:
            if neighbor != word and neighbor not in related:
                related.add(neighbor)
    # 添加 RELATED 边
    for neighbor in list(related)[:8]:
        cg.add_triple(word, "RELATED", neighbor, confidence=0.3, source="dict_import")
        edges_added += 1
    if edges_added % 5000 == 0:
        print(f"   {edges_added} 条边...")

elapsed = time.time() - t0
print(f"   新增边: {edges_added} 耗时: {elapsed:.1f}s")

# 保存
print("\n保存概念图...")
cg.save(cg_path)
print(f"   ✅ 导入完成!")
print(f"   节点: {start_nodes} → {len(cg.nodes)}")
print(f"   三元组: {cg.total_triples}")
print(f"   增长: +{added}节点 +{edges_added}边")
