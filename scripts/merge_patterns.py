#!/usr/bin/env python3
"""合并概念模式 + 序列_3 + 序列_5 → 最终 DragonField 缓存"""
import torch, os, sys

concept_file = "data/models/dragon_field_patterns.pt"
seq3_file    = "data/models/dragon_field_seq3_clean.pt"
seq5_file    = "data/models/dragon_field_patterns_w5.pt"
output_file  = "data/models/dragon_field_patterns.pt"

# Backup existing
if os.path.exists(output_file):
    backup = output_file + ".backup"
    os.rename(output_file, backup)
    print(f"Backup: {backup}")

# Load all
concept = torch.load(concept_file, map_location='cpu')
c_vecs, c_ids, c_subs = concept['vectors'], concept['ids'], concept['subjects']
c_types = concept.get('pattern_types', ['concept'] * len(c_ids))
print(f"Concept: {len(c_ids):,}")

seq3 = torch.load(seq3_file, map_location='cpu')
s3_vecs, s3_ids, s3_subs = seq3['vectors'], seq3['ids'], seq3['subjects']
s3_types = seq3.get('pattern_types', ['sequence_3'] * len(s3_ids))
print(f"Sequence_3: {len(s3_ids):,}")

seq5 = torch.load(seq5_file, map_location='cpu')
s5_vecs, s5_ids, s5_subs = seq5['vectors'], seq5['ids'], seq5['subjects']
s5_types = seq5.get('pattern_types', ['sequence_5'] * len(s5_ids))
print(f"Sequence_5: {len(s5_ids):,}")

# Merge
all_vecs = torch.cat([c_vecs, s3_vecs, s5_vecs], dim=0)
all_ids = c_ids + s3_ids + s5_ids
all_subs = c_subs + s3_subs + s5_subs
all_types = c_types + s3_types + s5_types

print(f"\nTotal: {len(all_ids):,} (concept={len(c_ids)}, seq3={len(s3_ids)}, seq5={len(s5_ids)})")
size_mb = all_vecs.element_size() * all_vecs.numel() / 1024**2
print(f"Size: {size_mb:.1f} MB")

torch.save({
    'vectors': all_vecs,
    'ids': all_ids,
    'subjects': all_subs,
    'pattern_types': all_types,
}, output_file)

print(f"Done: {output_file}")
