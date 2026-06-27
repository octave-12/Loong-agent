#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 场构建脚本 — 概念图 → Hopfield 模式矩阵
═══════════════════════════════════════════════════════

从 concept_graph.db 提取语义三元组 → BGE-large-zh 编码 → 写入场记忆。

过滤策略:
  - 排除噪音关系: POETIC_NEXT, HAS_PINYIN, POETIC_WITH, COOCCURS_IN
  - 排除低置信: conf < 0.3
  - 排除元标签: IS_A 中的 "中文词条" / "成语" 等

输出:
  data/models/dragon_field.safetensors  — 模式矩阵 (mmap可用)
  data/models/dragon_field_meta.json    — 元数据 (id/主体映射)

用法:
  python scripts/build_field.py                    # 全量构建 (需能访问 HuggingFace)
  HF_ENDPOINT=https://hf-mirror.com python scripts/build_field.py  # 国内镜像
  python scripts/build_field.py --dry-run          # 预览
"""

import sys
import os
import json
import time
import argparse
import logging
import sqlite3

import torch
import numpy as np

# ══ 项目路径 ══
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from loongpearl.core.dragon_field import DragonField

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('build_field')


# ── 配置 ──────────────────────────────────────────────────

# 需要保留的语义关系类型
SEMANTIC_RELATIONS = {
    'DEFINED_AS', 'IS_A', 'HAS', 'PART_OF', 'CAUSE',
    'COOCCURS_WITH', 'RELATED', 'FOLLOWS', 'OCCURS_IN',
    'PREVENTS',
}

# 需要排除的关系 (噪音)
EXCLUDED_RELATIONS = {
    'HAS_PINYIN',      # 拼音标注, 非语义
    'POETIC_NEXT',     # 诗词接龙, conf≈0.003
    'POETIC_WITH',     # 诗意关联
}

# 需要排除的 IS_A 元标签
META_LABELS = {"中文词条", "成语", "词语", "词条", "词汇", "汉字"}

# 置信度下限
MIN_CONFIDENCE = 0.3

# BGE 模型
BGE_MODEL = "BAAI/bge-large-zh"
EMBED_DIM = 1024
BATCH_SIZE = 512  # BGE 编码批大小

# 输出路径
OUTPUT_FIELD = os.path.join(PROJECT, 'data', 'models', 'dragon_field.safetensors')
OUTPUT_META = os.path.join(PROJECT, 'data', 'models', 'dragon_field_meta.json')
DB_PATH = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')


def load_triples(db_path: str) -> list:
    """从 SQLite 加载并过滤三元组"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # 统计总数
    total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    log.info(f"概念图总三元组: {total:,}")

    # 过滤查询
    placeholders = ','.join(['?'] * len(EXCLUDED_RELATIONS))
    query = f"""
        SELECT id, s, r, o, c
        FROM triples
        WHERE r NOT IN ({placeholders})
          AND c >= ?
    """

    params = list(EXCLUDED_RELATIONS) + [MIN_CONFIDENCE]
    cursor = conn.execute(query, params)

    triples = []
    excluded_meta = 0
    excluded_other = 0

    for row in cursor:
        rowid, s, r, o, c = row

        # 过滤 IS_A 元标签
        if r == 'IS_A' and o in META_LABELS:
            excluded_meta += 1
            continue

        # 过滤非语义关系
        if r not in SEMANTIC_RELATIONS and r not in ('DEFINED_AS',):
            excluded_other += 1
            continue

        triples.append({
            'id': rowid - 1,  # 转为 0-based
            'subject': s,
            'relation': r,
            'object': o,
            'confidence': c,
        })

    conn.close()
    log.info(
        f"有效语义三元组: {len(triples):,} "
        f"(过滤: 元标签{excluded_meta:,} 其他{excluded_other:,})"
    )
    return triples


def encode_triples(triples: list, device: str = 'cuda') -> torch.Tensor:
    """
    BGE 编码三元组 → 模式矩阵。

    编码文本: subject + " " + object (用空格连接，BGE 对空格分隔的短文本效果最好)
    """
    from sentence_transformers import SentenceTransformer

    log.info(f"加载 BGE 模型: {BGE_MODEL}")
    model = SentenceTransformer(BGE_MODEL, device=device)
    model.max_seq_length = 64  # 三元组文本很短

    total = len(triples)
    all_vectors = []

    log.info(f"开始编码 {total:,} 条三元组 (batch={BATCH_SIZE})...")
    t_start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = triples[batch_start:batch_end]

        # 构造编码文本
        texts = [f"{t['subject']} {t['object']}" for t in batch]

        # BGE 编码
        with torch.no_grad():
            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,  # L2 归一化
            )

        all_vectors.append(vectors.cpu())

        if (batch_start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t_start
            progress = batch_end / total * 100
            speed = batch_end / elapsed
            eta = (total - batch_end) / speed if speed > 0 else 0
            log.info(
                f"  进度: {progress:.0f}% ({batch_end:,}/{total:,}) "
                f"速度: {speed:.0f}条/s ETA: {eta:.0f}s"
            )

    elapsed = time.time() - t_start
    vectors = torch.cat(all_vectors, dim=0)
    log.info(f"编码完成: {vectors.shape} ({elapsed:.1f}s, {total/elapsed:.0f}条/s)")

    return vectors


def build_field(
    triples: list,
    vectors: torch.Tensor,
    output_path: str,
    meta_path: str,
):
    """创建 DragonField 并保存"""
    ids = [t['id'] for t in triples]
    subjects = [t['subject'] for t in triples]

    # 创建场
    field = DragonField(embed_dim=EMBED_DIM, beta=8.0)
    field.store_patterns(
        vectors.to(dtype=torch.float16),
        ids,
        subjects,
    )

    # 保存
    log.info(f"保存场到 {output_path}...")
    field.save(output_path)

    # 保存元数据 (轻量 JSON, 不含向量)
    meta = {
        'total_patterns': len(triples),
        'embed_dim': EMBED_DIM,
        'beta': 8.0,
        'model': BGE_MODEL,
        'relations': list(SEMANTIC_RELATIONS),
        'min_confidence': MIN_CONFIDENCE,
        'sample_triples': [
            {
                'subject': t['subject'],
                'relation': t['relation'],
                'object': t['object'],
                'confidence': t['confidence'],
            }
            for t in triples[:20]
        ],
    }

    # 统计各关系类型数量
    rel_counts = {}
    for t in triples:
        rel_counts[t['relation']] = rel_counts.get(t['relation'], 0) + 1
    meta['relation_counts'] = rel_counts

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 文件大小
    file_size_mb = os.path.getsize(output_path) / 1024**2
    log.info(f"元数据: {meta_path}")
    log.info(f"场文件: {output_path} ({file_size_mb:.1f} MB)")

    total_mb = vectors.element_size() * vectors.numel() / 1024**2
    log.info(f"模式数据: {total_mb:.1f} MB ({len(triples):,} × {EMBED_DIM})")
    log.info(f"各关系类型: {json.dumps(rel_counts, ensure_ascii=False)}")

    return field, meta


def main():
    parser = argparse.ArgumentParser(description="龙 场构建脚本")
    parser.add_argument(
        '--dry-run', action='store_true',
        help='只预览，不编码'
    )
    parser.add_argument(
        '--batch-size', type=int, default=BATCH_SIZE,
        help=f'BGE 编码批大小 (默认 {BATCH_SIZE})'
    )
    parser.add_argument(
        '--db', type=str, default=DB_PATH,
        help='concept_graph.db 路径'
    )
    parser.add_argument(
        '--output', type=str, default=OUTPUT_FIELD,
        help='输出场文件路径'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='编码设备'
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("龙 场构建 — 概念图 → Hopfield 模式矩阵")
    log.info("=" * 60)

    # 1. 加载三元组
    triples = load_triples(args.db)
    if not triples:
        log.error("没有有效三元组! 检查数据库和过滤条件。")
        return

    if args.dry_run:
        rel_counts = {}
        for t in triples:
            rel_counts[t['relation']] = rel_counts.get(t['relation'], 0) + 1
        log.info(f"[DRY RUN] 将编码 {len(triples):,} 条三元组")
        log.info(f"各关系类型: {json.dumps(rel_counts, ensure_ascii=False)}")
        log.info("没做实际操作。")
        return

    # 2. BGE 编码
    vectors = encode_triples(triples, device=args.device)

    # 3. 构建场
    field, meta = build_field(
        triples, vectors,
        args.output, OUTPUT_META,
    )

    log.info("✅ 场构建完成")
    log.info(f"   模式数: {field.num_patterns:,}")
    log.info(f"   维度: {field.embed_dim}")
    log.info(f"   数据类型: float16")


if __name__ == '__main__':
    main()
