#!/usr/bin/env python3
"""
构建序列模式 — Wikipedia 滑动窗口 → BGE 编码 → 存入 DragonField

Step 1: 从 zhwiki.db 读取文章，滑动窗口切分字符序列
Step 2: BGE 批量编码序列为 1024 维向量  
Step 3: 保存为 dragon_field_patterns.pt (概念模式 + 序列模式合并)

参数:
  --max-articles: 最大文章数 (默认50000, 内存友好)
  --window: 窗口大小 (默认3, 也支持5用于句级框架)
  --max-patterns: 序列模式总数上限 (默认300000)
  --concat: 是否与现有概念模式合并输出 (默认True)

输出: data/models/dragon_field_patterns.pt (合并后的完整模式缓存)
"""
import sys, os, time, argparse, sqlite3
import torch
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# BGE 模型路径 (本地缓存, 避免网络请求)
_MODEL_PATHS = [
    os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-large-zh/snapshots/b5d9f5c027e87b6f0b6fa4b614f8f9cdc45ce0e8"),
    os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/snapshots"),
]

def find_model():
    for p in _MODEL_PATHS:
        if os.path.isdir(p):
            return p
        # 尝试 glob snapshots
        base = os.path.dirname(p)
        if os.path.isdir(base):
            snaps = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
            if snaps:
                return os.path.join(base, snaps[0])
    raise FileNotFoundError("BGE model not found in cache. Run huggingface-cli download first.")


def sliding_windows(text: str, window: int = 3, step: int = 1, max_windows: int = 200):
    """从文本提取滑动窗口序列 (仅保留纯中文)"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    if len(chars) < window:
        return []
    windows = []
    for i in range(0, min(len(chars) - window + 1, max_windows), step):
        windows.append(''.join(chars[i:i+window]))
    return windows


def build_sequence_patterns(
    db_path: str = "data/wikipedia/zhwiki.db",
    max_articles: int = 50000,
    window: int = 3,
    max_patterns: int = 300000,
    concat_existing: bool = True,
    existing_cache: str = "data/models/dragon_field_patterns.pt",
    output_path: str = "data/models/dragon_field_patterns.pt",
):
    """主构建函数"""
    print("=" * 60)
    print(f"🧬 构建序列模式 (窗口={window}字, 上限={max_patterns}条)")
    print("=" * 60)

    # 1. 加载 BGE
    model_path = find_model()
    print(f"📖 加载 BGE 编码器: {model_path}")
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    print(f"   设备: {device}")

    # 2. 从 Wikipedia 读取文章 + 滑动窗口
    print(f"\n📊 读取 Wikipedia 文章 (上限 {max_articles} 篇)...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT title, text FROM articles WHERE char_count > 100 ORDER BY char_count DESC LIMIT ?",
        (max_articles,)
    )

    all_windows = []
    article_count = 0
    t0 = time.time()

    for row in cursor:
        text = row['text']
        windows = sliding_windows(text, window=window, step=1, max_windows=200)
        all_windows.extend(windows)
        article_count += 1

        if article_count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {article_count} 篇, {len(all_windows)} 窗口 "
                  f"({len(all_windows)/max(elapsed,1)/1000:.0f}k/s)")

    conn.close()
    print(f"\n   完成: {article_count} 篇 → {len(all_windows):,} 个窗口")

    # 3. 去重 + 按频率排序取 top + 过滤高频碎片
    print(f"\n🔍 去重 + 频率排序 + 过滤碎片...")
    window_counts = Counter(all_windows)

    # ★ 过滤高频碎片: 在过多文章出现的通用模式(维基元数据)
    max_freq = max(window_counts.values()) if window_counts else 1
    freq_threshold = max(3, max_freq * 0.05)  # 过滤 >5%最高频的
    boilerplate_prefixes = {'所屬', '間名', '現稱', '中的', '有的', '的一個', '於一'}
    boilerplate_suffixes = {'的', '為', '內'}

    filtered = []
    for w, c in window_counts.most_common(max_patterns * 2):
        if c > freq_threshold and c > 100:
            continue  # 过于高频的通用碎片
        # 过滤维基元数据碎片
        if any(w.startswith(p) for p in boilerplate_prefixes):
            continue
        if any(w.endswith(s) for s in boilerplate_suffixes) and len(w) <= 3:
            continue
        filtered.append((w, c))
        if len(filtered) >= max_patterns:
            break

    print(f"   唯一窗口: {len(window_counts):,} | 过滤后: {len(filtered):,}")
    sequences = [w for w, _ in filtered]
    frequencies = [c for _, c in filtered]

    # 4. BGE 批量编码
    print(f"\n🧠 BGE 编码 {len(sequences)} 条序列...")
    t0 = time.time()
    batch_size = 512
    vectors_list = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        with torch.no_grad():
            vecs = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
        vectors_list.append(vecs.cpu().to(torch.float16))

        if (i // batch_size) % 20 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(sequences)} ({i/len(sequences)*100:.0f}%) "
                  f"| {i/elapsed:.0f} seq/s")

    vectors = torch.cat(vectors_list, dim=0)
    elapsed = time.time() - t0
    print(f"   完成: {vectors.shape} | {elapsed:.1f}s | "
          f"{vectors.element_size()*vectors.numel()/1024**2:.1f} MB")

    # 5. 合并已存在的概念模式
    ids = list(range(len(sequences)))  # 序列模式用虚拟ID (负值)
    subjects = sequences
    types = [f'sequence_{window}'] * len(sequences)

    if concat_existing and os.path.exists(existing_cache) and existing_cache != output_path:
        print(f"\n🔗 合并现有概念模式: {existing_cache}")
        existing = torch.load(existing_cache, map_location='cpu')
        existing_vecs = existing['vectors']
        existing_ids = existing['ids']
        existing_subjects = existing['subjects']
        existing_types = existing.get('pattern_types', ['concept'] * len(existing_ids))

        print(f"   概念模式: {len(existing_ids):,} 条")
        print(f"   序列模式: {len(sequences):,} 条")

        # 合并
        all_vecs = torch.cat([existing_vecs, vectors], dim=0)
        all_ids = existing_ids + ids
        all_subjects = existing_subjects + subjects
        all_types = existing_types + types
    else:
        all_vecs = vectors
        all_ids = ids
        all_subjects = subjects
        all_types = types

    # 6. 保存
    print(f"\n💾 保存: {output_path}")
    print(f"   总模式: {len(all_ids):,}")
    print(f"   概念: {sum(1 for t in all_types if t == 'concept'):,}")
    print(f"   序列_3: {sum(1 for t in all_types if t == 'sequence_3'):,}")
    print(f"   序列_5: {sum(1 for t in all_types if t == 'sequence_5'):,}")
    print(f"   文件大小: {all_vecs.element_size() * all_vecs.numel() / 1024**2:.1f} MB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'vectors': all_vecs,
        'ids': all_ids,
        'subjects': all_subjects,
        'pattern_types': all_types,
    }, output_path)
    print(f"✅ 序列模式构建完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Wikipedia 序列模式")
    parser.add_argument("--max-articles", type=int, default=50000,
                       help="最大文章数 (默认50000)")
    parser.add_argument("--window", type=int, default=3,
                       help="窗口大小 (默认3)")
    parser.add_argument("--max-patterns", type=int, default=300000,
                       help="序列模式上限 (默认300000)")
    parser.add_argument("--no-concat", action="store_true",
                       help="不与现有概念模式合并")
    parser.add_argument("--db", type=str, default="data/wikipedia/zhwiki.db",
                       help="Wikipedia 数据库路径")
    parser.add_argument("--output", type=str,
                       default="data/models/dragon_field_patterns.pt",
                       help="输出路径")

    args = parser.parse_args()

    build_sequence_patterns(
        db_path=args.db,
        max_articles=args.max_articles,
        window=args.window,
        max_patterns=args.max_patterns,
        concat_existing=not args.no_concat,
        existing_cache="data/models/dragon_field_patterns.pt",
        output_path=args.output,
    )
