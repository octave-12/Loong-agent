#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠 概念图 → 能量景观 批量注入
═══════════════════════════════════════════════════════
将概念图 193 万三元组中的高价值字对注入能量景观。

用法:
    python scripts/inject_concept_graph.py                    # 全量 (conf≥0.5)
    python scripts/inject_concept_graph.py --min-conf 0.7     # 高置信度
    python scripts/inject_concept_graph.py --max-pairs 50000  # 限数量
    python scripts/inject_concept_graph.py --dry-run          # 只评估不写入
"""

import sys, os, json, time, argparse
from collections import defaultdict

# 确保项目路径
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import torch
import numpy as np

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.concept_graph import ConceptGraph
from loongpearl.learning.learner import DragonBallLearner

# 适合提取字对的关系类型
PAIR_RELATIONS = {
    'COOCCURS_WITH',  # subject/object 已是单汉字
    'IS_A',
    'RELATED',
    'PART_OF',
    'HAS',
    'CAUSE',
    'COOCCURS_IN',
    'POETIC_WITH',
}


def extract_pairs_from_triples(triples, char_to_idx: dict,
                                min_conf: float = 0.5,
                                max_pairs: int = 200000) -> list:
    """
    从三元组（dict或list）中提取字对索引。

    支持两种格式:
      - Triple 对象 (.subject, .relation, .object, .confidence)
      - 原始 dict ('s', 'r', 'o', 'c')

    Returns: [(ia, ib), ...] 去重字对索引
    """
    seen = set()
    pairs = []
    stats = defaultdict(int)

    # 统一迭代器
    if isinstance(triples, dict):
        items = triples.values()
    else:
        items = triples

    for t in items:
        # 兼容 Triple 对象和 dict
        if hasattr(t, 'relation'):
            rel = t.relation
            conf = t.confidence
            subj = t.subject
            obj = t.object
        else:
            rel = t.get('r', '')
            conf = t.get('c', 0)
            subj = t.get('s', '')
            obj = t.get('o', '')

        if not subj or not obj:
            continue

        # 提取首汉字
        ca = subj[0] if subj else ''
        cb = obj[0] if obj else ''

        if not ('\u4e00' <= ca <= '\u9fff'):
            # 多字符主题：尝试找第一个汉字
            for ch in subj:
                if '\u4e00' <= ch <= '\u9fff':
                    ca = ch
                    break
        if not ('\u4e00' <= cb <= '\u9fff'):
            for ch in obj:
                if '\u4e00' <= ch <= '\u9fff':
                    cb = ch
                    break

        ia = char_to_idx.get(ca)
        ib = char_to_idx.get(cb)

        if ia is None or ib is None:
            continue
        if ia == ib:
            continue  # 跳过自环

        key = (min(ia, ib), max(ia, ib))
        if key not in seen:
            seen.add(key)
            pairs.append((ia, ib))
            stats[rel] += 1

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    return pairs, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="概念图→能量景观 批量注入")
    parser.add_argument('--min-conf', type=float, default=0.5,
                        help='最低置信度 (default: 0.5)')
    parser.add_argument('--max-pairs', type=int, default=200000,
                        help='最大字对数 (default: 200000)')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='GPU batch size (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率 (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数 (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只评估不写入')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    args = parser.parse_args()

    # 检测可用设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU")
        args.device = 'cpu'

    device = torch.device(args.device)
    print(f"🐉 概念图 → 能量景观 批量注入")
    print(f"   设备: {device}")
    print(f"   置信度阈值: {args.min_conf}")
    print(f"   最大字对: {args.max_pairs}")
    print()

    # ── 注入锁: 通知守护进程本轮暂停，防止竞态写模型 ──
    lock_path = os.path.join(PROJECT, 'data', 'runtime', 'inject.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, 'w') as lf:
        lf.write(str(os.getpid()))
    print(f"🔒 写入注入锁: {lock_path}")

    try:
        # ── 1. 加载模型 ──
        print("加载字场...")
        t0 = time.time()
        field = HanziAnchorField.load(
            os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
            freeze=True
        )

        print("加载能量景观...")
        landscape = FreqEnergyLandscape.load(
            os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        )
        landscape = landscape.to(device)
        landscape.train()

        print("加载概念图...")
        cg_path = os.path.join(PROJECT, 'data/models/concept_graph.json')
        cg = ConceptGraph(field, landscape)
        cg.load(cg_path)

        print("初始化学习器...")
        learner = DragonBallLearner(landscape, field)

        load_time = time.time() - t0
        print(f"   加载耗时: {load_time:.1f}s\n")

        # ── 2. 提取字对 ──
        print(f"从概念图提取字对 (关系: {sorted(PAIR_RELATIONS)})...")
        t0 = time.time()

        pairs, pair_stats = extract_pairs_from_triples(
            cg.triples,
            field._char_to_idx,
            min_conf=args.min_conf,
            max_pairs=args.max_pairs,
        )

        extract_time = time.time() - t0
        print(f"   提取完成: {len(pairs)} 对 ({extract_time:.1f}s)")
        for rel, count in sorted(pair_stats.items(), key=lambda x: -x[1]):
            print(f"     {rel}: {count} 对")
        print()

        if not pairs:
            print("❌ 未找到符合条件的三元组，退出")
            return

        # ── 3. 注入前评估 ──
        print("注入前评估...")
        device_cpu = torch.device('cpu')
        anchors = field.anchors.to(device_cpu)

        # 采样评估
        sample_size = min(2000, len(pairs))
        import random
        sample = random.sample(pairs, sample_size)

        idx_a = torch.tensor([p[0] for p in sample], device=device_cpu)
        idx_b = torch.tensor([p[1] for p in sample], device=device_cpu)
        mids_cpu = (anchors[idx_a] + anchors[idx_b]) / 2.0
        mids = mids_cpu.to(device)

        landscape_eval = landscape.to(device)
        with torch.no_grad():
            e_before = landscape_eval(mids).mean().item()

        # 随机点对比
        random_vecs = torch.randn(sample_size, 1024, device=device)
        random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)
        with torch.no_grad():
            e_random = landscape_eval(random_vecs).mean().item()

        print(f"   已知字对能量: {e_before:+.2f}")
        print(f"   随机向量能量: {e_random:+.2f}")
        separation_before = e_random - e_before
        print(f"   分离度: {separation_before:.2f}\n")

        if args.dry_run:
            print("🔍 干运行模式，不修改模型")
            return

        # ── 4. 批量注入 ──
        print(f"开始注入 ({args.epochs} 轮, LR={args.lr}, batch={args.batch_size})...")
        t0 = time.time()

        total = len(pairs)
        idx_a = torch.tensor([p[0] for p in pairs])
        idx_b = torch.tensor([p[1] for p in pairs])

        # 低学习率 + 梯度裁剪，防止权重爆炸
        optimizer = torch.optim.Adam(landscape.parameters(), lr=args.lr)
        max_grad_norm = 1.0  # 梯度裁剪阈值

        for epoch in range(args.epochs):
            perm = torch.randperm(total)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, total, args.batch_size):
                batch_idx = perm[i:i + args.batch_size]
                batch_a = idx_a[batch_idx]
                batch_b = idx_b[batch_idx]

                # CPU 上计算中点 → GPU
                mids_batch_cpu = (anchors[batch_a] + anchors[batch_b]) / 2.0
                mids_batch = mids_batch_cpu.to(device)

                optimizer.zero_grad()

                # Hebbian 损失: 让字对能量尽量低，但要加 L2 正则防止权重爆炸
                energy = landscape(mids_batch)
                loss = energy.mean()
                # L2 正则化
                l2_reg = 0.0
                for param in landscape.parameters():
                    l2_reg += (param ** 2).sum()
                loss = loss + 1e-6 * l2_reg
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(landscape.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if n_batches % 20 == 0:
                    progress = (i + len(batch_idx)) / total * 100
                    print(f"  Epoch {epoch+1}: {progress:.0f}% | "
                          f"loss={loss.item():.4f} | "
                          f"avg_energy={energy.mean().item():+.2f}",
                          flush=True)

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1} 完成: avg_loss={avg_loss:.4f}")

        inject_time = time.time() - t0
        print(f"\n   注入耗时: {inject_time:.1f}s")

        # ── 5. 注入后评估 ──
        print("\n注入后评估...")
        with torch.no_grad():
            e_after = landscape_eval(mids).mean().item()

        separation_after = e_random - e_after
        print(f"   已知字对能量: {e_after:+.2f}")
        print(f"   随机向量能量: {e_random:+.2f}")
        print(f"   分离度: {separation_before:.2f} → {separation_after:.2f} "
              f"{'↑' if separation_after > separation_before else '↓'}"
              f"{abs(separation_after - separation_before):.2f})")

        # ── 6. 保存 ──
        landscape = landscape.to('cpu')
        model_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        landscape.save(model_path)
        print(f"\n💾 模型已保存: {model_path}")

        # 备份
        backup_path = model_path + '.post_cg_inject'
        landscape.save(backup_path)
        print(f"💾 备份已保存: {backup_path}")

        print(f"\n✅ 完成: {len(pairs)} 字对注入，分离度 {separation_before:.1f}→{separation_after:.1f}")

    finally:
        # ── 清理注入锁 ──
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print(f"🔓 注入锁已释放: {lock_path}")


if __name__ == '__main__':
    main()
