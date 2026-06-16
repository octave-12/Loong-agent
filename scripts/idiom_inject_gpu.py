#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠成语 GPU 批量注入 (idiom_inject_gpu.py)
==========================================
利用 RTX 3060 12GB 将 29K 成语的字对关联批量注入能量景观。

硬件: RTX 3060 12GB VRAM | 20 核 CPU | 15GB RAM
策略: 全部字对一次加载到 GPU → 大 batch 梯度下降 → Hebbian 固化

用法:
    python idiom_inject_gpu.py                    # 全量注入
    python idiom_inject_gpu.py --batch 20000      # 自定义 batch 大小
    python idiom_inject_gpu.py --dry-run           # 只评估不写入
"""

import sys, os, json, time, math, argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="龙珠成语 GPU 批量注入")
    parser.add_argument('--batch', type=int, default=16000, help='GPU batch size')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--dry-run', action='store_true', help='只评估不保存')
    parser.add_argument('--idiom-limit', type=int, default=0, help='限制成语数量(0=全部)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🐉 龙珠成语 GPU 注入")
    print(f"   设备: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"   Batch: {args.batch} | Epochs: {args.epochs} | LR: {args.lr}")
    print()
    
    # ── 1. 加载 ──
    print("加载字场...")
    t0 = time.time()
    field = HanziAnchorField.load(
        os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
        freeze=True
    )
    
    print("加载能量景观 → GPU...")
    ls = FreqEnergyLandscape.load(
        os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    )
    ls = ls.to(device)
    ls.train()
    
    print("加载成语...")
    with open(os.path.join(PROJECT, 'data/dicts/idioms.json'), encoding='utf-8') as f:
        idioms = json.load(f)
    
    if args.idiom_limit > 0:
        idioms = idioms[:args.idiom_limit]
    
    load_time = time.time() - t0
    print(f"   加载耗时: {load_time:.1f}s\n")
    
    # ── 2. 构建字对 → GPU 张量 ──
    print(f"构建字对 (GPU 加速)...")
    t0 = time.time()
    
    anchors_cpu = field.anchors  # (94117, 1024) on CPU
    char_to_idx = field._char_to_idx
    
    # 快速构建索引对列表
    pair_indices = []
    skipped = 0
    for idiom in idioms:
        chars = list(idiom)
        if len(chars) != 4:
            skipped += 1
            continue
        try:
            idxs = [char_to_idx[ch] for ch in chars]
        except KeyError:
            skipped += 1
            continue
        # 三对: (0,1), (1,2), (2,3)
        pair_indices.extend([(idxs[0], idxs[1]), (idxs[1], idxs[2]), (idxs[2], idxs[3])])
    
    total_pairs = len(pair_indices)
    print(f"   有效成语: {len(idioms) - skipped} | 字对总数: {total_pairs}")
    
    # 转为 GPU 张量 (total_pairs, 2) — int64 索引
    pair_tensor = torch.tensor(pair_indices, dtype=torch.long, device='cpu')
    
    # 预计算所有中点向量 → GPU
    print(f"   预计算中点向量 ({total_pairs} × 1024)...")
    idx_a = pair_tensor[:, 0]
    idx_b = pair_tensor[:, 1]
    
    # 分批搬运到 GPU（避免一次性占用太多 CPU→GPU 带宽）
    chunk_size = 50000
    mid_chunks = []
    for i in range(0, total_pairs, chunk_size):
        end = min(i + chunk_size, total_pairs)
        chunk_a = anchors_cpu[idx_a[i:end]].to(device)
        chunk_b = anchors_cpu[idx_b[i:end]].to(device)
        mid_chunk = (chunk_a + chunk_b) / 2.0
        mid_chunks.append(mid_chunk)
    
    all_mids = torch.cat(mid_chunks, dim=0)  # (N, 1024) on GPU
    del mid_chunks, anchors_cpu
    torch.cuda.empty_cache()
    
    build_time = time.time() - t0
    vram_used = all_mids.element_size() * all_mids.numel() / 1e6
    print(f"   中点张量: {all_mids.shape}, {vram_used:.0f}MB VRAM")
    print(f"   构建耗时: {build_time:.1f}s\n")
    
    # ── 3. 评估注入前 ──
    print("注入前评估...")
    with torch.no_grad():
        sample_size = min(1000, total_pairs)
        indices = torch.randperm(total_pairs, device=device)[:sample_size]
        e_before = ls(all_mids[indices]).mean().item()
        # 随机点对比
        random_vecs = torch.randn(sample_size, 1024, device=device)
        random_vecs = random_vecs / random_vecs.norm(dim=1, keepdim=True)
        e_random = ls(random_vecs).mean().item()
    
    print(f"   已知字对能量: {e_before:+.2f}")
    print(f"   随机向量能量: {e_random:+.2f}")
    print(f"   分离度: {e_random - e_before:.2f}\n")
    
    if args.dry_run:
        print("🔍 干运行模式，不修改模型")
        return
    
    # ── 4. GPU 批量梯度下降 ──
    print(f"GPU 训练 ({args.epochs} epochs × {total_pairs//args.batch + 1} batches)...")
    optimizer = torch.optim.Adam(ls.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    target = torch.tensor([-15.0], device=device)  # 目标能量：-15
    
    t_train = time.time()
    
    for epoch in range(args.epochs):
        # 每 epoch 打乱顺序
        perm = torch.randperm(total_pairs, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, total_pairs, args.batch):
            batch_idx = perm[i:i + args.batch]
            batch_mids = all_mids[batch_idx]
            
            energies = ls(batch_mids)
            loss = criterion(energies, target.expand_as(energies))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # 评估
        with torch.no_grad():
            e_now = ls(all_mids[indices]).mean().item()
        
        elapsed = time.time() - t_train
        print(f"   Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
              f"能量={e_now:+.2f} | {elapsed:.0f}s")
    
    train_time = time.time() - t_train
    
    # ── 5. 注入后评估 ──
    print(f"\n注入后评估...")
    with torch.no_grad():
        e_after = ls(all_mids[indices]).mean().item()
    
    print(f"   注入前: {e_before:+.2f}")
    print(f"   注入后: {e_after:+.2f}")
    print(f"   降低: {e_before - e_after:.2f}")
    print(f"   训练耗时: {train_time:.1f}s")
    print(f"   吞吐量: {total_pairs * args.epochs / train_time:.0f} 对/秒")
    
    # ── 6. 保存 ──
    save_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
    backup_path = save_path + '.backup'
    
    # 备份原模型
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(save_path, backup_path)
        print(f"\n💾 已备份原模型: {backup_path}")
    
    torch.save({
        'model_state_dict': ls.state_dict(),
        'dim': 1024,
        'version': 'idiom_injected_v2',
        'idioms_injected': len(idioms),
        'pairs_learned': total_pairs,
        'epochs': args.epochs,
        'energy_before': e_before,
        'energy_after': e_after,
    }, save_path)
    print(f"💾 已保存: {save_path}")
    
    total_time = time.time() - load_time - build_time - train_time + load_time + build_time + train_time
    print(f"\n✅ 完成! 总耗时: {load_time + build_time + train_time:.1f}s")


if __name__ == '__main__':
    main()
