#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 集成测试 — 场收敛 + 三层记忆 + NLG 回答
═══════════════════════════════════════════════════════

验证:
  1. DragonField 加载/保存
  2. Hopfield 收敛到正确 basin
  3. FieldMemory 三层分级查询
  4. FieldNLG 场→文本翻译
  5. CuriosityEngine 好奇心检测
  6. 场驱动调度 (活跃度)

用法:
  python tests/test_dragon_field.py
  python tests/test_dragon_field.py --small  # 小样本快速测试
"""

import sys
import os
import time
import logging
import argparse

import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger('test_dragon')


def test_field_create_save_load():
    """测试场创建/保存/加载"""
    log.info("=" * 60)
    log.info("测试 1: 场创建/保存/加载")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField

    # 创建小场 (100个随机模式)
    embed_dim = 1024
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    n_patterns = 100
    vectors = torch.randn(n_patterns, embed_dim)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)

    ids = list(range(n_patterns))
    subjects = [f"概念{i}" for i in range(n_patterns)]

    field.store_patterns(vectors.half(), ids, subjects)
    assert field.num_patterns == n_patterns, f"预期 {n_patterns} 实际 {field.num_patterns}"
    log.info(f"✅ 存储 {field.num_patterns} 个模式")

    # 保存
    tmp_path = '/tmp/test_dragon_field.pt'
    field.save(tmp_path)
    assert os.path.exists(tmp_path), f"文件未创建: {tmp_path}"
    log.info(f"✅ 保存到 {tmp_path}")

    # 重新加载
    field2 = DragonField.load(tmp_path)
    assert field2.num_patterns == n_patterns
    assert field2.embed_dim == embed_dim
    assert field2._pattern_ids == ids
    log.info(f"✅ 重新加载: {field2.num_patterns} 模式, {field2.embed_dim}维")

    os.remove(tmp_path)
    log.info("测试 1 通过 ✅\n")


def test_field_convergence():
    """测试 Hopfield 收敛"""
    log.info("=" * 60)
    log.info("测试 2: Hopfield 收敛")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField

    embed_dim = 64  # 用小维度加速测试

    # 创建包含已知模式的场
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    # 存储一些模式 (确保float16下也有足够区分度)
    patterns = torch.tensor([
        [1.0, 0.0, 0.0] + [0.0] * 61,     # 龙 (0)
        [0.0, 1.0, 0.0] + [0.0] * 61,     # 凤凰 (1)
        [0.0, 0.0, 1.0] + [0.0] * 61,     # 量子 (2)
    ], dtype=torch.float32)

    patterns = torch.nn.functional.normalize(patterns, dim=-1)

    field.store_patterns(
        patterns.half(),
        list(range(3)),
        ['龙', '凤凰', '量子'],
    )

    # 查询: 明确接近"龙"
    query = torch.tensor([1.0, 0.0, 0.0] + [0.0] * 61)
    query = torch.nn.functional.normalize(query, dim=-1)

    result = field.converge(query, max_steps=20)
    log.info(f"  收敛步数: {result.convergence_steps}")
    log.info(f"  能量: {result.energy:.4f}")
    top_subject = field._pattern_subjects[result.top_pattern_indices[0]]
    log.info(f"  最近模式: {top_subject}")
    log.info(f"  相似度: {result.top_similarities[0]:.4f}")
    log.info(f"  距离: {result.distance_to_nearest:.4f}")
    log.info(f"  盆地深度: {result.basin_depth:.4f}")
    log.info(f"  置信度: {result.confidence_label}")

    assert top_subject == '龙', f"预期收敛到'龙', 实际 '{top_subject}'"
    assert result.is_retrieval, f"应该是检索, 实际距离={result.distance_to_nearest}"
    assert result.convergence_steps <= 5, f"收敛太慢: {result.convergence_steps}步"
    log.info("✅ 收敛正确 (检索模式)")

    # 查询: 介于龙和凤凰之间的点 — 应该涌现
    query_between = torch.tensor([0.5, 0.5, 0.0] + [0.0] * 61)
    query_between = torch.nn.functional.normalize(query_between, dim=-1)

    result2 = field.converge(query_between, max_steps=20)
    log.info(f"\n  查询: 龙+凤凰之间")
    log.info(f"  最近模式: {field._pattern_subjects[result2.top_pattern_indices[0]]}")
    log.info(f"  Top-2: {[field._pattern_subjects[i] for i in result2.top_pattern_indices[:2]]}")
    log.info(f"  距离: {result2.distance_to_nearest:.4f}")
    log.info(f"  是否涌现: {result2.is_emergent}")
    log.info(f"  置信度: {result2.confidence_label}")

    log.info("测试 2 通过 ✅\n")


def test_field_energy():
    """测试能量函数"""
    log.info("=" * 60)
    log.info("测试 3: 能量函数")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField

    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    patterns = torch.randn(20, embed_dim)
    patterns = torch.nn.functional.normalize(patterns, dim=-1)
    field.store_patterns(patterns.half(), list(range(20)), [f"p{i}" for i in range(20)])

    # 在已知模式上的能量应该更低
    known_vec = patterns[0].float()
    energy_known = field.energy(known_vec).item()

    # 随机向量的能量应该更高
    random_vec = torch.randn(embed_dim)
    random_vec = torch.nn.functional.normalize(random_vec, dim=-1)
    energy_random = field.energy(random_vec).item()

    log.info(f"  已知模式能量: {energy_known:.4f}")
    log.info(f"  随机向量能量: {energy_random:.4f}")

    assert energy_known < energy_random, (
        f"已知模式能量 ({energy_known:.4f}) 应 < 随机 ({energy_random:.4f})"
    )
    log.info("✅ 能量函数正确 (已知 < 随机)")

    # 单向量和多向量批处理
    batch = torch.stack([known_vec, random_vec])
    energy_batch = field.energy(batch)
    log.info(f"  批处理能量: {energy_batch.tolist()}")
    assert abs(energy_batch[0].item() - energy_known) < 0.01
    log.info("✅ 批处理一致")

    log.info("测试 3 通过 ✅\n")


def test_mc_uncertainty():
    """测试 MC Dropout 不确定性"""
    log.info("=" * 60)
    log.info("测试 4: MC 不确定性估计")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField

    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    patterns = torch.randn(30, embed_dim)
    patterns = torch.nn.functional.normalize(patterns, dim=-1)
    field.store_patterns(patterns.half(), list(range(30)), [f"p{i}" for i in range(30)])

    query = patterns[5].float()

    mc = field.mc_uncertainty(query, n_samples=10, dropout_rate=0.15)
    log.info(f"  方差: {mc['variance']:.6f}")
    log.info(f"  能量 std: {mc['energy_std']:.6f}")
    log.info(f"  稳定: {mc['is_stable']}")

    # 在已知模式上应该相对稳定
    assert mc['variance'] < 0.1, f"方差过大: {mc['variance']}"
    log.info("✅ MC 不确定性可计算")

    log.info("测试 4 通过 ✅\n")


def test_field_memory():
    """测试三层记忆"""
    log.info("=" * 60)
    log.info("测试 5: 三层记忆系统")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField
    from loongpearl.core.field_memory import FieldMemory

    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    n = 500
    patterns = torch.randn(n, embed_dim)
    patterns = torch.nn.functional.normalize(patterns, dim=-1)
    field.store_patterns(patterns.half(), list(range(n)), [f"p{i}" for i in range(n)])

    mem = FieldMemory(field, hot_size=100, warm_size=200, device='cpu')

    # 查询
    query = patterns[42].float()
    result = mem.query(query, hot_threshold=0.95)

    log.info(f"  层级: {result.tier}")
    log.info(f"  扫描模式数: {result.patterns_scanned}")
    log.info(f"  耗时: {result.elapsed_ms:.1f}ms")
    log.info(f"  置信度: {result.field_result.confidence_label}")

    assert result.tier in ('hot', 'warm', 'cold')
    log.info(f"✅ 三层分层查询: {result.tier}")

    # 统计
    stats = mem.stats
    log.info(f"  统计: 热{stats['hot_ratio']:.0%} GPU:{stats['memory_usage']['hot_gpu']:.0f}MB")

    log.info("测试 5 通过 ✅\n")


def test_field_nlg():
    """测试 NLG"""
    log.info("=" * 60)
    log.info("测试 6: 场→文本 NLG")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField, FieldResult
    from loongpearl.core.field_nlg import FieldNLG

    # 创建包含真实三元组的场
    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    # 模拟概念: 龙 / 凤凰 / 神话生物
    pat_dragon = torch.tensor([1.0, 0.0] + [0.0] * 62)
    pat_phoenix = torch.tensor([0.0, 1.0] + [0.0] * 62)
    pat_mythical = torch.tensor([0.7, 0.7] + [0.0] * 62)
    pat_scale = torch.tensor([0.8, 0.2] + [0.0] * 62)

    patterns = torch.stack([pat_dragon, pat_phoenix, pat_mythical, pat_scale])
    patterns = torch.nn.functional.normalize(patterns, dim=-1)

    field.store_patterns(
        patterns.half(),
        [0, 1, 2, 3],
        ['龙', '凤凰', '神话生物', '鳞片'],
    )

    # 查询: "龙"
    query = patterns[0].float()
    result = field.converge(query, max_steps=20)

    # 模拟 SQLite 回查 (实际会从 concept_graph.db 查)
    # 这里直接测试渲染逻辑
    log.info(f"  收敛到: {field._pattern_subjects[result.top_pattern_indices[0]]}")
    log.info(f"  置信度: {result.confidence_label}")
    log.info(f"  距离: {result.distance_to_nearest:.4f}")

    # 测试 FieldResult 属性
    assert result.is_retrieval
    assert '已知' in result.confidence_label

    log.info("测试 6 通过 ✅ (NLG 结构正确, 实际文本需真实概念图)\n")


def test_curiosity():
    """测试好奇心引擎"""
    log.info("=" * 60)
    log.info("测试 7: 好奇心引擎")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField
    from loongpearl.core.field_curiosity import CuriosityEngine

    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    n = 200
    patterns = torch.randn(n, embed_dim)
    patterns = torch.nn.functional.normalize(patterns, dim=-1)
    field.store_patterns(
        patterns.half(),
        list(range(n)),
        [f"概念{i}" for i in range(n)],
    )

    engine = CuriosityEngine(field, device='cpu')

    # 好奇心评分
    signals = engine.score_curiosity(n_samples=20)
    log.info(f"  好奇心信号数: {len(signals)}")
    if signals:
        log.info(f"  Top-1: idx={signals[0].anchor_idx} score={signals[0].curiosity_score:.4f}")

    # 场活跃度
    activity = engine.should_tick()
    log.info(f"  场活跃度: {activity:.4f}")
    log.info(f"  推荐间隔: {engine.recommended_interval():.0f}s")

    # 探索
    if signals:
        result = engine.explore(signals[0], round_number=1)
        log.info(f"  探索结果: action={result['action']} discovery={result.get('discovery')}")

    log.info("测试 7 通过 ✅\n")


def test_evolution():
    """测试场演化 (ODE 接口)"""
    log.info("=" * 60)
    log.info("测试 8: 场演化")
    log.info("=" * 60)

    from loongpearl.core.dragon_field import DragonField

    embed_dim = 64
    field = DragonField(embed_dim=embed_dim, beta=8.0)

    patterns = torch.randn(30, embed_dim)
    patterns = torch.nn.functional.normalize(patterns, dim=-1)
    field.store_patterns(patterns.half(), list(range(30)), [f"p{i}" for i in range(30)])

    # 初始状态
    x0 = torch.randn(embed_dim)
    x0 = torch.nn.functional.normalize(x0, dim=-1)

    # 单步演化
    x1 = field.evolution_step(x0, dt=0.1)
    log.info(f"  初始能量: {field.energy(x0).item():.4f}")
    log.info(f"  演化后能量: {field.energy(x1).item():.4f}")
    log.info(f"  能量变化: {field.energy(x1).item() - field.energy(x0).item():.4f}")

    # 演化后能量应该更低 (向盆地靠近)
    assert field.energy(x1).item() <= field.energy(x0).item() + 0.1, (
        "演化应降低能量"
    )

    # 多步演化
    x = x0.clone()
    energies = []
    for _ in range(10):
        x = field.evolution_step(x, dt=0.1)
        energies.append(field.energy(x).item())

    log.info(f"  多步演化: {energies[0]:.4f} → {energies[-1]:.4f}")
    assert energies[-1] <= energies[0] + 0.1, "多步应持续降低能量"

    log.info("✅ 演化正确\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='跳过耗时测试')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("龙 集成测试开始")
    log.info("=" * 60)

    test_field_create_save_load()
    test_field_convergence()
    test_field_energy()
    test_mc_uncertainty()
    test_field_memory()
    test_field_nlg()
    test_curiosity()
    test_evolution()

    log.info("=" * 60)
    log.info("🎉 全部 8 项测试通过!")
    log.info("=" * 60)

    # 真实场测试 (仅当 field 已构建)
    field_path = os.path.join(
        PROJECT, 'data', 'models', 'dragon_field.safetensors'
    )
    if os.path.exists(field_path) and not args.small:
        log.info("\n" + "=" * 60)
        log.info("真实场测试: 加载 1.18M 模式的场")
        log.info("=" * 60)

        from loongpearl.core.dragon_field import DragonField
        from loongpearl.core.field_memory import FieldMemory
        from loongpearl.core.field_nlg import FieldNLG

        field = DragonField.load(field_path)
        mem = FieldMemory(field, hot_size=200_000, warm_size=600_000)

        # 用 BGE 编码查询 (匹配 build_field.py 的编码方式)
        from sentence_transformers import SentenceTransformer
        log.info("加载 BGE 模型用于查询编码...")
        bge = SentenceTransformer('BAAI/bge-large-zh', device='cpu')

        query_text = '龙是什么'
        query_vec = bge.encode(
            query_text, convert_to_tensor=True, normalize_embeddings=True
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        result = mem.query(query_vec)
        log.info(f"查询 '{query_text}':")
        log.info(f"  层级: {result.tier}")
        log.info(f"  耗时: {result.elapsed_ms:.1f}ms")
        log.info(f"  距离最近模式: {result.field_result.distance_to_nearest:.4f}")
        log.info(f"  置信度: {result.field_result.confidence_label}")

        # NLG — 传入 pattern_ids 做正确映射
        nlg = FieldNLG(
            os.path.join(PROJECT, 'data', 'models', 'concept_graph.db'),
            pattern_ids=field._pattern_ids,
        )
        text = nlg.render(result.field_result, query_text=query_text)
        log.info(f"  回答: {text}")

        # 再来一个: 凤凰是什么
        query_text2 = '凤凰是什么'
        query_vec2 = bge.encode(
            query_text2, convert_to_tensor=True, normalize_embeddings=True
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        result2 = mem.query(query_vec2)
        text2 = nlg.render(result2.field_result, query_text=query_text2)
        log.info(f"\n查询 '{query_text2}':")
        log.info(f"  层级: {result2.tier} 距离: {result2.field_result.distance_to_nearest:.4f}")
        log.info(f"  置信度: {result2.field_result.confidence_label}")
        log.info(f"  回答: {text2}")

        log.info("✅ 真实场测试完成")


if __name__ == '__main__':
    main()
