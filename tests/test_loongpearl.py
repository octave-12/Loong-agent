#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠端到端测试（test_loongpearl.py）
=====================================
测试龙珠系统的全部核心功能：初始化、自知无知、知识查询、汉字推理。
"""

import sys
import os
import time

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/d/soso/projects/Loong-agent/Loong-pearl")

from loongpearl.interaction.engine import LoongPearl, QueryResult


def print_section(title: str):
    """打印测试段落标题"""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def test_initialization():
    """测试1: 初始化龙珠"""
    print_section("测试1: 初始化龙珠")
    
    t0 = time.time()
    loongpearl = LoongPearl()
    loongpearl.initialize(verbose=True)
    elapsed = time.time() - t0
    
    print(f"\n  ✅ 初始化成功 ({elapsed:.1f}秒)")
    print(f"     字场: {loongpearl.zichang.num_hanzi} 汉字")
    print(f"     维度: {loongpearl.zichang.embed_dim}")
    
    return loongpearl


def test_self_ignorance(loongpearl: LoongPearl):
    """测试2: 自知无知检测"""
    print_section("测试2: 自知无知检测")
    
    test_cases = [
        ("量子计算机", "应检测为未知（远离任何锚点区域）"),
        ("人工智能", "应检测为已知或未知（取决于播种覆盖）"),
        ("数据结构", "应检测为已知或未知"),
        ("xyzzy_not_a_real_concept_12345", "应检测为未知（无意义文本）"),
    ]
    
    for question, expected in test_cases:
        # 不做 auto_learn，纯检测
        result = loongpearl.query(question, auto_learn=False, verbose=False)
        status = "✅已知" if result.is_known else "❓未知"
        print(f"  {status}  \"{question}\"")
        print(f"     置信度: {result.confidence:.3f} | {result.diagnosis}")
        if result.nearest_chars:
            print(f"     最近字: {', '.join(result.nearest_chars[:3])}")


def test_knowledge_query(loongpearl: LoongPearl):
    """测试3: 知识查询（已知领域）"""
    print_section("测试3: 知识查询")
    
    # 使用已播种的高频字相关概念
    test_questions = [
        "火",
        "水",
        "人工智能",
        "机器学习",
    ]
    
    for q in test_questions:
        result = loongpearl.query(q, auto_learn=False, verbose=False)
        status = "✅已知" if result.is_known else "❓未知"
        print(f"  {status}: \"{q}\"")
        print(f"     置信度: {result.confidence:.3f}")
        print(f"     答案: {result.answer_text}")
        if result.is_known:
            print(f"     最近字: {result.nearest_chars}")
            print(f"     能量: {result.energy:.2f} | 步数: {result.steps} | "
                  f"收敛: {result.converged}")


def test_char_reasoning(loongpearl: LoongPearl):
    """测试4: 汉字间推理"""
    print_section("测试4: 汉字间推理")
    
    test_pairs = [
        ("火", "水"),    # 反义/对立
        ("大", "小"),    # 反义
        ("日", "月"),    # 相关（天体）
        ("人", "机"),    # 现代相关（人机）
    ]
    
    for a, b in test_pairs:
        result = loongpearl.reason_between(a, b, steps=50)
        
        if 'error' in result:
            print(f"  ❌ {a}↔{b}: {result['error']}")
            continue
        
        converged = "✅" if result['converged_to_target'] else "❌"
        print(f"  {converged} {a}↔{b}: "
              f"收敛到 {result['nearest_chars'][:3]}, "
              f"能垒={result['path_barrier']:.3f}, "
              f"步数={result['steps']}")


def test_find_nearest(loongpearl: LoongPearl):
    """测试5: 快速汉字检索"""
    print_section("测试5: 快速汉字检索")
    
    texts = ["太阳", "计算机", "爱情", "算法"]
    
    for text in texts:
        chars = loongpearl.find_nearest_chars(text, k=5)
        top = [f"{ch}({sim:.2f})" for ch, sim in chars[:5]]
        print(f"  \"{text}\" → {' | '.join(top)}")


def test_auto_learn(loongpearl: LoongPearl):
    """测试6: 自动学习（触发 Ollama）"""
    print_section("测试6: 自动学习（Ollama）")
    
    # 用一个不太可能被播种的概念测试自动学习
    novel_concept = "量子纠缠"
    
    print(f"  查询未知概念: \"{novel_concept}\"（自动学习模式）")
    result = loongpearl.query(novel_concept, auto_learn=True, verbose=True)
    
    print(f"\n  最终结果: {result}")
    print(f"  答案: {result.answer_text}")
    print(f"  学习次数: {loongpearl.total_learned}")


def test_stats(loongpearl: LoongPearl):
    """测试7: 统计报告"""
    print_section("测试7: 统计报告")
    
    stats = loongpearl.get_stats()
    
    print(f"  总查询次数: {stats['total_queries']}")
    print(f"  已知次数:   {stats['total_known']}")
    print(f"  学习次数:   {stats['total_learned']}")
    print(f"  已知率:     {stats['known_ratio']:.1%}")
    
    if 'learner' in stats:
        ls = stats['learner']
        print(f"  学习统计:   {ls.get('total_learns', 0)}次学习, "
              f"{ls.get('total_decays', 0)}次衰减")


# ============================================================================
# 主测试入口
# ============================================================================

def run_all_tests():
    """运行全部端到端测试"""
    print("=" * 60)
    print("🐉  龙珠初代 · 端到端测试")
    print("=" * 60)
    
    total_start = time.time()
    
    try:
        # 测试1: 初始化
        loongpearl = test_initialization()
        
        # 测试2-7: 功能测试
        test_self_ignorance(loongpearl)
        test_knowledge_query(loongpearl)
        test_char_reasoning(loongpearl)
        test_find_nearest(loongpearl)
        test_auto_learn(loongpearl)
        test_stats(loongpearl)
        
    except Exception as e:
        print(f"\n  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'═' * 60}")
    print(f"  ✅ 全部测试完成 ({total_elapsed:.1f}秒)")
    print(f"{'═' * 60}")
    
    return True


def run_quick_test():
    """快速冒烟测试（不触发 Ollama）"""
    print("=" * 60)
    print("🐉  龙珠初代 · 快速冒烟测试")
    print("=" * 60)
    
    loongpearl = LoongPearl()
    loongpearl.initialize(verbose=True)
    
    # 只测核心路径：编码 → 自知无知 → 快速检索
    print("\n  快速测试: 编码 + 自知无知 + 检索")
    
    vec = loongpearl._encode("测试文本")
    print(f"  ✅ 编码: shape={vec.shape}")
    
    check = loongpearl.learner.check_knowledge(vec)
    print(f"  ✅ 自知无知: is_known={check['is_known']}, conf={check['confidence']:.3f}")
    
    chars = loongpearl.find_nearest_chars("龙珠", k=5)
    print(f"  ✅ 检索: {[c for c,_ in chars]}")
    
    print("\n  ✅ 冒烟测试通过！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="龙珠端到端测试")
    parser.add_argument('--quick', action='store_true', help='快速冒烟测试（不触发Ollama）')
    parser.add_argument('--test', type=str, help='运行指定测试 (init|ignorance|query|reason|nearest|learn|stats|all)')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.test:
        loongpearl = LoongPearl()
        loongpearl.initialize(verbose=True)
        
        tests = {
            'init': lambda: print("初始化已在上方完成"),
            'ignorance': lambda: test_self_ignorance(loongpearl),
            'query': lambda: test_knowledge_query(loongpearl),
            'reason': lambda: test_char_reasoning(loongpearl),
            'nearest': lambda: test_find_nearest(loongpearl),
            'learn': lambda: test_auto_learn(loongpearl),
            'stats': lambda: test_stats(loongpearl),
            'all': lambda: run_all_tests(),
        }
        
        if args.test in tests:
            tests[args.test]()
        else:
            print(f"未知测试: {args.test}")
            print(f"可用: {', '.join(tests.keys())}")
    else:
        run_all_tests()
