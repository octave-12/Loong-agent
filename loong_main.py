#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠统一入口 (loong_main.py)
═══════════════════════════════
一个主进程，一次加载字场+景观(368MB)，三种模式：

  python loong_main.py --daemon      7×24自主学习守护进程
  python loong_main.py --seed        成语种子注入(一次性)
  python loong_main.py --verify      收敛精度检验(100成语infer)
  python loong_main.py --once        单轮学习(调试用)

优势:
  - 字场(368MB)只加载一次，避免多进程重复加载
  - 单例锁：同模式不重复启动
  - 统一日志：所有模式共用 logs/loong_main.log

架构:
  加载层(一次性) → mode: daemon | seed | verify | once
"""

import sys
import os
import time
import argparse
import atexit
import signal
import logging

import torch

# 项目路径
PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.learning.autonomous_learner import AutonomousLearner

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_FILE = os.path.join(PROJECT, 'logs', 'loong_main.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger('loong_main')


# ============================================================================
# 单例锁
# ============================================================================

def singleton_lock(name: str) -> bool:
    """获取单例锁，返回 True=成功获得锁，False=已有实例运行"""
    lock_file = os.path.join(PROJECT, 'logs', f'{name}.pid')
    try:
        with open(lock_file) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, 0)
            log.error(f"❌ {name} 已在运行 (PID {old_pid})")
            return False
        except OSError:
            pass
    except (FileNotFoundError, ValueError):
        pass
    with open(lock_file, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock_file) and os.remove(lock_file))
    return True


# ============================================================================
# 模型加载（一次性，所有模式共用）
# ============================================================================

def load_models():
    """加载字场 + 能量景观 + 学习器（隔离失败，任何一个加载失败都不影响其他）"""
    log.info("🐉 加载龙珠模型...")
    t0 = time.time()

    field = None
    landscape = None
    learner = None

    # 字场（必须——没有字场无法工作）
    try:
        field = HanziAnchorField.load(
            os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
            freeze=True
        )
        log.info(f"   字场:{field.num_hanzi}字 嵌入:{field.embed_dim}d")
    except Exception as e:
        log.error(f"❌ 字场加载失败: {e}")
        sys.exit(1)

    # 能量景观（可选——无景观时概念图仍可用，只是无能量评分）
    try:
        landscape = FreqEnergyLandscape.load(
            os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        ).to(DEVICE).eval()
        log.info(f"   景观: 已加载")
    except Exception as e:
        log.warning(f"⚠️ 能量景观加载失败({e})，概念图将使用默认能量值")
        landscape = None

    # 学习器（可选——无学习器时自主学习不可用）
    if landscape is not None:
        try:
            learner = DragonBallLearner(landscape, field, device=DEVICE)
            learner.calibrate()
            log.info(f"   学习器: 已就绪")
        except Exception as e:
            log.warning(f"⚠️ 学习器初始化失败({e})，自主学习不可用")
            learner = None
    else:
        log.warning("⚠️ 无能量景观，跳过学习器初始化")

    elapsed = time.time() - t0
    log.info(f"   总耗时:{elapsed:.1f}s")
    return field, landscape, learner


# ============================================================================
# 模式1: 守护进程
# ============================================================================

def run_daemon(field, landscape, learner, args):
    """7×24自主学习守护进程"""
    from scripts.auto_learn import AutoLearningDaemon
    
    daemon = AutoLearningDaemon(
        scan_interval=args.interval,
        max_learn_per_round=args.max_learn,
        factors=None,
        from_landscape=True,
        start_stage=args.stage,
    )
    # 注入已加载的模型（避免重复加载）
    daemon.field = field
    daemon.landscape = landscape
    daemon.learner = learner
    
    daemon.run_daemon(max_rounds=args.max_rounds)


# ============================================================================
# 模式2: 种子注入
# ============================================================================

def run_seed(field, landscape, learner, args):
    """成语种子注入"""
    from scripts.idiom_inject_gpu import main as inject_main
    # 重建命令行参数
    sys.argv = [
        'idiom_inject_gpu.py',
        '--batch', str(args.batch),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
    ]
    if args.dry_run:
        sys.argv.append('--dry-run')
    inject_main()


# ============================================================================
# 模式3: 验证检验
# ============================================================================

def run_verify(field, landscape, learner, args):
    """收敛精度检验——用 infer() 测试100个成语的收敛正确率"""
    import json
    import random
    
    log.info("=" * 50)
    log.info("🔬 收敛精度检验: 100成语 infer() 测试")
    log.info("=" * 50)
    
    random.seed(42)
    
    # 加载成语列表
    idiom_path = os.path.join(PROJECT, 'data/dicts/idioms.json')
    with open(idiom_path, encoding='utf-8') as f:
        all_idioms = json.load(f)
    
    # 筛选字场中存在的四字成语
    valid_idioms = []
    for idiom in all_idioms:
        chars = list(idiom)
        if len(chars) != 4:
            continue
        if all(c in field._char_to_idx for c in chars):
            valid_idioms.append(idiom)
    
    log.info(f"有效成语: {len(valid_idioms)}/{len(all_idioms)}")
    
    # 采样100个
    sample = random.sample(valid_idioms, min(100, len(valid_idioms)))
    
    landscape.eval()
    correct = 0
    total = 0
    results = []
    
    for idiom in sample:
        chars = list(idiom)
        # 用前两个字的中点作为查询，看是否收敛到后两字附近
        # 或者用成语的中点看收敛到哪个字
        idx_a = field._char_to_idx[chars[0]]
        idx_b = field._char_to_idx[chars[1]]
        idx_c = field._char_to_idx[chars[2]]
        idx_d = field._char_to_idx[chars[3]]
        
        # 成语中点向量
        mid = (field.anchors[idx_a] + field.anchors[idx_b] + 
               field.anchors[idx_c] + field.anchors[idx_d]) / 4.0
        
        with torch.no_grad():
            # 找到能量景观中mid周围的最近锚点
            distances = torch.cosine_similarity(
                mid.unsqueeze(0).to(DEVICE),
                field.anchors.to(DEVICE), dim=1
            )
            top_indices = torch.topk(distances, 4).indices.cpu().tolist()
            top_chars = [field.hanzi_list[i] for i in top_indices]
        
        # 检验：top-4 中是否包含成语中的任意字
        idiom_chars = set(chars)
        overlap = idiom_chars & set(top_chars)
        
        total += 1
        if overlap:
            correct += 1
        
        if total <= 10 or total % 20 == 0:
            marker = '✅' if overlap else '❌'
            log.info(f"  {marker} {idiom} → 最近字: {top_chars[:4]} "
                    f"命中: {len(overlap)}/4")
    
    accuracy = correct / total * 100 if total > 0 else 0
    log.info(f"\n📊 检验结果: {correct}/{total} = {accuracy:.1f}%")
    log.info(f"   （top-4最近锚点中包含成语中至少1个字）")
    
    if accuracy < 50:
        log.warning("⚠️ 精度偏低，建议增加种子注入轮数或调整学习率")
    elif accuracy < 80:
        log.info("🟡 精度中等，持续守护学习可提升")
    else:
        log.info("🟢 精度良好，景观已形成有效盆地")


# ============================================================================
# 模式4: 单次学习（调试用）
# ============================================================================

def run_once(field, landscape, learner, args):
    """单轮学习"""
    from scripts.auto_learn import AutoLearningDaemon
    
    daemon = AutoLearningDaemon(
        scan_interval=9999,
        max_learn_per_round=args.max_learn,
        factors=None,
        from_landscape=True,
    )
    daemon.field = field
    daemon.landscape = landscape
    daemon.learner = learner
    
    daemon.run_round()


# ============================================================================
# 模式5: 自组织语言 — 前缀补全
# ============================================================================

def run_generate(field, landscape, args):
    """自组织语言：给定前缀，补全为成语/词语"""
    from loongpearl.core.sequence_energy import SequenceEnergy
    
    seq = SequenceEnergy(field, landscape, device=DEVICE)
    prefix = args.generate
    
    log.info(f"🔤 自组织语言: '{prefix}' → ?")
    
    results = seq.complete(prefix, top_n=10)
    
    if results:
        log.info(f"\n  {'前缀':<6} → {'补全':<12} {'能量':>8}")
        log.info(f"  {'-'*6}   {'-'*12} {'-'*8}")
        for full, energy in results:
            log.info(f"  {prefix:<6} → {full:<12} {energy:>8.1f}")
    else:
        log.info(f"  ⚠️ 未找到匹配 '{prefix}' 的成语")
    
    # 也显示关联成语（包含前缀但不以前缀开头）
    from_dict = seq.complete_from_dict(prefix, top_n=5)
    if from_dict:
        log.info(f"\n  包含'{prefix}'的成语:")
        for full, energy in from_dict[:5]:
            log.info(f"    {full} ({energy:.1f})")


# ============================================================================
# 概念图持久化路径
# ============================================================================

CONCEPT_GRAPH_BASE = os.path.join(PROJECT, 'data', 'models', 'concept_graph')

# ============================================================================
# 模式6: 概念图推理
# ============================================================================

def run_reason(field, landscape, args):
    """多学科概念图推理 — 完整版：加载/构建/推理/归纳/评估"""
    from loongpearl.core.concept_graph import ConceptGraph, Relation

    concept = args.reason

    # 尝试加载持久化概念图
    cg = ConceptGraph(field, landscape)
    loaded = False
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        try:
            cg.load(CONCEPT_GRAPH_BASE)
            loaded = True
            log.info(f"📂 加载已持久化概念图: {cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组")
        except Exception as e:
            log.warning(f"加载失败({e})，将重新构建")

    if not loaded:
        log.info("🆕 构建新概念图（全学科种子 + 归纳推理）...")
        counts = cg.seed_all_domains()
        for dom, n in counts.items():
            log.info(f"   {dom}: {n}个概念集")
        log.info(f"   初始: {cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组")

        # 归纳推理
        inferred = cg.induce()
        log.info(f"   🧩 归纳推理: +{len(inferred)}条推断三元组")

        # 矛盾检测
        conflicts = cg.detect_contradictions()
        if conflicts:
            log.warning(f"   ⚠️ 发现 {len(conflicts)} 个矛盾")

        # 持久化
        try:
            cg.save(CONCEPT_GRAPH_BASE)
            log.info(f"   💾 已保存至 {CONCEPT_GRAPH_BASE}")
        except Exception as e:
            log.warning(f"   保存失败: {e}")

    log.info(f"\n🧠 概念图推理: '{concept}'")
    log.info(f"   节点:{cg.stats()['nodes']} 三元组:{cg.stats()['triples']}")

    # 1. 直接关联
    log.info(f"\n📋 '{concept}' 的直接关联 (按能量排序):")
    for r in cg.query(concept, max_results=10, sort_by="energy"):
        arrow = "→" if r['direction'] == 'forward' else "←"
        log.info(f"   {concept} {arrow} [{r['relation']}] {r['concept']} "
                f"(能:{r['energy']:.1f} 信:{r['confidence']:.2f} [{r['source']}])")

    # 2. 多跳推理 (双向，任意关系)
    log.info(f"\n🔗 多跳推理 (双向, 任意关系, 3跳):")
    paths = cg.reason(concept, max_hops=3, direction='both')
    for p in paths[:8]:
        energy = cg._path_energy(p)
        log.info(f"   {' → '.join(p)} (能:{energy:.1f})")

    # 3. 按关系类型推理
    for rel in [Relation.PART_OF, Relation.IS_A, Relation.HAS]:
        paths = cg.reason(concept, relation=rel, max_hops=3, direction='both')
        if paths and len(paths[0]) > 1:
            log.info(f"\n🔗 [{rel}] 定向推理:")
            for p in paths[:5]:
                log.info(f"   {' → '.join(p)}")

    # 4. 自评估
    log.info(f"\n📊 概念图自评估:")
    report = cg.evaluate()
    log.info(f"   节点:{report['nodes']} 三元组:{report['triples']} 推断比:{report['inferred_ratio']:.2f}")
    log.info(f"   平均度:{report['avg_degree']} 一致性:{report['consistency']} 连通性:{report['connectivity']}")
    log.info(f"   均值信:{report['avg_confidence']} 矛盾:{report['conflicts_found']}")

    # 5. 扩展建议
    suggestions = cg.suggest_expansions(max_suggestions=10)
    if suggestions:
        log.info(f"\n💡 需要扩展的概念 (连接稀疏):")
        for s in suggestions:
            log.info(f"   {s['concept']} (度:{s['degree']}) — {s['reason']}")


# ============================================================================
# 模式7: 概念图管理
# ============================================================================

def run_concept(field, landscape, args):
    """概念图管理：构建/重建/评估/查询/扩展"""
    from loongpearl.core.concept_graph import ConceptGraph

    action = args.concept_action or 'build'

    if action == 'rebuild':
        # 删除旧文件重建
        for ext in ['.json', '_embeds.pt']:
            path = CONCEPT_GRAPH_BASE + ext
            if os.path.exists(path):
                os.remove(path)
        log.info("🗑️  已删除旧概念图")
        action = 'build'

    if action == 'build':
        cg = ConceptGraph(field, landscape)
        counts = cg.seed_all_domains()
        for dom, n in counts.items():
            log.info(f"   {dom}: {n}个概念集")
        inferred = cg.induce()
        log.info(f"   🧩 归纳推理: +{len(inferred)}条")
        cg.save(CONCEPT_GRAPH_BASE)
        log.info(f"   ✅ 构建完成: {cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组")

    elif action == 'eval':
        cg = ConceptGraph(field, landscape)
        if not os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            log.error("概念图不存在，请先 --concept build")
            return
        cg.load(CONCEPT_GRAPH_BASE)
        report = cg.evaluate()
        log.info(f"📊 概念图评估报告:")
        for k, v in report.items():
            if k != 'relations':
                log.info(f"   {k}: {v}")
        log.info(f"   关系分布: {report['relations']}")

        suggestions = cg.suggest_expansions(max_suggestions=20)
        if suggestions:
            log.info(f"\n💡 需扩展概念 (前20):")
            for s in suggestions:
                log.info(f"   {s['concept']} (度:{s['degree']}) — {s['reason']}")

    elif action == 'query':
        if not args.concept_query:
            log.error("请用 --concept-query '概念名' 指定查询概念")
            return
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        else:
            cg.seed_all_domains()
            cg.induce()

        for q in [args.concept_query]:
            log.info(f"\n📋 查询: '{q}'")
            for r in cg.query(q, max_results=15, sort_by="score"):
                arrow = "→" if r['direction'] == 'forward' else "←"
                log.info(f"   {q} {arrow} [{r['relation']}] {r['concept']} "
                        f"(信:{r['confidence']:.2f} [{r['source']}])")

    elif action == 'contradictions':
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        conflicts = cg.detect_contradictions()
        if conflicts:
            log.warning(f"⚠️ 发现 {len(conflicts)} 个矛盾:")
            for c in conflicts:
                log.warning(f"   {c['message']}")
        else:
            log.info("✅ 未发现矛盾")

    elif action == 'induce':
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        else:
            cg.seed_all_domains()
        inferred = cg.induce()
        log.info(f"🧩 归纳推理: {len(inferred)} 条新推断")
        for t in inferred[:20]:
            log.info(f"   💡 {t.subject} {t.relation} {t.object} (信:{t.confidence:.2f})")
        if inferred:
            cg.save(CONCEPT_GRAPH_BASE)
            log.info(f"💾 已保存")


# ============================================================================
# 模式8: 跨学科桥接
# ============================================================================

def run_bridge(field, landscape, args):
    """跨学科桥接：发现不同领域间的隐藏关联"""
    from loongpearl.core.concept_graph import ConceptGraph
    from loongpearl.core.cross_domain_bridge import CrossDomainBridgeEngine

    # 加载概念图
    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        cg.load(CONCEPT_GRAPH_BASE)
    else:
        log.info("先构建概念图...")
        cg.seed_all_domains()
        cg.induce()
        cg.save(CONCEPT_GRAPH_BASE)

    log.info(f"🌉 跨学科桥接引擎启动")
    log.info(f"   概念图: {cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组")

    bridge = CrossDomainBridgeEngine(field, landscape, cg)
    log.info(f"   领域: {len(bridge.domain_concepts)}个")
    for dom, concepts in sorted(bridge.domain_concepts.items()):
        if len(concepts) > 0:
            log.info(f"     {dom}: {len(concepts)}个概念")

    # 综合桥接（一次性运行四层+去重，避免重复计算）
    log.info(f"\n🌉 综合桥接 (四层去重合并)...")
    all_bridges = bridge.build_all_bridges(min_confidence=0.3, max_bridges=100)
    summary = bridge.summary(all_bridges)

    log.info(f"   ✅ 总计: {summary['total']} 座跨学科桥梁")
    log.info(f"   平均置信度: {summary['avg_confidence']:.3f}")
    log.info(f"   最高置信度: {summary['max_confidence']:.3f}")
    # 从合并结果推算各层贡献
    layer_counts = {'hanzi_share': 0, 'embed_proximity': 0, 'structural': 0, 'combined': 0}
    for b in all_bridges:
        t = b.bridge_type
        layer_counts[t] = layer_counts.get(t, 0) + 1
    log.info(f"   各层分布: 字素{layer_counts.get('hanzi_share',0)} 嵌入{layer_counts.get('embed_proximity',0)} 结构{layer_counts.get('structural',0)} 合并{layer_counts.get('combined',0)}")
    log.info(f"   桥接类型: {summary['bridge_types']}")
    log.info(f"   领域对分布:")

    for pair, count in summary['domain_pairs'].items():
        log.info(f"     {pair}: {count}座桥")

    log.info(f"\n🏆 Top 10 跨学科桥梁:")
    for i, b in enumerate(summary['top_bridges']):
        log.info(f"   {i+1}. {b['concepts']} ({b['domains']}) 信:{b['confidence']} [{b['type']}]")
        log.info(f"      {b['reason']}")

    # 写入概念图
    n_added = bridge.add_bridges_to_concept_graph(all_bridges, min_confidence=0.4)
    log.info(f"\n💾 已将 {n_added} 座高置信度桥梁写入概念图")

    if n_added > 0:
        # 写入后重新归纳推理（可能产生新的传递链）
        cg.induce()
        cg.save(CONCEPT_GRAPH_BASE)
        log.info(f"💾 概念图已更新保存 ({cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组)")

    # 显示连通性改善
    report = cg.evaluate()
    log.info(f"\n📊 桥接后评估:")
    log.info(f"   连通性: {report['connectivity']:.3f} (越高越好)")
    log.info(f"   平均度: {report['avg_degree']}")
    log.info(f"   三元组: {report['triples']} (推断比:{report['inferred_ratio']:.2f})")


# ============================================================================
# 模式9: 词级补全
# ============================================================================

def run_word(field, landscape, args):
    """词级补全：给定前缀，补全为多字词"""
    from loongpearl.core.concept_graph import ConceptGraph
    from loongpearl.core.word_energy import WordEnergy

    # 加载概念图
    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        cg.load(CONCEPT_GRAPH_BASE)

    we = WordEnergy(field, landscape, cg)

    prefix = args.word
    log.info(f"🔤 词级补全: '{prefix}'")
    log.info(f"   概念图: {cg.stats()['nodes']}节点 | 词引擎: {len(we._word_embeddings)}词")

    # 前缀匹配词
    matching = we.find_words_by_prefix(prefix, max_results=20)
    if matching:
        log.info(f"\n📋 以'{prefix}'开头的已知词 ({len(matching)}个):")
        for w in matching[:12]:
            log.info(f"   {w}")

    # 智能补全
    results = we.complete(prefix, top_n=10, beam_width=8)
    if results:
        log.info(f"\n🏆 词级补全 Top 10:")
        for i, (text, energy, source) in enumerate(results):
            log.info(f"   {i+1}. {text} (能:{energy:.1f} [{source}])")

    # 词链
    if len(prefix) >= 1 and prefix in cg.nodes:
        log.info(f"\n🔗 词链: '{prefix}' → ...")
        chains = we.chain(prefix, max_words=4)
        for c in chains[:5]:
            log.info(f"   {' → '.join(c)}")

    # 候选排序
    ranked = we.rank([
        f"{prefix}{suffix}" for suffix in 
        ["力学", "计算", "纠缠", "场论", "信息", "通信", "物理", "化学"]
    ])
    if ranked:
        log.info(f"\n📊 候选词能量排序:")
        for word, energy in ranked[:8]:
            log.info(f"   {word}: {energy:.1f}")


# ============================================================================
# 模式10: 闭环验证
# ============================================================================

def run_verify_loop(field, landscape, args):
    """闭环验证：验证概念图推断并修正置信度"""
    from loongpearl.core.concept_graph import ConceptGraph
    from loongpearl.learning.verify_loop import VerifyLoop

    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        cg.load(CONCEPT_GRAPH_BASE)

    vf = VerifyLoop(cg)

    # 统计当前推断
    total_inferred = sum(1 for t in cg.triples.values() if t.source == "infer")
    low_conf = sum(1 for t in cg.triples.values()
                   if t.source == "infer" and t.confidence < 0.4)

    log.info(f"🔄 闭环验证引擎启动")
    log.info(f"   总推断: {total_inferred} | 低置信度(<0.4): {low_conf}")

    # 批量验证
    report = vf.verify_all_inferred(max_verify=15, max_confidence=0.5)
    log.info(f"\n📊 验证结果:")
    log.info(f"   确认: {report['confirmed']} | 矛盾: {report['contradicted']} "
            f"| 不确定: {report['uncertain']}")
    for r in report['results'][:10]:
        marker = "✅" if r['verdict'] == 'confirmed' else ("❌" if r['verdict'] == 'contradicted' else "❓")
        log.info(f"   {marker} {r['triple']} ({r['confidence']})")

    # 保存修正后的概念图
    if report['confirmed'] > 0 or report['contradicted'] > 0:
        cg.save(CONCEPT_GRAPH_BASE)
        log.info(f"\n💾 概念图已更新保存 (置信度已修正)")


# ============================================================================
# 模式11: 能量金字塔
# ============================================================================

def run_pyramid(field, landscape, args):
    """能量金字塔：训练三级抽象景观"""
    from loongpearl.core.concept_graph import ConceptGraph
    from loongpearl.core.multi_level import EnergyPyramid

    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        cg.load(CONCEPT_GRAPH_BASE)
    else:
        cg.seed_all_domains()
        cg.induce()

    pyramid = EnergyPyramid(field, base_dim=1024, device=DEVICE)
    log.info(f"🔺 能量金字塔: 1024→512→256→128")
    log.info(f"   参数: {sum(p.numel() for p in pyramid.parameters()):,}")

    # 训练
    results = pyramid.train_all_levels(cg, epochs_per_level=150)

    # 测试
    log.info("\\n🔍 三级推理:")
    tests = [("电子","原子"),("原子","分子"),("细胞","器官"),("数学","物理"),("红楼梦","唐诗"),("量子","力学")]
    for a, b in tests:
        analysis = pyramid.analyze(a, b, cg)
        d = analysis.get('diagnosis', {})
        log.info(f"  {a}→{b}: e1={analysis.get('e1','?')} {d.get('word_level','')} "
                f"e2={analysis.get('e2','?')} {d.get('syntax_level','')} "
                f"e3={analysis.get('e3','?')} {d.get('concept_level','')} "
                f"总={analysis.get('total','?')}")

    # 保存
    pyramid_path = os.path.join(PROJECT, 'data', 'models', 'energy_pyramid.pt')
    pyramid.save(pyramid_path)
    log.info(f"\\n💾 金字塔已保存: {pyramid_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🐉 龙珠统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式:
  python loong_main.py --daemon       7×24守护进程
  python loong_main.py --seed         成语种子注入
  python loong_main.py --verify       收敛精度检验
  python loong_main.py --generate X   前缀补全(自组织语言)
  python loong_main.py --reason X     概念图推理(多学科)
  python loong_main.py --once         单轮学习(调试)

守护参数:
  --interval N    扫描间隔秒数(默认120)
  --max-learn N   每轮学习数(默认3)
  --max-rounds N  最大轮数(默认0=无限)
  --stage N       起始课程阶段(默认0=恢复进度)

种子参数:
  --batch N       GPU batch size(默认16000)
  --epochs N      训练轮数(默认3)
  --lr F          学习率(默认0.01)
  --dry-run       只评估不保存
        """
    )
    
    # 模式
    parser.add_argument('--daemon', '-d', action='store_true', help='守护进程模式')
    parser.add_argument('--seed', '-s', action='store_true', help='种子注入模式')
    parser.add_argument('--verify', '-v', action='store_true', help='验证检验模式')
    parser.add_argument('--once', '-o', action='store_true', help='单轮学习模式')
    parser.add_argument('--generate', '-g', type=str, default=None, metavar='PREFIX',
                       help='前缀补全(自组织语言), 如: --generate 画龙')
    parser.add_argument('--reason', '-r', type=str, default=None, metavar='CONCEPT',
                       help='概念图推理(多学科), 如: --reason 电子')
    parser.add_argument('--concept', '-c', dest='concept_action', type=str, nargs='?',
                        const='build', default=None, metavar='ACTION',
                        choices=['build', 'rebuild', 'eval', 'query', 'contradictions', 'induce'],
                        help='概念图管理: build/rebuild/eval/query/contradictions/induce')
    parser.add_argument('--concept-query', '-cq', type=str, default=None, metavar='CONCEPT',
                       help='概念图查询目标(配合 --concept query)')
    parser.add_argument('--bridge', '-b', dest='bridge_action', type=str, nargs='?',
                        const='full', default=None, metavar='ACTION',
                        choices=['full'],
                        help='跨学科桥接: 发现不同领域间的隐藏关联')
    parser.add_argument('--word', '-w', type=str, default=None, metavar='PREFIX',
                       help='词级补全(多字词), 如: --word 量子')
    parser.add_argument('--verify-loop', dest='verify_loop', action='store_true',
                       help='闭环验证: 验证概念图推断并修正置信度')
    parser.add_argument('--pyramid', '-p', action='store_true',
                       help='能量金字塔: 训练三级抽象景观并推理')
    
    # 守护参数
    parser.add_argument('--interval', '-i', type=int, default=120)
    parser.add_argument('--max-learn', '-m', type=int, default=3)
    parser.add_argument('--max-rounds', '-n', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    
    # 种子参数
    parser.add_argument('--batch', type=int, default=16000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    # 默认模式
    has_mode = args.daemon or args.seed or args.verify or args.once or \
               args.generate or args.reason or args.concept_action or \
               args.bridge_action or args.word or args.verify_loop or args.pyramid
    if not has_mode:
        parser.print_help()
        print("\n请指定一种模式: --daemon / --seed / --verify / --generate / "
              "--reason / --concept / --bridge / --word / --verify-loop / --once")
        sys.exit(1)
    
    # 单例锁
    if args.daemon:
        mode_name = 'daemon'
    elif args.seed:
        mode_name = 'seed'
    elif args.verify:
        mode_name = 'verify'
    elif args.once:
        mode_name = 'once'
    elif args.generate:
        mode_name = 'generate'
    elif args.concept_action:
        mode_name = 'concept'
    elif args.bridge_action:
        mode_name = 'bridge'
    elif args.word:
        mode_name = 'word'
    elif args.verify_loop:
        mode_name = 'verify_loop'
    elif args.pyramid:
        mode_name = 'pyramid'
    else:
        mode_name = 'reason'
    if not singleton_lock(f'loong_{mode_name}'):
        sys.exit(1)
    
    log.info(f"🐉 龙珠启动 — 模式: {mode_name}")
    
    # 加载模型
    field, landscape, learner = load_models()
    
    # 分发
    if args.daemon:
        run_daemon(field, landscape, learner, args)
    elif args.seed:
        run_seed(field, landscape, learner, args)
    elif args.verify:
        run_verify(field, landscape, learner, args)
    elif args.once:
        run_once(field, landscape, learner, args)
    elif args.generate:
        run_generate(field, landscape, args)
    elif args.reason:
        run_reason(field, landscape, args)
    elif args.concept_action:
        run_concept(field, landscape, args)
    elif args.bridge_action:
        run_bridge(field, landscape, args)
    elif args.word:
        run_word(field, landscape, args)
    elif args.verify_loop:
        run_verify_loop(field, landscape, args)
    elif args.pyramid:
        run_pyramid(field, landscape, args)
    
    log.info(f"🐉 龙珠退出")


if __name__ == '__main__':
    main()
