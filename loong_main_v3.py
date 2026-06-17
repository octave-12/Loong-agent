#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠统一入口 v3 — 8引擎全栈集成
═══════════════════════════════════════════════════════

新模式:
  --chat              交互式对话 (NLU→规划→查询→NLG 全链路)
  --parse "xxx"       测试解义器
  --render             测试化能器  
  --multiform "xxx"   测试万象格
  --fuzzy             测试模糊格
  --plan "xxx"        测试策应器
  --lang              测试万语锚
  --harvest           测试万象收
  --contra            测试矛盾解

守护循环增强:
  每轮:   学习 + 万象收增量采集
  每5轮:  桥接 + 矛盾检测消解
  每10轮: 闭环验证 + 模糊格证据重评估
  每20轮: 剪枝对齐 + 万象格跨格式验证

═══════════════════════════════════════════════════════
"""

import sys
import os
import time
import argparse
import atexit
import signal
import logging
import json

import torch

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

CONCEPT_GRAPH_BASE = os.path.join(PROJECT, 'data', 'models', 'concept_graph')


# ============================================================================
# 单例锁
# ============================================================================

def singleton_lock(name: str) -> bool:
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
# 模型加载
# ============================================================================

def load_models(lightweight=False):
    """加载模型。lightweight=True 时只加载字场（用于纯规则测试）"""
    log.info("🐉 加载龙珠模型...")
    t0 = time.time()

    field = None
    landscape = None
    learner = None

    try:
        field = HanziAnchorField.load(
            os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
            freeze=True
        )
        log.info(f"   字场:{field.num_hanzi}字 嵌入:{field.embed_dim}d")
    except Exception as e:
        log.error(f"❌ 字场加载失败: {e}")
        sys.exit(1)

    if lightweight:
        log.info(f"   总耗时:{time.time()-t0:.1f}s (轻量模式)")
        return field, None, None

    try:
        landscape = FreqEnergyLandscape.load(
            os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        ).to(DEVICE).eval()
        log.info(f"   景观: 已加载")
    except Exception as e:
        log.warning(f"⚠️ 能量景观加载失败({e})")
        landscape = None

    if landscape is not None:
        try:
            learner = DragonBallLearner(landscape, field, device=DEVICE)
            learner.calibrate()
            log.info(f"   学习器: 已就绪")
        except Exception as e:
            log.warning(f"⚠️ 学习器初始化失败({e})")
            learner = None

    elapsed = time.time() - t0
    log.info(f"   总耗时:{elapsed:.1f}s")
    return field, landscape, learner


def load_concept_graph(field, landscape):
    """加载概念图（所有引擎共用）"""
    from loongpearl.core.concept_graph import ConceptGraph
    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        try:
            cg.load(CONCEPT_GRAPH_BASE)
            log.info(f"   概念图: {cg.stats()['nodes']}节点 {cg.stats()['triples']}三元组")
            return cg
        except Exception as e:
            log.warning(f"   概念图加载失败: {e}")
    return cg


# ============================================================================
# 模式: 交互式对话 (全栈 8引擎)
# ============================================================================

def run_chat(field, landscape, args):
    """
    交互式对话模式 — NLU → 规划 → 多引擎查询 → NLG 全链路。
    使用 Orchestrator 调度 8 引擎。
    """
    from loongpearl.core.orchestrator import create_orchestrator

    orch = create_orchestrator(field, landscape)

    print("\n" + orch.status_report())
    print("\n输入 'quit' 退出, 'status' 查看状态\n")

    while True:
        try:
            query = input("🐉 龙珠> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q', '退出'):
            break
        if query.lower() == 'status':
            print(orch.status_report())
            continue

        # 全栈处理
        result = orch.dialogue(query)

        # 输出（根据路由类型显示不同格式）
        rtype = result.get('type', 'knowledge')
        debug = result.get('debug', {})

        if rtype == 'social':
            print(f"\n🐉 {result['output']}")
        elif rtype == 'chitchat':
            print(f"\n💭 {result['output']}")
        elif rtype == 'creative':
            ctype = debug.get('creative_type', '')
            emoji = "📜" if ctype == 'poetry' else "🔗" if ctype == 'idiom_chain' else "📖"
            print(f"\n{emoji} 创作结果:\n{result['output']}")
        elif rtype == 'knowledge':
            frame = debug.get('frame', {})
            print(f"\n📋 [{frame.get('type', '查询')}] "
                  f"{frame.get('subject', query)}")
            if 'cross_lang' in debug:
                print(f"🌐 {debug['cross_lang']}")
            if debug.get('context_enhanced'):
                print(f"💡 (上下文增强)")
            print(f"💬 {result['output']}")
            if 'cross_form' in debug:
                forms = [k for k, v in debug['cross_form'].items() if v]
                if forms:
                    print(f"🔗 万象格: {', '.join(forms)}")
        else:
            print(f"\n💬 {result['output']}")


# ============================================================================
# 模式: 解义器测试
# ============================================================================

def run_parse(field, landscape, args):
    from loongpearl.core.sem_parser import SemParser

    # 轻量模式: 不加载概念图，纯规则解析
    sp = SemParser(concept_graph=None)

    texts = args.parse if isinstance(args.parse, list) else [args.parse]
    for text in texts:
        frame = sp.parse(text)
        print(f"\n{'='*60}")
        print(f"📝 文本: {text}")
        print(f"   类型: {frame.question_type.name if frame.question_type else '陈述'}")
        print(f"   意图: {frame.intent.name if frame.intent else 'N/A'}")
        print(f"   主体: '{frame.subject}'")
        print(f"   谓词: '{frame.predicate}'")
        print(f"   客体: '{frame.object}'")
        print(f"   概念: {frame.concepts}")
        print(f"   修饰: {frame.modifiers}")
        print(f"   未知: {frame.unknown_terms}")
        print(f"   查询: {frame.structured_query}")


# ============================================================================
# 模式: 化能器测试
# ============================================================================

def run_render_test(field, landscape, args):
    from loongpearl.core.energy_decoder import EnergyDecoder

    decoder = EnergyDecoder()

    test_cases = [
        {
            "render_type": "explain_path",
            "subject": "量子纠缠",
            "path": ["量子纠缠", "量子态", "测量", "波函数坍缩"],
            "edges": [
                {"rel": "IS_A", "confidence": 0.95},
                {"rel": "CAUSE", "confidence": 0.87},
            ],
        },
        {
            "render_type": "compare",
            "compare_subjects": ["儒家", "道家"],
            "facts": [
                {"type": "common", "description": "都产生于先秦"},
                {"type": "difference", "description": "儒家入世vs道家出世"},
            ],
        },
    ]

    for i, tc in enumerate(test_cases):
        print(f"\n测试 {i+1}: {tc['render_type']}")
        print(decoder.render(tc))


# ============================================================================
# 模式: 万象格测试
# ============================================================================

def run_multiform(field, landscape, args):
    from loongpearl.core.multiform_kg import MultiFormKG, seed_multiform_kg

    mkg = MultiFormKG()
    seed_multiform_kg(mkg)
    mkg.print_stats()

    # 交互查询
    queries = args.multiform if isinstance(args.multiform, list) else [args.multiform] if args.multiform else ["秦朝", "科学方法", "温度"]
    for q in queries:
        print(f"\n🔍 跨格式推理: '{q}'")
        results = mkg.reason_across_forms(q)
        for key, val in results.items():
            if val:
                print(f"  [{key}]")
                if isinstance(val, list):
                    for v in val[:3]:
                        print(f"    {v}")
                elif isinstance(val, dict):
                    for k, v in list(val.items())[:3]:
                        print(f"    {k}: {v}")


# ============================================================================
# 模式: 模糊格测试
# ============================================================================

def run_fuzzy(field, landscape, args):
    from loongpearl.core.fuzzy_graph import FuzzyGraph

    fg = FuzzyGraph()
    fg.add_evidence("电子", "PART_OF", "原子", source="量子力学教材", mass=0.85)
    fg.add_evidence("电子", "PART_OF", "原子", source="化学教材", mass=0.92)
    fg.add_evidence("电子", "PART_OF", "原子", source="物理百科", mass=0.88)

    bel, pl = fg.uncertainty("电子", "PART_OF", "原子")
    print(f"\nD-S 证据融合结果:")
    print(f"  命题: 电子 PART_OF 原子")
    print(f"  信念: Bel={bel:.2%}  似然: Pl={pl:.2%}")
    print(f"  置信区间: [{bel:.2%}, {pl:.2%}]")

    decision = fg.decide("电子", "PART_OF", "原子")
    print(f"  决策: {decision['decision']} (质量: {decision['quality']})")

    conflicts = fg.detect_conflicts()
    print(f"  冲突: {len(conflicts)} 个")


# ============================================================================
# 模式: 策应器测试
# ============================================================================

def run_plan_test(field, landscape, args):
    from loongpearl.core.task_planner import TaskPlanner

    tp = TaskPlanner()

    queries = args.plan if isinstance(args.plan, list) else [args.plan] if args.plan else [
        "对比儒家和道家", "量子力学有哪些基本概念"
    ]
    for q in queries:
        plan = tp.plan(q)
        tp.print_plan(plan)


# ============================================================================
# 模式: 万语锚测试
# ============================================================================

def run_multilang(field, landscape, args):
    from loongpearl.core.multilang_anchor import MultiLangAnchor

    mla = MultiLangAnchor()
    mla.print_stats()

    # 中英互译
    tests = [
        ("量子力学", "zh"),
        ("artificial intelligence", "en"),
        ("Confucianism", "en"),
    ]
    for text, lang in tests:
        cids = mla.map_to_concepts(text, lang)
        print(f"\n'{text}' ({lang}) → {cids}")
        for cid in cids[:2]:
            zh = mla.get_concept_name(cid, "zh")
            en = mla.get_concept_name(cid, "en")
            print(f"   zh: {zh}  |  en: {en}")


# ============================================================================
# 模式: 万象收测试
# ============================================================================

def run_harvest(field, landscape, args):
    from loongpearl.core.harvester import KnowledgeHarvester

    h = KnowledgeHarvester()

    text = """
    电子是原子的一种组成部分。原子由质子和中子组成。
    量子力学是物理学的一个分支。光电效应导致电子逸出。
    细胞是生物体的基本单位。基因决定了生物的性状。
    """

    count = h.harvest_from_text(text, lang="zh", source="test")
    h.print_stats()
    print(f"\n文本采集: {count} 个三元组")


# ============================================================================
# 模式: 矛盾解测试
# ============================================================================

def run_contra(field, landscape, args):
    from loongpearl.core.contra_resolver import ContraResolver

    class MockCG:
        def __init__(self):
            self.triples = {}
        def add_triple(self, s, r, o, confidence=0.5, source="test"):
            if s not in self.triples:
                self.triples[s] = []
            self.triples[s].append((r, o, confidence, source))

    cg = MockCG()
    cg.add_triple("A", "IS_A", "B", 0.8)
    cg.add_triple("B", "IS_A", "C", 0.7)
    cg.add_triple("C", "IS_A", "A", 0.6)
    cg.add_triple("光", "IS_A", "粒子", 0.6)
    cg.add_triple("光", "OPPOSITE", "粒子", 0.2)

    cr = ContraResolver(cg)
    cr.detect_all()
    cr.print_report()

    print(f"\n消解...")
    for c in cr.conflicts:
        print(f"  {cr.resolve(c, strategy='confidence_based')}")

    summary = cr.get_summary()
    print(f"\n消解后: {summary}")


# ============================================================================
# 模式: 守护进程 (增强版 — 集成8引擎)
# ============================================================================

def run_daemon(field, landscape, learner, args):
    """7×24自主学习守护进程 — Orchestrator 调度 8 引擎"""
    from scripts.auto_learn import AutoLearningDaemon
    from loongpearl.core.orchestrator import create_orchestrator

    # 创建调度器（单一概念图引用，所有引擎共享）
    orch = create_orchestrator(field, landscape, learner)

    daemon = AutoLearningDaemon(
        scan_interval=args.interval,
        max_learn_per_round=args.max_learn,
        factors=None,
        from_landscape=True,
        start_stage=args.stage,
    )
    daemon.field = field
    daemon.landscape = landscape
    daemon.learner = learner

    # 🔑 把 autonomous learner 的概念图替换为 orch 的引用（同一个cg）
    if hasattr(daemon, 'autonomous') and daemon.autonomous:
        daemon.autonomous.concept_graph = orch.cg

    # 用 Orchestrator 增强每轮循环
    _original_run_round = daemon.run_round

    def enhanced_run_round():
        _original_run_round()

        round_num = daemon.rounds
        try:
            tick = orch.daemon_tick(round_num)

            # 日志输出本轮引擎活动
            if tick.get('pipeline', {}).get('acquired', 0) > 0:
                p = tick['pipeline']
                log.info(f"  📖 知识管线: 需求{p['demands_found']} "
                        f"采集{p['acquired']} "
                        f"注入{p['triples_added']}三元组 "
                        f"{p.get('total_bigrams',0)}字对")

            if tick.get('contra', {}).get('detected', 0) > 0:
                c = tick['contra']
                log.info(f"  ⚔️ 矛盾解: 检测{c['detected']} "
                        f"消解{c.get('resolved',0)} "
                        f"争议{c.get('kept_as_disputed',0)}")

            if tick.get('fuzzy', {}).get('d_s_feedbacks', 0) > 0:
                log.info(f"  🔄 模糊格: D-S回写{tick['fuzzy']['d_s_feedbacks']}条置信度")

            if tick.get('plan', {}).get('targets', 0) > 0:
                log.info(f"  📋 策应器: 规划{tick['plan']['targets']}个学习目标")

            if tick.get('multiform', {}).get('total_form_knowledge', 0) > 0:
                log.info(f"  📐 万象格: {tick['multiform']['total_form_knowledge']}条非三元组知识")

        except Exception as e:
            log.debug(f"  引擎调度异常: {e}")

    daemon.run_round = enhanced_run_round
    daemon.run_daemon(max_rounds=args.max_rounds)


# ============================================================================
# 保留原有模式 (seed/verify/once/generate/reason/concept/bridge/word/verify_loop/pyramid)
# ============================================================================

# 从原版导入
_original_functions = {}
try:
    from loong_main_original import (
        run_seed, run_verify, run_once, run_generate,
        run_reason, run_concept, run_bridge,
        run_word, run_verify_loop, run_pyramid,
    )
    _original_functions.update(locals())
except ImportError:
    pass


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🐉 龙珠统一入口 v3 — 8引擎全栈',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════
8引擎模式 (新):
  --chat              交互式对话 (NLU→NLG全链路)
  --parse "xxx"       解义器测试
  --render             化能器测试
  --multiform "xxx"   万象格测试
  --fuzzy             模糊格测试
  --plan "xxx"        策应器测试
  --lang              万语锚测试
  --harvest           万象收测试
  --contra            矛盾解测试

原有模式:
  --daemon / --seed / --verify / --once
  --generate / --reason / --concept / --bridge
  --word / --verify-loop / --pyramid
═══════════════════════════════════════════════════
        """
    )

    # 新模式
    parser.add_argument('--chat', action='store_true', help='交互式对话(全栈8引擎)')
    parser.add_argument('--parse', type=str, nargs='*', default=None, metavar='TEXT',
                       help='解义器测试')
    parser.add_argument('--render', action='store_true', help='化能器测试')
    parser.add_argument('--multiform', type=str, nargs='*', default=None, metavar='QUERY',
                       help='万象格测试')
    parser.add_argument('--fuzzy', action='store_true', help='模糊格测试')
    parser.add_argument('--plan', type=str, nargs='*', default=None, metavar='QUERY',
                       help='策应器测试')
    parser.add_argument('--lang', action='store_true', help='万语锚测试')
    parser.add_argument('--harvest', action='store_true', help='万象收测试')
    parser.add_argument('--contra', action='store_true', help='矛盾解测试')

    # 原有模式
    parser.add_argument('--daemon', '-d', action='store_true', help='守护进程模式')
    parser.add_argument('--seed', '-s', action='store_true', help='种子注入模式')
    parser.add_argument('--verify', '-v', action='store_true', help='验证检验模式')
    parser.add_argument('--once', '-o', action='store_true', help='单轮学习模式')
    parser.add_argument('--generate', '-g', type=str, default=None, metavar='PREFIX')
    parser.add_argument('--reason', '-r', type=str, default=None, metavar='CONCEPT')
    parser.add_argument('--concept', '-c', dest='concept_action', type=str, nargs='?',
                        const='build', default=None, metavar='ACTION',
                        choices=['build', 'rebuild', 'eval', 'query', 'contradictions', 'induce'])
    parser.add_argument('--concept-query', '-cq', type=str, default=None)
    parser.add_argument('--bridge', '-b', dest='bridge_action', type=str, nargs='?',
                        const='full', default=None, metavar='ACTION', choices=['full'])
    parser.add_argument('--word', '-w', type=str, default=None, metavar='PREFIX')
    parser.add_argument('--verify-loop', dest='verify_loop', action='store_true')
    parser.add_argument('--pyramid', '-p', action='store_true')

    # 守护参数
    parser.add_argument('--interval', '-i', type=int, default=120)
    parser.add_argument('--max-learn', '-m', type=int, default=3)
    parser.add_argument('--max-rounds', '-n', type=int, default=0)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--batch', type=int, default=16000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    # 检查新模式
    new_modes = {
        'chat': args.chat,
        'parse': args.parse is not None,
        'render_test': args.render,
        'multiform': args.multiform is not None,
        'fuzzy': args.fuzzy,
        'plan_test': args.plan is not None,
        'multilang': args.lang,
        'harvest': args.harvest,
        'contra': args.contra,
    }

    # 检查原有模式
    old_modes = {
        'daemon': args.daemon,
        'seed': args.seed,
        'verify': args.verify,
        'once': args.once,
        'generate': args.generate,
        'reason': args.reason,
        'concept': args.concept_action,
        'bridge': args.bridge_action,
        'word': args.word,
        'verify_loop': args.verify_loop,
        'pyramid': args.pyramid,
    }

    has_new_mode = any(new_modes.values())
    has_old_mode = any(old_modes.values())

    if not has_new_mode and not has_old_mode:
        parser.print_help()
        print("\n请指定一种模式。例如: python loong_main_v3.py --chat")
        sys.exit(1)

    # 确定模式名用于单例锁
    if has_new_mode:
        mode_name = [k for k, v in new_modes.items() if v][0]
    else:
        mode_name = [k for k, v in old_modes.items() if v][0]

    if not singleton_lock(f'loong_{mode_name}'):
        sys.exit(1)

    log.info(f"🐉 龙珠启动 — 模式: {mode_name}")

    # 轻量模式（只需要字场，不加载景观和学习器）
    lightweight_modes = {'parse', 'render_test', 'plan_test', 'multilang'}
    need_lightweight = mode_name in lightweight_modes or mode_name.startswith('plan')

    # 加载模型
    field, landscape, learner = load_models(lightweight=need_lightweight)

    # 分发到新模式
    if args.chat:
        run_chat(field, landscape, args)
        return
    elif args.parse is not None:
        run_parse(field, landscape, args)
        return
    elif args.render:
        run_render_test(field, landscape, args)
        return
    elif args.multiform is not None:
        run_multiform(field, landscape, args)
        return
    elif args.fuzzy:
        run_fuzzy(field, landscape, args)
        return
    elif args.plan is not None:
        run_plan_test(field, landscape, args)
        return
    elif args.lang:
        run_multilang(field, landscape, args)
        return
    elif args.harvest:
        run_harvest(field, landscape, args)
        return
    elif args.contra:
        run_contra(field, landscape, args)
        return

    # 分发到原有模式 (回退到原始 loong_main.py 的逻辑)
    if args.daemon:
        run_daemon(field, landscape, learner, args)
    elif args.seed:
        from scripts.idiom_inject_gpu import main as inject_main
        sys.argv = ['idiom_inject_gpu.py',
                    '--batch', str(args.batch),
                    '--epochs', str(args.epochs),
                    '--lr', str(args.lr)]
        if args.dry_run:
            sys.argv.append('--dry-run')
        inject_main()
    elif args.verify:
        # 原版 run_verify
        import random
        idiom_path = os.path.join(PROJECT, 'data/dicts/idioms.json')
        with open(idiom_path, encoding='utf-8') as f:
            all_idioms = json.load(f)
        random.seed(42)
        valid_idioms = [i for i in all_idioms if len(i) == 4 and all(c in field._char_to_idx for c in i)]
        sample = random.sample(valid_idioms, min(100, len(valid_idioms)))
        correct = 0
        for idiom in sample:
            chars = list(idiom)
            idxs = [field._char_to_idx[c] for c in chars]
            mid = sum(field.anchors[i] for i in idxs) / 4.0
            with torch.no_grad():
                dist = torch.cosine_similarity(mid.unsqueeze(0).to(DEVICE), field.anchors.to(DEVICE), dim=1)
                top = torch.topk(dist, 4).indices.cpu().tolist()
                hit = set(chars) & set(field.hanzi_list[i] for i in top)
                if hit:
                    correct += 1
        log.info(f"📊 检验: {correct}/{len(sample)} = {correct/len(sample)*100:.1f}%")
    elif args.once:
        from scripts.auto_learn import AutoLearningDaemon
        daemon = AutoLearningDaemon(scan_interval=9999, max_learn_per_round=args.max_learn,
                                    factors=None, from_landscape=True)
        daemon.field = field
        daemon.landscape = landscape
        daemon.learner = learner
        daemon.run_round()
    elif args.generate:
        from loongpearl.core.sequence_energy import SequenceEnergy
        seq = SequenceEnergy(field, landscape, device=DEVICE)
        results = seq.complete(args.generate, top_n=10)
        for full, energy in results:
            log.info(f"  {args.generate} → {full} ({energy:.1f})")
    elif args.reason:
        from loongpearl.core.concept_graph import ConceptGraph
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        paths = cg.reason(args.reason, max_hops=3, direction='both')
        for p in paths[:8]:
            log.info(f"  {' → '.join(p)}")
    elif args.concept_action:
        from loongpearl.core.concept_graph import ConceptGraph
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        elif args.concept_action == 'build':
            cg.seed_all_domains()
            cg.induce()
            cg.save(CONCEPT_GRAPH_BASE)
        log.info(f"  概念图: {cg.stats()}")
    elif args.bridge_action:
        from loongpearl.core.concept_graph import ConceptGraph
        from loongpearl.core.cross_domain_bridge import CrossDomainBridgeEngine
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        bridge = CrossDomainBridgeEngine(field, landscape, cg)
        all_bridges = bridge.build_all_bridges(min_confidence=0.3, max_bridges=100)
        log.info(f"  🌉 {len(all_bridges)} 座桥")
    elif args.word:
        from loongpearl.core.concept_graph import ConceptGraph
        from loongpearl.core.word_energy import WordEnergy
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        we = WordEnergy(field, landscape, cg)
        results = we.complete(args.word, top_n=10)
        for text, energy, source in results:
            log.info(f"  {text} ({energy:.1f})")
    elif args.verify_loop:
        from loongpearl.core.concept_graph import ConceptGraph
        from loongpearl.learning.verify_loop import VerifyLoop
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        vf = VerifyLoop(cg)
        report = vf.verify_all_inferred(max_verify=15)
        log.info(f"  验证: {report}")
    elif args.pyramid:
        from loongpearl.core.concept_graph import ConceptGraph
        from loongpearl.core.multi_level import EnergyPyramid
        cg = ConceptGraph(field, landscape)
        if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
            cg.load(CONCEPT_GRAPH_BASE)
        pyramid = EnergyPyramid(field, base_dim=1024, device=DEVICE)
        pyramid.train_all_levels(cg, epochs_per_level=150)

    log.info("🐉 龙珠退出")


if __name__ == '__main__':
    main()
