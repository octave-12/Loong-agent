#!/usr/bin/env python3
"""
龙珠 7×24 自主学习守护进程 (auto_learn.py v2)
═══════════════════════════════════════════════════
7个独立因子 × 分区并行扫描 × 优先级队列 → 持续自主学习

架构:
  检测层: MultiFactorDetector(7因子) → 盲区优先级队列
  学习层: AutonomousLearner → 搜索→提取→注入
  调度层: 取盲区→学习→验证→标记→重扫受影响分区
  持久层: 定期保存能量景观 + 学习日志

不依赖 Hermes 喂数据，不依赖 Ollama。
龙珠自己发现「我不知道什么」，自己去学。

检测模式:
  from_landscape=True (默认): 纯景观模式。
    能量景观是唯一知识源。idioms.json 仅作为训练引导的种子知识，
    不是运行时知识源。
  from_landscape=False (--use-idioms): 传统模式。
    使用 idioms.json 统计信息辅助检测。

用法:
    python scripts/auto_learn.py              # 扫描+学习一轮(纯景观)
    python scripts/auto_learn.py --daemon     # 7×24守护进程(纯景观)
    python scripts/auto_learn.py --use-idioms # 传统模式(使用idioms)
    python scripts/auto_learn.py --scan-only  # 只扫描不学习
    python scripts/auto_learn.py --daemon --interval 300  # 每5分钟一轮
    python scripts/auto_learn.py --daemon --stage 2       # 从阶段2开始
"""

import sys, os, json, time, argparse, signal, logging, threading
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.learning.autonomous_learner import AutonomousLearner
from loongpearl.learning.blindspot_detector import (
    MultiFactorDetector, BlindSpot, ScanResult
)
from loongpearl.learning.curriculum import BabyCurriculum

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 日志配置
LOG_DIR = Path(PROJECT) / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'auto_learn.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger('auto_learn')


# ============================================================================
# 自主学习守护进程
# ============================================================================

class AutoLearningDaemon:
    """
    7×24 自主学习守护进程。
    
    主循环:
      while True:
        1. 全因子扫描 → 发现盲区
        2. 从优先级队列取 top-N 个盲区
        3. 对每个盲区: 搜索 → 提取字对 → Hebbian注入
        4. 验证学习效果(能量变化)
        5. 标记已学习 → 更新队列
        6. 保存能量景观
        7. 休眠 interval 秒 → 下一轮
    
    检测模式:
      from_landscape=True (默认): 纯景观模式，所有检测从能量景观拓扑出发。
        能量景观是唯一知识源。idioms.json 仅作为种子知识/训练引导。
      from_landscape=False: 传统模式，使用 idioms.json 统计信息辅助检测。
    """
    
    def __init__(self, scan_interval: int = 120, max_learn_per_round: int = 3,
                 factors: list = None, from_landscape: bool = True,
                 start_stage: int = 0):
        self.scan_interval = scan_interval
        self.max_learn_per_round = max_learn_per_round
        self.active_factors = factors  # None = 全部
        self.from_landscape = from_landscape
        self.start_stage = start_stage  # 0 = 从保存的进度恢复
        
        # 状态
        self.running = True
        self.rounds = 0
        self.total_learned = 0
        self.total_injected_pairs = 0
        
        # 加载模型
        self._load_models()
        
        # 创建多因子检测器
        # idioms 作为种子知识/训练引导传入，非运行时知识源
        self.detector = MultiFactorDetector(
            self.field, self.landscape, self.idioms,
            num_partitions=8,
            from_landscape=from_landscape,
        )
        
        # 创建自主学习引擎
        self.autonomous = AutonomousLearner(
            self.field, self.landscape, self.learner
        )
        
        # 学习日志
        self.learn_log_path = LOG_DIR / 'learned_blindspots.json'
        self._load_learn_log()
        
        # 课程（婴儿式成长）
        self.curriculum = BabyCurriculum()
        if self.start_stage > 0 and self.start_stage <= 8:
            self.curriculum.current_stage = self.start_stage
        
        log.info(f"🐉 龙珠 7×24 自主学习守护进程就绪")
        log.info(f"   字场:{self.field.num_hanzi} 成语(种子):{len(self.idioms) if self.idioms else 0}")
        log.info(f"   模式:{'纯景观' if from_landscape else '传统(idioms)'}")
        log.info(f"   因子:{len(self.detector.factors)} 分区:{self.detector.num_partitions}")
        log.info(f"   间隔:{scan_interval}s 每轮学习:{max_learn_per_round}个")
        log.info(f"   课程: 阶段{self.curriculum.current_stage} ({self.curriculum.STAGES[self.curriculum.current_stage]}) "
                 f"已识{len(self.curriculum.known_chars)}字")
    
    def _load_models(self):
        """加载所有模型
        
        idioms.json = 种子知识/训练引导数据，非运行时知识源。
        在 from_landscape=True 模式下仍然加载它供课程系统使用，
        但检测器不依赖它进行盲区检测。
        """
        log.info("加载字场...")
        self.field = HanziAnchorField.load(
            os.path.join(PROJECT, 'data/models/zichang_94117_1024d.pt'),
            freeze=True
        )
        
        log.info("加载能量景观...")
        self.landscape = FreqEnergyLandscape.load(
            os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        ).to(DEVICE).eval()
        
        log.info("初始化学习器...")
        self.learner = DragonBallLearner(self.landscape, self.field, device=DEVICE)
        try:
            self.learner.calibrate()
        except Exception as e:
            log.warning(f"校准跳过: {e}")
        
        # idioms.json: 种子知识/训练引导，非运行时知识源
        # 纯景观模式下仍加载供课程系统参考，但检测器不使用
        idioms_path = os.path.join(PROJECT, 'data/dicts/idioms.json')
        if os.path.exists(idioms_path):
            with open(idioms_path, encoding='utf-8') as f:
                self.idioms = json.load(f)
        else:
            self.idioms = None
    
    def _load_learn_log(self):
        """加载已学习记录"""
        if self.learn_log_path.exists():
            with open(self.learn_log_path, encoding='utf-8') as f:
                self.learned_history = json.load(f)
        else:
            self.learned_history = []
    
    def _save_learn_log(self):
        """保存学习记录"""
        with open(self.learn_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.learned_history, f, ensure_ascii=False, indent=2)
    
    def scan(self) -> list:
        """全因子扫描 → 返回优先级队列"""
        log.info(f"\n{'='*50}")
        log.info(f"🔍 第 {self.rounds+1} 轮扫描")
        log.info(f"{'='*50}")
        
        t0 = time.time()
        results = self.detector.scan_all(
            parallel=False,  # 串行更稳定, GPU不能多线程抢
            factors=self.active_factors,
        )
        elapsed = time.time() - t0
        
        top = self.detector.top_gaps(15)
        
        if top:
            log.info(f"  优先学习队列 (前10):")
            for i, gap in enumerate(top[:10]):
                ev = str(gap.evidence)[:80]
                log.info(f"    {i+1}. '{gap.char}' [{gap.factor}] "
                         f"分数={gap.score:.1f} {ev}")
        else:
            log.info(f"  ✅ 无盲区! 知识完备")
        
        return top
    
    def learn_one(self, gap: BlindSpot) -> dict:
        """
        学习一个盲区——不再只搜单字，而是搜该字的词语簇。
        
        搜索策略:
          - 死路因子 → 搜索该字开头的成语
          - 梯度因子 → 三连搜: 成语+组词+常见搭配
          - 语义因子 → 搜索该字及其关联字的组合词
          - 其他 → 组词优先
        """
        char = gap.char
        factor = gap.factor
        
        # 构造多查询——不是搜孤立的字，而是搜字在词语中的用法
        queries = []
        if factor == 'dead_end':
            queries = [f"{char}字开头的成语有哪些", f"{char}成语大全"]
        elif factor == 'gradient':
            # 梯度异常的字：搜成语 + 组词 + 常见搭配，三管齐下
            queries = [
                f"{char} 成语 释义",           # 成语层面
                f"{char}字 组词 常见词语",      # 词语层面
                f"含有{char}字的词语 成语",     # 包含该字的词
            ]
        elif factor == 'semantic':
            related = gap.evidence.get('related_char', '') if isinstance(gap.evidence, dict) else ''
            if related:
                queries = [
                    f"{char}{related} 成语 词语",      # 关联字组合
                    f"{char}和{related} 成语 组词",     # 两者关系
                ]
            else:
                queries = [f"{char} 组词 成语 搭配"]
        elif factor == 'statistical':
            queries = [f"{char} 成语接龙", f"{char}字 常见词语"]
        elif factor == 'coverage':
            queries = [f"{char} 成语 用法 释义", f"{char} 组词"]
        elif factor == 'freshness':
            queries = [f"{char}字 组词 新词语"]
        else:
            queries = [f"{char} 组词 成语 搭配"]
        
        # 合并所有查询结果
        all_pairs = 0
        total_results = []
        for query in queries:
            log.info(f"\n  📖 学习: '{char}' [{factor}] → {query}")
            
            t0 = time.time()
            result = self.autonomous.learn_if_unknown(
                query_text=query,
                query_vec=None,
                auto_search=True,
            )
            elapsed = time.time() - t0
            
            pairs = result.get('pairs_learned', 0)
            all_pairs += pairs
            total_results.append(result)
            
            if result['status'] == 'learned':
                log.info(f"    ✅ {query[:20]}... 注入{pairs}对 | {elapsed:.1f}s")
            else:
                log.info(f"    ⚠️ {result['status']}: {result.get('message','')[:40]} | {elapsed:.1f}s")
        
        # 合并记录（取第一次搜索的结果状态）
        primary = total_results[0] if total_results else {'status': 'search_failed', 'pairs_learned': 0}
        
        # 记录
        log_entry = {
            'char': char,
            'factor': factor,
            'score': gap.score,
            'queries': queries,
            'status': primary['status'],
            'pairs_learned': all_pairs,
            'sources': primary.get('sources', []),
            'time': time.time(),
            'timestamp': time.time(),
        }
        self.learned_history.append(log_entry)
        
        self.total_injected_pairs += all_pairs
        
        if all_pairs > 0:
            log.info(f"    ✅ 学会: 共{len(queries)}次搜索 注入{all_pairs}对")
        else:
            log.info(f"    ⚠️ {primary['status']}: {primary.get('message','')[:40]}")
        
        self.total_learned += 1
        return primary
    
    def run_round(self):
        """执行一轮完整循环"""
        self.rounds += 1
        
        # 1. 扫描
        top_gaps = self.scan()
        
        if not top_gaps:
            # 没有盲区了, 加长休眠
            log.info(f"  💤 无盲区, 30分钟后重扫...")
            return
        
        # 2. 学习前N个
        learned_this_round = 0
        for i, gap in enumerate(top_gaps[:self.max_learn_per_round]):
            if gap.char in self.detector._learned:
                continue
            
            # 验证是否真的需要学(二次确认)
            # 跳过已学过的字
            learned_chars = {e['char'] for e in self.learned_history 
                           if e['status'] == 'learned'}
            if gap.char in learned_chars:
                self.detector.mark_learned(gap.char)
                continue
            
            self.learn_one(gap)
            self.detector.mark_learned(gap.char)
            learned_this_round += 1
            
            # 学习间隙(避免被搜索引擎封)
            if i < self.max_learn_per_round - 1:
                time.sleep(2)
        
        # 3. 衰减(废退) — 每轮对久未使用的知识施加衰减
        try:
            decay_result = self.learner.decay_step()
            if decay_result.get('decayed', 0) > 0:
                log.info(f"  📉 衰减: {decay_result.get('decayed',0)} 条弱化")
        except Exception:
            pass
        
        # 4. 保存——只在真正学到新知识时保存（防止无效覆盖已注入的深盆地）
        if learned_this_round > 0:
            save_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
            self.landscape.save(save_path)
            self._save_learn_log()
            # 同时保存概念图（如果有新提取的三元组）
            if hasattr(self, 'autonomous') and self.autonomous.total_concept_triples > 0:
                try:
                    self.autonomous.save_concept_graph()
                except Exception:
                    pass
            log.info(f"  💾 模型已保存 ({learned_this_round}字, 本轮注入{self.total_injected_pairs}对)")
        else:
            self._save_learn_log()  # 仍保存学习记录（失败原因等）
        
        # 5. 课程推进（每轮尝试，不抛异常不影响主流程）
        try:
            old_stage = self.curriculum.current_stage
            advanced = self.curriculum.advance_if_ready()
            if advanced:
                log.info(f"🐣 课程进阶！阶段 {old_stage} → {self.curriculum.current_stage} "
                         f"({self.curriculum.STAGES[self.curriculum.current_stage]})")
                self.curriculum.save_progress()
            elif self.curriculum.current_stage <= 4:
                try:
                    self.curriculum.learn_next_batch(10)
                    self.curriculum.save_progress()
                except Exception:
                    pass
        except Exception:
            pass
        
        # 6. 定期跨学科桥接（每5轮自动运行一次）
        try:
            bridge_interval = getattr(self, '_last_bridge_round', 0)
            if self.rounds - bridge_interval >= 5:
                self._last_bridge_round = self.rounds
                log.info(f"  🌉 第 {self.rounds} 轮: 自动跨学科桥接...")
                from loongpearl.core.cross_domain_bridge import CrossDomainBridgeEngine
                from loongpearl.core.concept_graph import ConceptGraph
                cg = self.autonomous.concept_graph
                engine = CrossDomainBridgeEngine(self.field, self.landscape, cg)
                bridges = engine.build_all_bridges(min_confidence=0.3, max_bridges=50)
                n_added = engine.add_bridges_to_concept_graph(bridges, min_confidence=0.4)
                if n_added > 0:
                    cg.induce()
                    self.autonomous.save_concept_graph()
                    report = cg.evaluate()
                    log.info(f"  🌉 桥接完成: +{n_added}桥, 连通性:{report['connectivity']:.3f}, "
                            f"三元组:{report['triples']}")
        except Exception as e:
            pass
        
        # 7. 定期剪枝（每20轮，清除积累的低质量推断）
        try:
            prune_interval = getattr(self, '_last_prune_round', -20)
            if self.rounds - prune_interval >= 20:
                self._last_prune_round = self.rounds
                cg = self.autonomous.concept_graph
                removed = cg.prune(min_confidence=0.1, min_evidence=0)
                if removed > 0:
                    self.autonomous.save_concept_graph()
                    log.info(f"  ✂️ 剪枝: 移除{removed}条低质量三元组, 剩余{cg.total_triples}条")

            # ★ 知识对齐已归入 orchestrator 统一调度
            # 概念图→能量景观的写入权仅属于大脑(orchestrator)
            # 此处不再直接调用 cg.align_to_landscape()
            pass
        except Exception as e:
            pass

        # 8. 定期闭环验证（每10轮自动验证一次推断三元组）
        try:
            verify_interval = getattr(self, '_last_verify_round', -10)
            if self.rounds - verify_interval >= 10:
                self._last_verify_round = self.rounds
                cg = self.autonomous.concept_graph
                from loongpearl.learning.verify_loop import VerifyLoop
                vf = VerifyLoop(cg)
                v_report = vf.verify_lowest_confidence(n=5)
                if v_report['confirmed'] > 0 or v_report['contradicted'] > 0:
                    self.autonomous.save_concept_graph()
                    log.info(f"  🔄 闭环验证: 确认{v_report['confirmed']}条 "
                            f"矛盾{v_report['contradicted']}条 不确定{v_report['uncertain']}条")
        except Exception as e:
            pass
        
        stats = self.detector.get_stats()
        log.info(f"\n  📊 本轮: 学{learned_this_round}个 | "
                 f"累计: 学{self.total_learned} 注入{self.total_injected_pairs}对 | "
                 f"剩余盲区: {stats['pending_gaps']}")
    
    def run_daemon(self, max_rounds: int = 0):
        """7×24 守护进程主循环"""
        log.info(f"\n🔄 7×24 守护进程启动")
        log.info(f"   每 {self.scan_interval}s 扫描一轮")
        log.info(f"   按 Ctrl+C 停止\n")
        
        # 注册信号处理
        def _shutdown(sig, frame):
            log.info(f"\n🛑 收到停止信号, 保存后退出...")
            self.running = False
        
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        
        round_count = 0
        while self.running:
            self.run_round()
            round_count += 1
            
            if max_rounds > 0 and round_count >= max_rounds:
                log.info(f"\n✅ 完成 {max_rounds} 轮, 退出")
                break
            
            if self.running:
                log.info(f"\n⏳ 休眠 {self.scan_interval}s...")
                time.sleep(self.scan_interval)
        
        # 最终保存
        save_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        self.landscape.save(save_path)
        self._save_learn_log()
        # 最终保存概念图
        if hasattr(self, 'autonomous'):
            try:
                self.autonomous.save_concept_graph()
            except Exception:
                pass
        
        log.info(f"\n📊 最终统计:")
        log.info(f"   总轮数: {self.rounds}")
        log.info(f"   总学习: {self.total_learned}")
        log.info(f"   注入对: {self.total_injected_pairs}")
        log.info(f"   {self.detector.get_stats()}")
        
        # 课程统计
        try:
            c = self.curriculum
            log.info(f"\n🐣 课程进度:")
            log.info(f"   阶段: {c.current_stage} ({c.STAGES.get(c.current_stage, '未知')})")
            log.info(f"   已识单字: {len(c.known_chars)}")
            log.info(f"   已学词语: {len(c.known_words)}")
            log.info(f"   掌握单字: {len(c.mastered_chars)}")
            c.save_progress()
        except Exception:
            pass


# ============================================================================
# CLI
# ============================================================================

def main():
    # 单例锁：防止重复启动
    PID_FILE = os.path.join(PROJECT, 'logs/auto_learn.pid')
    try:
        with open(PID_FILE) as f:
            old_pid = int(f.read().strip())
        try:
            os.kill(old_pid, 0)
            print(f"❌ 守护进程已在运行 (PID {old_pid})，先 kill {old_pid} 再重试")
            sys.exit(1)
        except OSError:
            pass  # 旧进程已死
    except (FileNotFoundError, ValueError):
        pass
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    # 注册退出清理
    import atexit
    atexit.register(lambda: os.path.exists(PID_FILE) and os.remove(PID_FILE))
    
    parser = argparse.ArgumentParser(
        description='龙珠 7×24 自主学习守护进程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_learn.py                    # 一轮扫描+学习(纯景观模式)
  python auto_learn.py --use-idioms       # 传统模式(使用idioms统计)
  python auto_learn.py --scan-only        # 只看盲区不学习
  python auto_learn.py --daemon           # 7×24守护进程
  python auto_learn.py --daemon -i 300    # 每5分钟一轮
  python auto_learn.py --daemon -n 10     # 只跑10轮
  python auto_learn.py --daemon --stage 2 # 从阶段2(组合字)开始
  python auto_learn.py -f statistical,dead_end  # 只用指定因子
        """
    )
    parser.add_argument('--daemon', '-d', action='store_true', 
                       help='守护进程模式(持续运行)')
    parser.add_argument('--scan-only', '-s', action='store_true',
                       help='只扫描不学习')
    parser.add_argument('--interval', '-i', type=int, default=120,
                       help='守护模式: 扫描间隔(秒), 默认120')
    parser.add_argument('--max-rounds', '-n', type=int, default=0,
                       help='守护模式: 最大轮数, 0=无限')
    parser.add_argument('--max-learn', '-m', type=int, default=3,
                       help='每轮最多学习几个盲区, 默认3')
    parser.add_argument('--factors', '-f', type=str, default=None,
                       help='指定因子(逗号分隔), 默认全部')
    parser.add_argument('--stage', type=int, default=0,
                       help='起始课程阶段(1-8), 0=从保存进度恢复, 默认0')
    parser.add_argument('--use-idioms', action='store_true',
                       help='使用 idioms.json 统计信息辅助检测(默认: 纯景观模式)')
    args = parser.parse_args()
    
    # 解析因子
    factors = None
    if args.factors:
        factors = [f.strip() for f in args.factors.split(',')]
    
    # 检测模式: 默认 from_landscape=True (纯景观)
    # idioms.json 是种子知识/训练引导，不是运行时知识源
    from_landscape = not args.use_idioms
    
    # 创建守护进程
    daemon = AutoLearningDaemon(
        scan_interval=args.interval,
        max_learn_per_round=args.max_learn,
        factors=factors,
        from_landscape=from_landscape,
        start_stage=args.stage,
    )
    
    if args.scan_only:
        daemon.scan()
        return
    
    if args.daemon:
        daemon.run_daemon(max_rounds=args.max_rounds)
    else:
        daemon.run_round()


if __name__ == '__main__':
    main()
