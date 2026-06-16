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

用法:
    python scripts/auto_learn.py              # 扫描+学习一轮
    python scripts/auto_learn.py --daemon     # 7×24守护进程
    python scripts/auto_learn.py --scan-only  # 只扫描不学习
    python scripts/auto_learn.py --daemon --interval 300  # 每5分钟一轮
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
    """
    
    def __init__(self, scan_interval: int = 120, max_learn_per_round: int = 3,
                 factors: list = None):
        self.scan_interval = scan_interval
        self.max_learn_per_round = max_learn_per_round
        self.active_factors = factors  # None = 全部
        
        # 状态
        self.running = True
        self.rounds = 0
        self.total_learned = 0
        self.total_injected_pairs = 0
        
        # 加载模型
        self._load_models()
        
        # 创建多因子检测器
        self.detector = MultiFactorDetector(
            self.field, self.landscape, self.idioms,
            num_partitions=8
        )
        
        # 创建自主学习引擎
        self.autonomous = AutonomousLearner(
            self.field, self.landscape, self.learner
        )
        
        # 学习日志
        self.learn_log_path = LOG_DIR / 'learned_blindspots.json'
        self._load_learn_log()
        
        log.info(f"🐉 龙珠 7×24 自主学习守护进程就绪")
        log.info(f"   字场:{self.field.num_hanzi} 成语:{len(self.idioms)}")
        log.info(f"   因子:{len(self.detector.factors)} 分区:{self.detector.num_partitions}")
        log.info(f"   间隔:{scan_interval}s 每轮学习:{max_learn_per_round}个")
    
    def _load_models(self):
        """加载所有模型"""
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
        
        with open(os.path.join(PROJECT, 'data/dicts/idioms.json'), encoding='utf-8') as f:
            self.idioms = json.load(f)
    
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
        学习一个盲区。
        
        搜索策略基于因子类型:
          - 死路因子 → 搜索 "{char}字开头的成语"
          - 统计因子 → 搜索 "{char} 成语 接龙"
          - 覆盖因子 → 搜索 "{char} 成语 用法"
          - 其他 → 搜索 "{char} 是什么意思"
        """
        char = gap.char
        factor = gap.factor
        
        # 构造搜索查询
        if factor == 'dead_end':
            query = f"{char}字开头的成语有哪些"
        elif factor == 'statistical':
            query = f"{char} 成语接龙 开头"
        elif factor == 'coverage':
            query = f"{char} 成语 用法 释义"
        elif factor == 'freshness':
            query = f"{char}字 组词 成语"
        else:
            query = f"{char} 成语 释义 出处"
        
        log.info(f"\n  📖 学习: '{char}' [{factor}] → {query}")
        
        t0 = time.time()
        result = self.autonomous.learn_if_unknown(
            query_text=query,
            query_vec=None,
            auto_search=True,
        )
        elapsed = time.time() - t0
        
        # 记录
        log_entry = {
            'char': char,
            'factor': factor,
            'score': gap.score,
            'query': query,
            'status': result['status'],
            'pairs_learned': result.get('pairs_learned', 0),
            'sources': result.get('sources', []),
            'time': elapsed,
            'timestamp': time.time(),
        }
        self.learned_history.append(log_entry)
        
        if result['status'] == 'learned':
            injected = result.get('pairs_learned', 0)
            self.total_injected_pairs += injected
            log.info(f"    ✅ 学会: 注入{injected}对 | {elapsed:.1f}s")
        else:
            log.info(f"    ⚠️ {result['status']}: {result.get('message','')} | {elapsed:.1f}s")
        
        self.total_learned += 1
        return result
    
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
        
        # 3. 保存
        save_path = os.path.join(PROJECT, 'data/models/energy_landscape_1024d.pt')
        self.landscape.save(save_path)
        self._save_learn_log()
        
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
        
        log.info(f"\n📊 最终统计:")
        log.info(f"   总轮数: {self.rounds}")
        log.info(f"   总学习: {self.total_learned}")
        log.info(f"   注入对: {self.total_injected_pairs}")
        log.info(f"   {self.detector.get_stats()}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='龙珠 7×24 自主学习守护进程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_learn.py                    # 一轮扫描+学习
  python auto_learn.py --scan-only        # 只看盲区不学习
  python auto_learn.py --daemon           # 7×24守护进程
  python auto_learn.py --daemon -i 300    # 每5分钟一轮
  python auto_learn.py --daemon -n 10     # 只跑10轮
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
    args = parser.parse_args()
    
    # 解析因子
    factors = None
    if args.factors:
        factors = [f.strip() for f in args.factors.split(',')]
    
    # 创建守护进程
    daemon = AutoLearningDaemon(
        scan_interval=args.interval,
        max_learn_per_round=args.max_learn,
        factors=factors,
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
