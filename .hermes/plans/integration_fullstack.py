#!/usr/bin/env python3
"""龙珠 v4 — 全栈集成测试（含被跳过的 Orchestrator query + daemon_tick_v2）"""
import sys, os, time, json, traceback

PROJECT = "/mnt/d/soso/projects/Loong-agent"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

import torch
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
logging.getLogger('orchestrator').setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")

RESULTS = []
PASS, FAIL = "✅", "❌"

def test(name, fn):
    t0 = time.time()
    try:
        fn()
        dur = time.time() - t0
        RESULTS.append((PASS, name, f"{dur:.2f}s"))
        print(f"  {PASS} {name} ({dur:.1f}s)")
    except Exception as e:
        dur = time.time() - t0
        RESULTS.append((FAIL, name, str(e)[:120]))
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()

# ═══════════════════════════════════════════════
print("=" * 60)
print("龙珠 v4 — 全栈集成测试 (Orchestrator)")
print("=" * 60)

# ── 1. 加载模型 ──
print("\n📦 1. 加载全栈模型")

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.core.orchestrator import create_orchestrator_with_sequential

FIELD = None
LANDSCAPE = None
LEARNER = None
ORCH = None

def _load_field():
    global FIELD
    FIELD = HanziAnchorField.load(
        os.path.join(PROJECT, "data/models/zichang_94117_1024d.pt"),
        freeze=True
    )
    print(f"    字场={FIELD.num_hanzi}字, {FIELD.embed_dim}d")

test("加载字场 (94117字, 369MB)", _load_field)

def _load_landscape():
    global LANDSCAPE
    LANDSCAPE = FreqEnergyLandscape.load(
        os.path.join(PROJECT, "data/models/energy_landscape_1024d.pt")
    ).to(DEVICE).eval()
    print(f"    景观=已加载")

test("加载能量景观", _load_landscape)

def _create_learner():
    global LEARNER
    LEARNER = DragonBallLearner(LANDSCAPE, FIELD, device=DEVICE)
    LEARNER.calibrate()
    print(f"    学习器=已就绪")

test("初始化学习器", _create_learner)

# ── 2. 创建 Orchestrator ──
print("\n🎯 2. 创建 Orchestrator")

def _create_orch():
    global ORCH
    ORCH = create_orchestrator_with_sequential(FIELD, LANDSCAPE, LEARNER)
    print(f"    创建成功")

test("create_orchestrator_with_sequential()", _create_orch)

# ── 3. orchestrator.query() 五步管道 ──
print("\n🔗 3. orchestrator.query() 五步管道测试")

def _query_dragon():
    r = ORCH.query("龙是什么")
    answer = str(r.get("answer", ""))[:50]
    print(f"    stage={r.get('stage')}, path={r.get('path')},"
          f" strategy={r.get('strategy')},"
          f" answer前50={repr(answer)}")

test("query('龙是什么')", _query_dragon)

def _query_quantum():
    r = ORCH.query("什么是量子")
    answer = str(r.get("answer", ""))[:60]
    print(f"    stage={r.get('stage')}, path={r.get('path')},"
          f" signal={r.get('signal')}, status={r.get('status')},"
          f" answer前60={repr(answer)}")

test("query('什么是量子')", _query_quantum)

def _query_compare():
    r = ORCH.query("原子和分子的区别")
    print(f"    stage={r.get('stage')}, path={r.get('path')},"
          f" strategy={r.get('strategy')},"
          f" signals={len(r.get('signal_results', []))}")

test("query('原子和分子的区别')", _query_compare)

def _query_poem():
    r = ORCH.query("写一首关于春天的诗")
    answer = str(r.get("answer", ""))[:50]
    print(f"    stage={r.get('stage')}, path={r.get('path')},"
          f" strategy={r.get('strategy')},"
          f" answer前50={repr(answer)}")

test("query('写一首关于春天的诗')", _query_poem)

def _query_hello():
    r = ORCH.query("你好")
    answer = str(r.get("answer", ""))[:40]
    print(f"    stage={r.get('stage')}, path={r.get('path')},"
          f" strategy={r.get('strategy')},"
          f" answer前40={repr(answer)}")

test("query('你好')", _query_hello)

# ── 4. daemon_tick_v2() 守护循环 ──
print("\n🔄 4. daemon_tick_v2() 守护循环测试")

def _tick1():
    tick = ORCH.daemon_tick_v2(1)
    print(f"    round={tick.get('round')}, events={tick.get('events_processed')},"
          f" threads={tick.get('active_threads')},"
          f" signals={tick.get('signals_handled', [])[:3]}")

test("daemon_tick_v2(round=1)", _tick1)

def _tick2():
    tick = ORCH.daemon_tick_v2(2)
    print(f"    round={tick.get('round')}, events={tick.get('events_processed')},"
          f" threads={tick.get('active_threads')},"
          f" pending={tick.get('pending_queue_size', 0)}")

test("daemon_tick_v2(round=2)", _tick2)

# ── 汇总 ──
print("\n" + "=" * 60)
print("全栈测试汇总")
print("=" * 60)

passed = sum(1 for r in RESULTS if r[0] == PASS)
failed = sum(1 for r in RESULTS if r[0] == FAIL)

print(f"\n{PASS} 通过: {passed}")
print(f"{FAIL} 失败: {failed}")
print(f"总计: {len(RESULTS)}")

if failed > 0:
    print(f"\n失败详情:")
    for status, name, detail in RESULTS:
        if status == FAIL:
            print(f"  {FAIL} {name}: {detail}")

report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "type": "fullstack",
    "total": len(RESULTS),
    "passed": passed,
    "failed": failed,
    "skipped": 0,
    "results": [{"status": s, "name": n, "detail": d} for s, n, d in RESULTS],
}
report_path = os.path.join(PROJECT, ".hermes", "plans", "test_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n报告已保存: {report_path}")

sys.exit(0 if failed == 0 else 1)
