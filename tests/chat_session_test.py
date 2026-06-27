#!/usr/bin/env python3
"""龙珠 v4 会话聊天测试 — 多类型交互"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.core.orchestrator import create_orchestrator

print("🐉 加载龙珠...")
t0 = time.time()
hf = HanziAnchorField.load("data/models/zichang_94117_1024d.pt")
el = FreqEnergyLandscape.load("data/models/energy_landscape_1024d.pt")
orch = create_orchestrator(field=hf, landscape=el)
print(f"✅ 加载完成 ({time.time()-t0:.1f}s)\n")

# ═══════════════════════════════════════════════
queries = [
    # 1. 日常闲聊
    ("闲聊-问候", "你好啊，今天心情怎么样？"),
    ("闲聊-自我介绍", "你是谁？介绍一下自己"),
    ("闲聊-能力边界", "你能帮我做什么？"),
    
    # 2. 知识查询（概念图内）
    ("知识-龙", "龙是什么？"),
    ("知识-量子", "量子力学的基本原理是什么？"),
    ("知识-李白", "李白是谁？他写过什么诗？"),
    ("知识-人工智能", "什么是人工智能？"),
    ("知识-深度学习", "深度学习和机器学习有什么区别？"),
    
    # 3. 成语
    ("成语-解释", "画龙点睛是什么意思？"),
    ("成语-接龙", "龙飞凤舞下一句是什么？"),
    ("成语-用法", "一鸣惊人用在哪里合适？"),
    
    # 4. 诗词
    ("诗词-春天", "写一首关于春天的诗"),
    ("诗词-月亮", "以月亮为题写一首诗"),
    
    # 5. 关系推理
    ("推理-因果", "为什么会下雨？"),
    ("推理-对比", "火和水有什么区别？"),
    ("推理-组成", "原子由什么组成？"),
    
    # 6. 边界/鲁棒性
    ("边界-生僻字", "龘是什么意思？"),
    ("边界-空输入", ""),
    ("边界-符号", "???"),
    ("边界-无意义", "xyzasdf1234"),
    
    # 7. 汉字推理
    ("汉字-太阳", "太阳怎么写？和哪些字有关？"),
    ("汉字-爱", "爱这个字怎么解释？"),
]

print("=" * 70)
print("🐉 龙珠 v4 会话测试")
print("=" * 70)

for cat, q in queries:
    print(f"\n{'─' * 60}")
    print(f"📌 [{cat}]")
    print(f"👤 问: {repr(q) if q == '' else q}")
    
    t1 = time.time()
    try:
        result = orch.query(q)
        elapsed = time.time() - t1
        
        status = result.get('status', '?')
        confidence = result.get('confidence', 0)
        answer = result.get('answer', '')
        route = result.get('route', '?')
        
        # 截断长回答
        display_answer = answer[:500] if len(answer) > 500 else answer
        
        print(f"🐉 答: {display_answer}")
        print(f"   ⏱ {elapsed:.1f}s | 状态:{status} | 置信:{confidence:.0%} | 路由:{route}")
        
    except Exception as e:
        print(f"   ❌ 异常: {e}")

print(f"\n{'=' * 70}")
print("✅ 会话测试完成")
