#!/usr/bin/env python3
"""
端到端测试: 龙珠自主学习回路
═══════════════════════════════════════
流程: 初始化 → 查询未知概念 → 触发联网搜索 → 提取字对 → 
     Hebbian注入 → 保存模型 → 重新查询验证

不依赖 Ollama，全程龙珠自主完成。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.learning.autonomous_learner import AutonomousLearner
from sentence_transformers import SentenceTransformer
import torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("🐉 龙珠自主学习回路 — 端到端测试")
print("=" * 60)
print(f"设备: {DEVICE}")
print()

# ── 1. 加载 ──
t0 = time.time()
print("[1/4] 加载字场...")
field = HanziAnchorField.load(f'{P}/data/models/zichang_94117_1024d.pt', freeze=True)

print("[2/4] 加载能量景观 → GPU...")
ls = FreqEnergyLandscape.load(f'{P}/data/models/energy_landscape_1024d.pt')
ls = ls.to(DEVICE)
ls.eval()

print("[3/4] 初始化学习器...")
learner = DragonBallLearner(ls, field, device=DEVICE)
try:
    learner.calibrate()
    print("  学习器: 已校准")
except Exception as e:
    print(f"  学习器: 校准跳过 ({e})")

print("[4/4] 创建自主学习引擎...")
al = AutonomousLearner(field, ls, learner)
print(f"  就绪!")

load_time = time.time() - t0
print(f"\n⏱ 加载耗时: {load_time:.1f}s\n")

# ── 2. 检测已知概念 vs 未知概念 ──
print("─" * 40)
print("📊 已知/未知检测")
print("─" * 40)

embed_model = SentenceTransformer('BAAI/bge-large-zh', device=DEVICE, local_files_only=True)

def encode(text):
    emb = embed_model.encode([text], normalize_embeddings=True)[0]
    return torch.from_numpy(emb).float().to(DEVICE)

# 已知概念（成语词典里有的）
known_queries = ['一飞冲天', '龙飞凤舞', '画龙点睛']
for q in known_queries:
    vec = encode(q)
    check = learner.check_knowledge(vec)
    status = "✅已知" if check['is_known'] else "❓未知"
    print(f"  {status} '{q}' conf={check['confidence']:.0%} energy={check.get('energy','?'):.1f}")

# 未知概念（不太可能在成语词典里的）
unknown_queries = ['量子纠缠', '神经网络', '机器学习']
for q in unknown_queries:
    vec = encode(q)
    check = learner.check_knowledge(vec)
    status = "✅已知" if check['is_known'] else "❓未知"
    print(f"  {status} '{q}' conf={check['confidence']:.0%} energy={check.get('energy','?'):.1f}")

# ── 3. 自主学习回路 ──
print()
print("─" * 40)
print("🔄 自主学习回路")
print("─" * 40)

# 选一个未知概念来测试
test_query = '量子纠缠'
print(f"\n📖 学习目标: '{test_query}'")

query_vec = encode(test_query)

# 3a. 学前的检测
check_before = learner.check_knowledge(query_vec)
print(f"  学前: is_known={check_before['is_known']}, "
      f"energy={check_before.get('energy','?'):.2f}, "
      f"conf={check_before['confidence']:.0%}")

# 3b. 触发自主学习
print(f"\n  🔍 全网搜索 '{test_query}'...")
t_search = time.time()

result = al.learn_if_unknown(
    query_text=test_query,
    query_vec=query_vec,
    auto_search=True,
)

search_time = time.time() - t_search
print(f"  搜索耗时: {search_time:.1f}s")
print(f"  状态: {result['status']}")
print(f"  注入字对: {result.get('pairs_learned', 0)}")
print(f"  来源: {result.get('sources', [])}")

# 3c. 学后验证
check_after = learner.check_knowledge(query_vec)
print(f"\n  学后: is_known={check_after['is_known']}, "
      f"energy={check_after.get('energy','?'):.2f}, "
      f"conf={check_after['confidence']:.0%}")

# 3d. 能量景观推理
if check_after.get('is_known'):
    print(f"\n  🧠 能量景观推理...")
    infer = ls.infer(query_vec, steps=50)
    resolved = ls.resolve(field, infer['state'], top_k=5)
    print(f"    收敛: {infer['converged']} | 步数: {infer['steps']} | 能量: {infer['energy']:.2f}")
    print(f"    最近汉字:")
    for ch, sim in resolved:
        print(f"      '{ch}' (sim={sim:.4f})")

# ── 4. 学习统计 ──
print()
print("─" * 40)
print("📊 自主学习统计")
print("─" * 40)
print(f"  总搜索: {al.total_searched}")
print(f"  总学习: {al.total_learned}")
print(f"  总注入: {al.total_injected}")

stats = learner.get_stats()
print(f"  学习器总学习: {stats['total_learns']}")
print(f"  学习器总衰减: {stats['total_decays']}")
if stats.get('top_active'):
    print(f"  最活跃字: {', '.join(stats['top_active'])}")

print(f"\n✅ 端到端测试完成!")

# ── 5. 保存更新后的景观 ──
if result['status'] == 'learned' and result.get('pairs_learned', 0) > 0:
    print(f"\n💾 保存更新后的能量景观...")
    # 备份原文件
    save_path = f'{P}/data/models/energy_landscape_1024d.pt'
    backup = save_path + '.auto_learn_backup'
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(save_path, backup)
        print(f"  已备份: {backup}")
    ls.save(save_path)
    print(f"  已保存: {save_path}")
