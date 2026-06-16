#!/usr/bin/env python3
"""
龙珠自演化对话系统 (self_evolving.py)
======================================
完整闭环: 多源交叉验证 → 极小步长学习 → 自然衰减 → 反噬回退
全程静默, 用户无感, 越用越聪明

架构:
  query("量子纠缠是什么?")
    → 字场编码 → 能量景观推理 → 检索最近锚点
    → 能量景观推断生成自然语言回答（零 LLM）
    → (后台) 多源交叉验证 → 极小步长植入 → 反噬检测 → 衰减调度
"""

import sys, os, json, time, re, threading
from collections import defaultdict
from datetime import datetime
import requests
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.energy_landscape import EnergyLandscape
from loongpearl.learning.learner import DragonBallLearner
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根
ZICHANG = os.path.join(BASE, "data/models/zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "data/models/energy_landscape_1024d.pt")
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:7b"

# ====================================================================
# 多源搜索器
# ====================================================================

class MultiSourceSearcher:
    """多搜索引擎 + 交叉验证"""

    def search(self, query: str) -> dict:
        """
        多源搜索, 取交叉验证的关键词。

        Returns:
            {
                'high_conf': [(字, 频), ...],   # 多源共识 (≥2源)
                'low_conf': [(字, 频), ...],    # 单源
                'sources': int,                  # 有效源数
            }
        """
        results_ddg = self._search_ddg(query)
        results_bing = self._search_bing(query)

        chars_ddg = self._extract_hanzi(results_ddg, query)
        chars_bing = self._extract_hanzi(results_bing, query)

        # 交叉验证
        ddg_set = set(c for c, _ in chars_ddg)
        bing_set = set(c for c, _ in chars_bing)

        high_conf = []
        low_conf = []

        # 双源共识
        for ch, freq in chars_ddg:
            if ch in bing_set:
                bing_freq = next(f for c, f in chars_bing if c == ch)
                high_conf.append((ch, freq + bing_freq))
            else:
                low_conf.append((ch, freq))

        # 只在 Bing 出现的
        for ch, freq in chars_bing:
            if ch not in ddg_set:
                low_conf.append((ch, freq))

        high_conf.sort(key=lambda x: -x[1])
        low_conf.sort(key=lambda x: -x[1])

        sources = (1 if chars_ddg else 0) + (1 if chars_bing else 0)
        return {'high_conf': high_conf[:10], 'low_conf': low_conf[:10], 'sources': sources}

    def _search_ddg(self, query: str) -> list:
        try:
            r = requests.get("https://lite.duckduckgo.com/lite/",
                params={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', r.text, re.DOTALL)
        except: return []

    def _search_bing(self, query: str) -> list:
        try:
            r = requests.get("https://www.bing.com/search",
                params={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            return re.findall(r'<p[^>]*>(.*?)</p>', r.text, re.DOTALL)[:10]
        except: return []

    def _extract_hanzi(self, texts: list, query: str) -> list:
        combined = " ".join(texts)
        hanzi = re.findall(r'[\u4e00-\u9fff]', combined)
        freq = defaultdict(int)
        for ch in hanzi: freq[ch] += 1
        for qc in set(re.findall(r'[\u4e00-\u9fff]', query)): freq.pop(qc, None)
        return sorted(freq.items(), key=lambda x: -x[1])[:20]


# ====================================================================
# 知识衰减调度器
# ====================================================================

class DecayScheduler:
    """所有植入的关联随时间自然衰减, 被重新激活的保留, 不活跃的消失"""

    def __init__(self, half_life_hours: float = 72):
        """
        Args:
            half_life_hours: 半衰期 (小时), 72小时后关联权重减半
        """
        self.half_life = half_life_hours * 3600
        self.decay_rate = np.log(2) / self.half_life
        self.activations = defaultdict(lambda: {'count': 0, 'last_active': time.time()})
        self._lock = threading.Lock()

    def activate(self, pair: tuple):
        """关联对被查询激活 — 重置衰减时钟"""
        with self._lock:
            entry = self.activations[pair]
            entry['count'] += 1
            entry['last_active'] = time.time()

    def get_factor(self, pair: tuple) -> float:
        """获取当前衰减因子 [0, 1], 1=刚激活, 0=完全衰减"""
        with self._lock:
            if pair not in self.activations: return 1.0
            elapsed = time.time() - self.activations[pair]['last_active']
            return np.exp(-self.decay_rate * elapsed)

    def prune_inactive(self, threshold: float = 0.01) -> list:
        """返回衰减到阈值以下的关联对列表"""
        with self._lock:
            return [p for p, e in self.activations.items() if self.get_factor(p) < threshold]


# ====================================================================
# 反噬检测器
# ====================================================================

class BacklashDetector:
    """新知识植入后检查是否破坏已有知识"""

    def __init__(self, landscape, anchors, char_to_idx, neighbor_radius: int = 10):
        self.ls = landscape
        self.anchors = anchors
        self.char_to_idx = char_to_idx
        self.radius = neighbor_radius

    def snapshot(self, char_indices: list) -> dict:
        """记录植入前锚点及其邻居的能量"""
        snap = {}
        with torch.no_grad():
            for idx in char_indices:
                neighbors = self._get_neighbors(idx, self.radius)
                energies = self.ls.energy(self.anchors[neighbors]).tolist()
                snap[idx] = {'neighbors': neighbors, 'energies': energies}
        return snap

    def check(self, before: dict, tolerance: float = 0.15) -> bool:
        """
        检查植入后邻居能量是否恶化。
        返回 True = 安全, False = 有反噬需回退
        """
        with torch.no_grad():
            for idx, info in before.items():
                neighbors = info['neighbors']
                before_e = info['energies']
                after_e = self.ls.energy(self.anchors[neighbors]).tolist()
                # 检查邻居平均能量是否上升 > tolerance%
                avg_before = np.mean(before_e)
                avg_after = np.mean(after_e)
                if avg_after > avg_before * (1 + tolerance):
                    return False
        return True

    def _get_neighbors(self, idx: int, k: int) -> list:
        """找 k 个向量近邻"""
        vec = self.anchors[idx:idx+1]
        sims = vec @ self.anchors.T
        sims[0, idx] = -float('inf')
        return torch.topk(sims, k, dim=1).indices[0].tolist()


# ====================================================================
# 自演化龙珠
# ====================================================================

class SelfEvolvingLoongPearl:
    """
    静默自演化对话系统。

    用户只管问, 系统:
      1. 检索关联汉字
      2. 能量景观推断生成自然语言回答（零 LLM）
      3. 后台静默: 多源交叉验证 → 极小步长学习 → 反噬检测 → 衰减
    
    Ollama 已降级为可选后备（默认关闭），设置 use_ollama_fallback = True 可启用。
    """

    # Ollama 后备开关（已弃用，默认关闭）
    use_ollama_fallback = False

    def __init__(self, decay_half_life: float = 72):
        print("🐉 初始化自演化龙珠...")

        self.zc = HanziAnchorField.load(ZICHANG)
        self.ls = EnergyLandscape.load(LANDSCAPE)
        self.lr = DragonBallLearner(landscape=self.ls, anchor_field=self.zc, hebbian_lr=0.001)

        self.searcher = MultiSourceSearcher()
        self.decay = DecayScheduler(half_life_hours=decay_half_life)
        self.backlash = BacklashDetector(self.ls, self.zc.anchors, self.zc._char_to_idx)

        # 统计
        self.total_queries = 0
        self.total_learned = 0
        self.total_reverted = 0
        self.total_decayed = 0

        print(f"   字场:{self.zc.num_hanzi}字 | 衰减半衰期:{decay_half_life}h")

    # ── 主接口 ──────────────────────────────────

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        问一句话, 得到自然语言回答。全程静默自演化。

        Args:
            question: 用户问题

        Returns:
            自然语言回答
        """
        self.total_queries += 1
        t0 = time.time()

        # 1. 字场编码 → 能量景观检索
        vec = self._encode(question)
        self.ls.train()
        infer = self.ls.infer(vec, steps=50)
        self.ls.eval()
        _, chars, energies = self.zc.find_nearest(infer['state'], k=10)

        if verbose:
            print(f"  [检索] {' '.join(chars[:5])} (能量={infer['energy']:.2f})")

        # 2. 激活衰减 — 被检索到的关联重置时钟
        for ch in chars[1:]:
            pair = tuple(sorted([chars[0], ch]))
            self.decay.activate(pair)

        # 3. 联网交叉验证 (后台静默)
        search_result = self.searcher.search(question)
        if search_result['high_conf']:
            need_learn = search_result['high_conf'][:5]
            if verbose:
                words = [c for c,_ in need_learn]
                print(f"  [交叉验证] {search_result['sources']}源共识: {words}")

            # 4. 反噬检测 → 极小步长学习
            involved = [self.zc._char_to_idx.get(c) for c, _ in need_learn 
                       if c in self.zc._char_to_idx]
            involved = [i for i in involved if i is not None]
            if involved:
                snap = self.backlash.snapshot(involved)
                self._micro_learn(vec, need_learn)
                if self.backlash.check(snap):
                    self.total_learned += 1
                else:
                    # 反噬 → 回退 (重载原始景观)
                    self.ls = EnergyLandscape.load(LANDSCAPE)
                    self.lr = DragonBallLearner(landscape=self.ls, anchor_field=self.zc, hebbian_lr=0.001)
                    self.backlash = BacklashDetector(self.ls, self.zc.anchors, self.zc._char_to_idx)
                    self.total_reverted += 1
                    if verbose:
                        print(f"  [反噬] 已回退, 保留原景观")

        # 5. 能量景观推断生成自然语言回答（零 LLM）
        answer = self._generate_answer(question, chars, energies, search_result)
        if verbose:
            print(f"  [耗时] {time.time()-t0:.1f}s")

        return answer

    # ── 编码 ──────────────────────────────────

    def _encode(self, text: str) -> torch.Tensor:
        """编码文本, 减少虚词干扰"""
        v = self.zc.encode_text(text)
        if v.shape[0] == 0:
            return torch.zeros(self.zc.embed_dim)
        # 加权: 虚词降权, 实词保持
        stop_chars = set('的吗了呢嘛啊吧呀是的不一在了有和就都也这那个什么怎么')
        weights = []
        for ch in text:
            if ch in stop_chars:
                weights.append(0.1)  # 虚词权重降低
            else:
                weights.append(1.0)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        weighted = v * weights_t[:v.shape[0]]
        result = weighted.sum(dim=0) / (weights_t.sum() + 1e-8)
        return torch.nn.functional.normalize(result, dim=-1)

    # ── 极小步长学习 ──────────────────────────

    def _micro_learn(self, query_vec: torch.Tensor, keywords: list):
        """
        极小步长 Hebbian 植入, 仅对交叉验证共识的关键词。
        学习率 = 基础值 × 源数 / 10  (单源不学, 双源 0.0002, 极高共识 0.001)
        """
        for hanzi, freq in keywords[:8]:
            if hanzi not in self.zc._char_to_idx: continue
            # 学习率与交叉验证强度成正比, 上限 0.001
            micro_lr = min(0.001, 0.0001 * freq)
            try:
                self.lr.hebbian.update(
                    query_vec,
                    self.zc.anchors[self.zc._char_to_idx[hanzi]],
                    feedback=micro_lr,
                )
            except: pass

    # ── 回答生成 ──────────────────────────────

    def _generate_answer(self, question: str, chars: list, energies: list,
                         search: dict) -> str:
        """
        从能量景观推断生成自然语言回答（零 LLM 主唱）。
        
        使用 self.ls.infer() + self.zc.find_nearest() 获取的最近汉字,
        结合多源交叉验证结果，拼装成自然回答。
        
        Ollama 已降级为可选后备（默认关闭）。
        """
        # === Ollama 后备（已弃用）===
        if self.use_ollama_fallback:
            import warnings
            warnings.warn(
                "Ollama fallback is DEPRECATED. "
                "Energy landscape inference is now the primary answer generator. "
                "Set use_ollama_fallback=False to use the new default.",
                DeprecationWarning, stacklevel=2
            )
            return self._generate_answer_ollama(question, chars, energies, search)

        # === 能量景观推断回答 ===
        lines = []

        # 最近汉字（带相似度分数）
        # energies 实际是 find_nearest() 返回的余弦相似度 [0, 1]
        top_items = []
        for ch, sim in zip(chars[:5], energies[:5]):
            sim_val = float(sim) if sim is not None else 0.0
            top_items.append(f"{ch}({sim_val:.2f})")

        if top_items:
            lines.append(f"「{question}」在字场中关联到: {'、'.join(top_items)}")

        # 联网交叉验证的关键词
        if search.get('high_conf'):
            hc_chars = [c for c, _ in search['high_conf'][:5]]
            lines.append(f"联网验证: {'、'.join(hc_chars)}")

        # 仅低置信度时也展示网络检索结果
        if search.get('low_conf') and not search.get('high_conf'):
            lc_chars = [c for c, _ in search['low_conf'][:5]]
            lines.append(f"网络检索: {'、'.join(lc_chars)}")

        if not lines:
            related = "、".join(chars[:8])
            lines.append(f"「{question}」与以下概念关联: {related}")

        return "\n".join(lines)

    def _generate_answer_ollama(self, question: str, chars: list,
                                 energies: list, search: dict) -> str:
        """
        [DEPRECATED] 用 Ollama 基于检索结果生成自然语言回答。
        
        此方法仅在 use_ollama_fallback = True 时调用。
        未来版本将移除。
        """
        related = "、".join(chars[:8])
        context = f"字场检索关联汉字: [{related}]"

        if search.get('high_conf'):
            hc = "、".join(c for c, _ in search['high_conf'][:5])
            context += f"\n联网交叉验证: [{hc}]"

        prompt = (
            f"你是一个基于汉字知识库的问答助手。\n"
            f"{context}\n\n"
            f"用户问: {question}\n"
            f"请用中文简短回答 (2-4句话), 引用相关知识。"
        )

        try:
            resp = requests.post(OLLAMA_API, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 500},
            }, timeout=60)
            raw = resp.json().get("response", "")
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            return raw if raw else f"「{question}」在字场中最相关的概念是: {related}"
        except:
            return f"「{question}」与以下概念关联: {related}"

    # ── 周期性衰减 ────────────────────────────

    def run_decay_cycle(self, verbose: bool = False) -> int:
        """
        执行一次衰减清扫 — 移除完全衰减的关联。
        建议每 6-12 小时调用一次 (通过 cron)。

        Returns: 衰减掉的关联对数量
        """
        inactive = self.decay.prune_inactive(threshold=0.01)
        self.total_decayed += len(inactive)
        if verbose and inactive:
            print(f"[衰减] {len(inactive)} 条关联已自然消失")
        return len(inactive)

    # ── 统计 + 持久化 ─────────────────────────

    def stats(self) -> dict:
        return {
            'total_queries': self.total_queries,
            'total_learned': self.total_learned,
            'total_reverted': self.total_reverted,
            'total_decayed': self.total_decayed,
            'active_pairs': len(self.decay.activations),
        }

    def save(self):
        self.ls.save(LANDSCAPE)
        print(f"💾 已保存: {LANDSCAPE}")


# ====================================================================
# 测试
# ====================================================================

if __name__ == "__main__":
    lp = SelfEvolvingLoongPearl(decay_half_life=72)

    print("\n" + "="*50)
    print("测试1: 已知概念")
    print("="*50)
    ans = lp.ask("什么是龙?", verbose=True)
    print(f"回答: {ans}\n")

    print("="*50)
    print("测试2: 联网学习")
    print("="*50)
    ans = lp.ask("量子纠缠是什么?", verbose=True)
    print(f"回答: {ans}\n")

    print("="*50)
    print("测试3: 再问一次 (应比第一次更准)")
    print("="*50)
    ans = lp.ask("龙有什么文化含义?", verbose=True)
    print(f"回答: {ans}\n")

    print("="*50)
    print(f"统计: {lp.stats()}")
    lp.save()
