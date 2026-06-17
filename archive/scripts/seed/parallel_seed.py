#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠并行播种器（parallel_seed.py）
==================================
用 concurrent.futures 多线程调用 Ollama，3-4 倍加速播种。

用法:
    python parallel_seed.py                          # 从剩余字列表继续播种
    python parallel_seed.py --workers 4              # 4 线程并行
    python parallel_seed.py --chars 1000             # 只播前1000个剩余字
    python parallel_seed.py --dry-run                # 干运行（不修改能量景观）

策略:
    - 每个线程独立调用 Ollama，互不干扰
    - Hebbian 更新加锁保护（能量景观参数是共享的）
    - 每 100 字保存断点 + 能量景观
"""

import sys, os, json, time, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import requests as _requests

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape
from loongpearl_learner import DragonBallLearner

# ====================================================================
# 配置
# ====================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ZICHANG_PATH = os.path.join(PROJECT_DIR, "zichang_94117_1024d.pt")
LANDSCAPE_PATH = os.path.join(PROJECT_DIR, "energy_landscape_1024d.pt")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "seed_parallel_checkpoint.json")
REMAINING_FILE = os.path.join(PROJECT_DIR, "remaining_chars.txt")

OLLAMA_MODEL = "deepseek-r1:7b"
OLLAMA_API = "http://localhost:11434/api/generate"

# ====================================================================
# 并行播种器
# ====================================================================

class ParallelSeeder:
    """多线程并行播种器"""

    def __init__(self, zichang, landscape, learner, workers=3):
        self.zichang = zichang
        self.landscape = landscape
        self.learner = learner
        self.workers = workers

        # 线程安全：Hebbian 更新锁
        self.hebbian_lock = threading.Lock()

        # 统计（线程安全用锁）
        self.stats_lock = threading.Lock()
        self.seeded = set()
        self.pairs = set()
        self.total_implanted = 0
        self.total_failed = 0
        self.start_time = None

    # ── 单个汉字播种 ──────────────────────────────────

    def _seed_one(self, ch: str, strength: float = 0.5) -> dict:
        """播种单个汉字（线程安全），返回统计"""
        try:
            # 调用 Ollama
            data = self._fetch_associations(ch)
            if not data:
                with self.stats_lock:
                    self.seeded.add(ch)
                    self.total_failed += 1
                return {"char": ch, "implanted": 0, "status": "empty"}

            # Hebbian 植入（加锁保护参数更新）
            idx = self.zichang._char_to_idx[ch]
            imp = 0

            with self.hebbian_lock:
                for item in data:
                    tgt = item.get("hanzi", "")
                    if not tgt or tgt not in self.zichang._char_to_idx or tgt == ch:
                        continue
                    pair = tuple(sorted([ch, tgt]))
                    if pair in self.pairs:
                        continue
                    try:
                        result = self.learner.hebbian.update(
                            self.zichang.anchors[idx],
                            self.zichang.anchors[self.zichang._char_to_idx[tgt]],
                            feedback=strength,
                        )
                        if result.get("status") != "skipped":
                            imp += 1
                            self.pairs.add(pair)
                    except Exception:
                        pass

            with self.stats_lock:
                self.seeded.add(ch)
                self.total_implanted += imp

            return {"char": ch, "implanted": imp, "status": "ok"}

        except Exception as e:
            with self.stats_lock:
                self.seeded.add(ch)
                self.total_failed += 1
            return {"char": ch, "implanted": 0, "status": "error", "error": str(e)}

    # ── Ollama 调用 ──────────────────────────────

    def _fetch_associations(self, ch: str, num: int = 3) -> list:
        """调用 Ollama 为单个汉字生成关联"""
        import re
        try:
            resp = _requests.post(
                OLLAMA_API,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": (
                        f'List {num} Chinese characters semantically related to "{ch}". '
                        f'Output ONLY: [{{"hanzi":"char","relation":"type"}}]'
                    ),
                    "stream": False,
                    "options": {"temperature": 0.6, "num_predict": 2000},
                },
                timeout=120,
            )
            raw = resp.json().get("response", "")
            return self._parse_json(raw)
        except Exception:
            return []

    def _parse_json(self, text: str) -> list:
        import re
        if not text:
            return []
        for m in re.finditer(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL):
            result = self._normalize(m.group(1))
            if result:
                return result
        s = text.find('[')
        e = text.rfind(']') + 1
        if s >= 0 and e > s:
            result = self._normalize(text[s:e])
            if result:
                return result
        return []

    def _normalize(self, js: str) -> list:
        try:
            data = json.loads(js)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        result = []
        for item in data:
            if isinstance(item, str):
                result.append({"hanzi": item, "relation": "未知"})
            elif isinstance(item, dict) and "hanzi" in item:
                result.append({"hanzi": item["hanzi"].strip(), "relation": item.get("relation", "未知")})
        return result

    # ── 批量播种 ──────────────────────────────

    def seed_batch(
        self,
        chars: list,
        strength: float = 0.5,
        checkpoint_interval: int = 100,
        dry_run: bool = False,
    ):
        """
        多线程并行播种。

        Args:
            chars: 待播种汉字列表
            strength: Hebbian 学习强度
            checkpoint_interval: 断点保存间隔（字数）
            dry_run: 干运行模式
        """
        self.start_time = time.time()

        # 加载断点
        if not dry_run and os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH) as f:
                ckpt = json.load(f)
            self.seeded = set(ckpt.get("chars", []))
            self.pairs = set(tuple(p) for p in ckpt.get("pairs", []))
            self.total_implanted = ckpt.get("total", 0)
            self.total_failed = ckpt.get("failed", 0)
            print(f"📂 断点恢复: {len(self.seeded)}字/{len(self.pairs)}对/{self.total_implanted}植入")

        # 过滤
        pending = [c for c in chars if c in self.zichang._char_to_idx and c not in self.seeded]
        if not pending:
            print("✅ 所有汉字已播种！")
            return self._summary()

        total = len(pending)
        print(f"\n🔥 并行播种启动: {total} 字 × {self.workers} 线程")
        if dry_run:
            print("⚠️  干运行模式：不修改能量景观")
        print(f"   速率预估: ~{6.8 * self.workers:.0f} 字/分钟")
        print(f"   ETA: {total / (6.8 * self.workers) / 60:.0f} 小时\n")

        # 主循环：分批提交任务
        last_checkpoint = len(self.seeded)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 使用队列逐步提交（避免一次性提交过多）
            char_iter = iter(pending)
            futures = {}

            # 初始填充
            for _ in range(min(self.workers * 2, total)):
                try:
                    ch = next(char_iter)
                    futures[executor.submit(self._seed_one, ch, strength)] = ch
                except StopIteration:
                    break

            while futures:
                # 等待任意一个完成
                done_futures = []
                for f in as_completed(futures):
                    done_futures.append(f)
                    break  # 只取一个

                for f in done_futures:
                    result = f.result()
                    del futures[f]

                    # 提交新任务
                    try:
                        ch = next(char_iter)
                        futures[executor.submit(self._seed_one, ch, strength)] = ch
                    except StopIteration:
                        pass

                # 进度 + 断点
                n_seeded = len(self.seeded)
                if n_seeded - last_checkpoint >= checkpoint_interval:
                    self._report_progress(total)
                    if not dry_run:
                        self._save_checkpoint()
                    # 同时保存能量景观
                    if not dry_run:
                        torch.save(self.landscape.state_dict(), LANDSCAPE_PATH.replace(".pt", "_autosave.pt"))
                    last_checkpoint = n_seeded

        # 最终保存
        if not dry_run:
            self._save_checkpoint()
            self.landscape.save(LANDSCAPE_PATH)
            # 删除自动保存
            auto = LANDSCAPE_PATH.replace(".pt", "_autosave.pt")
            if os.path.exists(auto):
                os.remove(auto)

        return self._summary()

    # ── 辅助 ──────────────────────────────

    def _report_progress(self, total: int):
        n = len(self.seeded)
        elapsed = time.time() - self.start_time
        rate = n / max(elapsed, 1) * 60
        remaining = total - n
        eta_min = remaining / max(rate, 0.01) if rate > 0 else 0
        print(f"  [{n}/{total}] {n/total*100:.1f}% | "
              f"{rate:.1f}字/min | 植入={self.total_implanted} | "
              f"eta={eta_min/60:.0f}h{eta_min%60:.0f}m")

    def _save_checkpoint(self):
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump({
                "chars": list(self.seeded),
                "pairs": [list(p) for p in self.pairs],
                "total": self.total_implanted,
                "failed": self.total_failed,
            }, f, ensure_ascii=False, indent=2)

    def _summary(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        n = len(self.seeded)
        return {
            "total_seeded": n,
            "total_implanted": self.total_implanted,
            "total_failed": self.total_failed,
            "total_pairs": len(self.pairs),
            "elapsed_hours": elapsed / 3600,
            "rate_per_minute": n / max(elapsed, 1) * 60,
        }


# ====================================================================
# 主入口
# ====================================================================

def get_remaining_chars() -> list:
    """获取所有未播种的汉字（从字场中排除已播字）"""
    # 加载字场
    data = torch.load(ZICHANG_PATH, map_location="cpu", weights_only=True)
    all_chars = data["hanzi_list"]

    # 加载已播种
    seeded = set()
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        seeded.update(ckpt.get("chars", []))
    else:
        # 从旧的 result 文件收集
        for fname in ["seed_v7_result.json", "seed_v7b_result.json",
                       "seed_v6_result.json", "seed_final.json"]:
            path = os.path.join(PROJECT_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                if "chars" in d:
                    seeded.update(d["chars"])

    # 过滤
    remaining = [c for c in all_chars if c in all_chars and c not in seeded]
    print(f"全量: {len(all_chars)} | 已播: {len(seeded)} | 剩余: {len(remaining)}")
    return remaining


def main():
    parser = argparse.ArgumentParser(description="龙珠并行播种器")
    parser.add_argument("--workers", "-w", type=int, default=3,
                        help="并行线程数 (默认: 3)")
    parser.add_argument("--chars", "-n", type=int, default=None,
                        help="播种字数上限 (默认: 全部剩余)")
    parser.add_argument("--strength", "-s", type=float, default=0.5,
                        help="Hebbian 学习强度 (默认: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式（不修改能量景观）")
    parser.add_argument("--reset", action="store_true",
                        help="清除断点从头开始")

    args = parser.parse_args()

    if args.reset and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("🗑️  断点已清除")

    print("=" * 60)
    print("🐉 龙珠并行播种器")
    print("=" * 60)
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  线程: {args.workers}")
    print(f"  强度: {args.strength}")

    # 1. 加载
    print("\n[1/3] 加载字场...")
    zichang = HanziAnchorField.load(ZICHANG_PATH)

    print("[2/3] 加载能量景观...")
    landscape = EnergyLandscape.load(LANDSCAPE_PATH)
    landscape.eval()

    print("[3/3] 初始化学习器...")
    learner = DragonBallLearner(
        landscape=landscape,
        anchor_field=zichang,
        hebbian_lr=0.001,
    )

    # 2. 获取剩余汉字
    remaining = get_remaining_chars()
    if args.chars:
        remaining = remaining[:args.chars]

    # 3. 播种
    seeder = ParallelSeeder(zichang, landscape, learner, workers=args.workers)
    result = seeder.seed_batch(
        chars=remaining,
        strength=args.strength,
        checkpoint_interval=100,
        dry_run=args.dry_run,
    )

    # 4. 报告
    print(f"\n{'═' * 60}")
    print(f"✅ 播种完成！")
    print(f"  播种字数: {result['total_seeded']}")
    print(f"  植入关联: {result['total_implanted']}")
    print(f"  唯一对:   {result['total_pairs']}")
    print(f"  失败数:   {result['total_failed']}")
    print(f"  耗时:     {result['elapsed_hours']:.1f} 小时")
    print(f"  速率:     {result['rate_per_minute']:.1f} 字/分钟")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
