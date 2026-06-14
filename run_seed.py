#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠知识播种启动脚本（run_seed.py）
====================================
批量调用 Ollama 为汉字生成语义关联，注入能量景观。

用法:
    python run_seed.py                    # 交互式选择播种范围
    python run_seed.py --chars 500        # 播种前500个汉字
    python run_seed.py --chars 3755       # 播种全部 GB2312 一级汉字
    python run_seed.py --resume           # 从断点继续
    python run_seed.py --dry-run 10       # 干运行（只生成关联，不修改能量景观）

依赖:
    - Ollama 服务运行中 (localhost:11434)
    - deepseek-r1:7b 模型已加载
    - 字场 zichang_94117_1024d.pt 已存在
    - 能量景观 energy_landscape_1024d.pt 已存在
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from sentence_transformers import SentenceTransformer

from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape
from loongpearl_learner import DragonBallLearner
from loongpearl_seeder import OllamaSeeder, AssociationGenerator, OllamaClient


# ============================================================================
# 配置
# ============================================================================

# 默认路径
DEFAULT_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ZICHANG = os.path.join(DEFAULT_MODEL_DIR, "zichang_94117_1024d.pt")
DEFAULT_LANDSCAPE = os.path.join(DEFAULT_MODEL_DIR, "energy_landscape_1024d.pt")
DEFAULT_FREQ_FILE = os.path.join(DEFAULT_MODEL_DIR, "hanzi_top3500.txt")
DEFAULT_PROGRESS = os.path.join(DEFAULT_MODEL_DIR, "seed_progress.json")

# Ollama 配置
DEFAULT_MODEL = "deepseek-r1:7b"
DEFAULT_API = "http://localhost:11434"


# ============================================================================
# 播种器（基于已验证的 v7 模式）
# ============================================================================

class DragonBallSeeder:
    """
    龙珠播种器 —— 封装播种流程，支持断点续传。
    
    复用已验证的模式:
      - 英文 prompt + num_predict=2000
      - 兼容字符串/对象 JSON 数组
      - 每 50 字进度报告，每 100 字保存断点
    """
    
    def __init__(
        self,
        zichang: HanziAnchorField,
        landscape: EnergyLandscape,
        learner: DragonBallLearner,
        embed_model: SentenceTransformer,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API,
    ):
        self.zichang = zichang
        self.landscape = landscape
        self.learner = learner
        self.embed_model = embed_model
        self.api_url = f"{api_url}/api/generate"
        self.model = model
        
        # 统计
        self.seeded = set()
        self.pairs = set()
        self.total_implanted = 0
        self.total_failed = 0
    
    def seed(
        self,
        chars: list,
        num_associations: int = 3,
        hebbian_strength: float = 0.5,
        delay: float = 0.05,
        checkpoint_file: str = None,
        verbose: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        批量播种汉字关联。
        
        Args:
            chars: 汉字列表（按优先级排序）
            num_associations: 每个字生成的关联数
            hebbian_strength: Hebbian 学习强度
            delay: 每字间延迟（秒）
            checkpoint_file: 断点文件路径
            verbose: 是否打印进度
            dry_run: 干运行模式（不修改能量景观）
        
        Returns:
            播种统计字典
        """
        import re
        
        # 断点恢复
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            self.seeded = set(ckpt.get('chars', []))
            self.pairs = set(tuple(p) for p in ckpt.get('pairs', []))
            self.total_implanted = ckpt.get('total', 0)
            print(f"断点恢复: {len(self.seeded)}字/{len(self.pairs)}对/{self.total_implanted}植入")
        
        # 过滤：去重 + 只在字场中的字
        pending = []
        for c in chars:
            if c in self.zichang._char_to_idx and c not in self.seeded:
                pending.append(c)
        
        if not pending:
            print("所有汉字已播种！")
            return self._stats(0, 0)
        
        print(f"\n待播种: {len(pending)} 字 (已播: {len(self.seeded)})")
        if dry_run:
            print("⚠️  干运行模式：只生成关联，不修改能量景观\n")
        
        t0 = time.time()
        newly_seeded = 0
        
        for i, ch in enumerate(pending):
            try:
                # 步骤1: 调用 Ollama 生成关联
                data = self._fetch_associations(ch, num_associations)
                
                if not data:
                    self.seeded.add(ch)
                    self.total_failed += 1
                    if verbose and i < 3:
                        print(f"  [{i+1}] '{ch}': ⚠️ Ollama 返回空")
                    continue
                
                # 步骤2: 植入学习
                idx = self.zichang._char_to_idx[ch]
                imp = 0
                
                if not dry_run:
                    for item in data:
                        tgt = item.get('hanzi', '')
                        if not tgt or tgt not in self.zichang._char_to_idx or tgt == ch:
                            continue
                        
                        pair = tuple(sorted([ch, tgt]))
                        if pair in self.pairs:
                            continue
                        
                        try:
                            result = self.learner.hebbian.update(
                                self.zichang.anchors[idx],
                                self.zichang.anchors[self.zichang._char_to_idx[tgt]],
                                feedback=hebbian_strength,
                            )
                            if result.get('status') != 'skipped':
                                imp += 1
                                self.pairs.add(pair)
                        except Exception:
                            pass
                
                self.seeded.add(ch)
                self.total_implanted += imp
                newly_seeded += 1
                
                # 进度报告
                if verbose and (len(self.seeded) % 50 == 0 or len(self.seeded) <= 5):
                    total_done = len(self.seeded)
                    total_target = len(chars)
                    elapsed = time.time() - t0
                    rate = total_done / max(elapsed, 1) * 60
                    remaining = total_target - total_done
                    eta = remaining / max(rate, 0.01) if rate > 0 else 0
                    
                    mode = "[DRY]" if dry_run else ""
                    print(f"  {mode}[{total_done}/{total_target}] '{ch}':{imp}植入 "
                          f"| {rate:.1f}/min | eta={eta/60:.0f}h{eta%60:.0f}m "
                          f"| 累计={self.total_implanted}")
                
                # 保存断点
                if not dry_run and checkpoint_file and len(self.seeded) % 100 == 0:
                    self._save_checkpoint(checkpoint_file)
                    print(f"  💾 进度已保存 ({len(self.seeded)}字)")
            
            except Exception as e:
                self.seeded.add(ch)
                self.total_failed += 1
                if verbose and i < 5:
                    print(f"  [{i+1}] '{ch}': ❌ {e}")
            
            # 延迟控制
            if delay > 0 and i < len(pending) - 1:
                time.sleep(delay)
        
        # 最终保存
        if not dry_run and checkpoint_file:
            self._save_checkpoint(checkpoint_file)
        
        return self._stats(time.time() - t0, newly_seeded)
    
    def _fetch_associations(self, ch: str, num: int = 3) -> list:
        """调用 Ollama 为单个汉字生成关联"""
        import requests as _requests
        import re
        
        resp = _requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": (f'List {num} Chinese characters semantically related to "{ch}". '
                           f'Output ONLY: [{{"hanzi":"char","relation":"type"}}]'),
                "stream": False,
                "options": {"temperature": 0.6, "num_predict": 2000},
            },
            timeout=120,
        )
        
        raw = resp.json().get("response", "")
        return self._parse_json(raw)
    
    def _parse_json(self, text: str) -> list:
        """从 Ollama 响应中提取 JSON"""
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
        except:
            return []
        
        if not isinstance(data, list):
            return []
        
        result = []
        for item in data:
            if isinstance(item, str):
                result.append({"hanzi": item, "relation": "未知"})
            elif isinstance(item, dict) and 'hanzi' in item:
                result.append({
                    "hanzi": item['hanzi'].strip(),
                    "relation": item.get('relation', '未知'),
                })
        return result
    
    def _save_checkpoint(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'chars': list(self.seeded),
                'pairs': [list(p) for p in self.pairs],
                'total': self.total_implanted,
                'failed': self.total_failed,
            }, f, ensure_ascii=False, indent=2)
    
    def _stats(self, elapsed: float, newly_seeded: int) -> dict:
        total = len(self.seeded)
        return {
            'total_seeded': total,
            'newly_seeded': newly_seeded,
            'total_implanted': self.total_implanted,
            'total_failed': self.total_failed,
            'total_pairs': len(self.pairs),
            'elapsed_minutes': elapsed / 60,
            'rate_per_minute': total / max(elapsed, 1) * 60,
            'associations_per_char': self.total_implanted / max(total, 1),
        }


# ============================================================================
# 生成汉字列表
# ============================================================================

def get_char_list(num_chars: int = 500) -> list:
    """
    获取指定数量的高频汉字列表。
    
    优先级:
      1. hanzi_top3500.txt (GB2312 一级字，已生成)
      2. hanzi_list.txt (94117 Unicode 全量)
    """
    # 尝试从频序文件加载
    if os.path.exists(DEFAULT_FREQ_FILE):
        with open(DEFAULT_FREQ_FILE) as f:
            chars = list(f.read().strip())
        return chars[:num_chars]
    
    # 兜底：从字场文件加载
    import torch
    data = torch.load(DEFAULT_ZICHANG, map_location='cpu', weights_only=True)
    chars = data['hanzi_list'][:num_chars]
    print(f"⚠️  未找到频序文件，使用 Unicode 序前 {len(chars)} 字")
    return chars


# ============================================================================
# 主入口
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="龙珠知识播种 —— 用 Ollama 为能量景观播种汉字语义关联",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_seed.py                        # 交互式
  python run_seed.py --chars 500            # 播种前500字
  python run_seed.py --chars 3755           # 播种全部GB2312一级汉字
  python run_seed.py --resume               # 从断点继续
  python run_seed.py --dry-run 10           # 干运行10字
  python run_seed.py --char-file hanzi_top3500.txt   # 从指定文件播种
        """,
    )
    
    parser.add_argument('--chars', '-n', type=int, default=None,
                        help='播种字数')
    parser.add_argument('--resume', action='store_true',
                        help='从断点继续')
    parser.add_argument('--dry-run', type=int, default=None,
                        help='干运行N字（只生成关联，不修改能量景观）')
    parser.add_argument('--char-file', '-f', type=str,
                        help='自定义汉字列表文件（每行一字或连续字符串）')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help=f'Ollama 模型 (默认: {DEFAULT_MODEL})')
    parser.add_argument('--strength', '-s', type=float, default=0.5,
                        help='Hebbian 学习强度 (默认: 0.5)')
    parser.add_argument('--delay', '-d', type=float, default=0.05,
                        help='每字间延迟秒数 (默认: 0.05)')
    parser.add_argument('--reset', action='store_true',
                        help='清除断点从头开始')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='详细输出')
    
    args = parser.parse_args()
    
    # 确定播种字数
    num_chars = args.chars or 500
    is_dry_run = args.dry_run is not None
    if is_dry_run:
        num_chars = args.dry_run
    
    print("=" * 60)
    print("🌱 龙珠知识播种")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  字数: {num_chars}")
    print(f"  强度: {args.strength}")
    if is_dry_run:
        print(f"  ⚠️  干运行模式：不修改能量景观")
    
    # 加载字场
    print("\n[1/4] 加载字场...")
    zichang = HanziAnchorField.load(DEFAULT_ZICHANG)
    print(f"      字场: {zichang.num_hanzi} 汉字")
    
    # 加载能量景观
    print("[2/4] 加载能量景观...")
    landscape = EnergyLandscape.load(DEFAULT_LANDSCAPE)
    landscape.eval()
    print(f"      能量景观: 已加载 (dim={landscape.embed_dim})")
    
    # 创建学习器
    print("[3/4] 初始化学习器...")
    learner = DragonBallLearner(
        landscape=landscape,
        anchor_field=zichang,
        hebbian_lr=0.001,
    )
    print(f"      学习器: 就绪")
    
    # 加载汉字列表
    print("[4/4] 准备汉字列表...")
    if args.char_file and os.path.exists(args.char_file):
        with open(args.char_file) as f:
            chars = list(f.read().strip())
        print(f"      从文件加载: {len(chars)} 字")
    else:
        chars = get_char_list(num_chars)
        chars = chars[:num_chars]
        print(f"      汉字列表: {len(chars)} 字")
    
    # 断点管理
    checkpoint_file = DEFAULT_PROGRESS
    if args.reset and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("      旧断点已清除")
    
    # 播种
    print(f"\n{'─' * 40}")
    
    seeder = DragonBallSeeder(
        zichang=zichang,
        landscape=landscape,
        learner=learner,
        embed_model=None,  # seeder 不需要编码模型
        model=args.model,
    )
    
    result = seeder.seed(
        chars=chars,
        num_associations=3,
        hebbian_strength=args.strength,
        delay=args.delay,
        checkpoint_file=checkpoint_file if not is_dry_run else None,
        dry_run=is_dry_run,
    )
    
    # 保存能量景观
    if not is_dry_run and result['newly_seeded'] > 0:
        print(f"\n[5/5] 保存能量景观...")
        landscape.save(DEFAULT_LANDSCAPE)
        print(f"      已保存至 {DEFAULT_LANDSCAPE}")
    
    # 报告
    print(f"\n{'═' * 60}")
    print(f"✅ 播种完成！")
    print(f"  播种字数: {result['total_seeded']}")
    print(f"  新增字数: {result['newly_seeded']}")
    print(f"  植入关联: {result['total_implanted']}")
    print(f"  唯一对:   {result['total_pairs']}")
    print(f"  失败次数: {result['total_failed']}")
    print(f"  耗时:     {result['elapsed_minutes']:.1f} 分钟")
    print(f"  速率:     {result['rate_per_minute']:.1f} 字/分钟")
    print(f"  密度:     {result['associations_per_char']:.1f} 关联/字")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
