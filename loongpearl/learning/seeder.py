#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠知识播种器（loongpearl_seeder.py）—— 用Ollama为能量景观播种初始关联
=======================================================================
在字场和能量景观之上，调用 DeepSeek-R1 批量生成汉字间的语义关联，
通过 Hebbian 学习将这些关联植入能量景观，形成初始知识网络。

核心流程:
  1. 遍历汉字列表（优先高频字）
  2. 对每个汉字，调用 Ollama 生成 5 个语义关联字
  3. 用 HebbianLearner 将每个关联植入能量景观
  4. 持久化进度，支持断点续传

依赖: requests, json, torch, zichang, energy_landscape, loongpearl_learner

作者: Hermes + 李泽坤
版本: 1.0.0 (初代龙珠)
"""

import requests
import json
import time
import os
import re
import sys
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR


# ============================================================================
# 第一部分：Ollama 客户端
# ============================================================================

class OllamaClient:
    """Ollama API 封装，处理 DeepSeek-R1 的特殊响应格式"""
    
    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_predict: int = 2000,
        timeout: int = 120,
    ):
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        
        # 统计
        self.total_calls = 0
        self.total_failures = 0
        self.total_tokens = 0
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用 Ollama 生成文本。
        
        Args:
            prompt: 提示词
            max_retries: 最大重试次数
        
        Returns:
            生成的文本，失败返回空字符串
        """
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.num_predict,
                            "top_p": 0.9,
                        }
                    },
                    timeout=self.timeout,
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    self.total_calls += 1
                    self.total_tokens += data.get("eval_count", 0)
                    return text
                else:
                    self.total_failures += 1
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        
            except requests.exceptions.Timeout:
                self.total_failures += 1
                if attempt < max_retries - 1:
                    time.sleep(3)
            except Exception as e:
                self.total_failures += 1
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return ""
    
    def extract_json(self, text: str) -> Optional[List[Dict]]:
        """
        从 DeepSeek-R1 响应中提取 JSON 数组。
        
        处理多种格式:
          - 纯 JSON: [{"hanzi":...}]
          - Markdown代码块: ```json [{"hanzi":...}] ```
          - 混合文本中的 JSON 片段
        
        Args:
            text: 原始响应文本
        
        Returns:
            解析后的字典列表，失败返回 None
        """
        if not text:
            return None
        
        # 处理 DeepSeek-R1 的  answer/ response 标签
        for tag in [' answer', ' response']:
            if tag in text:
                idx = text.find(tag) + len(tag)
                text = text[idx:].strip()
        
        # 尝试多种提取策略
        candidates = []
        
        # 策略1: 提取 ```json ... ``` 代码块
        for match in re.finditer(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL):
            candidates.append(match.group(1))
        
        # 策略2: 提取最外层 [...] 
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start and start < end:
            candidates.append(text[start:end])
        
        if not candidates:
            return None
        
        # 尝试解析每个候选（按优先级）
        for candidate in candidates:
            result = self._parse_json_candidate(candidate)
            if result is not None:
                return result
        
        return None
    
    def _parse_json_candidate(self, json_str: str) -> Optional[List[Dict]]:
        """尝试解析单个 JSON 候选字符串，支持自动修复。
        
        返回统一格式: [{"hanzi": str, "relation": str}, ...]
        兼容字符串数组输入: ["炎","热","明"] → [{"hanzi":"炎","relation":"未知"}, ...]
        """
        # 尝试直接解析
        for attempt in range(3):
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return self._normalize_items(data)
                return None
            except json.JSONDecodeError:
                if attempt == 0:
                    json_str = re.sub(r'\s+', ' ', json_str)
                elif attempt == 1:
                    json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
                    json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")
                else:
                    cleaned = []
                    for ch in json_str:
                        if ord(ch) < 128 or ch in '[]{},:':
                            cleaned.append(ch)
                        elif ch in '，。！？；：':
                            cleaned.append(',')
                    json_str = ''.join(cleaned)
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            return self._normalize_items(data)
                    except:
                        pass
        return None
    
    def _normalize_items(self, data: List) -> List[Dict]:
        """标准化JSON数组元素：字符串→对象，保留已有对象"""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append({"hanzi": item, "relation": "未知"})
            elif isinstance(item, dict) and 'hanzi' in item and item['hanzi'].strip():
                result.append({
                    "hanzi": item['hanzi'].strip(),
                    "relation": item.get('relation', '未知'),
                })
        return result
    
    def get_stats(self) -> Dict:
        """获取调用统计"""
        return {
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'total_tokens': self.total_tokens,
            'success_rate': (self.total_calls - self.total_failures) / max(self.total_calls, 1),
        }


# ============================================================================
# 第二部分：语义关联生成器
# ============================================================================

class AssociationGenerator:
    """汉字语义关联生成器 —— 调用 Ollama 为单个汉字生成关联网络"""
    
    # 关联类型定义
    RELATION_TYPES = [
        "同义",    # 近义词
        "反义",    # 反义词
        "上下位",  # 整体-部分 / 类别-实例
        "因果",    # 因果关系
        "属性",    # 属性/特征
        "组成",    # 组成部分
        "搭配",    # 常见搭配/组词
        "形近",    # 字形相近
        "音近",    # 读音相近
    ]
    
    def __init__(self, client: OllamaClient):
        self.client = client
        self.cache: Dict[str, List[Dict]] = {}  # 汉字→关联列表缓存
    
    def _build_prompt(self, hanzi: str, num: int = 5) -> str:
        """构建提示词（英文格式，模型返回对象数组更稳定）"""
        return (
            f'List {num} Chinese characters semantically related to "{hanzi}". '
            f'Output ONLY a JSON array: [{{"hanzi":"char","relation":"type"}}]'
        )
    
    def generate(self, hanzi: str, num: int = 5, use_cache: bool = True) -> List[Dict]:
        """
        为单个汉字生成语义关联列表。
        
        Args:
            hanzi: 目标汉字
            num: 生成的关联数量
            use_cache: 是否使用缓存
        
        Returns:
            关联列表 [{"hanzi": str, "relation": str, "evidence": str}, ...]
        """
        # 检查缓存
        if use_cache and hanzi in self.cache:
            return self.cache[hanzi]
        
        prompt = self._build_prompt(hanzi, num)
        response = self.client.generate(prompt)
        
        if not response:
            return []
        
        associations = self.client.extract_json(response)
        
        if associations:
            # 过滤无效关联
            valid = []
            for item in associations:
                if isinstance(item, dict) and 'hanzi' in item:
                    target = item['hanzi'].strip()
                    if target and len(target) == 1:  # 确保是单个汉字
                        valid.append({
                            'hanzi': target,
                            'relation': item.get('relation', '未知'),
                            'evidence': item.get('evidence', ''),
                        })
            
            if use_cache:
                self.cache[hanzi] = valid
            return valid
        
        return []
    
    def generate_batch(
        self, 
        hanzi_list: List[str], 
        num: int = 5,
        delay: float = 0.3,
        verbose: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        批量为多个汉字生成关联。
        
        Args:
            hanzi_list: 汉字列表
            num: 每个字的关联数
            delay: 请求间隔（秒）
            verbose: 是否打印进度
        
        Returns:
            {汉字: 关联列表} 字典
        """
        results = {}
        failed = []
        start_time = time.time()
        
        for i, hanzi in enumerate(hanzi_list):
            try:
                assoc = self.generate(hanzi, num=num)
                if assoc:
                    results[hanzi] = assoc
                else:
                    failed.append(hanzi)
                
                if verbose and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (len(hanzi_list) - i - 1)
                    print(f"  [{i+1}/{len(hanzi_list)}] {hanzi}: {len(assoc)}关联 | "
                          f"elapsed={elapsed:.0f}s | eta={eta/60:.0f}min")
                
                if i < len(hanzi_list) - 1 and delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                failed.append(hanzi)
                if verbose:
                    print(f"  [{i+1}] {hanzi}: ERROR - {e}")
        
        if verbose and failed:
            print(f"\n  失败: {len(failed)}/{len(hanzi_list)} 个汉字")
        
        return results


# ============================================================================
# 第三部分：知识播种器
# ============================================================================

class OllamaSeeder:
    """
    知识播种器 —— 用 Ollama 为龙珠能量景观播种初始知识关联。
    
    工作流程:
      1. 加载字场、能量景观、Hebbian学习器
      2. 遍历汉字列表，调用 Ollama 生成语义关联
      3. 对每个有效关联，用 Hebbian 学习降低两字间的能量
      4. 持久化进度（支持中断后继续）
    
    性能估算:
      - 单字处理时间: ~8s (含API调用+学习)
      - 100字: ~13分钟
      - 500字 (推荐首轮): ~67分钟
      - 1000字: ~2.2小时
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        num_associations: int = 5,
        hebbian_strength: float = 0.5,
        progress_dir: str = None,
    ):
        """
        初始化播种器。
        
        Args:
            model: Ollama 模型名
            num_associations: 每个汉字生成的关联数
            hebbian_strength: Hebbian 学习强度
            progress_dir: 进度文件目录
        """
        self.model = model
        self.num_associations = num_associations
        self.hebbian_strength = hebbian_strength
        
        # 初始化 Ollama 客户端
        self.client = OllamaClient(model=model)
        self.generator = AssociationGenerator(self.client)
        
        # 进度管理
        self.progress_dir = progress_dir or os.path.dirname(os.path.abspath(__file__))
        self.progress_file = os.path.join(self.progress_dir, "data/runtime/seed_progress.json")
        self.seeded_indices: Set[int] = set()
        self.seeded_pairs: Set[Tuple[str, str]] = set()
        self.total_associations = 0
        self.total_failures = 0
        
        # 字场和学习器引用（在 seed() 中设置）
        self.zichang = None
        self.learner = None
    
    def _load_progress(self) -> bool:
        """加载播种进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.seeded_indices = set(data.get('indices', []))
                self.seeded_pairs = set(tuple(p) for p in data.get('pairs', []))
                self.total_associations = data.get('total_associations', 0)
                self.total_failures = data.get('total_failures', 0)
                
                print(f"[进度恢复] 已完成 {len(self.seeded_indices)} 个汉字, "
                      f"{self.total_associations} 个关联, "
                      f"{self.total_failures} 个失败")
                return True
            except Exception as e:
                print(f"[警告] 进度文件损坏: {e}，从头开始")
        return False
    
    def _save_progress(self):
        """保存播种进度"""
        os.makedirs(self.progress_dir, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'indices': list(self.seeded_indices),
                'pairs': [list(p) for p in self.seeded_pairs],
                'total_associations': self.total_associations,
                'total_failures': self.total_failures,
                'timestamp': time.time(),
            }, f, ensure_ascii=False, indent=2)
    
    def seed(
        self,
        zichang: 'HanziAnchorField',       # type: ignore
        landscape: 'EnergyLandscape',      # type: ignore
        learner: 'HebbianLearner',         # type: ignore
        max_hanzi: int = 500,
        start_index: int = 0,
        delay: float = 0.2,
        verbose: bool = True,
    ) -> Dict:
        """
        批量播种初始知识关联。
        
        遍历前 max_hanzi 个汉字（从 start_index 开始），
        对每个汉字调用 Ollama 生成关联，然后用 Hebbian 学习植入。
        
        Args:
            zichang: 字场实例
            landscape: 能量景观实例
            learner: Hebbian 学习器实例
            max_hanzi: 最多播种多少个汉字
            start_index: 起始索引（跳过前N个）
            delay: 每字间延迟（秒）
            verbose: 是否打印详细进度
        
        Returns:
            播种统计字典
        """
        self.zichang = zichang
        self.learner = learner
        
        # 加载进度
        self._load_progress()
        
        total = min(max_hanzi, len(zichang.hanzi_list))
        start_time = time.time()
        batch_report_interval = max(5, min(20, total // 10))
        
        print(f"\n{'='*60}")
        print(f"龙珠知识播种开始")
        print(f"  模型:      {self.model}")
        print(f"  目标字数:  {total} (从索引 {start_index} 开始)")
        print(f"  每字关联:  {self.num_associations}")
        print(f"  学习强度:  {self.hebbian_strength}")
        print(f"  延迟:      {delay}s/字")
        print(f"  已播种:    {len(self.seeded_indices)} 字")
        print(f"  已有关联:  {self.total_associations}")
        print(f"  {'='*60}\n")
        
        newly_seeded = 0
        newly_associations = 0
        
        # 准备待处理的汉字列表
        hanzi_to_process = []
        for i in range(start_index, min(start_index + total, len(zichang.hanzi_list))):
            if i not in self.seeded_indices:
                hanzi_to_process.append((i, zichang.hanzi_list[i]))
        
        if not hanzi_to_process:
            print("所有汉字已播种完毕！")
            return self._build_stats(start_time, 0, 0)
        
        print(f"待处理: {len(hanzi_to_process)} 个新汉字\n")
        
        for batch_idx, (idx, hanzi) in enumerate(hanzi_to_process):
            try:
                # 步骤1: 生成关联
                associations = self.generator.generate(
                    hanzi, num=self.num_associations
                )
                
                if not associations:
                    self.total_failures += 1
                    self.seeded_indices.add(idx)
                    if verbose and len(associations) == 0:
                        print(f"  [{idx}] '{hanzi}': 无关联 (Ollama返回空)")
                    continue
                
                # 步骤2: 植入关联
                implanted = 0
                for assoc in associations:
                    target_char = assoc.get('hanzi', '')
                    if not target_char or target_char not in zichang._char_to_idx:
                        continue
                    
                    target_idx = zichang._char_to_idx[target_char]
                    
                    # 去重检查
                    pair = tuple(sorted([hanzi, target_char]))
                    if pair in self.seeded_pairs:
                        continue
                    
                    # 获取锚点向量
                    q_vec = zichang.anchors[idx]
                    a_vec = zichang.anchors[target_idx]
                    
                    # Hebbian 学习：降低两字之间的能量
                    result = learner.hebbian.update(
                        q_vec, a_vec, feedback=self.hebbian_strength
                    )
                    
                    if result.get('status') != 'skipped':
                        implanted += 1
                        self.seeded_pairs.add(pair)
                
                self.seeded_indices.add(idx)
                newly_seeded += 1
                newly_associations += implanted
                self.total_associations += implanted
                
                if verbose and (batch_idx + 1) % batch_report_interval == 0:
                    elapsed = time.time() - start_time
                    processed = len(self.seeded_indices)
                    remaining = len(hanzi_to_process) - batch_idx - 1
                    eta = (elapsed / max(batch_idx + 1, 1)) * remaining
                    rate = (batch_idx + 1) / max(elapsed, 1)
                    
                    print(f"  [{processed}/{start_index + total}] "
                          f"'{hanzi}': {implanted}植入 | "
                          f"速率={rate:.1f}字/min | "
                          f"eta={eta/60:.0f}min | "
                          f"累计={self.total_associations}关联")
                
                # 保存进度
                if (batch_idx + 1) % (batch_report_interval * 2) == 0:
                    self._save_progress()
                
            except Exception as e:
                self.total_failures += 1
                self.seeded_indices.add(idx)  # 标记为已处理（跳过）
                if verbose:
                    print(f"  [{idx}] '{hanzi}': ERROR - {e}")
            
            # API 调用间隔
            if delay > 0:
                time.sleep(delay)
        
        # 最终保存
        self._save_progress()
        
        return self._build_stats(start_time, newly_seeded, newly_associations)
    
    def _build_stats(self, start_time: float, newly_seeded: int, newly_assoc: int) -> Dict:
        """构建统计报告"""
        elapsed = time.time() - start_time
        
        return {
            'status': 'completed',
            'model': self.model,
            'total_seeded': len(self.seeded_indices),
            'newly_seeded': newly_seeded,
            'total_associations': self.total_associations,
            'newly_implanted': newly_assoc,
            'total_failures': self.total_failures,
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'rate_per_minute': (newly_seeded / max(elapsed, 1)) * 60,
            'associations_per_char': self.total_associations / max(len(self.seeded_indices), 1),
            'ollama_stats': self.client.get_stats(),
            'cache_size': len(self.generator.cache),
        }
    
    def get_progress(self) -> Dict:
        """获取当前播种进度"""
        if not self.zichang:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'in_progress' if self.seeded_indices else 'not_started',
            'seeded_chars': len(self.seeded_indices),
            'total_chars': len(self.zichang.hanzi_list),
            'seeded_pairs': len(self.seeded_pairs),
            'total_associations': self.total_associations,
            'total_failures': self.total_failures,
            'progress_pct': 100 * len(self.seeded_indices) / len(self.zichang.hanzi_list),
        }
    
    def reset_progress(self):
        """清除进度（从头开始）"""
        self.seeded_indices = set()
        self.seeded_pairs = set()
        self.total_associations = 0
        self.total_failures = 0
        self.generator.cache = {}
        
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            print("进度已清除")


# ============================================================================
# 第四部分：便捷函数
# ============================================================================

def quick_seed(
    zichang_path: str,
    energy_path: str,
    max_hanzi: int = 100,
    hebbian_strength: float = 0.5,
    model: str = "deepseek-r1:7b",
) -> Dict:
    """
    快速播种：加载字场和能量景观，播种指定数量的汉字。
    
    Args:
        zichang_path: 字场文件路径
        energy_path: 能量景观文件路径
        max_hanzi: 最多播种字数
        hebbian_strength: Hebbian 学习强度
        model: Ollama 模型
    
    Returns:
        播种统计
    """
    sys.path.insert(0, os.path.dirname(zichang_path))
    import loongpearl.core.zichang
    from loongpearl.core.energy_landscape import EnergyLandscape
    from loongpearl.learning.learner import DragonBallLearner
    
    # 加载
    zf = zichang.HanziAnchorField.load(zichang_path)
    landscape = EnergyLandscape.load(energy_path)
    learner = DragonBallLearner(landscape, zf)
    
    # 播种
    seeder = OllamaSeeder(
        model=model,
        num_associations=5,
        hebbian_strength=hebbian_strength,
    )
    
    return seeder.seed(zf, landscape, learner, max_hanzi=max_hanzi)


# ============================================================================
# 第五部分：主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="龙珠知识播种器 —— 用Ollama为能量景观播种初始知识关联",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--zichang', '-z', type=str, required=True,
                        help='字场文件路径')
    parser.add_argument('--energy', '-e', type=str, required=True,
                        help='能量景观文件路径')
    parser.add_argument('--max-hanzi', '-n', type=int, default=500,
                        help='最多播种汉字数 (默认: 500)')
    parser.add_argument('--model', '-m', type=str, default='deepseek-r1:7b',
                        help='Ollama模型名 (默认: deepseek-r1:7b)')
    parser.add_argument('--strength', '-s', type=float, default=0.5,
                        help='Hebbian学习强度 (默认: 0.5)')
    parser.add_argument('--delay', '-d', type=float, default=0.2,
                        help='请求间隔秒数 (默认: 0.2)')
    parser.add_argument('--reset', action='store_true',
                        help='清除旧进度从头开始')
    parser.add_argument('--dry-run', action='store_true',
                        help='只生成关联不植入学习')
    parser.add_argument('--progress', action='store_true',
                        help='只显示当前进度')
    
    args = parser.parse_args()
    
    # 加载模块
    sys.path.insert(0, os.path.dirname(args.zichang))
    import loongpearl.core.zichang
    from loongpearl.core.energy_landscape import EnergyLandscape
    from loongpearl.learning.learner import DragonBallLearner
    
    zf = zichang.HanziAnchorField.load(args.zichang)
    landscape = EnergyLandscape.load(args.energy)
    learner = DragonBallLearner(landscape, zf)
    
    seeder = OllamaSeeder(
        model=args.model,
        num_associations=5,
        hebbian_strength=args.strength,
    )
    
    if args.progress:
        seeder._load_progress()
        print(seeder.get_progress())
        sys.exit(0)
    
    if args.reset:
        seeder.reset_progress()
    
    if args.dry_run:
        print("干运行模式：只生成关联，不修改能量景观")
        # 只测试前5个字
        for i in range(min(5, len(zf.hanzi_list))):
            hanzi = zf.hanzi_list[i]
            assoc = seeder.generator.generate(hanzi, num=5)
            print(f"  '{hanzi}': {len(assoc)} 关联")
            for a in assoc[:3]:
                print(f"    -> {a['hanzi']} [{a['relation']}]")
        sys.exit(0)
    
    # 执行播种
    stats = seeder.seed(
        zf, landscape, learner,
        max_hanzi=args.max_hanzi,
        delay=args.delay,
    )
    
    print(f"\n{'='*60}")
    print("播种完成！")
    print(f"  播种字数: {stats['newly_seeded']}")
    print(f"  植入关联: {stats['newly_implanted']}")
    print(f"  累计关联: {stats['total_associations']}")
    print(f"  失败次数: {stats['total_failures']}")
    print(f"  总耗时:   {stats['elapsed_minutes']:.1f} 分钟")
    print(f"  播种速率: {stats['rate_per_minute']:.1f} 字/分钟")
    print(f"  关联密度: {stats['associations_per_char']:.1f} 关联/字")
    print(f"  模型调用: {stats['ollama_stats']['total_calls']} "
          f"(成功率 {stats['ollama_stats']['success_rate']:.1%})")
    print(f"{'='*60}")
