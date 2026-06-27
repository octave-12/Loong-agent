#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙 序列场 — 按语义盆分区的字间转移概率
═══════════════════════════════════════════════════════

双场架构的下半场:
  语义场 (DragonField/Hopfield) → 回答 "属于什么概念" → 返回 basin_id
  序列场 (SequenceField/Markov) → 回答 "这个概念的字怎么排" → walk 生成

存储: 每个概念盆内的 bigram 频率表 (轻量, 无 BGE 编码)
  basin "龙" → { "龙": {"是": 142, "的": 89, "有": 67}, ... }

构建: Wikipedia 句子 → DragonField.converge → 找盆 → 存 bigram
推理: walk(basin_subject, seed_chars, length) → Markov 采样子序列
"""

import re
import os
import math
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict

log = logging.getLogger(__name__)


class SequenceField:
    """
    按 DragonField 语义盆分区的 Markov 序列表。

    每个盆独立存储:
      - forward:  char_a → {char_b: count, ...}   (前向转移)
      - backward: char_b → {char_a: count, ...}   (后向转移, 用于补全前缀)

    Attributes:
        _basin_forward:  dict[basin_key, dict[char, Counter]]
        _basin_backward: dict[basin_key, dict[char, Counter]]
        _basin_metadata: dict[basin_key, {'total_bigrams': int, 'vocab_size': int}]
        _vocab:          set — 全局字表
        _total_bigrams:  int — 全局 bigram 总数
    """

    def __init__(self):
        self._basin_forward: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._basin_backward: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._basin_metadata: Dict[str, dict] = {}
        self._vocab: Set[str] = set()
        self._total_bigrams: int = 0

    # ══════════════════════════════════════════════════════════════
    # 构建: Wikipedia → converge → 存 bigram
    # ══════════════════════════════════════════════════════════════

    def ingest_sentence(
        self,
        text: str,
        basin_subject: str,
        weight: float = 1.0,
    ):
        """
        将一句话的 bigram 存入指定盆。

        Args:
            text: 句子文本 (只取汉字)
            basin_subject: 语义盆标识 (DragonField 收敛到的概念主体)
            weight: 权重 (默认1.0, 高质量句子可提高)
        """
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(chars) < 2:
            return

        basin = basin_subject
        count = max(1, int(weight))
        fw = self._basin_forward[basin]
        bw = self._basin_backward[basin]

        for i in range(len(chars) - 1):
            a, b = chars[i], chars[i + 1]
            fw[a][b] += count
            bw[b][a] += count
            self._vocab.add(a)
            self._vocab.add(b)
            self._total_bigrams += count

    def ingest_sentence_multi_basin(
        self,
        text: str,
        top_subjects: List[Tuple[str, float]],
    ):
        """
        将一句话存入多个盆 (按相似度加权)。

        Args:
            text: 句子文本
            top_subjects: [(basin_subject, similarity), ...] 按相似度降序
        """
        if not top_subjects:
            return
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(chars) < 2:
            return

        # 归一化权重
        total_sim = sum(s for _, s in top_subjects)
        if total_sim <= 0:
            return

        for subject, sim in top_subjects:
            weight = sim / total_sim
            # 只对显著相似的盆存入
            if weight < 0.05:
                continue
            self.ingest_sentence(text, subject, weight=weight * len(top_subjects))

    # ══════════════════════════════════════════════════════════════
    # 推理: Markov walk
    # ══════════════════════════════════════════════════════════════

    def walk(
        self,
        basin_subject: str,
        seed_chars: List[str],
        length: int = 15,
        temperature: float = 0.8,
        direction: str = 'forward',
        fallback_basins: List[str] = None,
    ) -> str:
        """
        在指定盆内做 Markov 采样，生成字序列。

        Args:
            basin_subject: 语义盆标识
            seed_chars: 种子字列表 (从查询提取)
            length: 目标生成长度
            temperature: 采样温度 (0=贪心, 1=按频率, >1=更多样)
            direction: 'forward'(向后接) 或 'backward'(向前补)
            fallback_basins: 当前盆无数据时回退的盆列表

        Returns:
            生成的汉字序列
        """
        table = self._basin_forward if direction == 'forward' else self._basin_backward
        basin_table = table.get(basin_subject)

        # 回退: 找相似盆或全局统计
        if not basin_table:
            if fallback_basins:
                for fb in fallback_basins:
                    basin_table = table.get(fb)
                    if basin_table:
                        basin_subject = fb
                        break
            if not basin_table:
                # 最后回退: 合并所有盆的统计
                basin_table = self._merge_all_basins(direction)
                if not basin_table:
                    return ''.join(seed_chars[:length])

        result = list(seed_chars)
        current = seed_chars[-1] if seed_chars else None

        for _ in range(length):
            if current is None:
                break
            next_chars = basin_table.get(current)
            if not next_chars or sum(next_chars.values()) == 0:
                # 当前字无后继 → 从盆内随机选一字
                all_keys = list(basin_table.keys())
                if not all_keys:
                    break
                current = random.choice(all_keys)
                result.append(current)
                continue

            # 温度采样
            chosen = self._sample_with_temperature(next_chars, temperature)
            if chosen is None:
                break
            result.append(chosen)
            current = chosen

        return ''.join(result)

    def walk_bidirectional(
        self,
        basin_subject: str,
        seed_chars: List[str],
        length: int = 15,
        temperature: float = 0.8,
        fallback_basins: List[str] = None,
    ) -> str:
        """
        双向游走: 从种子字同时向前后扩展。

        适合查询如 "龙是什么":
          - backward 从 "是" 向前找 → "龙是"
          - forward  从 "是" 向后找 → "是什么..."
        """
        mid = len(seed_chars) // 2 if seed_chars else 0
        prefix = seed_chars[:mid] if mid > 0 else []
        suffix = seed_chars[mid:] if mid < len(seed_chars) else seed_chars

        # 向后扩展 (找前缀)
        if prefix:
            prefix_result = self.walk(
                basin_subject, list(reversed(prefix)),
                length=length // 2, temperature=temperature,
                direction='backward', fallback_basins=fallback_basins,
            )
            prefix_result = ''.join(reversed(prefix_result))
        else:
            prefix_result = ''

        # 向前扩展 (找后缀)
        suffix_result = self.walk(
            basin_subject, suffix,
            length=length - len(prefix_result),
            temperature=temperature,
            direction='forward', fallback_basins=fallback_basins,
        )

        return prefix_result + suffix_result

    # ══════════════════════════════════════════════════════════════
    # 查询 & 统计
    # ══════════════════════════════════════════════════════════════

    def get_basin_keys(self) -> List[str]:
        """返回所有盆标识"""
        return list(self._basin_forward.keys())

    def basin_stats(self, basin_subject: str) -> dict:
        """返回指定盆的统计信息"""
        fw = self._basin_forward.get(basin_subject, {})
        total = sum(sum(c.values()) for c in fw.values())
        vocab = set(fw.keys()) | {b for c in fw.values() for b in c}
        return {
            'basin': basin_subject,
            'total_bigrams': total,
            'vocab_size': len(vocab),
            'has_data': total > 0,
        }

    def global_stats(self) -> dict:
        """返回全局统计"""
        return {
            'num_basins': len(self._basin_forward),
            'total_bigrams': self._total_bigrams,
            'vocab_size': len(self._vocab),
            'top_basins': sorted(
                [(k, sum(sum(c.values()) for c in v.values()))
                 for k, v in self._basin_forward.items()],
                key=lambda x: -x[1]
            )[:10],
        }

    # ══════════════════════════════════════════════════════════════
    # 持久化
    # ══════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存序列场到磁盘 (JSON 格式, 可增量更新)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 转换为可序列化格式
        data = {
            'version': 1,
            'basins': {},
            'vocab': list(self._vocab),
            'total_bigrams': self._total_bigrams,
        }

        for basin_key, fw in self._basin_forward.items():
            bw = self._basin_backward.get(basin_key, {})
            entry = {'forward': {}, 'backward': {}}
            for ch_a, cnt in fw.items():
                entry['forward'][ch_a] = dict(cnt)
            for ch_b, cnt in bw.items():
                entry['backward'][ch_b] = dict(cnt)
            data['basins'][basin_key] = entry

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        log.info(f"序列场已保存: {path} ({len(data['basins'])}个盆, "
                 f"{self._total_bigrams} bigrams)")

    @classmethod
    def load(cls, path: str) -> 'SequenceField':
        """从磁盘加载序列场"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sf = cls()
        sf._total_bigrams = data.get('total_bigrams', 0)
        sf._vocab = set(data.get('vocab', []))

        for basin_key, entry in data.get('basins', {}).items():
            fw = entry.get('forward', {})
            bw = entry.get('backward', {})
            for ch_a, cnt_dict in fw.items():
                sf._basin_forward[basin_key][ch_a] = Counter(cnt_dict)
            for ch_b, cnt_dict in bw.items():
                sf._basin_backward[basin_key][ch_b] = Counter(cnt_dict)

        log.info(f"序列场已加载: {path} ({len(data['basins'])}个盆)")
        return sf

    # ══════════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _sample_with_temperature(
        counter: Counter, temperature: float = 1.0
    ) -> Optional[str]:
        """带温度的加权采样"""
        items = list(counter.items())
        if not items:
            return None

        if temperature <= 0.01:
            # 贪心: 选最高频
            return max(items, key=lambda x: x[1])[0]

        keys, counts = zip(*items)
        total = sum(counts)

        if temperature == 1.0:
            # 直接按频率采样
            probs = [c / total for c in counts]
        else:
            # 温度调整
            logits = [math.log(max(c, 1)) / temperature for c in counts]
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            exp_sum = sum(exp_logits)
            probs = [e / exp_sum for e in exp_logits]

        return random.choices(keys, weights=probs, k=1)[0]

    def _merge_all_basins(self, direction: str = 'forward') -> Dict[str, Counter]:
        """合并所有盆的统计作为回退"""
        table = defaultdict(Counter)
        basins = (
            self._basin_forward if direction == 'forward'
            else self._basin_backward
        )
        for basin_table in basins.values():
            for ch, cnt in basin_table.items():
                table[ch].update(cnt)
        return dict(table)

    def prune_rare(self, min_count: int = 1):
        """删除低频 bigram (减少噪音)"""
        removed = 0
        for basin_key in list(self._basin_forward.keys()):
            fw = self._basin_forward[basin_key]
            for ch_a in list(fw.keys()):
                for ch_b, cnt in list(fw[ch_a].items()):
                    if cnt < min_count:
                        del fw[ch_a][ch_b]
                        removed += 1
                if not fw[ch_a]:
                    del fw[ch_a]
        log.info(f"序列场剪枝: 移除 {removed} 个低频 bigram (min_count={min_count})")

    def merge(self, other: 'SequenceField'):
        """合并另一个 SequenceField 的数据"""
        for basin_key, fw in other._basin_forward.items():
            for ch_a, cnt in fw.items():
                self._basin_forward[basin_key][ch_a].update(cnt)
        for basin_key, bw in other._basin_backward.items():
            for ch_b, cnt in bw.items():
                self._basin_backward[basin_key][ch_b].update(cnt)
        self._vocab.update(other._vocab)
        self._total_bigrams += other._total_bigrams
