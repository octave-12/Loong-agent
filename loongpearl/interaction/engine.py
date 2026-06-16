#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠主类（loongpearl.py）—— 整合字场、能量景观、学习机制的知识内核
====================================================================

龙珠 = 字场（锚点基底） + 能量景观（吸引子网络） + 学习器（用进废退）

    查询流程:
      文本 → 编码为1024维向量 → 自知无知检测
        ├─ 已知 → 能量景观梯度下降 → 收敛到吸引子 → 映射回汉字
        └─ 未知 → 可选触发Ollama学习 → 注入能量景观 → 重新查询

    学习流程:
      Hebbian 更新（用进） + 衰减调度（废退） + 自知无知校准

核心组件:
    HanziAnchorField   — 94117个汉字嵌入锚点（永久冻结）
    EnergyLandscape    — 可微分吸引子网络（1024 → 3072 → 3072 → 1536 → 1）
    DragonBallLearner  — Hebbian学习 + 自知无知 + 衰减调度
    OllamaSeeder       — 用DeepSeek-R1批量播种语义关联

作者: Hermes + 李泽坤
版本: 1.0.0 (初代龙珠)
"""

import torch
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import requests
from sentence_transformers import SentenceTransformer

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loongpearl.core.zichang import HanziAnchorField
from loongpearl.core.freq_landscape import FreqEnergyLandscape
from loongpearl.learning.learner import DragonBallLearner, HebbianLearner
from loongpearl.learning.autonomous_learner import AutonomousLearner
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR
from loongpearl.utils.compute_sandbox import ComputeSandbox


# ============================================================================
# 数据类：查询结果
# ============================================================================

@dataclass
class QueryResult:
    """龙珠查询的结构化结果"""
    question: str                          # 原始问题
    is_known: bool = False                 # 系统是否"知道"这个问题的领域
    confidence: float = 0.0                # 自知无知置信度 (0~1)
    answer_text: str = ""                  # 人类可读的答案
    nearest_chars: List[str] = field(default_factory=list)   # 最近汉字列表
    similarities: List[float] = field(default_factory=list)  # 相似度列表
    energy: float = 0.0                    # 收敛后的能量值
    steps: int = 0                         # 梯度下降步数
    converged: bool = False                # 是否收敛
    diagnosis: str = ""                    # 自知无知诊断文本
    
    def __repr__(self):
        status = "✅已知" if self.is_known else "❓未知"
        return (f"QueryResult({status} conf={self.confidence:.2%} "
                f"energy={self.energy:.2f} steps={self.steps})")


# ============================================================================
# 龙珠主类
# ============================================================================

class LoongPearl:
    """
    龙珠 —— 以汉字为锚点的确定性知识内核。
    
    完整的查询-推理-学习循环:
    
        query("量子计算") → 编码 → 自知无知检测
          ├─ 已知 → infer → resolve → 返回最近汉字
          └─ 未知 → learn_from_ollama → 注入知识 → 重新查询
    
    典型用法:
    
        loongpearl = LoongPearl()
        loongpearl.initialize()
        
        result = loongpearl.query("人工智能")
        print(result.answer_text)  # 「智」是知识网络中最相关的概念...
    """
    
    # 默认文件路径
    DEFAULT_ZICHANG = "data/models/zichang_94117_1024d.pt"
    DEFAULT_LANDSCAPE = "data/models/energy_landscape_1024d.pt"
    DEFAULT_EMBED_MODEL = "BAAI/bge-large-zh"
    
    def __init__(
        self,
        model_dir: str = None,
        embed_dim: int = 1024,
        embed_model: str = None,
        device: str = "cpu",
    ):
        """
        初始化龙珠（不加载模型，需调用 initialize()）。
        
        Args:
            model_dir: 模型文件目录，默认当前脚本所在目录
            embed_dim: 嵌入维度（1024，对应 BAAI/bge-large-zh）
            embed_model: 编码模型名（SentenceTransformer 格式）
            device: 计算设备
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = model_dir
        self.embed_dim = embed_dim
        self.embed_model_name = embed_model or self.DEFAULT_EMBED_MODEL
        self.device = device
        
        # 文件路径
        self.zichang_path = os.path.join(model_dir, self.DEFAULT_ZICHANG)
        self.landscape_path = os.path.join(model_dir, self.DEFAULT_LANDSCAPE)
        
        # 核心组件（initialize() 中加载）
        self.zichang: Optional[HanziAnchorField] = None
        self.landscape: Optional[FreqEnergyLandscape] = None
        self.learner: Optional[DragonBallLearner] = None
        self.autonomous_learner: Optional[AutonomousLearner] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self._sandbox: Optional[ComputeSandbox] = None
        
        self.initialized = False
        
        # 统计
        self.total_queries = 0
        self.total_known = 0
        self.total_learned = 0
    
    # ------------------------------------------------------------------
    # 计算沙盒（懒加载）
    # ------------------------------------------------------------------
    
    @property
    def sandbox(self) -> ComputeSandbox:
        """计算沙盒懒加载 —— 首次访问时创建"""
        if self._sandbox is None:
            self._sandbox = ComputeSandbox(timeout=10)
        return self._sandbox
    
    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    
    def initialize(self, verbose: bool = True) -> 'LoongPearl':
        """
        初始化龙珠：加载字场、能量景观、学习器、编码模型。
        
        加载顺序:
          1. 字场 (zichang_94117_1024d.pt, ~369 MB)
          2. 能量景观 (energy_landscape_1024d.pt, ~67 MB)
          3. 学习器 (DragonBallLearner)
          4. 编码模型 (BAAI/bge-large-zh, ~1.3 GB, 首次下载)
        
        Returns:
            self（支持链式调用）
        """
        if self.initialized:
            return self
        
        printer = print if verbose else lambda *a, **kw: None
        
        # 1. 加载字场
        printer("[1/4] 加载字场...")
        self.zichang = HanziAnchorField.load(self.zichang_path)
        printer(f"      字场: {self.zichang.num_hanzi} 汉字, {self.zichang.embed_dim}维")
        
        # 2. 加载能量景观
        printer("[2/4] 加载能量景观...")
        self.landscape = FreqEnergyLandscape.load(self.landscape_path)
        self.landscape.eval()
        self.landscape.to(self.device)
        printer(f"      能量景观: 已加载 (dim={self.landscape.embed_dim})")
        
        # 3. 创建学习器
        printer("[3/4] 初始化学习器...")
        self.learner = DragonBallLearner(
            landscape=self.landscape,
            anchor_field=self.zichang,
            hebbian_lr=0.001,
            device=self.device,
        )
        
        # 校准自知无知检测器（在锚点上采样建立参考分布）
        try:
            self.learner.calibrate()
            printer(f"      学习器: 已校准")
        except Exception as e:
            printer(f"      学习器: 校准跳过 ({e})")
        
        # 3.5 创建自主学习引擎
        self.autonomous_learner = AutonomousLearner(
            zichang=self.zichang,
            landscape=self.landscape,
            learner=self.learner,
        )
        printer(f"      自主学习引擎: 就绪")
        
        # 4. 加载编码模型
        printer("[4/4] 加载编码模型...")
        self.embed_model = SentenceTransformer(
            self.embed_model_name,
            device=self.device,
            local_files_only=True,
        )
        printer(f"      编码模型: {self.embed_model_name}（本地缓存）")
        
        self.initialized = True
        printer(f"\n🐉 龙珠初始化完成！{self.zichang.num_hanzi} 汉字锚点就绪\n")
        
        return self
    
    # ------------------------------------------------------------------
    # 文本编码
    # ------------------------------------------------------------------
    
    def _encode(self, text: str) -> torch.Tensor:
        """
        将文本编码为嵌入向量。
        
        Args:
            text: 任意文本
        
        Returns:
            (embed_dim,) 浮点张量
        """
        embedding = self.embed_model.encode([text], normalize_embeddings=True)[0]
        return torch.from_numpy(embedding).float().to(self.device)
    
    # ------------------------------------------------------------------
    # 查询（核心入口）
    # ------------------------------------------------------------------
    
    def query(
        self,
        question: str,
        auto_learn: bool = True,
        infer_steps: int = 50,
        verbose: bool = False,
    ) -> QueryResult:
        """
        查询龙珠——完整的查询-推理-学习循环。
        
        流程:
          1. 文本编码为 1024 维向量
          2. 自知无知检测（梯度+能量+距离三信号综合）
          3. 如果已知 → 能量景观梯度下降 → 收敛 → 映射到最近汉字
          4. 如果未知 → 可选触发 Ollama 学习 → 注入知识 → 重新查询
        
        Args:
            question: 查询文本
            auto_learn: 未知时是否自动触发学习
            infer_steps: 梯度下降最大步数
            verbose: 是否详细输出
        
        Returns:
            QueryResult 结构化结果
        """
        if not self.initialized:
            raise RuntimeError("龙珠未初始化，请先调用 initialize()")
        
        self.total_queries += 1
        
        # 步骤1: 编码查询
        query_vec = self._encode(question)
        
        # 步骤1.5: 计算沙盒检测 —— 数学类问题直接计算，不经过能量景观
        if self.sandbox.is_math_question(question):
            answer = self.sandbox.calculate(question)
            return QueryResult(
                question=question,
                is_known=True,
                confidence=1.0,
                answer_text=answer,
                diagnosis="计算沙盒",
            )
        
        # 步骤2: 自知无知检测
        check_result = self.learner.check_knowledge(query_vec)
        is_known = check_result['is_known']
        confidence = check_result['confidence']
        
        if verbose:
            print(f"  自知无知: {check_result['diagnosis']} (conf={confidence:.2%})")
        
        # 步骤3a: 未知 → 自主学习（不依赖 Ollama）
        if not is_known and auto_learn:
            if verbose:
                print(f"  触发自主学习: '{question}' → 联网搜索...")
            
            # 优先走自主搜索学习（零 LLM）
            learn_result = self.autonomous_learner.learn_if_unknown(
                query_text=question,
                query_vec=query_vec,
                auto_search=True,
            )
            
            if learn_result['status'] == 'learned':
                self.total_learned += 1
                # 保存更新后的能量景观
                try:
                    self.save_landscape()
                except Exception:
                    pass
                # 学习后重新查询
                return self.query(question, auto_learn=False, infer_steps=infer_steps, verbose=verbose)
            
            # 自主搜索失败 → 回退 Ollama（如果可用）
            if learn_result['status'] == 'search_failed':
                if verbose:
                    print(f"  联网搜索未找到知识，回退 Ollama...")
                learned = self.learn_from_ollama(question)
                if learned:
                    self.total_learned += 1
                    return self.query(question, auto_learn=False, infer_steps=infer_steps, verbose=verbose)
            else:
                # 学习失败，返回"不知道"
                return QueryResult(
                    question=question,
                    is_known=False,
                    confidence=confidence,
                    answer_text="这个知识我还没学到，Ollama 也未返回有效信息。",
                    diagnosis=check_result.get('diagnosis', ''),
                    nearest_chars=check_result.get('nearest_chars', [])[:3],
                )
        
        # 步骤3b: 未知且不学习 → 直接返回
        if not is_known:
            return QueryResult(
                question=question,
                is_known=False,
                confidence=confidence,
                answer_text="这个知识我还没学到。需要我现在去学习吗？",
                diagnosis=check_result.get('diagnosis', ''),
                nearest_chars=check_result.get('nearest_chars', [])[:3],
            )
        
        # 步骤4: 已知 → 能量景观推理
        self.total_known += 1
        
        infer_result = self.landscape.infer(
            query_vec,
            steps=infer_steps,
            lr=0.02,
            convergence_threshold=1e-5,
        )
        
        # 步骤5: 映射到最近汉字
        resolved = self.landscape.resolve(
            self.zichang,
            infer_result['state'],
            top_k=5,
        )
        
        # 提取结果
        nearest_chars = [ch for ch, _ in resolved]
        similarities = [sim for _, sim in resolved]
        
        # 用进：强化查询→收敛点的路径（提升未来查询效率）
        try:
            self.learner.learn(query_vec, infer_result['state'], feedback=0.3)
        except Exception:
            pass  # 强化失败不影响主流程
        
        # 格式化答案
        answer_text = self._format_answer(nearest_chars, similarities, infer_result)
        
        return QueryResult(
            question=question,
            is_known=True,
            confidence=confidence,
            answer_text=answer_text,
            nearest_chars=nearest_chars,
            similarities=similarities,
            energy=infer_result['energy'],
            steps=infer_result['steps'],
            converged=infer_result['converged'],
            diagnosis=check_result.get('diagnosis', ''),
        )
    
    # ------------------------------------------------------------------
    # Ollama 学习
    # ------------------------------------------------------------------
    
    def learn_from_ollama(self, text: str, max_retries: int = 2) -> bool:
        """
        调用 Ollama (DeepSeek-R1) 学习新概念，将知识注入能量景观。
        
        流程:
          1. 构建提示词，让模型提取核心关键词（单汉字）
          2. 解析返回的 JSON
          3. 对每个关联字 → Hebbian 学习降低路径能量
        
        Args:
            text: 要学习的概念
            max_retries: 最大重试次数
        
        Returns:
            是否成功学习到新知识
        """
        for attempt in range(max_retries):
            try:
                # 调用 Ollama
                prompt = (
                    f'Analyze the concept "{text}". '
                    f'List 3-5 core Chinese characters (single char each) '
                    f'semantically related to this concept. '
                    f'Output ONLY: [{{"hanzi":"char","relation":"type","evidence":"reason"}}]'
                )
                
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "deepseek-r1:7b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.6, "num_predict": 2000},
                    },
                    timeout=120,
                )
                
                if resp.status_code != 200:
                    continue
                
                raw = resp.json().get("response", "")
                data = self._extract_json(raw)
                
                if not data:
                    continue
                
                # 注入能量景观
                query_vec = self._encode(text)
                implanted = 0
                
                for item in data:
                    hanzi = item.get('hanzi', '').strip()
                    if not hanzi or len(hanzi) != 1:
                        continue
                    if hanzi not in self.zichang._char_to_idx:
                        continue
                    
                    target_idx = self.zichang._char_to_idx[hanzi]
                    target_vec = self.zichang.anchors[target_idx].to(self.device)
                    
                    result = self.learner.learn(query_vec, target_vec, feedback=0.5)
                    if result.get('status') != 'skipped':
                        implanted += 1
                
                if implanted > 0:
                    # 学习后保存能量景观
                    self.save_landscape()
                    return True
                
            except Exception:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
        
        return False
    
    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        """从 Ollama 响应中提取 JSON，兼容字符串/对象数组"""
        import re
        
        if not text:
            return None
        
        # 策略1: 代码块
        for m in re.finditer(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL):
            parsed = self._try_parse_json(m.group(1))
            if parsed:
                return parsed
        
        # 策略2: 最外层 [...]
        s = text.find('[')
        e = text.rfind(']') + 1
        if s >= 0 and e > s:
            parsed = self._try_parse_json(text[s:e])
            if parsed:
                return parsed
        
        return None
    
    def _try_parse_json(self, js: str) -> Optional[List[Dict]]:
        """解析 JSON，标准化为对象数组"""
        try:
            data = json.loads(js)
        except json.JSONDecodeError:
            return None
        
        if not isinstance(data, list):
            return None
        
        result = []
        for item in data:
            if isinstance(item, str):
                result.append({"hanzi": item, "relation": "未知"})
            elif isinstance(item, dict) and 'hanzi' in item:
                result.append({
                    "hanzi": item['hanzi'].strip(),
                    "relation": item.get('relation', '未知'),
                })
        return result if result else None
    
    # ------------------------------------------------------------------
    # 文本 → 最近汉字（快速查询，不做自知无知检测）
    # ------------------------------------------------------------------
    
    def find_nearest_chars(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        快速找到与文本最相关的 k 个汉字（跳过自知无知检测）。
        
        Args:
            text: 输入文本
            k: 返回数量
        
        Returns:
            [(汉字, 相似度), ...]
        """
        if not self.initialized:
            raise RuntimeError("龙珠未初始化")
        
        vec = self._encode(text)
        _, chars, sims = self.zichang.find_nearest(vec, k=k)
        return list(zip(chars, sims.tolist()))
    
    # ------------------------------------------------------------------
    # 汉字间推理
    # ------------------------------------------------------------------
    
    def reason_between(self, char_a: str, char_b: str, steps: int = 50) -> Dict:
        """
        在两个汉字之间进行能量景观推理——从 char_a 出发，
        沿梯度下降到吸引子，观察是否收敛到 char_b。
        
        可用于判断两个概念之间的关联强度。
        
        Args:
            char_a: 起始汉字
            char_b: 目标汉字
            steps: 梯度下降步数
        
        Returns:
            推理结果字典
        """
        if not self.initialized:
            raise RuntimeError("龙珠未初始化")
        
        if char_a not in self.zichang._char_to_idx:
            return {'error': f"'{char_a}'不在字场中"}
        if char_b not in self.zichang._char_to_idx:
            return {'error': f"'{char_b}'不在字场中"}
        
        a_idx = self.zichang._char_to_idx[char_a]
        b_idx = self.zichang._char_to_idx[char_b]
        
        a_vec = self.zichang.anchors[a_idx].to(self.device)
        b_vec = self.zichang.anchors[b_idx].to(self.device)
        
        # 从中点出发推理
        mid_vec = (a_vec + b_vec) / 2.0
        mid_vec = torch.nn.functional.normalize(mid_vec, dim=-1)
        
        infer_result = self.landscape.infer(mid_vec, steps=steps)
        resolved = self.landscape.resolve(self.zichang, infer_result['state'], top_k=3)
        
        # 检查是否收敛到 char_b
        converged_to_b = resolved[0][0] == char_b
        
        # 计算两字间的初始能量
        import torch.nn.functional as F
        with torch.no_grad():
            energy_a = self.landscape.energy(a_vec.unsqueeze(0)).item()
            energy_b = self.landscape.energy(b_vec.unsqueeze(0)).item()
            energy_mid = self.landscape.energy(mid_vec.unsqueeze(0)).item()
        
        return {
            'from': char_a,
            'to': char_b,
            'converged_to_target': converged_to_b,
            'nearest_chars': [ch for ch, _ in resolved],
            'similarities': [sim for _, sim in resolved],
            'energy_a': energy_a,
            'energy_b': energy_b,
            'energy_mid': energy_mid,
            'path_barrier': energy_mid - min(energy_a, energy_b),  # 路径能垒
            'steps': infer_result['steps'],
            'converged': infer_result['converged'],
        }
    
    # ------------------------------------------------------------------
    # 格式化
    # ------------------------------------------------------------------
    
    def _format_answer(
        self,
        chars: List[str],
        sims: List[float],
        infer_result: Dict,
    ) -> str:
        """格式化推理结果为人类可读文本"""
        top = chars[0] if chars else "?"
        related = chars[1:4] if len(chars) >= 4 else chars[1:]
        related_str = '、'.join(related) if related else '无'
        
        return (
            f"「{top}」是知识网络中最相关的概念。"
            f"（相关：{related_str}）"
            f"[能量:{infer_result['energy']:.2f}, 步数:{infer_result['steps']}]"
        )
    
    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    
    def save_landscape(self, path: str = None):
        """保存能量景观到文件"""
        self.landscape.save(path or self.landscape_path)
    
    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    
    def get_stats(self) -> Dict:
        """获取龙珠运行统计"""
        stats = {
            'initialized': self.initialized,
            'total_queries': self.total_queries,
            'total_known': self.total_known,
            'total_learned': self.total_learned,
            'known_ratio': self.total_known / max(self.total_queries, 1),
        }
        
        if self.zichang:
            stats['num_hanzi'] = self.zichang.num_hanzi
            stats['embed_dim'] = self.zichang.embed_dim
        
        if self.learner:
            stats['learner'] = self.learner.get_stats()
        
        return stats
    
    def __repr__(self):
        status = "🐉 已初始化" if self.initialized else "⏳ 未初始化"
        if self.initialized:
            status += f" ({self.zichang.num_hanzi}字)"
        return f"LoongPearl({status})"


# ============================================================================
# 便捷函数
# ============================================================================

def quick_start(model_dir: str = None, verbose: bool = True) -> LoongPearl:
    """
    快速启动龙珠（一行代码初始化）。
    
    用法:
        loongpearl = quick_start()
        result = loongpearl.query("人工智能")
    """
    loongpearl = LoongPearl(model_dir=model_dir)
    loongpearl.initialize(verbose=verbose)
    return loongpearl


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="龙珠 —— 汉字锚点知识内核")
    parser.add_argument('query', nargs='?', type=str, help='查询文本')
    parser.add_argument('--model-dir', '-d', type=str, help='模型目录')
    parser.add_argument('--no-learn', action='store_true', help='未知时不自动学习')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    loongpearl = quick_start(model_dir=args.model_dir, verbose=args.verbose)
    
    if args.query:
        result = loongpearl.query(
            args.query,
            auto_learn=not args.no_learn,
            verbose=args.verbose,
        )
        print(result)
        print(f"\n答案: {result.answer_text}")
    else:
        # 交互模式
        print("\n🐉 龙珠交互模式（输入 'quit' 退出）\n")
        while True:
            try:
                q = input("> ").strip()
                if q.lower() in ('quit', 'exit', 'q'):
                    break
                if not q:
                    continue
                
                result = loongpearl.query(q)
                print(f"  {result}")
                print(f"  {result.answer_text}\n")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
