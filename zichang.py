#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字场模块（zichang.py）—— 龙珠知识内核的汉字锚点嵌入矩阵
================================================================
为初代龙珠构建完整的汉字锚点嵌入矩阵（字场）。
字场是龙珠的核心引擎——一个由所有CJK汉字嵌入向量构成的、永久冻结的能量景观基底。

依赖: torch, numpy, requests, unicodedata, sentence_transformers
嵌入模型: BAAI/bge-small-zh (512维, 本地推理, 首选)
          Ollama + nomic-embed-text (768维, 备选, 不推荐用于单字)
数据源: Unicode CJK统一汉字区间 (覆盖GB 18030-2022全部87887字)

重要发现: nomic-embed-text 对CJK单字区分力极弱(仅~1%唯一嵌入)，
          必须使用中文原生模型。BAAI/bge-small-zh 完美解决。

作者: Hermes + 李泽坤
版本: 1.0.0 (初代龙珠)
"""

import torch
import numpy as np
import requests
import time
import os
import sys
import json
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# ============================================================================
# 第一部分：汉字字表生成 —— 从Unicode CJK区间提取全部有效汉字
# ============================================================================

# GB 18030-2022 所覆盖的 CJK 统一汉字 Unicode 区间
# 这些区间共同构成了 87887 个编码汉字的完整集合
CJK_UNICODE_RANGES = [
    # (名称, 起始码点, 结束码点, 槽位数)
    ("CJK统一汉字基本区",  0x4E00,  0x9FFF ),   # 20,992 字 (U+4E00~U+9FFF)
    ("CJK统一汉字扩展A",   0x3400,  0x4DBF ),   #  6,592 字 (U+3400~U+4DBF)
    ("CJK统一汉字扩展B",   0x20000, 0x2A6DF),   # 42,720 字 (U+20000~U+2A6DF)
    ("CJK统一汉字扩展C",   0x2A700, 0x2B73F),   #  4,160 字 (U+2A700~U+2B73F)
    ("CJK统一汉字扩展D",   0x2B740, 0x2B81F),   #    224 字 (U+2B740~U+2B81F)
    ("CJK统一汉字扩展E",   0x2B820, 0x2CEAF),   #  5,776 字 (U+2B820~U+2CEAF)
    ("CJK统一汉字扩展F",   0x2CEB0, 0x2EBEF),   #  7,488 字 (U+2CEB0~U+2EBEF)
    ("CJK统一汉字扩展G",   0x30000, 0x3134F),   #  4,944 字 (U+30000~U+3134F)
    ("CJK统一汉字扩展H",   0x31350, 0x323AF),   #  4,192 字 (U+31350~U+323AF, Unicode 15.0+)
    ("CJK统一汉字扩展I",   0x2EBF0, 0x2EE5F),   #    624 字 (U+2EBF0~U+2EE5F, Unicode 15.1+)
    ("CJK兼容汉字",        0xF900,  0xFAFF ),   #    512 字 (U+F900~U+FAFF)
    ("CJK兼容汉字补充",    0x2F800, 0x2FA1F),   #    544 字 (U+2F800~U+2FA1F)
    ("康熙部首",           0x2F00,  0x2FDF ),   #    224 字 (U+2F00~U+2FDF)
    ("CJK笔划",            0x31C0,  0x31EF ),   #     48 字 (U+31C0~U+31EF)
]

# 总槽位数: 99,056，实际有效汉字约 94,000+（含未分配码点过滤后）


def generate_hanzi_list(
    ranges: List[Tuple[str, int, int]] = None,
    deduplicate: bool = True
) -> List[str]:
    """
    从Unicode CJK区间生成完整汉字列表。
    
    遍历所有CJK统一汉字Unicode区间，过滤掉未分配(Cn)、代理对(Cs)、
    纯控制字符(Cc)等无效码点，返回全部有效汉字字符。
    
    Args:
        ranges: Unicode区间列表，默认使用CJK_UNICODE_RANGES
        deduplicate: 是否去重（兼容区可能与基本区重复）
    
    Returns:
        汉字字符串列表，按Unicode码点排序
    """
    if ranges is None:
        ranges = [(name, s, e) for name, s, e, _ in CJK_UNICODE_RANGES 
                  if len(name) < 100]  # 兼容不带槽位数的格式
    
    # 重组为标准三元组
    normalized_ranges = []
    for item in ranges:
        if len(item) == 4:
            normalized_ranges.append((item[1], item[2]))  # (start, end)
        elif len(item) == 3:
            normalized_ranges.append((item[1], item[2]))
        else:
            normalized_ranges.append((item[0], item[1]))
    
    all_chars = []
    seen = set() if deduplicate else None
    
    for start, end in normalized_ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            cat = unicodedata.category(ch)
            
            # 过滤无效码点：
            #   Cn = 未分配 (Not Assigned)
            #   Cs = 代理对 (Surrogate, 仅用于UTF-16编码)
            #   Cc = 纯控制字符 (Control)
            if cat in ('Cn', 'Cs', 'Cc'):
                continue
            
            if deduplicate:
                if ch not in seen:
                    seen.add(ch)
                    all_chars.append(ch)
            else:
                all_chars.append(ch)
    
    return all_chars


def save_hanzi_list(hanzi_list: List[str], path: str):
    """将汉字列表保存为文本文件（每行一个汉字）"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for hz in hanzi_list:
            f.write(hz + '\n')
    print(f"汉字列表已保存: {path} ({len(hanzi_list)} 字)")


def load_hanzi_list(path: str) -> List[str]:
    """从文本文件加载汉字列表（每行一个汉字）"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n\r') for line in f if line.strip()]


# ============================================================================
# 第二部分：HanziAnchorField 字场核心类
# ============================================================================

class HanziAnchorField:
    """
    字场 —— 由全部CJK汉字嵌入向量构成的能量景观基底。
    
    字场是龙珠知识内核的确定性锚点层。每个汉字在768维嵌入空间中
    占据一个固定的坐标位置，所有知识的表达、检索和演化都在这个
    预冻结的基底上进行。字场本身是永久冻结的（frozen=True），
    不会被后续学习过程修改，确保知识的可追溯性和确定性。
    
    核心属性:
        hanzi_list:  汉字列表 (N个字符)
        num_hanzi:   汉字总数
        embed_dim:   嵌入向量维度 (默认768 = nomic-embed-text)
        anchors:     锚点矩阵 (num_hanzi × embed_dim), torch.Tensor
        frozen:      是否已冻结（构建完成后为True，禁止修改）
    
    核心方法:
        build_from_ollama():  调用Ollama批量嵌入API构建字场
        save() / load():      持久化/加载字场
        find_nearest():       余弦相似度检索最近汉字
        find_by_chars():      按字符查找锚点向量
        encode_text():        将文本映射为锚点向量序列
    """
    
    def __init__(
        self,
        hanzi_list: List[str],
        embed_dim: int = 1024,
        model_name: str = "BAAI/bge-large-zh",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        初始化字场。
        
        Args:
            hanzi_list: 汉字列表（按Unicode码点排序）
            embed_dim: 嵌入向量维度，默认1024（BAAI/bge-large-zh）
            model_name: 嵌入模型名称
            ollama_base_url: Ollama API地址
        """
        self.hanzi_list = list(hanzi_list)  # 防御性拷贝
        self.num_hanzi = len(hanzi_list)
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        # 锚点矩阵：None表示尚未构建
        self.anchors: Optional[torch.Tensor] = None
        
        # 冻结标志：构建完成后置True，禁止修改
        self.frozen = False
        
        # 构建统计
        self.build_stats: Dict = {
            'total': self.num_hanzi,
            'success': 0,
            'failed': 0,
            'elapsed_seconds': 0,
            'batch_size': 0,
        }
        
        # 汉字→索引映射（用于O(1)查找）
        self._char_to_idx: Dict[str, int] = {}
    
    def _build_index(self):
        """构建汉字→索引的快速查找表"""
        self._char_to_idx = {ch: i for i, ch in enumerate(self.hanzi_list)}
    
    # ------------------------------------------------------------------
    # 批量嵌入构建
    # ------------------------------------------------------------------
    
    def build_from_ollama(
        self,
        save_path: Optional[str] = None,
        resume: bool = True,
        batch_size: int = 200,
        request_delay: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        progress_interval: int = 1000,
        checkpoint_interval: int = 5000,
    ) -> 'HanziAnchorField':
        """
        调用Ollama批量嵌入API，为全部汉字生成嵌入向量。
        
        使用 /api/embed 批量端点（一次请求处理 batch_size 个汉字），
        相比逐字调用 /api/embeddings，速度提升约 batch_size 倍。
        
        支持断点续建：如果检测到 .temp 检查点文件，从上次中断位置继续。
        
        Args:
            save_path: 最终保存路径（.pt文件），也用于检查点命名
            resume: 是否启用断点续建（检查 .temp 文件）
            batch_size: 每批处理的汉字数量（建议100-500）
            request_delay: 批次间延迟（秒），避免压垮Ollama
            max_retries: 每批最大重试次数
            retry_delay: 重试间隔（秒）
            progress_interval: 进度打印间隔（每N字）
            checkpoint_interval: 检查点保存间隔（每N字）
        
        Returns:
            self（链式调用支持）
        """
        start_time = time.time()
        
        # 初始化锚点矩阵
        anchors = np.zeros((self.num_hanzi, self.embed_dim), dtype=np.float32)
        start_idx = 0
        
        # 断点续建：尝试加载检查点
        if resume and save_path:
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                try:
                    data = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    saved_anchors = data['anchors'].numpy()
                    saved_count = data['count']
                    anchors[:saved_count] = saved_anchors[:saved_count]
                    start_idx = saved_count
                    
                    elapsed = time.time() - start_time
                    pct = 100 * start_idx / self.num_hanzi
                    print(f"[断点续建] 从第 {start_idx}/{self.num_hanzi} 个汉字继续 "
                          f"({pct:.1f}%)，已耗时 {elapsed:.0f}s")
                except Exception as e:
                    print(f"[警告] 检查点加载失败: {e}，从头开始构建")
                    start_idx = 0
        
        # 批量嵌入构建主循环
        api_url = f"{self.ollama_base_url}/api/embed"
        total_batches = (self.num_hanzi - start_idx + batch_size - 1) // batch_size
        batch_count = 0
        failed_chars = []
        
        print(f"\n{'='*60}")
        print(f"字场构建开始")
        print(f"  汉字总数: {self.num_hanzi}")
        print(f"  嵌入维度: {self.embed_dim}")
        print(f"  嵌入模型: {self.model_name}")
        print(f"  批次大小: {batch_size}")
        print(f"  预估批次: {total_batches}")
        print(f"  起始位置: {start_idx}")
        print(f"{'='*60}\n")
        
        i = start_idx
        while i < self.num_hanzi:
            batch_start = time.time()
            batch_end = min(i + batch_size, self.num_hanzi)
            batch_chars = self.hanzi_list[i:batch_end]
            batch_count += 1
            
            # 调用Ollama批量嵌入API（带重试）
            embeddings = None
            last_error = None
            
            for retry in range(max_retries):
                try:
                    resp = requests.post(
                        api_url,
                        json={
                            "model": self.model_name,
                            "input": batch_chars
                        },
                        timeout=60  # 批量请求给予更长的超时
                    )
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        embeddings = result.get("embeddings", [])
                        
                        if len(embeddings) != len(batch_chars):
                            print(f"  [警告] 批次 {batch_count}: "
                                  f"期望 {len(batch_chars)} 个嵌入，实际收到 {len(embeddings)} 个")
                            # 截断或补零
                            while len(embeddings) < len(batch_chars):
                                embeddings.append([0.0] * self.embed_dim)
                            embeddings = embeddings[:len(batch_chars)]
                        
                        break  # 成功，跳出重试循环
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        
                except requests.exceptions.Timeout:
                    last_error = "请求超时"
                except requests.exceptions.ConnectionError:
                    last_error = "连接失败（Ollama可能未运行）"
                except Exception as e:
                    last_error = str(e)
                
                if retry < max_retries - 1:
                    print(f"  [重试 {retry+1}/{max_retries}] 批次 {batch_count}: {last_error}")
                    time.sleep(retry_delay * (retry + 1))  # 递增退避
            
            # 处理结果
            if embeddings is not None:
                for j, emb in enumerate(embeddings):
                    if len(emb) == self.embed_dim:
                        anchors[i + j] = np.array(emb, dtype=np.float32)
                    else:
                        # 维度不匹配：零向量占位
                        anchors[i + j] = np.zeros(self.embed_dim, dtype=np.float32)
                        failed_chars.append(self.hanzi_list[i + j])
            else:
                # 整批失败：全部零向量
                print(f"  [失败] 批次 {batch_count} ({i}-{batch_end-1}): {last_error}")
                for j in range(len(batch_chars)):
                    failed_chars.append(self.hanzi_list[i + j])
            
            batch_elapsed = time.time() - batch_start
            
            # 进度显示
            current_count = batch_end
            if current_count % progress_interval < batch_size or current_count == self.num_hanzi:
                total_elapsed = time.time() - start_time
                pct = 100 * current_count / self.num_hanzi
                eta = (total_elapsed / (current_count - start_idx)) * (self.num_hanzi - current_count) if current_count > start_idx else 0
                print(f"[进度] {current_count}/{self.num_hanzi} ({pct:.1f}%) | "
                      f"批次 {batch_count}/{total_batches} | "
                      f"耗时 {total_elapsed:.0f}s | "
                      f"预计剩余 {eta:.0f}s | "
                      f"批次耗时 {batch_elapsed:.1f}s")
            
            # 检查点保存
            if save_path and current_count % checkpoint_interval == 0:
                self._save_checkpoint(save_path, anchors, current_count)
            
            i = batch_end
            
            # 批次间延迟（避免压垮Ollama）
            if i < self.num_hanzi and request_delay > 0:
                time.sleep(request_delay)
        
        # 构建完成
        total_elapsed = time.time() - start_time
        self.anchors = torch.from_numpy(anchors).float()
        self.anchors.requires_grad = False  # 永久冻结
        self.frozen = True
        self._build_index()
        
        # 统计
        self.build_stats = {
            'total': self.num_hanzi,
            'success': self.num_hanzi - len(failed_chars),
            'failed': len(failed_chars),
            'elapsed_seconds': total_elapsed,
            'batch_size': batch_size,
            'total_batches': batch_count,
            'failed_chars': failed_chars[:100],  # 最多记录前100个失败字
        }
        
        # 最终保存
        if save_path:
            self.save(save_path)
            # 清理检查点文件
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        
        print(f"\n{'='*60}")
        print(f"字场构建完成！")
        print(f"  总汉字数: {self.num_hanzi}")
        print(f"  成功嵌入: {self.build_stats['success']}")
        print(f"  失败个数: {self.build_stats['failed']}")
        print(f"  总耗时:   {total_elapsed:.0f}s ({total_elapsed/60:.1f}分钟)")
        print(f"  平均速度: {self.num_hanzi/total_elapsed:.1f} 字/秒")
        print(f"  状态:     {'永久冻结' if self.frozen else '未冻结'}")
        print(f"{'='*60}\n")
        
        return self
    
    def _save_checkpoint(self, save_path: str, anchors: np.ndarray, count: int):
        """保存检查点（用于断点续建）"""
        try:
            ckpt_path = save_path + ".temp"
            torch.save({
                'anchors': torch.from_numpy(anchors[:count]).float(),
                'count': count,
                'hanzi_list': self.hanzi_list[:count],
                'embed_dim': self.embed_dim,
                'total': self.num_hanzi,
                'timestamp': time.time(),
            }, ckpt_path)
        except Exception as e:
            print(f"  [警告] 检查点保存失败: {e}")
    
    # ------------------------------------------------------------------
    # 单字嵌入构建（备选，逐字调用 /api/embeddings）
    # ------------------------------------------------------------------
    
    def build_from_ollama_single(
        self,
        save_path: Optional[str] = None,
        resume: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> 'HanziAnchorField':
        """
        逐字调用Ollama嵌入API构建字场（备选方案，较慢）。
        
        相比 build_from_ollama() 的批量模式，此方法每个汉字单独请求一次
        /api/embeddings，速度约为批量模式的 1/batch_size。
        仅当批量API不可用时作为备选。
        
        Args:
            save_path: 最终保存路径
            resume: 是否断点续建
            max_retries: 每字最大重试次数
            retry_delay: 重试间隔
        
        Returns:
            self
        """
        start_time_tracker = time.time()
        
        anchors = np.zeros((self.num_hanzi, self.embed_dim), dtype=np.float32)
        start_idx = 0
        
        # 断点续建
        if resume and save_path:
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                try:
                    data = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    anchors[:data['count']] = data['anchors'][:data['count']].numpy()
                    start_idx = data['count']
                    print(f"[断点续建] 从第 {start_idx} 个汉字继续")
                except Exception as e:
                    print(f"[警告] 检查点加载失败: {e}")
        
        api_url = f"{self.ollama_base_url}/api/embeddings"
        failed_count = 0
        
        print(f"\n逐字嵌入构建开始: {self.num_hanzi} 汉字 (从 {start_idx} 开始)")
        
        for i in range(start_idx, self.num_hanzi):
            hanzi = self.hanzi_list[i]
            success = False
            
            for retry in range(max_retries):
                try:
                    resp = requests.post(
                        api_url,
                        json={"model": self.model_name, "prompt": hanzi},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        embedding = resp.json()["embedding"]
                        if len(embedding) == self.embed_dim:
                            anchors[i] = np.array(embedding, dtype=np.float32)
                            success = True
                            break
                except Exception:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay)
            
            if not success:
                failed_count += 1
                if failed_count <= 10:
                    print(f"  [失败] 汉字 '{hanzi}' (第{i}个)")
            
            # 进度（每100字）
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time_tracker
                pct = 100 * (i + 1) / self.num_hanzi
                eta = elapsed / (i + 1 - start_idx) * (self.num_hanzi - i - 1) if i + 1 > start_idx else 0
                print(f"  进度: {i+1}/{self.num_hanzi} ({pct:.1f}%) | "
                      f"耗时 {elapsed:.0f}s | 预计剩余 {eta:.0f}s")
            
            # 检查点（每500字）
            if (i + 1) % 500 == 0 and save_path:
                self._save_checkpoint(save_path, anchors, i + 1)
        
        total_elapsed = time.time() - start_time_tracker
        self.anchors = torch.from_numpy(anchors).float()
        self.anchors.requires_grad = False
        self.frozen = True
        self._build_index()
        
        self.build_stats = {
            'total': self.num_hanzi,
            'success': self.num_hanzi - failed_count,
            'failed': failed_count,
            'elapsed_seconds': total_elapsed,
            'batch_size': 1,
            'total_batches': self.num_hanzi,
        }
        
        if save_path:
            self.save(save_path)
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        
        print(f"\n逐字构建完成: {self.num_hanzi} 字, "
              f"成功 {self.num_hanzi - failed_count}, "
              f"失败 {failed_count}, "
              f"耗时 {total_elapsed:.0f}s")
        
        return self
    
    # ------------------------------------------------------------------
    # 本地中文模型推理（BAAI/bge-small-zh，推荐）
    # ------------------------------------------------------------------
    
    def build_from_local(
        self,
        model_path: str,
        save_path: Optional[str] = None,
        resume: bool = True,
        batch_size: int = 512,
        device: str = "cpu",
        progress_interval: int = 5000,
        checkpoint_interval: int = 20000,
    ) -> 'HanziAnchorField':
        """
        使用本地 SentenceTransformer 模型构建字场（推荐方法）。
        
        与 Ollama 方法不同，此方法使用本地 GPU/CPU 推理 BAAI/bge-small-zh
        等中文原生模型。这些模型的 tokenizer 能正确识别每个 CJK 汉字，
        产生的嵌入向量具有真正的语义区分力。
        
        Args:
            model_path: 本地模型路径或 HuggingFace 模型名
            save_path: 最终保存路径
            resume: 是否断点续建
            batch_size: 批量大小（建议 256-1024）
            device: 推理设备
            progress_interval: 进度打印间隔
            checkpoint_interval: 检查点保存间隔
        
        Returns:
            self
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers: pip install sentence-transformers"
            )
        
        start_time = time.time()
        
        # 初始化锚点矩阵
        anchors = np.zeros((self.num_hanzi, self.embed_dim), dtype=np.float32)
        start_idx = 0
        
        # 断点续建
        if resume and save_path:
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                try:
                    data = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    saved_anchors = data['anchors'].numpy()
                    saved_count = data['count']
                    anchors[:saved_count] = saved_anchors[:saved_count]
                    start_idx = saved_count
                    elapsed = time.time() - start_time
                    pct = 100 * start_idx / self.num_hanzi
                    print(f"[断点续建] 从第 {start_idx}/{self.num_hanzi} 个汉字继续 "
                          f"({pct:.1f}%)，已耗时 {elapsed:.0f}s")
                except Exception as e:
                    print(f"[警告] 检查点加载失败: {e}，从头开始构建")
                    start_idx = 0
        
        # 加载模型
        print(f"加载本地模型: {model_path} ...")
        t0 = time.time()
        if device == "cpu":
            model = SentenceTransformer(model_path, device="cpu")
        else:
            model = SentenceTransformer(model_path, device=device)
        print(f"模型加载完成 ({time.time() - t0:.1f}s)，维度: {model.get_embedding_dimension()}")
        
        # 验证维度
        model_dim = model.get_embedding_dimension()
        if model_dim != self.embed_dim:
            print(f"[警告] 模型维度({model_dim})与字场维度({self.embed_dim})不一致，"
                  f"使用模型维度 {model_dim}")
            self.embed_dim = model_dim
            anchors = np.zeros((self.num_hanzi, self.embed_dim), dtype=np.float32)
        
        # 批量编码
        total_batches = (self.num_hanzi - start_idx + batch_size - 1) // batch_size
        batch_count = 0
        failed_count = 0
        
        print(f"\n{'='*60}")
        print(f"字场构建开始（本地模型）")
        print(f"  汉字总数: {self.num_hanzi}")
        print(f"  嵌入维度: {self.embed_dim}")
        print(f"  嵌入模型: {model_path}")
        print(f"  批次大小: {batch_size}")
        print(f"  预估批次: {total_batches}")
        print(f"  起始位置: {start_idx}")
        print(f"{'='*60}\n")
        
        i = start_idx
        while i < self.num_hanzi:
            batch_start = time.time()
            batch_end = min(i + batch_size, self.num_hanzi)
            batch_chars = self.hanzi_list[i:batch_end]
            batch_count += 1
            
            try:
                batch_embs = model.encode(
                    batch_chars,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                anchors[i:batch_end] = batch_embs
            except Exception as e:
                print(f"  [失败] 批次 {batch_count} ({i}-{batch_end-1}): {e}")
                failed_count += len(batch_chars)
            
            batch_elapsed = time.time() - batch_start
            
            # 进度
            current_count = batch_end
            if current_count % progress_interval < batch_size or current_count == self.num_hanzi:
                total_elapsed = time.time() - start_time
                pct = 100 * current_count / self.num_hanzi
                if current_count > start_idx:
                    eta = total_elapsed / (current_count - start_idx) * (self.num_hanzi - current_count)
                else:
                    eta = 0
                print(f"[进度] {current_count}/{self.num_hanzi} ({pct:.1f}%) | "
                      f"批次 {batch_count}/{total_batches} | "
                      f"耗时 {total_elapsed:.0f}s | "
                      f"预计剩余 {eta:.0f}s | "
                      f"批次耗时 {batch_elapsed:.1f}s")
            
            # 检查点
            if save_path and current_count % checkpoint_interval == 0:
                self._save_checkpoint(save_path, anchors, current_count)
            
            i = batch_end
        
        # 完成
        total_elapsed = time.time() - start_time
        self.anchors = torch.from_numpy(anchors).float()
        self.anchors.requires_grad = False
        self.frozen = True
        self._build_index()
        
        self.build_stats = {
            'total': self.num_hanzi,
            'success': self.num_hanzi - failed_count,
            'failed': failed_count,
            'elapsed_seconds': total_elapsed,
            'batch_size': batch_size,
            'total_batches': batch_count,
            'method': 'local',
            'model_path': model_path,
        }
        
        if save_path:
            self.save(save_path)
            ckpt_path = save_path + ".temp"
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        
        print(f"\n{'='*60}")
        print(f"字场构建完成！")
        print(f"  总汉字数: {self.num_hanzi}")
        print(f"  成功嵌入: {self.build_stats['success']}")
        print(f"  失败个数: {self.build_stats['failed']}")
        print(f"  总耗时:   {total_elapsed:.0f}s ({total_elapsed/60:.1f}分钟)")
        print(f"  平均速度: {self.num_hanzi/total_elapsed:.1f} 字/秒")
        print(f"  状态:     {'永久冻结' if self.frozen else '未冻结'}")
        print(f"{'='*60}\n")
        
        return self
    
    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    
    def save(self, path: str):
        """
        保存字场到磁盘。
        
        保存格式: PyTorch .pt 文件，包含:
            - hanzi_list: 汉字列表
            - anchors: 锚点矩阵 (num_hanzi × embed_dim)
            - embed_dim: 嵌入维度
            - num_hanzi: 汉字总数
            - frozen: 冻结状态
            - model_name: 使用的嵌入模型
            - build_stats: 构建统计信息
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        torch.save({
            'hanzi_list': self.hanzi_list,
            'anchors': self.anchors,
            'embed_dim': self.embed_dim,
            'num_hanzi': self.num_hanzi,
            'frozen': True,
            'model_name': self.model_name,
            'build_stats': self.build_stats,
        }, path)
        
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"字场已保存: {path} ({file_size_mb:.1f} MB)")
    
    @classmethod
    def load(cls, path: str, freeze: bool = True) -> 'HanziAnchorField':
        """
        从磁盘加载字场。
        
        Args:
            path: .pt 文件路径
            freeze: 是否冻结锚点矩阵（默认True）
        
        Returns:
            HanziAnchorField 实例
        """
        data = torch.load(path, map_location='cpu', weights_only=True)
        
        # 创建实例（不经过 __init__ 以避免重复构建索引）
        instance = cls.__new__(cls)
        instance.hanzi_list = data['hanzi_list']
        instance.anchors = data['anchors']
        instance.embed_dim = data['embed_dim']
        instance.num_hanzi = data['num_hanzi']
        instance.frozen = data.get('frozen', True)
        instance.model_name = data.get('model_name', 'nomic-embed-text')
        instance.ollama_base_url = 'http://localhost:11434'
        instance.build_stats = data.get('build_stats', {})
        
        if freeze:
            instance.anchors.requires_grad = False
            instance.frozen = True
        
        instance._build_index()
        
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"字场已加载: {path} ({file_size_mb:.1f} MB)")
        print(f"  汉字总数: {instance.num_hanzi}")
        print(f"  嵌入维度: {instance.embed_dim}")
        print(f"  嵌入模型: {instance.model_name}")
        print(f"  冻结状态: {'永久冻结' if instance.frozen else '未冻结'}")
        
        return instance
    
    # ------------------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------------------
    
    def find_nearest(
        self,
        query_vec: torch.Tensor,
        k: int = 5,
        exclude_chars: List[str] = None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        找到与查询向量余弦相似度最高的k个汉字。
        
        这是字场的核心检索操作。给定一个查询向量（可以是任意文本的嵌入、
        知识节点的激活向量等），返回在字场中与之最相似的k个汉字锚点。
        
        Args:
            query_vec: 查询向量 (embed_dim,) 或 (1, embed_dim)
            k: 返回最近邻数量
            exclude_chars: 排除的汉字列表（如已选中的字）
        
        Returns:
            (indices, chars, similarities) 三元组:
                indices:    最近汉字在hanzi_list中的索引 (LongTensor, k)
                chars:      最近汉字字符列表
                similarities: 余弦相似度值 (FloatTensor, k)
        """
        if self.anchors is None:
            raise RuntimeError("字场尚未构建，请先调用 build_from_ollama() 或 load()")
        
        # 确保查询向量是正确的形状
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)  # (embed_dim,) -> (1, embed_dim)
        
        with torch.no_grad():
            # 计算与全部锚点的余弦相似度
            # cosine_similarity(x1, x2, dim=1): x1(N,D) × x2(M,D) -> (N,M)
            similarities = torch.cosine_similarity(
                query_vec.float(),  # (1, D)
                self.anchors,       # (N, D)
                dim=1               # -> (N,)
            )
            
            # 排除指定字符
            if exclude_chars:
                exclude_indices = [self._char_to_idx.get(ch) for ch in exclude_chars]
                exclude_indices = [i for i in exclude_indices if i is not None]
                if exclude_indices:
                    similarities[exclude_indices] = -1.0  # 设为最小值以排除
            
            # 取Top-K
            actual_k = min(k, self.num_hanzi)
            top_k = torch.topk(similarities, actual_k)
            
            indices = top_k.indices
            chars = [self.hanzi_list[i] for i in indices]
            values = top_k.values
        
        return indices, chars, values
    
    def find_by_chars(self, chars: List[str]) -> torch.Tensor:
        """
        按汉字字符查找对应的锚点向量。
        
        Args:
            chars: 汉字字符列表
        
        Returns:
            锚点向量矩阵 (len(chars), embed_dim)，未找到的字返回零向量
        """
        if self.anchors is None:
            raise RuntimeError("字场尚未构建")
        
        vectors = []
        for ch in chars:
            idx = self._char_to_idx.get(ch)
            if idx is not None:
                vectors.append(self.anchors[idx])
            else:
                vectors.append(torch.zeros(self.embed_dim))
        
        return torch.stack(vectors) if vectors else torch.empty(0, self.embed_dim)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        将文本映射为锚点向量序列。
        
        提取文本中所有汉字字符，返回它们在字场中对应的锚点向量序列。
        非汉字字符（标点、数字、英文等）将被跳过。
        
        Args:
            text: 输入文本
        
        Returns:
            锚点向量序列 (M, embed_dim)，M为文本中的汉字数量
        """
        vectors = []
        for ch in text:
            idx = self._char_to_idx.get(ch)
            if idx is not None:
                vectors.append(self.anchors[idx])
        
        if not vectors:
            return torch.empty(0, self.embed_dim)
        return torch.stack(vectors)
    
    def expand_energy(
        self,
        seed_vec: torch.Tensor,
        temperature: float = 0.1,
        top_k: int = 10,
        exclude_chars: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        能量扩散：从种子向量出发，找到字场中能量相近的汉字。
        
        这是龙珠知识扩展的核心操作。给定一个种子向量（如知识激活后的
        融合向量），找出字场中与它共振的汉字，作为下一步展开的候选。
        
        Args:
            seed_vec: 种子向量 (embed_dim,)
            temperature: 温度参数（越大越分散，越小越集中）
            top_k: 返回候选数量
            exclude_chars: 排除的汉字
        
        Returns:
            [(汉字, 能量值), ...] 列表，按能量降序排列
        """
        indices, chars, similarities = self.find_nearest(
            seed_vec, k=top_k, exclude_chars=exclude_chars
        )
        
        # 应用温度缩放
        similarities = similarities / max(temperature, 1e-6)
        
        # Softmax 归一化为能量分布
        energies = torch.softmax(similarities, dim=0)
        
        return [(chars[i], energies[i].item()) for i in range(len(chars))]
    
    # ------------------------------------------------------------------
    # 信息查询
    # ------------------------------------------------------------------
    
    def __len__(self) -> int:
        return self.num_hanzi
    
    def __repr__(self) -> str:
        status = "已冻结" if self.frozen else "未构建" if self.anchors is None else "构建中"
        return (f"HanziAnchorField({self.num_hanzi}汉字, "
                f"{self.embed_dim}维, {status}, "
                f"模型={self.model_name})")
    
    def summary(self) -> str:
        """返回字场详细摘要"""
        lines = [
            f"字场摘要",
            f"  {'='*40}",
            f"  汉字总数:   {self.num_hanzi}",
            f"  嵌入维度:   {self.embed_dim}",
            f"  嵌入模型:   {self.model_name}",
            f"  锚点矩阵:   {self.anchors.shape if self.anchors is not None else '未构建'}",
            f"  冻结状态:   {'永久冻结' if self.frozen else '未冻结'}",
            f"  内存占用:   {self._memory_mb():.1f} MB",
        ]
        if self.build_stats:
            lines.append(f"  构建统计:")
            lines.append(f"    成功:     {self.build_stats.get('success', '?')}")
            lines.append(f"    失败:     {self.build_stats.get('failed', '?')}")
            lines.append(f"    耗时:     {self.build_stats.get('elapsed_seconds', 0):.0f}s")
        return '\n'.join(lines)
    
    def _memory_mb(self) -> float:
        """估算锚点矩阵内存占用（MB）"""
        if self.anchors is None:
            return 0.0
        return self.anchors.numel() * self.anchors.element_size() / (1024 * 1024)


# ============================================================================
# 第三部分：便捷函数
# ============================================================================

def create_zichang(
    hanzi_list_path: Optional[str] = None,
    output_path: Optional[str] = None,
    embed_dim: int = 1024,
    model_name: str = "BAAI/bge-large-zh",
    ollama_url: str = "http://localhost:11434",
    batch_size: int = 200,
    resume: bool = True,
) -> HanziAnchorField:
    """
    一键创建字场的便捷函数。
    
    自动完成: 生成汉字列表 → 创建HanziAnchorField → 批量嵌入 → 保存
    
    Args:
        hanzi_list_path: 已有汉字列表路径（None则自动从Unicode生成）
        output_path: 字场输出路径（.pt文件）
        embed_dim: 嵌入维度
        model_name: Ollama嵌入模型
        ollama_url: Ollama API地址
        batch_size: 批量大小
        resume: 是否断点续建
    
    Returns:
        构建完成的 HanziAnchorField 实例
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║         龙 珠 字 场 构 建 器                         ║")
    print("║         Hanzi Anchor Field Builder                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    
    # 第一步：生成或加载汉字列表
    if hanzi_list_path and os.path.exists(hanzi_list_path):
        print(f"[1/3] 加载汉字列表: {hanzi_list_path}")
        hanzi_list = load_hanzi_list(hanzi_list_path)
        print(f"      已加载 {len(hanzi_list)} 个汉字")
    else:
        print(f"[1/3] 从Unicode CJK区间生成汉字列表...")
        hanzi_list = generate_hanzi_list()
        print(f"      已生成 {len(hanzi_list)} 个有效CJK汉字")
        
        # 保存字表
        if output_path:
            list_path = output_path.replace('.pt', '_list.txt')
            save_hanzi_list(hanzi_list, list_path)
    
    # 第二步：创建字场实例
    print(f"\n[2/3] 创建字场实例...")
    zichang = HanziAnchorField(
        hanzi_list=hanzi_list,
        embed_dim=embed_dim,
        model_name=model_name,
        ollama_base_url=ollama_url,
    )
    print(f"      {zichang}")
    
    # 第三步：构建嵌入
    print(f"\n[3/3] 构建嵌入矩阵（批量模式, batch_size={batch_size}）...")
    zichang.build_from_ollama(
        save_path=output_path,
        resume=resume,
        batch_size=batch_size,
    )
    
    print(f"\n✅ 字场构建完毕！")
    if output_path:
        print(f"   输出文件: {output_path}")
    
    return zichang


# ============================================================================
# 第四部分：测试与验证
# ============================================================================

def test_zichang(zichang: HanziAnchorField):
    """对字场进行基本功能测试"""
    print("\n" + "="*60)
    print("字场功能测试")
    print("="*60)
    
    # 测试1: 基本属性
    print(f"\n[测试1] 基本属性:")
    print(f"  汉字数: {len(zichang)}")
    print(f"  维度:   {zichang.embed_dim}")
    print(f"  冻结:   {zichang.frozen}")
    assert zichang.frozen, "字场应为冻结状态"
    assert zichang.anchors is not None, "锚点矩阵不应为空"
    print("  ✅ 通过")
    
    # 测试2: 按字符查找
    print(f"\n[测试2] 字符→向量查找:")
    test_chars = ['龙', '珠', '知', '识', '核', '心']
    vecs = zichang.find_by_chars(test_chars)
    assert vecs.shape == (len(test_chars), zichang.embed_dim), f"形状错误: {vecs.shape}"
    # 验证非零（前几个高频字应该有有效嵌入）
    non_zero = (vecs.abs().sum(dim=1) > 0).sum().item()
    print(f"  测试字: {test_chars}")
    print(f"  向量形状: {vecs.shape}")
    print(f"  有效嵌入: {non_zero}/{len(test_chars)}")
    print("  ✅ 通过")
    
    # 测试3: 文本编码
    print(f"\n[测试3] 文本→锚点序列:")
    text = "龙珠知识内核"
    seq = zichang.encode_text(text)
    assert seq.shape[0] == len(text), f"编码长度错误: {seq.shape[0]} != {len(text)}"
    print(f"  输入: '{text}'")
    print(f"  输出: {seq.shape}")
    print("  ✅ 通过")
    
    # 测试4: 最近邻检索
    print(f"\n[测试4] 余弦相似度检索:")
    query = zichang.find_by_chars(['龙'])  # (1, 768)
    indices, chars, sims = zichang.find_nearest(query, k=10)
    print(f"  查询字: '龙'")
    print(f"  最近10字: {chars}")
    print(f"  相似度:   {[f'{s:.3f}' for s in sims[:5]]}...")
    assert '龙' in chars[:3], f"'龙'应该在Top-3结果中，实际: {chars[:3]}"
    print("  ✅ 通过")
    
    # 测试5: 能量扩散
    print(f"\n[测试5] 能量扩散:")
    seed = zichang.find_by_chars(['知'])
    candidates = zichang.expand_energy(seed.squeeze(), temperature=0.5, top_k=10)
    print(f"  种子: '知'")
    print(f"  扩散结果: {[(ch, f'{e:.4f}') for ch, e in candidates[:5]]}")
    print("  ✅ 通过")
    
    # 测试6: 内存占用
    print(f"\n[测试6] 资源占用:")
    mem_mb = zichang._memory_mb()
    print(f"  锚点矩阵: {mem_mb:.1f} MB")
    print(f"  形状:     {zichang.anchors.shape}")
    print(f"  数据类型: {zichang.anchors.dtype}")
    print("  ✅ 通过")
    
    print(f"\n{'='*60}")
    print("全部测试通过 ✅")
    print("="*60)


# ============================================================================
# 第五部分：主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="龙珠字场构建器 —— 为全部CJK汉字生成锚点嵌入矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整构建（生成字表 + 批量嵌入 + 保存）
  python zichang.py --output zichang_94117.pt
  
  # 使用已有字表构建
  python zichang.py --hanzi-list hanzi_list.txt --output zichang.pt
  
  # 从已有字场加载并测试
  python zichang.py --load zichang_94117.pt --test
  
  # 断点续建
  python zichang.py --output zichang.pt --resume
  
  # 生成字表（仅生成，不嵌入）
  python zichang.py --gen-list-only --output hanzi_list.txt
        """
    )
    
    parser.add_argument(
        '--output', '-o', type=str,
        help='字场输出路径 (.pt文件)'
    )
    parser.add_argument(
        '--hanzi-list', '-l', type=str,
        help='已有汉字列表文件路径（不指定则自动从Unicode生成）'
    )
    parser.add_argument(
        '--load', type=str,
        help='加载已有字场文件（跳过构建）'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='加载后运行功能测试'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=200,
        help='批量嵌入大小 (默认: 200)'
    )
    parser.add_argument(
        '--gen-list-only', action='store_true',
        help='仅生成汉字列表，不构建嵌入'
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='禁用断点续建（从头开始）'
    )
    parser.add_argument(
        '--embed-dim', '-d', type=int, default=768,
        help='嵌入维度 (默认: 768)'
    )
    parser.add_argument(
        '--model', '-m', type=str, default='nomic-embed-text',
        help='Ollama嵌入模型名称 (默认: nomic-embed-text)'
    )
    parser.add_argument(
        '--ollama-url', type=str, default='http://localhost:11434',
        help='Ollama API地址 (默认: http://localhost:11434)'
    )
    
    args = parser.parse_args()
    
    # 模式1: 仅生成字表
    if args.gen_list_only:
        hanzi_list = generate_hanzi_list()
        print(f"生成 {len(hanzi_list)} 个有效CJK汉字")
        
        output_path = args.output or 'hanzi_list.txt'
        save_hanzi_list(hanzi_list, output_path)
        
        # 打印统计
        print(f"\n各区间统计:")
        for name, start, end in [(n, s, e) for n, s, e, _ in CJK_UNICODE_RANGES]:
            count = sum(1 for hz in hanzi_list if start <= ord(hz) <= end)
            if count > 0:
                print(f"  {name}: {count}")
        sys.exit(0)
    
    # 模式2: 加载已有字场
    if args.load:
        zichang = HanziAnchorField.load(args.load)
        print(zichang.summary())
        if args.test:
            test_zichang(zichang)
        sys.exit(0)
    
    # 模式3: 完整构建
    if not args.output:
        parser.error("构建模式需要指定 --output 路径")
    
    zichang = create_zichang(
        hanzi_list_path=args.hanzi_list,
        output_path=args.output,
        embed_dim=args.embed_dim,
        model_name=args.model,
        ollama_url=args.ollama_url,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    
    # 自动测试
    if args.test:
        test_zichang(zichang)
    
    print("\n" + zichang.summary())
