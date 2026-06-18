#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双重知识提取器：正则优先 + LLM 兜底。

用于自主学习中从 Web 搜索结果提取结构化三元组。
正则处理标准句式（快速、确定），正则无法匹配时调用 Ollama LLM（灵活、高覆盖）。
"""

import re
import logging
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)


class DualExtractor:
    """正则优先 + LLM 兜底的知识提取器"""

    # ── 正则模板：从文本中提取三元组 ──

    _REGEX_PATTERNS = [
        # IS_A: "X是Y" / "X是一种Y"
        (r'([\u4e00-\u9fff]{1,6})是(?:一[种类个位名])?([\u4e00-\u9fff]{1,8})',
         'IS_A', 0.7),
        # DEFINED_AS: "X是指Y" / "X定义为Y"
        (r'([\u4e00-\u9fff]{1,6})(?:是指|定义为|亦称|又叫|也称)([\u4e00-\u9fff]{1,8})',
         'DEFINED_AS', 0.7),
        # PART_OF: "X属于Y" / "X是Y的一部分"
        (r'([\u4e00-\u9fff]{1,6})属于([\u4e00-\u9fff]{1,8})',
         'PART_OF', 0.6),
        # RELATED: "X和Y" / "X与Y" (两短词并列)
        (r'([\u4e00-\u9fff]{2,4})和([\u4e00-\u9fff]{2,4})(?:都是|均为|一样)',
         'RELATED', 0.5),
        # COOCCURS_WITH: "X的Y" / "X之Y"
        (r'([\u4e00-\u9fff]{1,4})的([\u4e00-\u9fff]{1,4})',
         'COOCCURS_WITH', 0.3),
    ]

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen2.5:3b"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    # ── 正则提取 ──

    def extract_regex(self, text: str) -> List[Tuple[str, str, str, float]]:
        """正则提取三元组 → [(s, r, o, conf), ...]"""
        triples = []
        for pattern, relation, conf in self._REGEX_PATTERNS:
            for match in re.finditer(pattern, text):
                s, o = match.group(1), match.group(2)
                # 过滤纯标点/数字
                if re.search(r'[\u4e00-\u9fff]', s) and re.search(r'[\u4e00-\u9fff]', o):
                    triples.append((s, relation, o, conf))
        return triples

    # ── LLM 兜底 ──

    def extract_llm(self, text: str) -> List[Tuple[str, str, str, float]]:
        """Ollama LLM 提取三元组"""
        try:
            import requests
            prompt = (
                "从以下文本中提取知识三元组。每行格式: 主语|关系|宾语\n"
                "关系只能是: IS_A, PART_OF, RELATED, COOCCURS_WITH, DEFINED_AS\n"
                f"文本: {text[:2000]}\n"
                "三元组:"
            )
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 256, "temperature": 0.1},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                response_text = resp.json().get("response", "")
                triples = []
                for line in response_text.strip().split("\n"):
                    parts = line.strip().split("|")
                    if len(parts) == 3:
                        s, r, o = parts
                        r = r.strip()
                        if r in ('IS_A', 'PART_OF', 'RELATED', 'COOCCURS_WITH', 'DEFINED_AS'):
                            triples.append((s.strip(), r, o.strip(), 0.5))  # LLM提取低置信
                return triples
        except Exception as e:
            log.debug(f"LLM提取失败: {e}")
        return []

    # ── 双重提取 ──

    def extract(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        双重提取: 正则优先(conf=0.3-0.7) → 正则无结果 → LLM兜底(conf=0.5)

        Returns: [(subject, relation, object, confidence), ...]
        """
        triples = self.extract_regex(text)
        if triples:
            return triples

        # 正则无结果 → LLM 兜底
        log.info(f"正则无结果，尝试LLM提取 (文本{len(text)}字)")
        llm_triples = self.extract_llm(text)
        return llm_triples if llm_triples else []
