#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合化解码器 — 模板 + LLM 润色。

简单回答（置信度高、单条事实）→ 纯模板，快速确定。
复杂回答（多跳推理、低置信）→ 模板生成骨架 + LLM 润色。

LLM 润色时注入推理路径作为硬约束，防止编造。
LLM 不可用时自动回退到纯模板。
"""

import logging
from typing import Dict, Optional

log = logging.getLogger(__name__)


class HybridDecoder:
    """模板优先 + LLM 润色的混合解码器"""

    def __init__(self, template_decoder=None,
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen2.5:3b"):
        from loongpearl.core.energy_decoder import EnergyDecoder
        self.template = template_decoder or EnergyDecoder()
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def decode(self, render_input: Dict, complexity: str = 'auto') -> str:
        """
        解码渲染输入为自然语言。

        Args:
            render_input: 模板森林输入
            complexity: 'simple' | 'complex' | 'auto'

        Returns:
            自然语言文本
        """
        # 自动判断复杂度
        if complexity == 'auto':
            complexity = self._judge_complexity(render_input)

        skeleton = self.template.render(render_input)

        if complexity == 'simple':
            return skeleton

        # 复杂回答: LLM 润色
        return self._llm_polish(skeleton, render_input)

    def _judge_complexity(self, render_input: Dict) -> str:
        """判断渲染复杂度"""
        facts = render_input.get('facts', [])
        render_type = render_input.get('render_type', 'fact_statement')

        # 单条高置信 → 简单
        if len(facts) <= 1 and render_type in ('fact_statement', 'list_related'):
            return 'simple'

        # 多跳路径/对比 → 复杂
        if render_type in ('explain_path', 'compare', 'table'):
            return 'complex'

        return 'simple' if len(facts) <= 2 else 'complex'

    def _llm_polish(self, skeleton: str, render_input: Dict) -> str:
        """LLM 润色骨架文本，注入推理路径约束"""
        try:
            import requests

            # 构建约束提示词
            facts_summary = ""
            for fact in render_input.get('facts', [])[:5]:
                subj = fact.get('subject', '')
                rel = fact.get('relation', '')
                obj = fact.get('object', '')
                conf = fact.get('confidence', 1.0)
                facts_summary += f"- {subj} {rel} {obj} (置信度={conf:.2f})\n"

            prompt = (
                "你是一个严谨的知识助手。请基于以下事实将骨架文本润色为流畅的中文。\n"
                "严格约束:\n"
                "1. 不要编造任何骨架中没有的事实\n"
                "2. 不要添加任何骨架中没有出现的概念\n"
                "3. 保持原意，只改善流畅度和自然度\n"
                "4. 如果骨架已足够好，直接返回原文\n\n"
                f"原始事实:\n{facts_summary}\n"
                f"骨架文本:\n{skeleton}\n\n"
                "润色后:"
            )

            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 256, "temperature": 0.3},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                polished = resp.json().get("response", "").strip()
                if polished and len(polished) > len(skeleton) * 0.3:
                    return polished
        except Exception as e:
            log.debug(f"LLM润色失败: {e}")

        # 回退到纯模板
        return skeleton
