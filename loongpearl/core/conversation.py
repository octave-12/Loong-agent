#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠对话引擎 (Conversation) — 多轮状态 + 社交模式 + 闲聊路由
════════════════════════════════════════════════════════════════════════

在解义器+化能器基础上，增加多轮对话管理：
  1. 对话状态追踪 (上下文记忆)
  2. 社交模式识别 (问候/告别/感谢/确认)
  3. 话题跟随 (从上一轮继承主题)
  4. 闲聊路由 (社交→NLU查询→兜底)

════════════════════════════════════════════════════════════════════════
设计原则
════════════════════════════════════════════════════════════════════════

  确定性闲聊 ≠ LLM式闲聊。
  我们不生成新内容——我们从概念图中提取相关事实，用模板组织。
  闲聊的本质是"低信息密度但有社交功能的语言交换"，
  可以通过话题跟随 + 关联展开来实现。

════════════════════════════════════════════════════════════════════════
"""

import re
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque


# ═══════════════════════════════════════════════════════════════════════
# 对话状态
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DialogueState:
    """多轮对话状态"""
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    active_topic: Optional[str] = None    # 当前话题
    mentioned_concepts: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_intent: Optional[str] = None
    user_name: Optional[str] = None       # 如果用户自报姓名
    
    def add_turn(self, user_input: str, system_response: str,
                 concepts: List[str] = None):
        self.turn_count += 1
        self.history.append({
            "user": user_input,
            "system": system_response,
            "concepts": concepts or [],
            "time": time.time(),
        })
        if concepts:
            self.mentioned_concepts.extend(concepts)
            self.mentioned_concepts = self.mentioned_concepts[-20:]  # 保留最近20个
    
    def set_topic(self, topic: str):
        self.active_topic = topic
    
    def recent_concepts(self, n: int = 5) -> List[str]:
        """获取最近谈到的概念"""
        all_concepts = []
        for turn in list(self.history)[-3:]:
            all_concepts.extend(turn.get('concepts', []))
        return list(dict.fromkeys(all_concepts))[-n:]  # 去重保序


# ═══════════════════════════════════════════════════════════════════════
# 社交模式库
# ═══════════════════════════════════════════════════════════════════════

_SOCIAL_PATTERNS = {
    # (正则模式, 类型, 回复模板列表)
    "greeting": {
        "patterns": [
            r'^(你好|您好|嗨|hi|hello|hey|早|早上好|下午好|晚上好)',
            r'^(好久不见|又见面了)',
        ],
        "responses": [
            "你好！我是龙珠，一个确定性知识引擎。有什么想了解的吗？",
            "嗨！今天想聊什么？我可以帮你查概念、做对比、解释因果关系。",
            "你好！我基于94117个汉字锚点和10万+概念图运行。问点什么吧？",
        ],
    },
    "farewell": {
        "patterns": [
            r'(再见|拜拜|bye|告辞|回头见|下次聊|晚安)',
        ],
        "responses": [
            "再见！有问题随时回来。",
            "拜拜，下次继续探索知识的宇宙。",
            "好的，我们改天再聊。",
        ],
    },
    "thanks": {
        "patterns": [
            r'(谢谢|感谢|多谢|thank|辛苦了)',
        ],
        "responses": [
            "不客气！能帮上忙就好。",
            "应该的。还有其他问题吗？",
            "不用谢。知识就是要分享的。",
        ],
    },
    "self_intro": {
        "patterns": [
            r'(你是谁|你叫什么|介绍一下你自己|你是做什么的|你能做什么)',
        ],
        "responses": [
            "我是龙珠，一个确定性知识引擎。我的核心是94117个汉字锚点构成的字场，"
            "以及10万+节点的概念图。我不用LLM——所有推理都基于确定性的能量计算和图遍历。"
            "\n\n我可以：解释概念、对比事物、追溯因果、列出关联、验证事实。试着问我一个问题吧！",
        ],
    },
    "capability": {
        "patterns": [
            r'(你会什么|你能干嘛|你有什么功能|你能)',
        ],
        "responses": [
            "我可以做这些事：\n"
            "• 解释概念（什么是量子力学）\n"
            "• 对比分析（儒家和道家有什么区别）\n"
            "• 因果追溯（为什么会有四季）\n"
            "• 列出关联（电子有哪些属性）\n"
            "• 验证真伪（光是粒子吗）\n"
            "• 诗词创作（写一首关于春天的五言诗）\n\n"
            "试试看？",
        ],
    },
    "agree": {
        "patterns": [
            r'^(对|没错|是的|确实|嗯|好|ok|yes|right)',
        ],
        "responses": [
            "好的。",
            "嗯，继续。",
            "是的。",
        ],
    },
    "confused": {
        "patterns": [
            r'(不懂|不明白|什么意思|没听懂|再说一遍)',
        ],
        "responses": [
            "抱歉，让我换个方式解释。",
            "我重新组织一下语言。",
            "好的，我换个角度说。",
        ],
    },
}

# 编译正则
_SOCIAL_COMPILED = {}
for intent, data in _SOCIAL_PATTERNS.items():
    _SOCIAL_COMPILED[intent] = {
        "patterns": [re.compile(p) for p in data["patterns"]],
        "responses": data["responses"],
    }


# ═══════════════════════════════════════════════════════════════════════
# 闲聊话题库 — 低信息密度但有社交功能的话题
# ═══════════════════════════════════════════════════════════════════════

_CHITCHAT_TOPICS = {
    "天气": {
        "patterns": [r'(天气|下雨|晴天|刮风|下雪|热|冷|温度)'],
        "templates": [
            "关于{concept}，我虽然不能感知天气，但可以告诉你：{concept}是气象学中的重要概念。"
            "想了解更多关于气候的知识吗？",
        ],
    },
    "时间": {
        "patterns": [r'(几点了|今天几号|星期几|什么日子)'],
        "responses": [
            "我没有实时时钟，但如果你问历史上的时间——比如'唐朝什么时候建立的'，我可以精确到年份。",
        ],
    },
    "情绪": {
        "patterns": [r'(开心|难过|无聊|累|烦|焦虑|兴奋|生气)'],
        "responses": [
            "听起来你现在的状态有点特别。要不要转移一下注意力，学点新知识？",
            "情绪波动是人之常情。要不我们聊点有趣的——比如'蝴蝶效应'是怎么回事？",
        ],
    },
}

_CHITCHAT_COMPILED = {}
for topic, data in _CHITCHAT_TOPICS.items():
    _CHITCHAT_COMPILED[topic] = {
        "patterns": [re.compile(p) for p in data["patterns"]],
        "responses": data.get("responses", []),
        "templates": data.get("templates", []),
    }


# ═══════════════════════════════════════════════════════════════════════
# 对话引擎主类
# ═══════════════════════════════════════════════════════════════════════

class ConversationEngine:
    """
    对话引擎 — 管理多轮对话状态、社交模式、闲聊路由。
    
    不替代 NLU/NLG，而是在其上增加一层对话管理。
    """

    def __init__(self, orchestrator=None):
        self.orch = orchestrator
        self.state = DialogueState()
        self._fallback_topics = [
            "量子力学", "人工智能", "进化论", "相对论", "儒家思想",
            "黑洞", "光合作用", "区块链", "唐诗", "元素周期表",
        ]
        self._used_fallbacks = set()

    def respond(self, user_input: str) -> Dict[str, Any]:
        """
        处理一轮对话。返回 {"output": str, "type": str, "concepts": [...]}
        
        路由优先级:
          1. 社交模式 → 直接回复
          2. 闲聊话题 → 轻量响应
          3. 知识查询 → 返回增强后的查询 (由Orchestrator继续处理)
          4. 兜底 → 话题推荐
        
        注意: 知识查询不在此处理——由调用方(Orchestrator)负责。
        """
        user_input = user_input.strip()
        result = {"output": "", "type": "unknown", "concepts": [],
                  "enhanced_query": user_input}

        # ── 路由1: 社交模式 ──
        social = self._match_social(user_input)
        if social:
            result["type"] = "social"
            result["output"] = social
            self.state.add_turn(user_input, social)
            return result

        # ── 路由2: 闲聊话题 ──
        chitchat = self._match_chitchat(user_input)
        if chitchat:
            result["type"] = "chitchat"
            result["output"] = chitchat
            self.state.add_turn(user_input, chitchat)
            return result

        # ── 路由3: 知识查询 — 补充上下文后返回给Orchestrator ──
        enhanced = self._add_context(user_input)
        result["type"] = "knowledge"
        result["enhanced_query"] = enhanced
        # 先用轻量解析提取概念（不进全栈）
        if self.orch and hasattr(self.orch, 'sem_parser'):
            try:
                frame = self.orch.sem_parser.parse(enhanced)
                result["concepts"] = frame.concepts
                self.state.add_turn(user_input, "[知识查询]", frame.concepts)
            except Exception:
                self.state.add_turn(user_input, "[知识查询]")
        return result

    def _match_social(self, text: str) -> Optional[str]:
        """匹配社交模式"""
        for intent, data in _SOCIAL_COMPILED.items():
            for pattern in data["patterns"]:
                if pattern.search(text):
                    return random.choice(data["responses"])
        return None

    def _match_chitchat(self, text: str) -> Optional[str]:
        """匹配闲聊话题"""
        for topic, data in _CHITCHAT_COMPILED.items():
            for pattern in data["patterns"]:
                m = pattern.search(text)
                if m:
                    if data.get("responses"):
                        return random.choice(data["responses"])
                    elif data.get("templates"):
                        concept = m.group(0) if m.groups() else text[:10]
                        return random.choice(data["templates"]).format(concept=concept)
        return None

    def _add_context(self, text: str) -> str:
        """为省略/代词查询补充上下文"""
        # 检测是否只有代词或简单追问
        pronouns = {'它', '他', '她', '这个', '那个', '这', '那', '这些', '那些'}
        followups = {'为什么', '怎么', '然后呢', '还有呢', '继续', '接着说', '详细点'}

        words = set(text)
        is_pronoun = any(p in text for p in pronouns) and len(text) <= 3
        is_followup = any(f in text for f in followups) and len(text) <= 5

        if is_pronoun or is_followup:
            recent = self.state.recent_concepts(3)
            if recent:
                if is_followup:
                    return f"{recent[-1]}{text}"
                else:
                    return f"{recent[-1]} {text}"

        return text

    def _generate_fallback(self) -> str:
        """生成兜底回复"""
        # 从概念图中随机推荐话题
        if self.orch and hasattr(self.orch, 'cg') and self.orch.cg:
            candidates = []
            for s in list(self.orch.cg.triples.keys())[:1000]:
                if len(s) >= 3 and s not in self._used_fallbacks:
                    deg = len(self.orch.cg.triples.get(s, []))
                    if deg >= 5:
                        candidates.append((s, deg))

            if candidates:
                candidates.sort(key=lambda x: -x[1])
                fresh = [c for c, _ in candidates if c not in self._used_fallbacks]
                if fresh:
                    topic = random.choice(fresh[:20])
                    self._used_fallbacks.add(topic)
                    return (
                        f"我不太确定你想问什么。要不聊聊'{topic}'？"
                        f"这个概念我了解得比较多。"
                    )

        # 终极兜底
        fresh = [t for t in self._fallback_topics if t not in self._used_fallbacks]
        if not fresh:
            self._used_fallbacks.clear()
            fresh = self._fallback_topics
        topic = random.choice(fresh)
        self._used_fallbacks.add(topic)
        return f"我没理解你的意思。要不要换个话题？比如'{topic}'？"

    def status(self) -> Dict[str, Any]:
        """返回对话状态摘要"""
        return {
            "turns": self.state.turn_count,
            "active_topic": self.state.active_topic,
            "recent_concepts": self.state.recent_concepts(5),
            "last_intent": self.state.last_intent,
        }
