#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠对话引擎 (Conversation) — 多轮状态 + 社交模式 + 闲聊路由 + L0感知升级
════════════════════════════════════════════════════════════════════════

在解义器+化能器基础上，增加多轮对话管理：
  1. 对话状态追踪 (上下文记忆)
  2. 社交模式识别 (问候/告别/感谢/确认)
  3. 话题跟随 (从上一轮继承主题)
  4. 闲聊路由 (社交→NLU查询→兜底)
  5. ★ L0 感知升级: 多轮记忆 + 用户偏好 (跨会话持久化)

════════════════════════════════════════════════════════════════════════
设计原则
════════════════════════════════════════════════════════════════════════

  确定性闲聊 ≠ LLM式闲聊。
  我们不生成新内容——我们从概念图中提取相关事实，用模板组织。
  闲聊的本质是"低信息密度但有社交功能的语言交换"，
  可以通过话题跟随 + 关联展开来实现。

L0 感知升级:
  用户数据持久化到 data/user_profiles.json，跨会话保留。
  每次对话后自动记录；启动时加载历史。
  回复时检查用户历史，提及过去查询，根据偏好调整风格。

════════════════════════════════════════════════════════════════════════
"""

import re
import time
import random
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════
# ★ L0: UserProfile — 跨会话用户记忆
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """用户画像：记录多轮对话历史与偏好"""
    user_id: str
    name: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    topic_interests: Dict[str, int] = field(default_factory=dict)   # {话题: 提及次数}
    query_history: List[Tuple[float, str, str]] = field(default_factory=list)  # [(时间, 查询, 意图), ...]
    preferences: Dict[str, str] = field(default_factory=dict)        # {'response_style': 'concise'}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "topic_interests": self.topic_interests,
            "query_history": self.query_history,
            "preferences": self.preferences,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=d.get("user_id", ""),
            name=d.get("name", ""),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            topic_interests=d.get("topic_interests", {}),
            query_history=[(t, q, i) for t, q, i in d.get("query_history", [])],
            preferences=d.get("preferences", {}),
        )


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
    
    ★ L0 感知升级: 多轮记忆 + 用户偏好
      - 用户数据持久化到 data/user_profiles.json
      - respond() 可接收 user_id 参数，检查历史提及过去查询
      - 根据 preferences 调整回复风格
    """

    # ★ 默认最大查询历史条数
    MAX_QUERY_HISTORY = 50
    # ★ 默认清理阈值（天）
    DEFAULT_MAX_AGE_DAYS = 30

    def __init__(self, orchestrator=None, profiles_path: str = None):
        self.orch = orchestrator
        self.state = DialogueState()
        self._fallback_topics = [
            "量子力学", "人工智能", "进化论", "相对论", "儒家思想",
            "黑洞", "光合作用", "区块链", "唐诗", "元素周期表",
        ]
        self._used_fallbacks = set()
        
        # ★ L0: 用户画像持久化
        self._profiles: Dict[str, UserProfile] = {}
        self._profiles_path = profiles_path or os.path.join(
            PROJECT_ROOT, "data", "user_profiles.json"
        )
        self._load_profiles()

    # ═══════════════════════════════════════════════════════════════════
    # ★ L0: 持久化辅助方法
    # ═══════════════════════════════════════════════════════════════════

    def _load_profiles(self):
        """从 JSON 文件加载用户画像"""
        if os.path.exists(self._profiles_path):
            try:
                with open(self._profiles_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for uid, data in raw.items():
                    self._profiles[uid] = UserProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError, IOError):
                self._profiles = {}

    def _save_profiles(self):
        """将用户画像保存到 JSON 文件"""
        os.makedirs(os.path.dirname(self._profiles_path), exist_ok=True)
        data = {uid: p.to_dict() for uid, p in self._profiles.items()}
        with open(self._profiles_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # ★ L0: 用户记忆 API
    # ═══════════════════════════════════════════════════════════════════

    def remember_user(self, user_id: str, query: str, intent: str = "",
                      concepts: List[str] = None):
        """
        记录用户的一轮对话。

        Args:
            user_id: 用户标识符
            query: 用户查询文本
            intent: 意图分类 (e.g. 'knowledge', 'social', 'chitchat')
            concepts: 涉及的概念列表 (用于更新话题兴趣)
        """
        now = time.time()
        
        # 获取或创建用户画像
        profile = self._profiles.get(user_id)
        if profile is None:
            profile = UserProfile(
                user_id=user_id,
                first_seen=now,
                last_seen=now,
            )
            self._profiles[user_id] = profile
        else:
            profile.last_seen = now
        
        # 记录查询历史（保留最近 MAX_QUERY_HISTORY 条）
        profile.query_history.append((now, query, intent))
        if len(profile.query_history) > self.MAX_QUERY_HISTORY:
            profile.query_history = profile.query_history[-self.MAX_QUERY_HISTORY:]
        
        # 更新话题兴趣
        for c in (concepts or []):
            profile.topic_interests[c] = profile.topic_interests.get(c, 0) + 1
        
        # 持久化
        self._save_profiles()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像，不存在返回 None"""
        return self._profiles.get(user_id)

    def suggest_topics(self, user_id: str) -> List[str]:
        """
        基于用户兴趣推荐话题。

        返回: 兴趣度排序的话题列表 [("话题", 提及次数), ...]
        """
        profile = self._profiles.get(user_id)
        if not profile or not profile.topic_interests:
            # 无历史——返回兜底话题
            return self._fallback_topics[:5]
        
        # 按提及次数降序排列
        sorted_topics = sorted(
            profile.topic_interests.items(),
            key=lambda x: -x[1]
        )
        return [t for t, _ in sorted_topics[:10]]

    def forget_old_history(self, max_age_days: int = None):
        """
        清理所有用户超过指定天数的旧查询记录。
        
        仅清理 query_history 中的旧条目；不影响 topic_interests 和 preferences。

        Args:
            max_age_days: 最大保留天数，默认 DEFAULT_MAX_AGE_DAYS
        """
        if max_age_days is None:
            max_age_days = self.DEFAULT_MAX_AGE_DAYS
        
        cutoff = time.time() - (max_age_days * 86400)
        cleaned = 0
        
        for profile in self._profiles.values():
            old_len = len(profile.query_history)
            profile.query_history = [
                (t, q, i) for t, q, i in profile.query_history
                if t >= cutoff
            ]
            cleaned += old_len - len(profile.query_history)
        
        # 移除没有任何记录的空白画像
        empty_users = [
            uid for uid, p in self._profiles.items()
            if not p.query_history and not p.topic_interests
        ]
        for uid in empty_users:
            del self._profiles[uid]
        
        if cleaned > 0:
            self._save_profiles()
        
        return cleaned

    # ═══════════════════════════════════════════════════════════════════
    # ★ L0: 增强的 respond() — 支持用户感知
    # ═══════════════════════════════════════════════════════════════════

    def respond(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """
        处理一轮对话。返回 {"output": str, "type": str, "concepts": [...]}
        
        路由优先级:
          1. 社交模式 → 直接回复
          2. 闲聊话题 → 轻量响应
          3. 知识查询 → 返回增强后的查询 (由Orchestrator继续处理)
          4. 兜底 → 话题推荐
        
        ★ L0 增强 (user_id 非 None 时):
          - 回复前检查用户历史，若问过类似问题则提及
          - 根据 preferences 调整回复风格
        
        注意: 知识查询不在此处理——由调用方(Orchestrator)负责。
        """
        user_input = user_input.strip()
        result = {"output": "", "type": "unknown", "concepts": [],
                  "enhanced_query": user_input}

        # ── 路由1: 社交模式 ──
        social = self._match_social(user_input)
        if social:
            # ★ L0: 如果在问候且用户有历史，可个性化
            if user_id:
                profile = self._profiles.get(user_id)
                if profile and profile.query_history and "你好" in user_input:
                    last_query = profile.query_history[-1][1]
                    social += f"\n\n对了，上次我们聊到了「{last_query[:20]}」，要继续吗？"
                # 记录本轮
                self.remember_user(user_id, user_input, "social")
            
            result["type"] = "social"
            result["output"] = social
            self.state.add_turn(user_input, social)
            return result

        # ── 路由2: 闲聊话题 ──
        chitchat = self._match_chitchat(user_input)
        if chitchat:
            if user_id:
                self.remember_user(user_id, user_input, "chitchat")
            
            result["type"] = "chitchat"
            result["output"] = chitchat
            self.state.add_turn(user_input, chitchat)
            return result

        # ★ L0: 检查用户历史，发现相似查询
        history_hint = ""
        if user_id:
            profile = self._profiles.get(user_id)
            if profile:
                similar = self._find_similar_query(user_input, profile)
                if similar:
                    history_hint = similar
                # ★ 根据 preferences 调整风格提示
                style_hint = profile.preferences.get("response_style", "")
            else:
                style_hint = ""
        else:
            style_hint = ""

        # ── 路由3: 知识查询 — 补充上下文后返回给Orchestrator ──
        enhanced = self._add_context(user_input)
        
        # ★ L0: 附加历史感知提示
        if history_hint:
            enhanced = f"{enhanced}\n[历史上下文: {history_hint}]"
        
        result["type"] = "knowledge"
        result["enhanced_query"] = enhanced
        
        # 先用轻量解析提取概念（不进全栈）
        concepts = []
        if self.orch and hasattr(self.orch, 'sem_parser'):
            try:
                frame = self.orch.sem_parser.parse(enhanced)
                concepts = frame.concepts
                result["concepts"] = concepts
            except Exception:
                pass
        
        self.state.add_turn(user_input, "[知识查询]", concepts)
        
        # ★ L0: 记录本轮对话到用户画像
        if user_id:
            self.remember_user(user_id, user_input, "knowledge", concepts)
        
        # ★ L0: 如果是偏好 concise，追加风格标记
        if style_hint == "concise":
            result["_prefer_concise"] = True
        
        return result

    def _find_similar_query(self, query: str, profile: UserProfile) -> Optional[str]:
        """
        在用户历史中寻找与当前查询相似的过往查询。
        
        策略: 提取 query 中的关键词（2字及以上的中文词），与历史匹配。
        """
        # 提取中文关键词（2字及以上）
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        if not keywords:
            return None
        
        best = None
        best_score = 0
        
        for ts, past_query, intent in reversed(profile.query_history[-30:]):
            if intent == "social":
                continue  # 跳过社交对话
            past_kw = set(re.findall(r'[\u4e00-\u9fff]{2,}', past_query))
            overlap = sum(1 for kw in keywords if kw in past_kw)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best = past_query[:30]
        
        if best:
            return f"您上次问过类似问题: 「{best}」"
        return None

    # ═══════════════════════════════════════════════════════════════════
    # 原有方法（不变）
    # ═══════════════════════════════════════════════════════════════════

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
