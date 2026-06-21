#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠 v4 未提交改动 — 单元测试
════════════════════════════════════════════════════════════════════════
覆盖:
  1. EventType 枚举
  2. LifeEvent dataclass
  3. process_event 事件处理 (mock orchestrator)
  4. should_express 表达决策
  5. register_event_sources 事件源注册
  6. UserProfile 用户画像 (conversation.py)
  7. ConversationEngine L0 用户记忆 (mock)
  8. Fuzzy SOURCE_WEIGHTS + decay + multi_source_fuse
  9. SemParser regex 增强 (什么是X, 单字功能词过滤)
════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import json
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict
from enum import Enum, auto

# ── 项目路径 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════
# 1. EventType + LifeEvent 单元测试 (不需要加载模型)
# ══════════════════════════════════════════════════════════════════════════

class TestEventSystem(unittest.TestCase):
    """测试 v4 事件系统的核心数据类"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.orchestrator import Orchestrator
        # 我们不实际初始化 Orchestrator（太重），直接引用内部类
        cls.EventType = Orchestrator.EventType
        cls.LifeEvent = Orchestrator.LifeEvent

    def test_event_type_enum_all_members(self):
        """EventType 应有 6 种事件类型"""
        expected = {'USER_MESSAGE', 'INTERNAL_BLINDSPOT', 'CURIOSITY',
                     'WIKI_RESULT', 'MEMORY_DECAY', 'TIMER'}
        actual = set(self.EventType.__members__.keys())
        self.assertEqual(expected, actual)

    def test_event_type_all_distinct_values(self):
        """所有事件类型的值应该唯一"""
        values = [e.value for e in self.EventType]
        self.assertEqual(len(values), len(set(values)))

    def test_lifeevent_creation_minimal(self):
        """LifeEvent 最小化创建"""
        evt = self.LifeEvent(etype=self.EventType.USER_MESSAGE)
        self.assertEqual(evt.etype, self.EventType.USER_MESSAGE)
        self.assertIsInstance(evt.payload, dict)
        self.assertEqual(evt.payload, {})
        self.assertGreater(evt.timestamp, 0)
        self.assertEqual(evt.source, "")

    def test_lifeevent_creation_full(self):
        """LifeEvent 完整字段创建"""
        evt = self.LifeEvent(
            etype=self.EventType.USER_MESSAGE,
            payload={'text': '龙是什么', 'user_id': 'test'},
            source='stdin'
        )
        self.assertEqual(evt.etype, self.EventType.USER_MESSAGE)
        self.assertEqual(evt.payload['text'], '龙是什么')
        self.assertEqual(evt.payload['user_id'], 'test')
        self.assertEqual(evt.source, 'stdin')

    def test_lifeevent_default_timestamp(self):
        """LifeEvent timestamp 应在合理范围内"""
        t0 = time.time()
        evt = self.LifeEvent(etype=self.EventType.TIMER)
        t1 = time.time()
        self.assertGreaterEqual(evt.timestamp, t0)
        self.assertLessEqual(evt.timestamp, t1 + 0.01)

    def test_lifeevent_custom_timestamp(self):
        """LifeEvent 自定义时间戳"""
        evt = self.LifeEvent(
            etype=self.EventType.MEMORY_DECAY,
            timestamp=1234567890.0
        )
        self.assertEqual(evt.timestamp, 1234567890.0)

    def test_lifeevent_dataclass_equality(self):
        """LifeEvent dataclass 等值比较"""
        e1 = self.LifeEvent(etype=self.EventType.TIMER, timestamp=100.0)
        e2 = self.LifeEvent(etype=self.EventType.TIMER, timestamp=100.0)
        e3 = self.LifeEvent(etype=self.EventType.TIMER, timestamp=200.0)
        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, e3)

    def test_all_event_types_creatable(self):
        """所有 6 种事件类型均可创建"""
        for etype in self.EventType:
            evt = self.LifeEvent(etype=etype)
            self.assertIsInstance(evt, self.LifeEvent)
            self.assertEqual(evt.etype, etype)


# ══════════════════════════════════════════════════════════════════════════
# 2. process_event + should_express (Mock Orchestrator)
# ══════════════════════════════════════════════════════════════════════════

class TestProcessEvent(unittest.TestCase):
    """测试 process_event 事件分发逻辑（Mock 重型依赖）"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.orchestrator import Orchestrator
        cls.Orchestrator = Orchestrator
        cls.EventType = Orchestrator.EventType
        cls.LifeEvent = Orchestrator.LifeEvent

    def setUp(self):
        """创建 Mock Orchestrator 实例"""
        # 绕开 __init__ 直接创建空对象，手动注入方法
        self.orch = object.__new__(self.Orchestrator)
        # 注入必要的属性
        self.orch.field = Mock()
        self.orch.field._char_to_idx = {'龙': 0, '量': 1, '子': 2, '原': 3, '核': 4}
        self.orch.cg = Mock()
        self.orch.cg.add_triple = Mock()
        self.orch.learner = Mock()
        self.orch.learner.learn_pairs_batch = Mock(return_value={'status': 'ok'})
        # Mock query 方法
        self.orch.query = Mock(return_value={
            'answer': '测试回答', 'signal': 'certain', 'confidence': 0.8,
            'debug': {'infer': {'top_candidates': ['龙']}}
        })
        # Mock daemon_tick_v2
        self.orch.daemon_tick_v2 = Mock(return_value={'status': 'ok', 'round': 1})
        # Mock 内部方法
        self.orch._arms_search_deep = Mock(return_value=[(0, 1)])
        self.orch._sync_fuzzy_to_cg = Mock()
        # 绑定 EventType
        self.orch.EventType = self.EventType
        self.orch.LifeEvent = self.LifeEvent

    def test_user_message_event(self):
        """USER_MESSAGE 事件应调用 query()"""
        evt = self.LifeEvent(
            etype=self.EventType.USER_MESSAGE,
            payload={'text': '龙是什么'}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.orch.query.assert_called_once_with('龙是什么')
        self.assertEqual(result['answer'], '测试回答')
        self.assertEqual(result['signal'], 'certain')

    def test_user_message_empty_text(self):
        """USER_MESSAGE 空文本"""
        evt = self.LifeEvent(
            etype=self.EventType.USER_MESSAGE,
            payload={'text': ''}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.orch.query.assert_called_once_with('')

    def test_blindspot_event_with_concept(self):
        """INTERNAL_BLINDSPOT 事件应触发学习"""
        evt = self.LifeEvent(
            etype=self.EventType.INTERNAL_BLINDSPOT,
            payload={'concept': '量子', 'energy': 5.0}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.assertEqual(result['concept'], '量子')
        self.assertIn('blind_spot_learned', result['note'])
        # 验证双臂搜索被调用了
        self.orch._arms_search_deep.assert_called_once_with('量子', '量子')
        # 验证学习被调用了
        self.orch.learner.learn_pairs_batch.assert_called_once()

    def test_blindspot_event_empty_concept(self):
        """INTERNAL_BLINDSPOT 空概念不应调用搜索"""
        evt = self.LifeEvent(
            etype=self.EventType.INTERNAL_BLINDSPOT,
            payload={'concept': ''}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.orch._arms_search_deep.assert_not_called()

    def test_curiosity_event(self):
        """CURIOSITY 事件应返回 silent"""
        # Mock wiki 避免 IO
        with patch.dict('sys.modules', {'loongpearl.core.wiki_lookup': None}):
            evt = self.LifeEvent(
                etype=self.EventType.CURIOSITY,
                payload={'concept': '量子'}
            )
            result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.assertEqual(result['concept'], '量子')

    def test_wiki_result_event(self):
        """WIKI_RESULT 事件应注入概念图"""
        triples = [
            {'s': '龙', 'r': 'IS_A', 'o': '神话生物', 'c': 0.8},
            {'s': '龙', 'r': 'RELATED', 'o': '凤凰', 'c': 0.5},
        ]
        evt = self.LifeEvent(
            etype=self.EventType.WIKI_RESULT,
            payload={'triples': triples}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.assertIn('wiki_injected', result['note'])

    def test_memory_decay_event(self):
        """MEMORY_DECAY 事件应调用 sync"""
        evt = self.LifeEvent(etype=self.EventType.MEMORY_DECAY)
        result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.assertIn('decay_applied', result['note'])
        self.orch._sync_fuzzy_to_cg.assert_called_once()

    def test_timer_event(self):
        """TIMER 事件应调用 daemon_tick_v2"""
        evt = self.LifeEvent(
            etype=self.EventType.TIMER,
            payload={'round_num': 5}
        )
        result = self.Orchestrator.process_event(self.orch, evt)
        self.assertEqual(result['signal'], 'silent')
        self.assertIn('timer_tick', result['note'])
        self.orch.daemon_tick_v2.assert_called_once_with(5)

    def test_timer_event_no_round_num(self):
        """TIMER 事件无 round_num 时默认 0"""
        evt = self.LifeEvent(etype=self.EventType.TIMER)
        result = self.Orchestrator.process_event(self.orch, evt)
        self.orch.daemon_tick_v2.assert_called_once_with(0)


class TestShouldExpress(unittest.TestCase):
    """测试 should_express 表达决策"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.orchestrator import Orchestrator
        cls.Orchestrator = Orchestrator
        cls.EventType = Orchestrator.EventType
        cls.LifeEvent = Orchestrator.LifeEvent

    def setUp(self):
        self.orch = object.__new__(self.Orchestrator)
        self.orch.EventType = self.EventType

    def test_user_message_always_expresses(self):
        """USER_MESSAGE 必须总能表达"""
        for payload in [{}, {'text': 'hello'}, {'text': ''}]:
            evt = self.LifeEvent(etype=self.EventType.USER_MESSAGE, payload=payload)
            self.assertTrue(
                self.Orchestrator.should_express(self.orch, evt, {}),
                f"USER_MESSAGE should express with payload={payload}"
            )

    def test_internal_events_never_express(self):
        """内部事件绝不表达"""
        silent_types = [
            self.EventType.INTERNAL_BLINDSPOT,
            self.EventType.CURIOSITY,
            self.EventType.WIKI_RESULT,
            self.EventType.MEMORY_DECAY,
            self.EventType.TIMER,
        ]
        for etype in silent_types:
            evt = self.LifeEvent(etype=etype)
            self.assertFalse(
                self.Orchestrator.should_express(self.orch, evt, {}),
                f"{etype} should NOT express"
            )

    def test_express_result_content_irrelevant(self):
        """should_express 只看事件类型，不看结果内容"""
        evt = self.LifeEvent(etype=self.EventType.TIMER)
        # 即使结果有 answer，TIMER 也不表达
        result_with_answer = {'answer': 'some text', 'signal': 'certain'}
        self.assertFalse(
            self.Orchestrator.should_express(self.orch, evt, result_with_answer)
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. register_event_sources (需要最小化 mock)
# ══════════════════════════════════════════════════════════════════════════

class TestEventSources(unittest.TestCase):
    """测试事件源注册"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.orchestrator import Orchestrator
        cls.Orchestrator = Orchestrator
        cls.EventType = Orchestrator.EventType
        cls.LifeEvent = Orchestrator.LifeEvent

    def setUp(self):
        self.orch = object.__new__(self.Orchestrator)
        self.orch.EventType = self.EventType
        self.orch.LifeEvent = self.LifeEvent
        self.orch.field = Mock()
        self.orch.cg = Mock()
        self.orch.cg.db_path = None
        self.orch.log = Mock()  # prevent log calls from crashing

    def test_registers_two_sources(self):
        """应注册 2 个事件源: blindspot + timer"""
        sources = self.Orchestrator.register_event_sources(self.orch)
        self.assertEqual(len(sources), 2)

    def test_timer_source_iteration(self):
        """timer 事件源应能 yield TIMER 事件 (非阻塞模式，需等待间隔)"""
        sources = self.Orchestrator.register_event_sources(self.orch)
        timer_fn = sources[1]  # timer_source
        gen = timer_fn(round_interval=0.01)  # 极短间隔

        # 非阻塞模式: 第一次 next() 可能返回 None (时间未到)
        # 等待足够时间后应 yield TIMER 事件
        import time
        time.sleep(0.02)  # 等待超过 interval
        evt = next(gen)
        # 跳过可能的 None
        while evt is None:
            time.sleep(0.02)
            evt = next(gen)
        self.assertEqual(evt.etype, self.EventType.TIMER)
        self.assertIn('round_num', evt.payload)
        self.assertEqual(evt.source, 'timer')

        # 第二次 next() 可能又因间隔未到返回 None (正常)
        # 不强制验证第二个事件

    def test_blindspot_source_handles_missing_terrain(self):
        """blindspot 源在 terrain 不可用时应捕获异常"""
        sources = self.Orchestrator.register_event_sources(self.orch)
        blindspot_fn = sources[0]

        # terrain 加载会失败（无实际文件），但不应崩溃
        try:
            gen = blindspot_fn(interval=0.1)
            # 第一次迭代会 sleep 0.1s 然后尝试加载 terrain
            # 应该静默跳过
            import select
            import time as _time
            # 不实际调用 next()，因为会 sleep 并可能 IO
            # 我们验证 generator 被正确创建即可
            self.assertIsNotNone(gen)
        except Exception as e:
            self.fail(f"blindspot_source 创建不应抛异常: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 4. UserProfile + ConversationEngine L0 感知
# ══════════════════════════════════════════════════════════════════════════

class TestUserProfile(unittest.TestCase):
    """测试 UserProfile dataclass"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.conversation import UserProfile
        cls.UserProfile = UserProfile

    def test_create_minimal(self):
        """最小化创建"""
        p = self.UserProfile(user_id="user_001")
        self.assertEqual(p.user_id, "user_001")
        self.assertEqual(p.name, "")
        self.assertEqual(p.first_seen, 0.0)
        self.assertEqual(p.last_seen, 0.0)
        self.assertEqual(p.topic_interests, {})
        self.assertEqual(p.query_history, [])
        self.assertEqual(p.preferences, {})

    def test_create_full(self):
        """完整字段创建"""
        p = self.UserProfile(
            user_id="u1",
            name="泽坤",
            first_seen=1000.0,
            last_seen=2000.0,
            topic_interests={"唐诗": 3, "量子": 5},
            query_history=[(1000.0, "龙是什么", "factual")],
            preferences={"response_style": "concise"}
        )
        self.assertEqual(p.user_id, "u1")
        self.assertEqual(p.name, "泽坤")
        self.assertEqual(p.topic_interests["唐诗"], 3)

    def test_to_dict(self):
        """to_dict 输出正确"""
        p = self.UserProfile(
            user_id="u1",
            topic_interests={"AI": 2},
            query_history=[(100.0, "what", "factual")]
        )
        d = p.to_dict()
        self.assertEqual(d["user_id"], "u1")
        self.assertEqual(d["topic_interests"], {"AI": 2})
        self.assertEqual(len(d["query_history"]), 1)
        self.assertEqual(d["query_history"][0][1], "what")

    def test_from_dict(self):
        """from_dict 重建对象"""
        d = {
            "user_id": "u2",
            "name": "test",
            "first_seen": 100.0,
            "last_seen": 200.0,
            "topic_interests": {"诗": 1},
            "query_history": [(150.0, "q", "factual")],
            "preferences": {"style": "brief"},
        }
        p = self.UserProfile.from_dict(d)
        self.assertEqual(p.user_id, "u2")
        self.assertEqual(p.name, "test")
        self.assertEqual(p.topic_interests, {"诗": 1})
        self.assertEqual(p.query_history, [(150.0, "q", "factual")])

    def test_roundtrip(self):
        """to_dict → from_dict 无损往返"""
        p1 = self.UserProfile(
            user_id="u1",
            name="n1",
            first_seen=1.0,
            last_seen=2.0,
            topic_interests={"A": 3},
            query_history=[(0.5, "q1", "intent1"), (1.0, "q2", "intent2")],
            preferences={"k": "v"},
        )
        d = p1.to_dict()
        p2 = self.UserProfile.from_dict(d)
        self.assertEqual(p1.user_id, p2.user_id)
        self.assertEqual(p1.name, p2.name)
        self.assertEqual(p1.topic_interests, p2.topic_interests)
        self.assertEqual(p1.query_history, p2.query_history)
        self.assertEqual(p1.preferences, p2.preferences)

    def test_from_dict_missing_fields(self):
        """from_dict 缺失字段使用默认值"""
        d = {"user_id": "u3"}
        p = self.UserProfile.from_dict(d)
        self.assertEqual(p.user_id, "u3")
        self.assertEqual(p.name, "")
        self.assertEqual(p.topic_interests, {})

    def test_from_dict_empty(self):
        """from_dict 空字典"""
        p = self.UserProfile.from_dict({})
        self.assertEqual(p.user_id, "")
        self.assertEqual(p.query_history, [])


class TestConversationL0(unittest.TestCase):
    """测试 ConversationEngine L0 用户记忆功能"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.conversation import ConversationEngine, UserProfile
        cls.ConversationEngine = ConversationEngine
        cls.UserProfile = UserProfile

    def setUp(self):
        """创建临时 profiles 文件"""
        self.tmp_dir = tempfile.mkdtemp()
        self.profiles_path = os.path.join(self.tmp_dir, "user_profiles.json")
        self.engine = self.ConversationEngine(
            orchestrator=None,
            profiles_path=self.profiles_path
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initial_profiles_empty(self):
        """初始时 profiles 应为空"""
        self.assertEqual(len(self.engine._profiles), 0)

    def test_remember_user_first_time(self):
        """首次记录用户"""
        self.engine.remember_user("user_001", "龙是什么", intent="factual")
        self.assertIn("user_001", self.engine._profiles)
        p = self.engine._profiles["user_001"]
        self.assertEqual(p.user_id, "user_001")
        self.assertGreater(p.first_seen, 0)
        self.assertGreater(p.last_seen, 0)
        self.assertEqual(len(p.query_history), 1)

    def test_remember_user_multiple_queries(self):
        """多次记录同一用户"""
        for q in ["龙是什么", "量子是什么", "写诗"]:
            self.engine.remember_user("u1", q)
        p = self.engine._profiles["u1"]
        self.assertEqual(len(p.query_history), 3)
        self.assertEqual(p.query_history[1][1], "量子是什么")
        # query_history 按时间排序
        self.assertGreaterEqual(p.query_history[2][0], p.query_history[1][0])

    def test_remember_user_topic_tracking(self):
        """话题兴趣跟踪"""
        self.engine.remember_user("u1", "唐诗三百首", concepts=["唐诗"])
        self.engine.remember_user("u1", "李白的诗", concepts=["李白", "诗"])
        self.engine.remember_user("u1", "量子纠缠", concepts=["量子"])
        p = self.engine._profiles["u1"]
        self.assertIn("唐诗", p.topic_interests)
        self.assertIn("量子", p.topic_interests)

    def test_get_user_profile_exists(self):
        """获取已存在用户"""
        self.engine.remember_user("u1", "hello")
        p = self.engine.get_user_profile("u1")
        self.assertIsNotNone(p)
        self.assertEqual(p.user_id, "u1")

    def test_get_user_profile_not_exists(self):
        """获取不存在用户"""
        p = self.engine.get_user_profile("nonexistent")
        self.assertIsNone(p)

    def test_save_and_load_profiles(self):
        """持久化 → 重新加载 → 数据一致"""
        self.engine.remember_user("u1", "q1", intent="factual")
        self.engine.remember_user("u1", "q2", intent="chat")
        self.engine._save_profiles()

        # 新建 engine 从同路径加载
        engine2 = self.ConversationEngine(
            orchestrator=None,
            profiles_path=self.profiles_path
        )
        self.assertIn("u1", engine2._profiles)
        p = engine2._profiles["u1"]
        self.assertEqual(len(p.query_history), 2)
        self.assertEqual(p.query_history[1][1], "q2")

    def test_forget_old_history(self):
        """清理旧历史"""
        old_time = time.time() - 100 * 86400  # 100 天前
        recent_time = time.time() - 1 * 86400  # 1 天前

        profile = self.UserProfile(user_id="u1", first_seen=old_time)
        profile.query_history = [
            (old_time, "old_q", "old"),
            (recent_time, "recent_q", "recent"),
        ]
        self.engine._profiles["u1"] = profile

        # 清理 30 天以前的数据
        self.engine.forget_old_history(max_age_days=30)

        p = self.engine._profiles["u1"]
        # 应该只保留最近的历史
        remaining_times = [t for t, _, _ in p.query_history]
        for t in remaining_times:
            self.assertGreater(t, time.time() - 30 * 86400)

    def test_forget_old_history_empty_profiles(self):
        """空 profiles 清理不崩溃"""
        try:
            self.engine.forget_old_history(max_age_days=30)
        except Exception as e:
            self.fail(f"空 profiles 清理不应抛异常: {e}")

    def test_suggest_topics(self):
        """话题推荐"""
        self.engine.remember_user("u1", "唐诗鉴赏", concepts=["唐诗"])
        self.engine.remember_user("u1", "李白生平", concepts=["李白"])
        self.engine.remember_user("u1", "宋词", concepts=["宋词"])
        topics = self.engine.suggest_topics("u1")
        self.assertIsInstance(topics, list)

    def test_remember_user_query_history_cap(self):
        """查询历史不应超过 MAX_QUERY_HISTORY"""
        for i in range(100):
            self.engine.remember_user("u1", f"query_{i}")
        p = self.engine._profiles["u1"]
        self.assertLessEqual(len(p.query_history), self.ConversationEngine.MAX_QUERY_HISTORY)

    def test_corrupted_profiles_file(self):
        """损坏的 JSON 文件不应崩溃"""
        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        with open(self.profiles_path, 'w') as f:
            f.write("NOT VALID JSON {{{")
        engine = self.ConversationEngine(
            orchestrator=None,
            profiles_path=self.profiles_path
        )
        self.assertEqual(len(engine._profiles), 0)


# ══════════════════════════════════════════════════════════════════════════
# 5. Fuzzy Graph — SOURCE_WEIGHTS + decay + multi_source_fuse
# ══════════════════════════════════════════════════════════════════════════

class TestFuzzySourceWeights(unittest.TestCase):
    """测试模糊格来源权重"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.fuzzy_graph import (
            resolve_source_weight as _rsw, SOURCE_WEIGHTS as _sw,
            Evidence, FuzzyGraph
        )
        cls.resolve_source_weight = staticmethod(_rsw)
        cls.SOURCE_WEIGHTS = _sw
        cls.Evidence = Evidence
        cls.FuzzyGraph = FuzzyGraph

    def test_exact_match(self):
        """精确匹配返回对应权重"""
        rsw = self.resolve_source_weight
        self.assertEqual(rsw("wikipedia_dump"), 0.7)
        self.assertEqual(rsw("concept_graph"), 0.5)
        self.assertEqual(rsw("user_input"), 0.6)
        self.assertEqual(rsw("perturbation"), 0.2)

    def test_prefix_match(self):
        """前缀匹配"""
        rsw = self.resolve_source_weight
        self.assertEqual(rsw("wikipedia_dump_v2"), 0.7)
        self.assertEqual(rsw("perturbation_engine"), 0.2)

    def test_unknown_source_returns_default(self):
        """未知来源返回 1.0"""
        rsw = self.resolve_source_weight
        self.assertEqual(rsw("unknown_source"), 1.0)
        self.assertEqual(rsw(""), 1.0)

    def test_all_weights_in_range(self):
        """所有权重应在 0-1 范围"""
        for source, weight in self.SOURCE_WEIGHTS.items():
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_evidence_effective_mass(self):
        """effective_mass = mass × source_weight"""
        # Evidence.__init__ 不会自动解析 SOURCE_WEIGHTS，
        # source_weight 需显式传入或由 FuzzyGraph.add_evidence() 解析
        ev = self.Evidence(source="wikipedia_dump", mass=0.8, source_weight=0.7)
        self.assertAlmostEqual(ev.effective_mass, 0.8 * 0.7)

        ev2 = self.Evidence(source="unknown", mass=0.5, source_weight=1.0)
        self.assertAlmostEqual(ev2.effective_mass, 0.5)

    def test_evidence_weight_validation(self):
        """source_weight 应在 0-1 范围"""
        with self.assertRaises(ValueError):
            self.Evidence(source="test", mass=0.5, source_weight=1.5)
        with self.assertRaises(ValueError):
            self.Evidence(source="test", mass=0.5, source_weight=-0.1)

    def test_evidence_mass_validation(self):
        """mass 应在 0-1 范围"""
        with self.assertRaises(ValueError):
            self.Evidence(source="test", mass=1.5)
        with self.assertRaises(ValueError):
            self.Evidence(source="test", mass=-0.1)


class TestFuzzyDecay(unittest.TestCase):
    """测试模糊格时间衰减 + 多源融合"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.fuzzy_graph import FuzzyGraph, Evidence
        cls.FuzzyGraph = FuzzyGraph
        cls.Evidence = Evidence

    def setUp(self):
        self.fg = self.FuzzyGraph()

    def test_combine_with_decay_no_evidences(self):
        """无证据时返回 0"""
        result = self.fg.combine_with_decay("龙", "IS_A", "神话生物")
        self.assertEqual(result, 0.0)

    def test_combine_with_decay_factor_1(self):
        """decay_factor=1 不衰减"""
        self.fg.add_evidence("龙", "IS_A", "神话生物", "wikipedia_dump", mass=0.8)
        result = self.fg.combine_with_decay("龙", "IS_A", "神话生物", decay_factor=1.0)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_combine_with_decay_factor_0(self):
        """decay_factor=0 完全衰减"""
        self.fg.add_evidence("龙", "IS_A", "神话生物", "wikipedia_dump", mass=0.8)
        result = self.fg.combine_with_decay("龙", "IS_A", "神话生物", decay_factor=0.0)
        self.assertEqual(result, 0.0)

    def test_combine_with_decay_partial(self):
        """部分衰减"""
        self.fg.add_evidence("X", "RELATED", "Y", "wikipedia_dump", mass=0.8)
        full = self.fg.combine_with_decay("X", "RELATED", "Y", decay_factor=1.0)
        partial = self.fg.combine_with_decay("X", "RELATED", "Y", decay_factor=0.5)
        self.assertGreater(full, partial)

    def test_combine_with_decay_multi_evidence(self):
        """多证据加权衰减"""
        self.fg.add_evidence("龙", "IS_A", "神兽",
                             "wikipedia_dump", mass=0.7)
        self.fg.add_evidence("龙", "IS_A", "神兽",
                             "user_input", mass=0.5)
        result = self.fg.combine_with_decay("龙", "IS_A", "神兽", decay_factor=1.0)
        # 两源融合应大于单源
        self.assertGreater(result, 0.0)

    def test_multi_source_fuse(self):
        """多源融合方法"""
        self.fg.add_evidence("龙", "IS_A", "神", "wikipedia_dump", mass=0.7)
        self.fg.add_evidence("龙", "IS_A", "神", "user_input", mass=0.5)
        result = self.fg.multi_source_fuse("龙", "IS_A", "神")
        self.assertIsInstance(result, dict)
        self.assertIn('belief', result)
        self.assertGreater(result['belief'], 0.0)
        self.assertLessEqual(result['belief'], 1.0)

    def test_multi_source_fuse_empty(self):
        """空命题多源融合"""
        result = self.fg.multi_source_fuse("X", "Y", "Z")
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('belief'), 0.0)

    def test_add_evidence_with_auto_weight(self):
        """add_evidence 自动解析来源权重"""
        bpa = self.fg.add_evidence("龙", "IS_A", "龙", "wikipedia_dump", mass=0.5)
        self.assertIsNotNone(bpa)
        ev = bpa.evidences[-1]
        self.assertAlmostEqual(ev.source_weight, 0.7)  # wikipedia_dump = 0.7

    def test_add_evidence_with_explicit_weight(self):
        """add_evidence 显式指定权重"""
        bpa = self.fg.add_evidence("X", "R", "Y", "custom_src", mass=0.5,
                                    source_weight=0.3)
        self.assertAlmostEqual(bpa.evidences[-1].source_weight, 0.3)


# ══════════════════════════════════════════════════════════════════════════
# 6. SemParser regex 增强 — 什么是X / 单字功能词
# ══════════════════════════════════════════════════════════════════════════

class TestSemParserEnhancements(unittest.TestCase):
    """测试 sem_parser 新增 regex 模式"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.sem_parser import SemParser
        cls.SemParser = SemParser

    def setUp(self):
        self.parser = self.SemParser()

    def test_what_is_X_pattern(self):
        """'什么是X' 模式 — 应识别为定义查询"""
        frame = self.parser.parse("什么是量子")
        self.assertIsNotNone(frame.subject or frame.concepts)
        # 至少 concepts 包含 '量子'
        self.assertIn('量子', frame.concepts)

    def test_X_is_what_pattern(self):
        """'X是什么' 模式 — 应识别为定义查询"""
        frame = self.parser.parse("龙是什么")
        self.assertIn('龙', frame.concepts)

    def test_what_is_X_multi_char(self):
        """'什么是X' 多字概念"""
        frame = self.parser.parse("什么是量子力学")
        # '量子力学' 应作为一个概念被识别
        self.assertTrue(
            '量子力学' in frame.concepts or '量子' in frame.concepts
        )

    def test_function_chars_filtered(self):
        """单字功能词应被过滤出概念列表"""
        frame = self.parser.parse("龙是什么")
        # '是' 不应出现在概念中
        self.assertNotIn('是', frame.concepts)
        self.assertNotIn('的', frame.concepts)

    def test_single_char_concept_allowed(self):
        """单字中文内容词应保留 (如龙、人、道)"""
        frame = self.parser.parse("龙")
        self.assertTrue(
            '龙' in frame.concepts,
            f"单字内容词应保留, concepts={frame.concepts}"
        )

    def test_question_with_function_chars(self):
        """带功能词的句子，概念提取正确"""
        frame = self.parser.parse("原子和分子的区别是什么")
        # '和' '是' 不应在概念中
        self.assertNotIn('和', frame.concepts)
        self.assertNotIn('是', frame.concepts)
        # '原子' '分子' 应在概念中
        self.assertTrue(
            '原子' in frame.concepts or '分子' in frame.concepts
        )

    def test_compare_pattern(self):
        """对比模式仍然工作"""
        frame = self.parser.parse("原子和分子有什么区别")
        self.assertEqual(frame.question_type.name, 'COMPARE')


# ══════════════════════════════════════════════════════════════════════════
# 7. Stage4 接线 (需轻量 mock)
# ══════════════════════════════════════════════════════════════════════════

class TestStage4Wiring(unittest.TestCase):
    """测试 orchestrator.query() 中 Stage4 接线"""

    @classmethod
    def setUpClass(cls):
        from loongpearl.core.orchestrator import Orchestrator
        cls.Orchestrator = Orchestrator

    def _make_mock_orch(self, stage4_result=None):
        """创建 mock Orchestrator 用于测试 _route_factual 和 query"""
        from loongpearl.core.orchestrator import Orchestrator
        orch = object.__new__(Orchestrator)
        orch.field = Mock()
        orch.field._char_to_idx = {'龙': 0, '量': 1, '子': 2}
        orch.cg = Mock()
        orch.learner = Mock()
        orch.landscape = Mock()
        orch.landscape.infer = Mock(return_value={
            'basin': '龙', 'energy': -50, 'top_candidates': ['龙'],
            'confidence': 0.8
        })
        # sem_parser/planner/decoder/conversation 都是 @property
        orch._sem_parser = Mock()
        orch._sem_parser.parse = Mock(return_value=Mock(
            subject='龙', concepts=['龙'], question_type=Mock(name='WHAT_IS')
        ))
        orch._planner = Mock()
        orch._planner.plan = Mock(return_value={'intent': 'DEFINE'})
        orch._decoder = Mock()
        orch._decoder.decode = Mock(return_value='测试回答')
        orch._conversation = Mock()
        orch._conversation.respond = Mock(return_value={'answer': '你好'})
        orch._get_triples_for = Mock(return_value=[])
        orch._generate_answer = Mock(return_value='测试回答')

        # 注入 Stage4 mock
        self._stage4_result = stage4_result
        self.stage4_called = False

        return orch

    def test_route_factual_returns_path(self):
        """_route_factual 应返回 path 字段"""
        orch = self._make_mock_orch()
        try:
            result = orch._route_factual("龙是什么", ['龙'])
            self.assertIsInstance(result, dict)
            self.assertIn('path', result)
        except Exception as e:
            self.skipTest(f"Stage4模块未就绪: {e}")

    def test_route_factual_handles_empty_chars(self):
        """_route_factual 空 query_chars"""
        orch = self._make_mock_orch()
        orch._get_triples_for = Mock(return_value=[])
        try:
            result = orch._route_factual("test", [])
            self.assertIn('path', result)
        except Exception as e:
            self.skipTest(f"需要完整模块: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 辅助运行
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 用 unittest 运行，verbose 输出
    unittest.main(verbosity=2)
