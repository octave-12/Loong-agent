#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠调度器 (Orchestrator) v3 — 信号驱动闭环推理中枢
═══════════════════════════════════════════════════

五步推理管道:
  第一步: 解义器 → 编码查询为向量 + 问题类型分流
  第二步: 策应器 → 制定推理策略 (知识查询/计算/闲聊/创意)
  第三步: 能量景观 → 梯度下降推理 → 发出四种信号之一
  第四步: 信号处理 → 大脑发信号，手脚去执行，知识写回大脑
  第五步: 化能器 → 生成自然语言回答 + 置信度标签

旧的对话路由(dialogue)保留向后兼容。
新增 query() 方法实现信号驱动的五步管道。
"""

import sys
import os
import time
import re
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple

import torch

import json

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

log = logging.getLogger('orchestrator')


class Orchestrator:
    """
    龙珠调度器 v3 — 信号驱动闭环推理。

    核心变更:
      - dialogue() 保留，向后兼容旧的四层路由
      - query() 新增，实现五步信号驱动管道
      - _handle_signal() 实现「大脑发信号→手脚执行→写回大脑」闭环
    """

    # ── 信号处理配置 ──
    MAX_SIGNAL_ITERATIONS = 3     # 最多迭代3次

    def __init__(self, field, landscape, concept_graph, learner=None):
        self.field = field
        self.landscape = landscape
        self.cg = concept_graph
        self.learner = learner

        # 延迟初始化所有引擎
        self._sem_parser = None
        self._decoder = None
        self._mkg = None
        self._fuzzy = None
        self._planner = None
        self._multilang = None
        self._harvester = None
        self._contra = None
        self._conversation = None
        self._creative = None
        self._pipeline = None
        self._searcher = None  # ★ 双臂: WebSearcher (惰性加载)

        # ★ 优先学习管道: 交互模式"超出知识范围"→写入队列→守护模式插队处理
        self._pending_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'runtime', 'pending_queries.json'
        )
        self._pending_queries: List[Dict] = []  # [{query_text, signal, ts, attempts, top_candidate}]
        self._pending_lock = threading.Lock()    # ★ 并发安全锁
        self._load_pending_queries()

        self._init_time = time.time()
        self._round_stats = {
            'harvested_triples': 0,
            'conflicts_resolved': 0,
            'd_s_feedbacks': 0,
            'cross_lang_lookups': 0,
        }

    # ═══════════════════════════════════════════════════════════════════
    # ★ 优先学习管道: 待决问题队列 — 交互→守护实时桥梁
    # ═══════════════════════════════════════════════════════════════════

    def _load_pending_queries(self):
        """从磁盘加载未解决的待决问题队列"""
        try:
            os.makedirs(os.path.dirname(self._pending_path), exist_ok=True)
            if os.path.exists(self._pending_path):
                with open(self._pending_path, 'r', encoding='utf-8') as f:
                    self._pending_queries = json.load(f)
                log.info(f"📋 加载待决队列: {len(self._pending_queries)}个未解决问题")
        except Exception:
            self._pending_queries = []

    def _save_pending_queries(self):
        """持久化待决问题队列到磁盘"""
        try:
            os.makedirs(os.path.dirname(self._pending_path), exist_ok=True)
            with open(self._pending_path, 'w', encoding='utf-8') as f:
                json.dump(self._pending_queries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"待决队列保存失败: {e}")

    def _add_pending_query(self, query_text: str, signal: str, top_candidate: str = ''):
        """
        ★ 将一个超出知识范围的查询加入优先学习队列。
        去重: 同一查询文本不重复入队。线程安全。
        """
        with self._pending_lock:
            for pq in self._pending_queries:
                if pq.get('query_text') == query_text:
                    pq['ts'] = time.time()
                    pq['signal'] = signal
                    pq['attempts'] = pq.get('attempts', 0)
                    self._save_pending_queries()
                    return
            self._pending_queries.append({
                'query_text': query_text,
                'signal': signal,
                'ts': time.time(),
                'attempts': 0,
                'top_candidate': top_candidate,
            })
            self._save_pending_queries()
        log.info(f"📌 待决入队: '{query_text[:40]}' (信号={signal})")

    def _resolve_pending_query(self, query_text: str):
        """★ 问题被解决后从队列移除"""
        before = len(self._pending_queries)
        self._pending_queries = [
            pq for pq in self._pending_queries
            if pq.get('query_text') != query_text
        ]
        if len(self._pending_queries) < before:
            self._save_pending_queries()
            log.info(f"✅ 待决出队: '{query_text[:40]}'")

    def _arms_search_deep(self, query_text: str, top_candidate: str = '') -> List[Tuple[int, int]]:
        """
        ★ 深度多策略搜索 — 交互模式回溯时和守护模式插队时共用。
        比 _arms_search_and_inject 更激进:
          1. 多查询策略 (复用 _build_search_queries 的7因子模板)
          2. Web 搜索 + 本地词典双引擎
          3. 查询文本自身的字对也纳入
        Returns: 去重后的字对索引列表
        """
        # 确定搜索关键词
        if not top_candidate:
            chars = re.findall(r'[\u4e00-\u9fff]', query_text)
            top_candidate = chars[0] if chars else '?'

        # 多查询策略: 对每个候选因子都生成查询
        all_pairs = []
        factors = ['dead_end', 'gradient', 'semantic', 'statistical', 'coverage', 'freshness']
        for factor in factors:
            queries = self._build_search_queries(top_candidate, factor)
            for sq in queries[:2]:  # 每因子最多2条查询
                try:
                    search_response = self.searcher.search(sq, max_results=5)
                    if search_response and search_response.results:
                        pairs = self._extract_pairs_from_search(search_response, sq)
                        all_pairs.extend(pairs)
                except Exception:
                    continue
                time.sleep(0.3)  # 避免被搜索引擎封

        # 本地词典补充
        try:
            local_pairs = self._extract_pairs_from_local_dicts(top_candidate)
            all_pairs.extend(local_pairs)
        except Exception:
            pass

        # 查询文本自身字对 (加权)
        try:
            query_chars = re.findall(r'[\u4e00-\u9fff]', query_text)
            for i in range(len(query_chars) - 1):
                a, b = query_chars[i], query_chars[i + 1]
                ia = self.field._char_to_idx.get(a)
                ib = self.field._char_to_idx.get(b)
                if ia is not None and ib is not None:
                    all_pairs.append((ia, ib))
        except Exception:
            pass

        # 去重
        seen = set()
        unique = []
        for p in all_pairs:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:300]

    # ═══════════════════════════════════════════════════════════════════
    # 延迟初始化（避免启动时全加载）
    # ═══════════════════════════════════════════════════════════════════

    @property
    def sem_parser(self):
        if self._sem_parser is None:
            from loongpearl.core.sem_parser import SemParser
            self._sem_parser = SemParser(self.cg)
        return self._sem_parser

    @property
    def decoder(self):
        if self._decoder is None:
            from loongpearl.core.hybrid_decoder import HybridDecoder
            self._decoder = HybridDecoder()
        return self._decoder

    @property
    def mkg(self):
        if self._mkg is None:
            from loongpearl.core.multiform_kg import MultiFormKG, seed_multiform_kg
            self._mkg = MultiFormKG(self.cg)
            seed_multiform_kg(self._mkg)
        return self._mkg

    @property
    def fuzzy(self):
        if self._fuzzy is None:
            from loongpearl.core.fuzzy_graph import FuzzyGraph
            self._fuzzy = FuzzyGraph(self.cg)
        return self._fuzzy

    @property
    def planner(self):
        if self._planner is None:
            from loongpearl.core.task_planner import TaskPlanner
            self._planner = TaskPlanner(self.cg, self.sem_parser, self.decoder)
        return self._planner

    @property
    def multilang(self):
        if self._multilang is None:
            from loongpearl.core.multilang_anchor import MultiLangAnchor
            self._multilang = MultiLangAnchor(self.cg, self.field)
        return self._multilang

    @property
    def harvester(self):
        if self._harvester is None:
            from loongpearl.core.harvester import KnowledgeHarvester
            self._harvester = KnowledgeHarvester(self.cg)
        return self._harvester

    @property
    def contra(self):
        if self._contra is None:
            from loongpearl.core.contra_resolver import ContraResolver
            self._contra = ContraResolver(self.cg)
        return self._contra

    @property
    def conversation(self):
        if self._conversation is None:
            from loongpearl.core.conversation import ConversationEngine
            self._conversation = ConversationEngine(self)
        return self._conversation

    @property
    def creative(self):
        if self._creative is None:
            from loongpearl.core.creative import CreativeEngine
            self._creative = CreativeEngine(self.field, self.cg, self.decoder)
        return self._creative

    @property
    def pipeline(self):
        if self._pipeline is None:
            from loongpearl.core.knowledge_pipeline import KnowledgePipeline
            self._pipeline = KnowledgePipeline(
                field=self.field,
                landscape=self.landscape,
                concept_graph=self.cg,
                orchestrator=self,
                learner=getattr(self, 'learner', None),
            )
        return self._pipeline

    @property
    def searcher(self):
        """★ 双臂: WebSearcher 惰性加载"""
        if self._searcher is None:
            from loongpearl.web.searcher import WebSearcher
            self._searcher = WebSearcher(timeout=15, cache_enabled=True)
        return self._searcher

    # ═══════════════════════════════════════════════════════════════════
    # 旧版对话路由 (保留向后兼容)
    # ═══════════════════════════════════════════════════════════════════

    def _get_triples_for(self, subject: str):
        """返回 (relation, object, conf, source) 列表 (SQLite优先, JSON回退)"""
        # ★ SQLite 加速: O(log N) 索引查询
        if hasattr(self, '_cgdb') and self._cgdb is not None:
            try:
                return self._cgdb.query_by_subject(subject, limit=200)
            except Exception:
                pass
        # JSON 回退
        results = []
        count = 0
        for key, t in self.cg.triples.items():
            if count >= 100000:
                break
            if hasattr(t, 'subject') and t.subject == subject:
                results.append((t.relation, t.object, t.confidence, t.source))
            count += 1
        return results

    def dialogue(self, query: str) -> Dict[str, Any]:
        """旧版四层路由 — 保留向后兼容"""
        result = {"output": "", "debug": {}, "type": "unknown"}

        conv_result = self.conversation.respond(query)
        if conv_result["type"] in ("social", "chitchat"):
            result["output"] = conv_result["output"]
            result["type"] = conv_result["type"]
            return result

        if conv_result.get("enhanced_query") and conv_result["enhanced_query"] != query:
            query = conv_result["enhanced_query"]
            result["debug"]["context_enhanced"] = True

        creative_output = self.creative.handle(query)
        if creative_output:
            result["output"] = creative_output
            result["type"] = "creative"
            result["debug"]["creative_type"] = (
                "poetry" if "《" in creative_output else
                "idiom_chain" if "→" in creative_output else
                "narrative"
            )
            self.conversation.state.add_turn(query, creative_output)
            return result

        result["type"] = "knowledge"

        lang = self.multilang.detect_language(query)
        result['debug']['lang'] = lang

        if lang != 'zh':
            cids = self.multilang.map_to_concepts(query, lang=lang)
            if cids:
                zh_name = self.multilang.get_concept_name(cids[0], 'zh')
                if zh_name:
                    result['debug']['cross_lang'] = f"{query} → {zh_name}"
                    self._round_stats['cross_lang_lookups'] += 1
                    query = zh_name

        frame = self.sem_parser.parse(query)
        for concept in frame.concepts[:5]:
            if concept and len(concept) >= 2:
                self.pipeline.feed_user_concept(concept, context=query)
        result['debug']['frame'] = {
            'type': frame.question_type.name if frame.question_type else '陈述',
            'intent': frame.intent.name if frame.intent else 'N/A',
            'subject': frame.subject,
            'object': frame.object,
            'concepts': frame.concepts,
        }

        plan = self.planner.plan(query)
        result['debug']['plan'] = {'type': plan.query_type, 'tasks': len(plan.tasks)}

        exec_results = self.planner.execute(plan)
        result['debug']['exec'] = {'results': len(exec_results.get('results', {}))}

        for concept in frame.concepts[:3]:
            cross = self.mkg.reason_across_forms(concept)
            if any(v for v in cross.values()):
                result['debug']['cross_form'] = {c: bool(v) for c, v in cross.items()}
                break

        if frame.intent and frame.intent.name in ('CHECK_TRUTH', 'FIND_PATH', 'DEFINE'):
            for concept in frame.concepts[:3]:
                if concept in self.cg.triples:
                    for rel, obj, conf, src in self._get_triples_for(concept)[:5]:
                        self.fuzzy.add_evidence(concept, rel, obj, source=src, mass=conf)

        self._sync_fuzzy_to_cg()

        render_input = self._build_render_input(frame, exec_results, query)
        result['output'] = self.decoder.decode(render_input)
        return result

    def _build_render_input(self, frame, exec_results, query) -> Dict:
        agg = exec_results.get('aggregated', {})
        if agg.get('comparison'):
            return {"render_type": "compare", "compare_subjects": frame.concepts[:2],
                    "facts": agg['comparison'].get('common', []) +
                             agg['comparison'].get('only_a', []) +
                             agg['comparison'].get('only_b', [])}
        elif agg.get('paths'):
            paths = agg['paths']
            edge_list = [{"rel": "RELATED", "confidence": 0.5}
                         for _ in range(len(paths[0]) - 1)] if paths else []
            return {"render_type": "explain_path", "subject": frame.subject or query,
                    "path": paths[0] if paths else [query], "edges": edge_list}
        elif agg.get('facts'):
            return {"render_type": "list_related", "subject": frame.subject or query,
                    "facts": agg['facts']}
        elif agg.get('table'):
            return {"render_type": "table", "facts": agg['table']}
        return {"render_type": "fact_statement",
                "facts": [{"relation": "RELATED", "object": c,
                           "subject": frame.subject or query}
                          for c in frame.concepts[:3]]}

    # ═══════════════════════════════════════════════════════════════════
    # ★ 新版: 五步信号驱动推理管道
    # ═══════════════════════════════════════════════════════════════════

    # ── 查询路由: 问题分类 + 路径选择 ──

    _FACTUAL_KEYWORDS = ['是什么', '什么是', '定义', '属于', '称作', '叫做',
                         '是不是', '是.*吗', '分类', '类别', '物种']
    _SEQUENTIAL_KEYWORDS = ['下一句', '下一字', '下一首', '补全', '接龙',
                            '后面.*什么', '接着.*什么']
    _RELATIONAL_KEYWORDS = ['区别', '不同', '差异', '相似', '相同', '比较',
                            '相比', 'vs', '对比', '关系']

    def _classify_query(self, text: str, frame) -> str:
        """分类查询类型: factual / sequential / relational / uncertain"""
        text_lower = text.lower()

        # 序列类: 查询含 "_" 或下划线（补全标记）→ sequential
        if '_' in text or '＿' in text or '___' in text:
            return 'sequential'

        # 关键词匹配
        for kw in self._SEQUENTIAL_KEYWORDS:
            if re.search(kw, text):
                return 'sequential'

        for kw in self._FACTUAL_KEYWORDS:
            if re.search(kw, text):
                return 'factual'

        for kw in self._RELATIONAL_KEYWORDS:
            if re.search(kw, text):
                return 'relational'

        # 帧类型辅助判断
        if frame and frame.question_type:
            qt = frame.question_type.name
            if qt in ('DEFINE', 'CHECK_TRUTH', 'YES_NO'):
                return 'factual'
            if qt in ('COMPARE', 'FIND_PATH'):
                return 'relational'

        return 'uncertain'

    def _route_query(self, text: str, query_chars: list,
                     query_vec: torch.Tensor, frame) -> Dict:
        """
        查询路由: 根据问题类型选择主路径。

        Returns:
            {'path': str, 'result': dict | None, 'fallback': bool}
            path ∈ {'concept_graph', 'poetic_next', 'energy_landscape', 'fuzzy_fusion'}
        """
        qtype = self._classify_query(text, frame)

        if qtype == 'factual':
            return self._route_factual(text, query_chars)
        elif qtype == 'sequential':
            return self._route_sequential(text, query_chars, query_vec)
        elif qtype == 'relational':
            return {'path': 'energy_landscape', 'result': None, 'fallback': False}
        else:
            return {'path': 'fuzzy_fusion', 'result': None, 'fallback': False}

    def _route_factual(self, text: str, query_chars: list) -> Dict:
        """事实类查询: 概念图 IS_A / DEFINED_AS 优先"""
        result = {'path': 'concept_graph', 'result': None, 'fallback': False}

        for ch in query_chars[:5]:
            triples = self._get_triples_for(ch)
            for rel, obj, conf, src in triples:
                if rel in ('IS_A', 'DEFINED_AS') and conf > 0.3:
                    if not result['result']:
                        result['result'] = []
                    result['result'].append({
                        'subject': ch, 'relation': rel,
                        'object': obj, 'confidence': conf
                    })

        # 概念图无结果 → 回退到能量景观
        if not result['result']:
            result['path'] = 'energy_landscape'
            result['fallback'] = True

        return result

    def _route_sequential(self, text: str, query_chars: list,
                          query_vec: torch.Tensor) -> Dict:
        """序列类查询: 概念图 POETIC_NEXT 优先 (SQLite 加速)"""
        result = {'path': 'poetic_next', 'result': None, 'fallback': False}

        last_char = query_chars[-1] if query_chars else ''
        if not last_char:
            result['fallback'] = True
            return result

        # ★ SQLite 直接查询 POETIC_NEXT
        next_chars = []
        if hasattr(self, '_cgdb') and self._cgdb is not None:
            try:
                next_chars = self._cgdb.query_poetic_next(last_char, min_conf=0.01)
            except Exception:
                pass

        # 回退: JSON _get_triples_for
        if not next_chars:
            triples = self._get_triples_for(last_char)
            for rel, obj, conf, src in triples:
                if rel == 'POETIC_NEXT' and conf > 0.01 and len(obj) == 1:
                    next_chars.append((obj, conf))
            next_chars.sort(key=lambda x: -x[1])

        if next_chars:
            result['result'] = {
                'anchor': last_char,
                'candidates': next_chars[:10],
                'source': 'concept_graph'
            }
            return result

        result['path'] = 'energy_landscape'
        result['fallback'] = True
        return result

    def query(self, question_text: str) -> Dict[str, Any]:
        """
        五步信号驱动推理 — 龙珠核心管道。

        第一步: 解义 — NLU编码 + 问题分类
        第二步: 策应 — 推理策略分流
        第三步: 能量景观 — 梯度下降推理 + 信号发射
        第四步: 信号处理 — 闭环吸收（大脑→手脚→写回）
        第五步: 化能 — 生成回答 + 置信度标签

        Returns:
            {'answer': str, 'signal': str, 'confidence': float, 'debug': dict}
        """
        result = {'answer': '', 'signal': 'certain', 'confidence': 0.0, 'debug': {}}

        # ── 第一步: 解义与编码 ──
        frame = self.sem_parser.parse(question_text)
        question_type = frame.question_type.name if frame.question_type else 'statement'
        result['debug']['frame'] = {
            'type': question_type,
            'subject': frame.subject,
            'concepts': frame.concepts,
        }

        # 构造查询向量：取概念词的字场锚点均值
        query_chars = re.findall(r'[\u4e00-\u9fff]', frame.subject or question_text)
        if not query_chars:
            query_chars = re.findall(r'[\u4e00-\u9fff]', question_text)

        # ── 第二步: 意图规划 (策应器分流) ──
        plan = self.planner.plan(question_text)
        result['debug']['plan_type'] = plan.query_type

        # ★ 计算类分流: 必须抢先于无汉字检查，计算表达式可能无汉字
        if self._is_compute_query(question_text):
            try:
                from loongpearl.utils.compute_sandbox import ComputeSandbox
                sandbox = ComputeSandbox()
                calc_answer = sandbox.calculate(question_text)
                result['answer'] = calc_answer
                result['signal'] = 'certain'
                result['confidence'] = 0.95
                result['debug']['plan_type'] = 'compute'
                return result
            except Exception as e:
                log.warning(f"  计算沙盒异常: {e}，回退到知识推理")

        # ★ 无汉字且非计算类 → 兜底渲染
        if not query_chars:
            result['answer'] = self.decoder.decode({
                'render_type': 'fact_statement',
                'facts': [{'relation': 'RELATED', 'object': question_text,
                           'subject': '未知'}],
            })
            return result

        # 查询向量 = 位置感知序列编码（保留语序）
        # 构建词库：从概念图提取已知双字词
        word_lexicon = None
        if hasattr(self, 'cg') and hasattr(self.cg, 'forward_index'):
            word_lexicon = {s for s in self.cg.forward_index.keys()
                          if len(s) == 2 and '\u4e00' <= s[0] <= '\u9fff'
                          and '\u4e00' <= s[1] <= '\u9fff'}

        if query_chars:
            query_vec = self.field.encode_sequence(
                query_chars, direction='forward', word_lexicon=word_lexicon
            )
            result['debug']['query_chars'] = query_chars[:5]
        else:
            # 无汉字但非计算类: 使用零向量兜底
            device = next(self.landscape.parameters()).device
            query_vec = torch.zeros(self.field.embed_dim, device=device)

        # 非知识查询直接走对应引擎
        conv_result = self.conversation.respond(question_text)
        if conv_result["type"] in ("social", "chitchat"):
            result['raw_answer'] = conv_result["output"]  # ★ 统一字段
            result['signal'] = 'certain'
            result['confidence'] = 1.0
            # ★ 模糊保护: 对话引擎输出经能量景观评估
            result['answer'] = self._generate_answer(
                question_text, result, source="chat"
            )
            return result

        creative_output = self.creative.handle(question_text)
        if creative_output:
            result['raw_answer'] = creative_output  # ★ 统一字段
            result['signal'] = 'certain'
            result['confidence'] = 0.9
            # ★ 模糊保护: 创意引擎输出经能量景观评估
            result['answer'] = self._generate_answer(
                question_text, result, source="creative"
            )
            return result

        # ── ★ 查询路由: 根据问题类型选择推理路径 ──
        route = self._route_query(question_text, query_chars, query_vec, frame)
        result['debug']['route'] = {
            'path': route['path'],
            'fallback': route.get('fallback', False),
            'qtype': self._classify_query(question_text, frame),
        }

        # 概念图直达 → 跳过能量推理
        if route['path'] == 'concept_graph' and route.get('result') and not route.get('fallback'):
            cg_result = route['result']
            facts = [{'relation': r['relation'], 'object': r['object'],
                      'subject': r['subject'], 'confidence': r['confidence']}
                     for r in cg_result[:5]]
            result['answer'] = self.decoder.decode({
                'render_type': 'list_related',
                'subject': question_text,
                'facts': facts,
            })
            result['signal'] = 'certain'
            result['confidence'] = max(r['confidence'] for r in cg_result[:5]) if cg_result else 0.3
            return result

        # 序列补全 → POETIC_NEXT 直达
        if route['path'] == 'poetic_next' and route.get('result') and not route.get('fallback'):
            seq_result = route['result']
            candidates = seq_result['candidates']
            if candidates:
                top = candidates[0]
                result['answer'] = f"「{seq_result['anchor']}」后最常接「{top[0]}」(置信度 {top[1]:.3f})"
                result['signal'] = 'certain'
                result['confidence'] = top[1]
                result['debug']['next_chars'] = candidates[:5]
                return result

        # ── 第三步: 能量景观梯度下降推理 ──
        infer_result = self.landscape.infer(
            query_vec,
            steps=50,
            lr=0.02,
            zichang=self.field,  # ★ 传入字场以启用信号发射
        )
        result['debug']['infer'] = {
            'signal': infer_result['signal'],
            'energy': infer_result['energy'],
            'steps': infer_result['steps'],
            'gradient_norm': infer_result.get('gradient_norm', 0),
            'top_candidates': infer_result.get('top_candidates', []),
        }

        # ── 第四步: 信号处理闭环 ──
        final_result = self._handle_signal(question_text, query_vec, infer_result)
        result['signal'] = final_result['signal']
        result['confidence'] = self._compute_confidence(final_result)
        result['debug']['final'] = {
            'signal': final_result['signal'],
            'signal_detail': final_result.get('signal_detail', ''),
            'energy': final_result['energy'],
        }

        # ── 第五步: 生成回答 ──
        result['answer'] = self._generate_answer(
            question_text, final_result, source="knowledge"
        )

        return result

    # ═══════════════════════════════════════════════════════════════════
    # ★ 第四步核心: 信号驱动的闭环处理
    # ═══════════════════════════════════════════════════════════════════

    def _handle_signal(
        self,
        query_text: str,
        query_vec: torch.Tensor,
        result: Dict,
    ) -> Dict:
        """
        根据能量景观发出的信号，调度对应手脚模块。

        闭环逻辑:
          blind_spot    → 🦾 双臂搜索 → 注入景观 → 重新推理
          conflict      → 🧬 身体裁决 → 强化/削弱 → 重新推理
          low_confidence → 🦶 双脚验证 → 强化路径 → 重新推理
          certain       → 直接返回

        回溯验证: 连续两轮相同信号且无改善 → 入队优先学习 + 最后一次深度尝试
        最多迭代 MAX_SIGNAL_ITERATIONS 次。

        ★ v2 变更: 回溯时不再直接丢弃，而是:
          1. 写入 pending_queries.json → 守护模式插队处理
          2. 触发最后一次深度多策略搜索 → 最后抢救
        """
        previous_signal = None       # ★ 回溯: 上轮信号
        same_signal_count = 0        # ★ 回溯: 相同信号连续出现次数

        for iteration in range(self.MAX_SIGNAL_ITERATIONS):
            signal = result.get('signal', 'certain')

            if signal == 'certain':
                # ★ 问题被解决 → 从待决队列移除
                self._resolve_pending_query(query_text)
                return result

            log.info(f"  📶 信号[{iteration+1}/{self.MAX_SIGNAL_ITERATIONS}]: "
                     f"{signal} — {result.get('signal_detail', '')[:80]}")

            # ★ 回溯检测: 连续两轮相同信号 → 入队 + 深度抢救
            if signal == previous_signal:
                same_signal_count += 1
                if same_signal_count >= 2:
                    # ★ 提取 top_candidate 用于深度搜索
                    top_cand = ''
                    if result.get('top_candidates'):
                        top_cand = result['top_candidates'][0]

                    # ★ 1. 入队优先学习 (守护模式会插队处理)
                    self._add_pending_query(query_text, signal, top_cand)

                    # ★ 2. 最后一次深度抢救: 全因子搜索 + 注入 + 重新推理
                    log.info(f"  🔄 最后一次深度抢救: '{query_text[:40]}'")
                    deep_pairs = self._arms_search_deep(query_text, top_cand)
                    if deep_pairs and self.learner:
                        try:
                            self.learner.learn_pairs_batch(deep_pairs, learning_rate=0.08)
                        except Exception:
                            pass
                    # 抢救后重新推理
                    result = self.landscape.infer(
                        query_vec, steps=80, lr=0.03, zichang=self.field
                    )
                    new_signal = result.get('signal', signal)
                    if new_signal == 'certain':
                        log.info(f"  ✅ 抢救成功! 信号 {signal}→certain")
                        self._resolve_pending_query(query_text)
                    else:
                        result['note'] = (f"经过{iteration+1}轮验证+深度抢救，"
                                         f"'{new_signal}'信号未改善，已加入优先学习队列")
                        log.warning(f"  ⚠️ 回溯终止: 连续{same_signal_count}轮'{signal}'"
                                   f"→抢救后仍为'{new_signal}'，已入队")
                    return result
            else:
                same_signal_count = 0
            previous_signal = signal

            # ── 信号分发 ──
            if signal == 'blind_spot':
                result = self._arms_search_and_inject(query_text, query_vec, result)

            elif signal == 'conflict':
                result = self._body_adjudicate(query_text, query_vec, result)

            elif signal == 'low_confidence':
                result = self._feet_verify_and_reinforce(query_text, query_vec, result)

            else:
                log.warning(f"  ⚠️ 未知信号类型 '{signal}'，静默返回")
                return result

            # ★ 知识写回后必须重新推理，检查新信号
            result = self.landscape.infer(
                query_vec, steps=50, lr=0.02, zichang=self.field
            )

        # 超过最大迭代次数，加入待决队列
        top_cand = result.get('top_candidates', [''])[0] if result.get('top_candidates') else ''
        self._add_pending_query(query_text, result.get('signal', 'unknown'), top_cand)
        log.warning(f"  ⚠️ 信号处理达到最大迭代次数({self.MAX_SIGNAL_ITERATIONS})，"
                    f"当前信号: {result.get('signal', '?')}，已入队优先学习")
        return result

    # ── 双臂: 盲区 → Web搜索 + 本地词典 ──

    def _arms_search_and_inject(
        self,
        query_text: str,
        query_vec: torch.Tensor,
        result: Dict,
    ) -> Dict:
        """
        🦾 双臂响应盲区信号 (v2 深度版):
        1. 多因子查询策略 → Web搜索 (6类因子 × 2查询)
        2. 本地成语/Unihan/CEDICT 词典查询
        3. 查询文本自身字对加权
        4. 提取字对 → Hebbian注入能量景观
        5. 重新梯度下降推理
        """
        # ★ 提取搜索关键词
        top_char = '?'
        if result.get('top_candidates'):
            top_char = result['top_candidates'][0]
        else:
            query_chars = [c for c in query_text if '\u4e00' <= c <= '\u9fff']
            if query_chars:
                top_char = query_chars[0]

        # ★ 深度多策略搜索
        pairs = self._arms_search_deep(query_text, top_char)

        # 去重并注入能量景观
        unique_pairs = list(set(pairs))
        if unique_pairs and self.learner:
            try:
                inject_result = self.learner.learn_pairs_batch(unique_pairs, learning_rate=0.05)
                log.info(f"  🦾 双臂注入: {inject_result.get('pairs_learned', 0)}字对 "
                        f"({len(unique_pairs)}去重), "
                        f"分离度 {inject_result.get('separation_before', 0):.1f}→"
                        f"{inject_result.get('separation_after', 0):.1f}")
            except Exception as e:
                log.warning(f"  双臂注入失败: {e}")

        # ★ 注入完成, 推理由 _handle_signal 统一执行
        return result

    # ── 身体: 冲突 → 概念图 + 矛盾解 + D-S 裁决 ──

    def _body_adjudicate(
        self,
        query_text: str,
        query_vec: torch.Tensor,
        result: Dict,
    ) -> Dict:
        """
        🧬 身体响应冲突信号:
        1. 概念图检索矛盾双方的证据链
        2. 万象格跨格式对比
        3. 模糊格 D-S 证据理论裁决
        4. 强化胜者路径，削弱败者路径
        """
        top_candidates = result.get('top_candidates', [])
        if len(top_candidates) < 2:
            return result

        char_a, char_b = top_candidates[0], top_candidates[1]

        # 1. 概念图: 查双方证据
        evidence_a = self._get_evidence_for(char_a)
        evidence_b = self._get_evidence_for(char_b)

        # 2. 万象格: 跨格式验证
        cross_a = self.mkg.reason_across_forms(char_a)
        cross_b = self.mkg.reason_across_forms(char_b)
        forms_a = sum(1 for v in cross_a.values() if v)
        forms_b = sum(1 for v in cross_b.values() if v)

        # 3. 模糊格 D-S 裁决: 证据写入后取融合置信度
        for rel, obj, conf, src in evidence_a:
            self.fuzzy.add_evidence(char_a, rel, obj, source=src, mass=conf)
        for rel, obj, conf, src in evidence_b:
            self.fuzzy.add_evidence(char_b, rel, obj, source=src, mass=conf)

        # ★ D-S 融合裁决: 用模糊格的 combined_mass 替代手工加权
        ds_a = self._compute_candidate_belief(char_a)
        ds_b = self._compute_candidate_belief(char_b)

        if ds_a > 0 or ds_b > 0:
            # D-S 融合有效: 信念质量 70% + 万象格覆盖 20% + 证据量 10%
            score_a = ds_a * 0.7 + min(forms_a * 0.1, 0.2) + min(len(evidence_a) * 0.02, 0.1)
            score_b = ds_b * 0.7 + min(forms_b * 0.1, 0.2) + min(len(evidence_b) * 0.02, 0.1)
            log.info(f"  🧬 D-S裁决: '{char_a}' belief={ds_a:.3f} vs '{char_b}' belief={ds_b:.3f}")
        else:
            # 回退: 模糊格无数据时用简单启发式
            score_a = len(evidence_a) * 0.3 + forms_a * 0.1
            score_b = len(evidence_b) * 0.3 + forms_b * 0.1
            log.info(f"  🧬 启发式裁决: '{char_a}' 证据{len(evidence_a)} vs '{char_b}' 证据{len(evidence_b)}")

        # 决策: 高分者为胜者
        if score_a >= score_b:
            winner, loser = char_a, char_b
        else:
            winner, loser = char_b, char_a

        # 4. 强化胜者路径 + 削弱败者路径
        query_chars = re.findall(r'[\u4e00-\u9fff]', query_text)
        if query_chars and self.learner:
            for qc in query_chars[:3]:
                try:
                    self.learner.learn_chars(qc, winner, strength=0.5)
                    self.learner.unlearn_chars(qc, loser, strength=0.3)
                except Exception as e:
                    log.debug(f"  路径更新失败({qc}): {e}")

        # 冲突标记写回概念图
        try:
            self.cg.add_triple(winner, "RELATED", loser,
                              confidence=0.2, source="adjudicated_conflict")
        except Exception:
            pass

        # ★ 注入完成, 推理由 _handle_signal 统一执行
        return result

    def _compute_candidate_belief(self, char: str) -> float:
        """
        ★ 计算某候选项在模糊格中的 D-S 聚合信念质量。

        遍历模糊格中所有以该字符为主体 (subject) 的命题，
        对它们的 combined_mass 取加权平均。
        无证据时返回 0.0。
        """
        total_mass = 0.0
        total_weight = 0.0
        for (s, r, o), bpa in self.fuzzy._bpas.items():
            if s == char and bpa.combined_mass > 0:
                evidence_count = len(bpa.evidences)
                weight = 1.0 + min(evidence_count, 5) * 0.1  # 多证据命题加权
                total_mass += bpa.combined_mass * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return total_mass / total_weight

    def _is_compute_query(self, text: str) -> bool:
        """
        ★ 检测是否为计算类请求（算术、数学表达式等）。

        匹配模式:
          - 纯算术: "1+1", "100*50", "2的10次方"
          - 算式: "计算 3.14*2", "sin(30)等于多少"
          - 数字为主且含运算符
        """
        import re as _re
        # 纯算术表达式: 数字+运算符
        if _re.search(r'[\d.]+\s*[\+\-\*/%\^]\s*[\d.]+', text):
            return True
        # 中文计算词 + 数字
        if _re.search(r'(计算|等于|多少|几)\s*[\d+\-*/]', text):
            return True
        # 函数调用: sin/cos/sqrt/log + 数字
        if _re.search(r'(sin|cos|tan|sqrt|log|abs|pow|exp)\s*\(\s*[\d.]+', text, _re.I):
            return True
        # 次方/开方
        if _re.search(r'[\d.]+的?\s*(次方|平方|立方|开方|开根)', text):
            return True
        return False

    # ── 双脚: 低置信 → Wikipedia + 词典验证 ──

    def _feet_verify_and_reinforce(
        self,
        query_text: str,
        query_vec: torch.Tensor,
        result: Dict,
    ) -> Dict:
        """
        🦶 双脚响应低置信信号:
        1. Wikipedia 权威百科采集
        2. 本地词典查询
        3. 强化验证后的路径
        4. 重新梯度下降推理
        """
        top_char = ''
        if result.get('top_candidates'):
            top_char = result['top_candidates'][0]
        else:
            qc = [c for c in query_text if '\u4e00' <= c <= '\u9fff']
            if qc:
                top_char = qc[0]
        if not top_char:
            return result

        verified = False

        # 1. Wikipedia 采集
        try:
            harvest_count = self.harvester.harvest_wikipedia(
                titles=[top_char], max_per_page=10
            )
            if harvest_count > 0:
                verified = True
                log.info(f"  🦶 Wikipedia验证: '{top_char}' 采集{harvest_count}条")
        except Exception as e:
            log.debug(f"  Wikipedia采集跳过: {e}")

        # 2. 本地词典查询 (成语)
        if not verified:
            try:
                idiom_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'data', 'dicts', 'idioms.json'
                )
                if os.path.exists(idiom_path):
                    import json
                    with open(idiom_path, encoding='utf-8') as f:
                        idioms = json.load(f)
                    matching = [i for i in idioms if top_char in i][:5]
                    if matching:
                        verified = True
                        log.info(f"  🦶 成语验证: '{top_char}' 匹配{len(matching)}条成语")
            except Exception as e:
                log.debug(f"  成语查询跳过: {e}")

        # 3. 强化路径
        if verified and self.learner:
            query_chars = re.findall(r'[\u4e00-\u9fff]', query_text)
            for qc in (query_chars[:2] if query_chars else [top_char]):
                try:
                    self.learner.learn_chars(qc, top_char, strength=0.4)
                except Exception:
                    pass

        # ★ 注入完成, 推理由 _handle_signal 统一执行
        return result

    # ═══════════════════════════════════════════════════════════════════
    # ★ 辅助方法: 知识提取与路径操作
    # ═══════════════════════════════════════════════════════════════════

    def _extract_pairs_from_search(self, search_response, context: str,
                                    use_llm: bool = True) -> List[Tuple[int, int]]:
        """从搜索结果中提取字对: 相邻字对 + DualExtractor 结构化三元组 → 字对

        Args:
            use_llm: 是否启用 LLM 兜底（守护模式禁用，防 Ollama 超时拖慢循环）
        """
        import re as _re
        from collections import Counter as _Counter

        pair_counter = _Counter()
        all_text = ' '.join(r.snippet for r in search_response.results[:5] if r.snippet)
        all_text += ' ' + (search_response.answer or '')

        # ── 原有: 相邻汉字对提取 ──
        cn_chars = _re.findall(r'[\u4e00-\u9fff]', all_text)
        for i in range(len(cn_chars) - 1):
            a, b = cn_chars[i], cn_chars[i + 1]
            if a in self.field._char_to_idx and b in self.field._char_to_idx:
                pair_counter[(a, b)] += 1

        query_chars = _re.findall(r'[\u4e00-\u9fff]', context)
        for i in range(len(query_chars) - 1):
            a, b = query_chars[i], query_chars[i + 1]
            if a in self.field._char_to_idx and b in self.field._char_to_idx:
                pair_counter[(a, b)] += 2

        # ── ★ DualExtractor: 结构化三元组 → 字对 (更高质量) ──
        if hasattr(self, '_dual_extractor') and self._dual_extractor:
            try:
                # 守护模式仅正则提取（快），聊天模式可 LLM 兜底
                if use_llm:
                    triples = self._dual_extractor.extract(all_text[:3000])
                else:
                    triples = self._dual_extractor.extract_regex(all_text[:3000])
                for s, r, o, conf in triples:
                    # 从三元组的主语和宾语提取字对
                    s_chars = _re.findall(r'[\u4e00-\u9fff]', s)
                    o_chars = _re.findall(r'[\u4e00-\u9fff]', o)
                    weight = int(conf * 5)  # 0.5→2, 0.7→3
                    for ca in s_chars:
                        for cb in o_chars:
                            if ca in self.field._char_to_idx and cb in self.field._char_to_idx:
                                pair_counter[(ca, cb)] += weight
            except Exception:
                pass

        pairs = []
        for (a, b), freq in pair_counter.most_common(50):
            ia = self.field._char_to_idx[a]
            ib = self.field._char_to_idx[b]
            pairs.append((ia, ib))
        return pairs

    def _extract_pairs_from_local_dicts(self, char: str) -> List[Tuple[int, int]]:
        """从本地成语词典提取包含某字的字对"""
        pairs = []
        try:
            idiom_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'dicts', 'idioms.json'
            )
            if os.path.exists(idiom_path):
                import json
                with open(idiom_path, encoding='utf-8') as f:
                    idioms = json.load(f)
                for idiom in idioms:
                    if char in idiom and len(idiom) >= 2:
                        for i in range(len(idiom) - 1):
                            a, b = idiom[i], idiom[i + 1]
                            ia = self.field._char_to_idx.get(a)
                            ib = self.field._char_to_idx.get(b)
                            if ia is not None and ib is not None:
                                pairs.append((ia, ib))
                        if len(pairs) >= 100:
                            break
        except Exception:
            pass
        return pairs

    def _get_evidence_for(self, concept: str) -> List[Tuple[str, str, float, str]]:
        """从概念图检索某概念的证据链"""
        evidence = []
        # 使用 forward_index 高效检索
        if concept in self.cg.forward_index:
            for obj, rel in self.cg.forward_index[concept].items():
                key = f"{concept}|{rel}|{obj}"
                triple = self.cg.triples.get(key)
                conf = triple.confidence if triple else 0.5
                src = triple.source if triple else "unknown"
                evidence.append((rel, obj, conf, src))
        # 反向检索
        for subj, edges in self.cg.forward_index.items():
            if concept in edges:
                rel = edges[concept]
                key = f"{subj}|{rel}|{concept}"
                triple = self.cg.triples.get(key)
                conf = triple.confidence if triple else 0.5
                src = triple.source if triple else "unknown"
                evidence.append((rel, subj, conf, src))
        return evidence[:10]

    def _inject_knowledge(self, knowledge_items: List[Dict], query_vec: torch.Tensor):
        """
        ★ 通过 Hebbian 学习将外部知识注入能量景观。

        Args:
            knowledge_items: [{'anchor': str, ...}, ...]
            query_vec: 查询向量
        """
        if not self.learner:
            return

        device = next(self.landscape.parameters()).device
        for item in knowledge_items:
            anchor_char = item.get('anchor', '')
            if anchor_char and anchor_char in self.field._char_to_idx:
                idx = self.field._char_to_idx[anchor_char]
                anchor_vec = self.field.anchors[idx].to(device)
                # Hebbian 强化: 降低查询向量→锚点中点的能量
                try:
                    self.learner.hebbian.update(query_vec.to(device), anchor_vec, feedback=0.3)
                except Exception:
                    pass

    def _reinforce_path(self, query_vec: torch.Tensor, anchor_char: str):
        """★ 强化路径: 降低查询向量到锚点的能量"""
        if not self.learner or anchor_char not in self.field._char_to_idx:
            return
        idx = self.field._char_to_idx[anchor_char]
        anchor_vec = self.field.anchors[idx].to(
            next(self.landscape.parameters()).device
        )
        try:
            self.learner.hebbian.update(
                query_vec.to(anchor_vec.device), anchor_vec, feedback=0.5
            )
        except Exception as e:
            log.debug(f"  路径强化失败({anchor_char}): {e}")

    def _weaken_path(self, query_vec: torch.Tensor, anchor_char: str):
        """★ 削弱路径: 提高查询向量到锚点的能量"""
        if not self.learner or anchor_char not in self.field._char_to_idx:
            return
        idx = self.field._char_to_idx[anchor_char]
        anchor_vec = self.field.anchors[idx].to(
            next(self.landscape.parameters()).device
        )
        try:
            self.learner.hebbian.update(
                query_vec.to(anchor_vec.device), anchor_vec, feedback=-0.3
            )
        except Exception as e:
            log.debug(f"  路径削弱失败({anchor_char}): {e}")

    # ═══════════════════════════════════════════════════════════════════
    # ★ 第五步: 化能器生成回答 — 置信度原生标注
    # ═══════════════════════════════════════════════════════════════════

    def _generate_answer(self, question: str, result: Dict,
                         source: str = "knowledge") -> str:
        """
        根据信号类型和置信度生成带标签的自然语言回答。

        Args:
            question: 原始问题
            result: 推理结果字典 (含 signal, top_candidates, state 等)
            source: 回答来源 — "knowledge"(能量景观推理) / "chat"(对话引擎) / "creative"(创意引擎)

        置信度标签:
          certain       → 直接回答
          low_confidence → ⚠️ 带不确定性提示
          blind_spot    → 🆕 刚学到的知识
          conflict      → ❓ 存在争议，列出主要观点

        ★ 模糊保护: 对话/创意引擎的输出必须经能量景观置信度评估
        """
        signal = result.get('signal', 'certain')

        # === 知识推理: 直接解码 ===
        if source == "knowledge":
            top_chars = result.get('top_candidates', [])
            converge_state = result.get('state')
            if converge_state is not None and top_chars:
                answer_body = self._decode_state(converge_state, top_chars, question)
            else:
                answer_body = f"关于「{question}」，目前的知识不完整。"

        # === 对话/创意: 模糊保护 — 编码→景观推理→置信度标注 ===
        else:
            raw_answer = result.get('raw_answer', '')
            if not raw_answer:
                raw_answer = f"关于「{question}」的回应"

            # 编码为向量并在能量景观中评估
            try:
                query_chars = re.findall(r'[\u4e00-\u9fff]', question)
                if query_chars:
                    device = next(self.landscape.parameters()).device
                    vecs = []
                    for ch in query_chars[:10]:
                        idx = self.field._char_to_idx.get(ch)
                        if idx is not None:
                            vecs.append(self.field.anchors[idx].to(device))
                    if vecs:
                        answer_vec = torch.stack(vecs).mean(dim=0)
                        verify_result = self.landscape.infer(
                            answer_vec, steps=30, lr=0.01,
                            zichang=self.field
                        )
                        verify_signal = verify_result.get('signal', 'certain')

                        if verify_signal == 'certain':
                            signal = 'certain'
                            answer_body = raw_answer
                        elif verify_signal == 'blind_spot':
                            signal = 'low_confidence'
                            result['note'] = '此回答为创意联想，龙珠对此领域的知识有限'
                            answer_body = raw_answer
                        elif verify_signal == 'conflict':
                            signal = 'low_confidence'
                            result['note'] = '回答涉及的概念存在知识冲突'
                            answer_body = raw_answer
                        else:
                            signal = 'low_confidence'
                            result['note'] = '以下回答仅供参考'
                            answer_body = raw_answer
                    else:
                        answer_body = raw_answer
                else:
                    answer_body = raw_answer
            except Exception:
                # 评估失败不影响主流程
                answer_body = raw_answer

        # === 置信度标签 ===
        if signal == 'certain':
            return answer_body
        elif signal == 'low_confidence':
            note = result.get('note', '以下回答仅供参考')
            return f"⚠️ {note}：\n{answer_body}"
        elif signal == 'blind_spot':
            return f"🆕 刚学到的知识：\n{answer_body}"
        elif signal == 'conflict':
            return f"❓ 存在争议，以下是主要观点：\n{answer_body}"
        return answer_body

    def _decode_state(self, state: torch.Tensor, candidates: List[str], question: str) -> str:
        """从收敛状态向量解码为自然语言回答"""
        try:
            # 化能器: 将向量映射为文字
            render_input = {
                "render_type": "fact_statement",
                "facts": [
                    {"relation": "RELATED", "object": c, "subject": question}
                    for c in candidates[:3]
                ],
            }
            return self.decoder.decode(render_input)
        except Exception:
            return f"关于「{question}」，最相关的概念是：{'、'.join(candidates[:3])}。"

    def _compute_confidence(self, result: Dict) -> float:
        """
        ★ 计算置信度 (0.0~1.0)。

        综合考虑:
          - 信号类型: certain=0.9, low_confidence=0.5, conflict=0.3, blind_spot=0.1
          - 能量值: 能量越低置信度越高
          - 梯度范数: 梯度越陡置信度越高
        """
        signal = result.get('signal', 'certain')
        energy = result.get('energy', 0.0)
        grad_norm = result.get('gradient_norm', 0.0)

        # 基础置信度（按信号类型）
        base_conf = {
            'certain': 0.9,
            'low_confidence': 0.5,
            'conflict': 0.3,
            'blind_spot': 0.1,
        }.get(signal, 0.5)

        # 能量修正: 低能量 (+0.1), 高能量 (-0.1)
        if energy < -10:
            energy_bonus = 0.1
        elif energy > 0:
            energy_bonus = -0.1
        else:
            energy_bonus = 0.0

        # 梯度修正: 陡峭梯度 (+0.05)
        grad_bonus = 0.05 if grad_norm > 0.01 else 0.0

        confidence = base_conf + energy_bonus + grad_bonus
        return max(0.0, min(1.0, confidence))

    # ═══════════════════════════════════════════════════════════════════
    # D-S 反馈闭环
    # ═══════════════════════════════════════════════════════════════════

    def _sync_fuzzy_to_cg(self):
        """将 D-S 融合后的置信度回写到概念图"""
        count = 0
        for (s, r, o), bpa in self.fuzzy._bpas.items():
            if bpa.combined_mass > 0:
                if s in self.cg.triples:
                    for i, (rel, obj, conf, src) in enumerate(self._get_triples_for(s)):
                        if rel == r and obj == o:
                            new_triple = (rel, obj, bpa.combined_mass, f"{src}+DS")
                            self._get_triples_for(s)[i] = new_triple
                            count += 1
        if count:
            self._round_stats['d_s_feedbacks'] += count
            log.debug(f"  🔄 D-S回写: {count} 条置信度更新")

    # ═══════════════════════════════════════════════════════════════════
    # 守护循环 v2 — 信号驱动统一循环
    # ═══════════════════════════════════════════════════════════════════

    def process_pending_queries(self) -> int:
        """
        ★ 优先学习管道核心: 处理待决问题队列。

        每轮 daemon_tick_v2 调用此方法，按优先级:
          1. 取出队列中最旧的 pending query
          2. 深度搜索 + 注入能量景观
          3. 重新梯度下降推理
          4. 信号改善 → 移出队列; 未改善 → attempts+1 保留
          5. attempts ≥ 5 且仍失败 → 移出队列 (放弃)

        Returns: 本轮成功解决的数量
        """
        # ★ 每次调用前从磁盘重载 — daemon 长期运行中交互模式可能写入了新问题
        self._load_pending_queries()

        if not self._pending_queries:
            return 0

        resolved = 0
        to_keep = []

        for pq in self._pending_queries:
            query_text = pq.get('query_text', '')
            signal = pq.get('signal', 'blind_spot')
            attempts = pq.get('attempts', 0)
            top_cand = pq.get('top_candidate', '')

            if not query_text:
                continue

            # 放弃: 5次失败后移除
            if attempts >= 5:
                log.info(f"  🗑️ 放弃待决: '{query_text[:40]}' (5次尝试失败)")
                continue

            # ★ 插队处理: 深度搜索 + 注入
            log.info(f"  📌 插队处理 [{attempts+1}/5]: '{query_text[:40]}'")
            try:
                deep_pairs = self._arms_search_deep(query_text, top_cand)
                if deep_pairs and self.learner:
                    self.learner.learn_pairs_batch(deep_pairs, learning_rate=0.08)
            except Exception as e:
                log.warning(f"  插队搜索失败: {e}")

            # 重新推理验证
            try:
                chars = re.findall(r'[\u4e00-\u9fff]', query_text)
                if chars:
                    device = next(self.landscape.parameters()).device
                    vecs = []
                    for ch in chars[:10]:
                        idx = self.field._char_to_idx.get(ch)
                        if idx is not None:
                            vecs.append(self.field.anchors[idx].to(device))
                    if vecs:
                        query_vec = torch.stack(vecs).mean(dim=0)
                        verify = self.landscape.infer(
                            query_vec, steps=50, lr=0.02, zichang=self.field
                        )
                        new_signal = verify.get('signal', signal)
                        if new_signal == 'certain':
                            resolved += 1
                            log.info(f"  ✅ 插队解决: '{query_text[:40]}' "
                                     f"信号 {signal}→certain")
                            continue  # 不加入 to_keep = 出队
                        else:
                            log.info(f"  ⏳ 仍为 '{new_signal}'，保留队列")
                else:
                    log.info(f"  ⏭️ 无汉字查询跳过: '{query_text[:40]}'")
                    to_keep.append(pq)  # 保留但标记
                    continue
            except Exception as e:
                log.warning(f"  插队推理失败: {e}")

            # 未解决 → attempts+1 保留
            pq['attempts'] = attempts + 1
            to_keep.append(pq)

        self._pending_queries = to_keep
        self._save_pending_queries()

        if resolved > 0:
            log.info(f"🎉 本轮插队解决: {resolved}个，剩余待决: {len(to_keep)}个")
        return resolved

    def daemon_tick(self, round_num: int) -> Dict[str, Any]:
        """旧守护循环 — 保留兼容, 委托给 daemon_tick_v2"""
        return self.daemon_tick_v2(round_num)

    def daemon_tick_v2(self, round_num: int) -> Dict[str, Any]:
        """
        v2 守护循环: 插队→扫描→搜索→注入→定期调度 统一信号驱动。

        每轮:
          0. ★ 优先插队: 处理交互模式遗留的待决问题 (最高优先级)
          1. 🧠 大脑扫描盲区 → 多因子检测
          2. 🦾 双臂搜索 + 脑当场吸收 → 搜索注入一气呵成
          3. 按轮次定期调度 (衰减/桥接/矛盾解/验证)

        Returns:
            {'scanned': N, 'learned': N, 'pairs_injected': N,
             'pending_resolved': N, 'separation': float, ...}
        """
        tick_report = {'scanned': 0, 'learned': 0, 'pairs_injected': 0,
                       'pending_resolved': 0}

        # ── 0. ★ 优先插队: 交互模式遗留的待决问题 ──
        try:
            tick_report['pending_resolved'] = self.process_pending_queries()
        except Exception as e:
            log.debug(f"  插队处理异常: {e}")

        # ── 1. 大脑扫描: 盲区检测 ──
        try:
            gaps = self._daemon_scan()
            tick_report['scanned'] = len(gaps)
        except Exception as e:
            log.warning(f"  盲区扫描异常: {e}")
            gaps = []

        # ── 2. 双臂搜索 + 脑当场吸收 ──
        if gaps:
            try:
                all_pairs = self._arms_search_batch(gaps[:3])
                if all_pairs and self.learner:
                    inject_result = self.learner.learn_pairs_batch(
                        all_pairs, learning_rate=0.05
                    )
                    tick_report['learned'] = len(gaps[:3])
                    tick_report['pairs_injected'] = inject_result.get(
                        'pairs_learned', len(all_pairs)
                    )
                    tick_report['separation_before'] = inject_result.get(
                        'separation_before', 0
                    )
                    tick_report['separation_after'] = inject_result.get(
                        'separation_after', 0
                    )
                    log.info(f"  🧠 吸收: {tick_report['pairs_injected']}字对 "
                            f"分离度 {tick_report.get('separation_before',0):.1f}→"
                            f"{tick_report.get('separation_after',0):.1f}")
                    # ★ EWC 正则: 拉回锚定参数
                    if hasattr(self.learner, 'ewc_regularize'):
                        try:
                            self.learner.ewc_regularize()
                        except Exception:
                            pass
            except Exception as e:
                log.warning(f"  双臂搜索/注入异常: {e}")

        # ── 3. 定期调度 ──
        # 每轮: 衰减
        try:
            if self.learner:
                decay_result = self.learner.decay_step()
                if decay_result.get('decayed', 0) > 0:
                    log.info(f"  📉 衰减: {decay_result['decayed']}条")
        except Exception:
            pass

        # 每5轮: 序列臂 + 桥接 + 矛盾解 + D-S回写 + 概念图→景观对齐
        if round_num % 5 == 0:
            # ★ 序列臂：学习有向字对方向性（低频运行，不干扰语义臂修复）
            if hasattr(self, '_directed_pairs') and self._directed_pairs:
                try:
                    seq_trained = self._train_sequential_arm()
                    if seq_trained > 0:
                        log.info(f"  序列臂: 训练 {seq_trained} 个有向关联")
                except Exception as e:
                    log.debug(f"  序列臂异常: {e}")
            try:
                import signal as _sig
                def _contra_timeout(signum, frame):
                    raise TimeoutError("矛盾解超时(30s)")
                _sig.signal(_sig.SIGALRM, _contra_timeout)
                _sig.alarm(30)
                try:
                    self._run_contra_safe_v2()
                finally:
                    _sig.alarm(0)
            except Exception as e:
                log.debug(f"  矛盾解异常: {e}")
            try:
                self._run_fuzzy_feedback_v2()
            except Exception as e:
                log.debug(f"  模糊格异常: {e}")
            # ★ 概念图→景观对齐: 每5轮注入高置信度知识
            try:
                self._run_prune_and_align()
            except Exception as e:
                log.debug(f'  对齐异常: {e}')

        # 每10轮: 闭环验证 + 策应器规划
        if round_num % 10 == 0:
            tick_report['plan'] = self._plan_learning_targets_v2()
            try:
                self._run_verify_loop()
            except Exception as e:
                log.debug(f"  验证异常: {e}")

        # 每20轮: 万象格 + 金字塔 (剪枝对齐已移至每5轮)
        if round_num % 20 == 0:
            try:
                tick_report['multiform'] = self._validate_multiform_v2()
            except Exception as e:
                log.debug(f"  万象格异常: {e}")
            try:
                p_result = self._train_pyramid()
                if p_result:
                    tick_report['pyramid'] = p_result
            except Exception as e:
                log.debug(f"  金字塔异常: {e}")

        # 每50轮: EWC Fisher更新 + 锚定参数采样
        if round_num % 50 == 0 and self.learner and hasattr(self.learner, 'update_ewc_fisher'):
            try:
                fisher_result = self.learner.update_ewc_fisher(n_samples=200)
                log.info(f"  ⚓ EWC锚定: {fisher_result['params_anchored']}参数 "
                        f"Fisher质量={fisher_result['total_fisher_mass']:.1f}")
            except Exception as e:
                log.warning(f"  EWC Fisher更新失败: {e}")

        return tick_report

    # ── 大脑扫描 ──

    def _daemon_scan(self) -> list:
        """
        🧠 全因子盲区扫描。使用 MultiFactorDetector 从能量景观拓扑检测
        7类知识缺口，返回优先级队列。
        """
        from loongpearl.learning.blindspot_detector import MultiFactorDetector
        if not hasattr(self, '_detector'):
            self._detector = MultiFactorDetector(
                self.field, self.landscape, None,
                num_partitions=8, from_landscape=True,
            )
        self._detector.scan_all(parallel=True)
        return self._detector.top_gaps(100)

    # ── 双臂搜索 ──

    def _arms_search_batch(self, gaps: list) -> list:
        """
        🦾 对盲区队列批量搜索，提取字对——不积压，直接返回。

        数据源优先级: SQLite加速 → 概念图邻接索引 → 网络搜索(DualExtractor) → 本地词典

        Returns: [(ia, ib), ...] 去重字对索引列表
        """
        all_pairs = []
        for gap in gaps[:3]:
            char = gap.char
            factor = getattr(gap, 'factor', 'unknown')

            # ── ★ 第一数据源: SQLite O(log N) 索引查询 1.93M 三元组 ──
            sqlite_pairs = []
            if hasattr(self, '_cgdb') and self._cgdb:
                try:
                    raw_pairs = self._cgdb.query_char_pairs(char, min_conf=0.1, limit=50)
                    for ca, cb, conf in raw_pairs:
                        ia = self.field._char_to_idx.get(ca)
                        ib = self.field._char_to_idx.get(cb)
                        if ia is not None and ib is not None:
                            sqlite_pairs.append((ia, ib))
                except Exception:
                    pass

            # ── 第二数据源: 概念图邻接索引 O(1) ──
            cg_pairs = []
            if hasattr(self.cg, 'get_char_pairs'):
                cg_pairs = self.cg.get_char_pairs(char, max_pairs=50)

            # 合并去重
            seen_local = set()
            combined = []
            for p in sqlite_pairs + cg_pairs:
                if p not in seen_local:
                    seen_local.add(p)
                    combined.append(p)

            if len(combined) >= 10:
                # 本地数据足够（≥10对）→ 跳过网络搜索
                all_pairs.extend(combined)
                continue

            # 本地不足 → 补充网络搜索（仅1个查询，不再3个）
            all_pairs.extend(combined)

            # 精简查询策略: 只用最有效的1个查询
            query = f"{char} 组词 成语 搭配"
            try:
                search_response = self.searcher.search(query, max_results=3)
                if search_response and search_response.results:
                    pairs = self._extract_pairs_from_search(
                        search_response, query, use_llm=False)
                    all_pairs.extend(pairs)
            except Exception:
                pass

            # 本地词典补充
            try:
                local_pairs = self._extract_pairs_from_local_dicts(char)
                all_pairs.extend(local_pairs)
            except Exception:
                pass

        # 去重
        seen = set()
        unique = []
        for p in all_pairs:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:200]

    def _build_search_queries(self, char: str, factor: str) -> list:
        """根据盲区因子构造多查询策略"""
        if factor == 'dead_end':
            return [f"{char}字开头的成语有哪些", f"{char}成语大全"]
        elif factor == 'gradient':
            return [f"{char} 成语 释义", f"{char}字 组词 常见词语",
                    f"含有{char}字的词语 成语"]
        elif factor == 'semantic':
            return [f"{char} 组词 成语 搭配"]
        elif factor == 'statistical':
            return [f"{char} 成语接龙", f"{char}字 常见词语"]
        elif factor == 'coverage':
            return [f"{char} 成语 用法 释义", f"{char} 组词"]
        elif factor == 'freshness':
            return [f"{char}字 组词 新词语"]
        else:
            return [f"{char} 组词 成语 搭配"]

    # ── 定期调度方法 v2 (修复 API 断裂: triples.keys → forward_index) ──

    def _harvest_from_wikipedia_v2(self) -> Dict:
        """Wikipedia 采集 — 使用 forward_index 选取候选节点"""
        candidates = list(self.cg.forward_index.keys())[:200]
        candidates = [s for s in candidates if len(
            self.cg.forward_index.get(s, {})
        ) >= 3]
        if not candidates:
            return {"collected": 0, "reason": "no candidates"}
        import random
        titles = random.sample(candidates, min(5, len(candidates)))
        try:
            total = self.harvester.harvest_wikipedia(titles=titles, max_per_page=30)
            result = {"collected": total, "titles": titles}
            if total > 0:
                for title in titles[:3]:
                    try:
                        frame = self.sem_parser.parse(title)
                        for c in frame.concepts:
                            if c not in self.cg.forward_index and len(c) >= 2:
                                self.cg.add_triple(c, "RELATED", title[:20],
                                                  confidence=0.3, source="harvest+nlu")
                    except Exception:
                        pass
                result['understood'] = True
            return result
        except Exception as e:
            return {"collected": 0, "error": str(e)}

    def _run_contra_safe_v2(self) -> Dict:
        """矛盾消解 — 使用 forward_index 检查证据"""
        conflicts = self.contra.detect_all()
        if not conflicts:
            return {"detected": 0, "resolved": 0}
        resolved = 0
        for c in conflicts:
            safe_to_purge = True
            for s, r, o, conf in c.involved_triples:
                evidence_count = 0
                if s in self.cg.forward_index:
                    evidence_count = sum(
                        1 for rel, obj, _, _ in self._get_triples_for(s)
                        if rel == r and obj == o
                    )
                if conf > 0.7 and evidence_count > 1:
                    safe_to_purge = False
                    break
            if safe_to_purge:
                self.contra.resolve(c, strategy="confidence_based")
                resolved += 1
            else:
                self.contra.resolve(c, strategy="keep_both")
        purged = self.contra.purge_resolved() if resolved > 0 else 0
        self._round_stats['conflicts_resolved'] += resolved
        return {"detected": len(conflicts), "resolved": resolved,
                "kept_as_disputed": len(conflicts) - resolved, "purged": purged}

    def _run_fuzzy_feedback_v2(self) -> Dict:
        """D-S回写 — 使用 forward_index 迭代节点"""
        added = 0
        for s in list(self.cg.forward_index.keys())[:1000]:
            for rel, obj, conf, src in self._get_triples_for(s):
                if 0.2 < conf < 0.7:
                    self.fuzzy.add_evidence(s, rel, obj, source=src, mass=conf)
                    added += 1
                    if added >= 200:
                        break
            if added >= 200:
                break
        self._sync_fuzzy_to_cg()
        fg_stats = self.fuzzy.stats()
        return {"evidences_added": added, "propositions": fg_stats['propositions'],
                "d_s_feedbacks": self._round_stats['d_s_feedbacks']}

    def _plan_learning_targets_v2(self) -> Dict:
        """策应器学习规划 — 使用 forward_index"""
        low_degree_nodes = []
        for s in list(self.cg.forward_index.keys())[:5000]:
            deg = len(self.cg.forward_index.get(s, {}))
            if 5 <= deg <= 20:
                low_degree_nodes.append(s)
        if not low_degree_nodes:
            return {"targets": 0}
        import random
        targets = random.sample(low_degree_nodes, min(5, len(low_degree_nodes)))
        plans = []
        for t in targets:
            try:
                plan = self.planner.plan(f"什么是{t}")
                plans.append({"concept": t, "query_type": plan.query_type,
                             "subtasks": len(plan.tasks)})
            except Exception:
                pass
        return {"targets": len(targets), "plans": plans[:3]}

    def _validate_multiform_v2(self) -> Dict:
        """万象格验证 — 使用 forward_index"""
        s = self.mkg.stats()
        missing = 0
        for name, process in self.mkg.processes.items():
            for step in process.steps:
                if step.action not in self.cg.forward_index:
                    self.cg.add_triple(step.action, "PART_OF", name,
                                      confidence=0.3, source="multiform_validate")
                    missing += 1
        timeline_missing = 0
        for name, tl in self.mkg.timelines.items():
            for ev in tl.events:
                if ev.name not in self.cg.forward_index:
                    self.cg.add_triple(ev.name, "OCCURS_IN", name,
                                      confidence=0.3, source="multiform_validate")
                    timeline_missing += 1
        return {"total_form_knowledge": s['total_form_knowledge'],
                "process_steps_missing": missing,
                "timeline_events_missing": timeline_missing,
                "written_back": missing + timeline_missing}

    def _run_verify_loop(self):
        """闭环验证 — 验证低置信度三元组"""
        try:
            from loongpearl.learning.verify_loop import VerifyLoop
            vf = VerifyLoop(self.cg)
            v_report = vf.verify_lowest_confidence(n=5)
            if v_report['confirmed'] > 0 or v_report['contradicted'] > 0:
                log.info(f"  🔄 闭环验证: 确认{v_report['confirmed']}条 "
                        f"矛盾{v_report['contradicted']}条")
        except Exception:
            pass

    def _run_prune_and_align(self):
        """剪枝 + 概念图→景观对齐"""
        try:
            removed = self.cg.prune(min_confidence=0.1, min_evidence=0)
            if removed > 0:
                log.info(f"  ✂️ 剪枝: 移除{removed}条低质量三元组, "
                        f"剩余{self.cg.total_triples}条")

            # 对齐: 高置信度概念图知识 → 能量景观 (每5轮2000对)
            pairs = self.cg.align_to_landscape(
                min_confidence=0.5, max_pairs=2000
            )
            if pairs and self.learner:
                self.learner.learn_pairs_batch(pairs, learning_rate=0.01)
                log.info(f"  🎯 知识对齐: {len(pairs)}对概念映射到能量景观")
                # ★ EWC 正则
                if hasattr(self.learner, 'ewc_regularize'):
                    try:
                        self.learner.ewc_regularize()
                    except Exception:
                        pass
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    # 序列臂：有向字对方向性训练
    # ═══════════════════════════════════════════════════════════════════

    def _load_directed_pairs(self):
        """加载有向字对（由外部脚本从 POETIC_NEXT 提取）"""
        import json
        from collections import defaultdict
        
        pairs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'models', 'directed_pairs.json'
        )
        
        if not os.path.exists(pairs_path):
            log.warning(f"有向字对文件不存在: {pairs_path}，序列臂禁用")
            self._directed_pairs = []
            self._anchor_to_positives = {}
            return
        
        with open(pairs_path, 'r') as f:
            self._directed_pairs = json.load(f)
        
        # 构建"锚点→正样本"映射
        self._anchor_to_positives = defaultdict(list)
        for src, tgt, conf in self._directed_pairs:
            self._anchor_to_positives[src].append(tgt)
        
        log.info(f"  序列臂: 已加载 {len(self._directed_pairs)} 对有向字对, "
                f"{len(self._anchor_to_positives)} 个锚点")

    def _train_sequential_arm(self) -> int:
        """
        序列臂训练：学习有向字对的方向性。
        
        非对称 Hebbian 注入：在锚点→正样本连线上建立下降梯度，
        在锚点→负样本中间推高能量。
        
        Returns:
            训练的字对数量
        """
        import random
        
        if not hasattr(self, '_directed_pairs') or not self._directed_pairs:
            return 0
        if not self.learner or not hasattr(self.learner, 'update_point'):
            return 0
        
        anchor_to_positives = self._anchor_to_positives
        hanzi_list = self.field.hanzi_list
        high_freq_chars = list(hanzi_list[:3500])
        
        # 每轮最多处理 200 个锚点
        anchor_items = list(anchor_to_positives.items())
        if len(anchor_items) > 200:
            anchor_items = random.sample(anchor_items, 200)
        
        trained = 0
        
        for anchor, positives in anchor_items:
            if anchor not in self.field._char_to_idx:
                continue
            
            anchor_idx = self.field._char_to_idx[anchor]
            anchor_vec = self.field.anchors[anchor_idx]
            
            for pos in positives[:3]:  # 每锚点最多 3 个正样本
                if pos not in self.field._char_to_idx:
                    continue
                
                pos_idx = self.field._char_to_idx[pos]
                pos_vec = self.field.anchors[pos_idx]
                
                # 构建负样本：高频字中排除锚点和正样本
                neg_pool = [c for c in high_freq_chars
                           if c != anchor and c not in positives][:50]
                if not neg_pool:
                    continue
                
                neg_char = random.choice(neg_pool)
                neg_idx = self.field._char_to_idx[neg_char]
                neg_vec = self.field.anchors[neg_idx]
                
                # 非对称 Hebbian：在 anchor→pos 连线上建下降梯度
                for alpha in [0.3, 0.5, 0.7, 0.85, 0.95]:
                    point = anchor_vec * (1 - alpha) + pos_vec * alpha
                    point = torch.nn.functional.normalize(point, dim=-1)
                    target_energy = -0.3 * alpha
                    self.learner.update_point(point, target_energy)
                
                # 负样本推高：anchor→neg 中点能量升高
                midpoint_neg = (anchor_vec + neg_vec) / 2.0
                midpoint_neg = torch.nn.functional.normalize(midpoint_neg, dim=-1)
                self.learner.update_point(midpoint_neg, target_energy=0.5)
                
                trained += 1
        
        return trained

    # ═══════════════════════════════════════════════════════════════════
    # 状态报告
    # ═══════════════════════════════════════════════════════════════════

    def status_report(self) -> str:
        cg_stats = self.cg.stats() if hasattr(self.cg, 'stats') else {}
        fg_stats = self.fuzzy.stats() if self._fuzzy else {}
        mkg_stats = self.mkg.stats() if self._mkg else {}
        ml_stats = self.multilang.stats() if self._multilang else {}
        lines = [
            "═══ 龙珠调度器 v3 状态 ═══",
            f"运行时间: {time.time() - self._init_time:.0f}s",
            f"信号迭代上限: {self.MAX_SIGNAL_ITERATIONS}",
            "",
            "📊 概念图:",
            f"  节点: {cg_stats.get('nodes', 'N/A')}  三元组: {cg_stats.get('triples', 'N/A')}",
            "",
            "🎲 模糊格(D-S):",
            f"  命题: {fg_stats.get('propositions', 0)}  证据: {fg_stats.get('total_evidences', 0)}",
            f"  高置信命题: {fg_stats.get('high_confidence_props', 0)}",
            "",
            "📐 万象格:",
            f"  过程: {mkg_stats.get('processes', 0)}  条件: {mkg_stats.get('conditionals', 0)}",
            f"  反事实: {mkg_stats.get('counterfactuals', 0)}  时间线: {mkg_stats.get('timelines', 0)}",
            "",
            "🌐 万语锚:",
            f"  跨语言概念: {ml_stats.get('total_concepts', 0)}",
            "",
            "📈 本轮统计:",
            f"  采集: {self._round_stats['harvested_triples']}",
            f"  冲突消解: {self._round_stats['conflicts_resolved']}",
            f"  D-S回写: {self._round_stats['d_s_feedbacks']}",
            f"  跨语言查: {self._round_stats['cross_lang_lookups']}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════

def create_orchestrator(field, landscape, learner=None) -> Orchestrator:
    """从已加载的模型创建一个调度器实例"""
    from loongpearl.core.concept_graph import ConceptGraph
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CONCEPT_GRAPH_BASE = os.path.join(_project_root, 'data', 'models', 'concept_graph')
    FALLBACK_BASE = os.path.join(_project_root, 'loongpearl', 'data', 'models', 'concept_graph')
    cg = ConceptGraph(field, landscape)
    
    # 尝试加载概念图（主路径 → 备份路径）
    loaded = False
    for base in [CONCEPT_GRAPH_BASE, FALLBACK_BASE]:
        if not os.path.exists(base + '.json'):
            continue
        try:
            cg.load(base)
            loaded = True
            if base != CONCEPT_GRAPH_BASE:
                log.info(f"概念图从备份加载: {base}.json")
            break
        except Exception as e:
            log.warning(f"概念图加载失败 ({base}): {e}")
    
    if not loaded:
        log.warning("⚠️ 概念图加载失败且无备份，以空概念图启动（仅在线学习可用）")
    elif cg.total_triples == 0:
        log.warning("⚠️ 概念图为空（0三元组），以空概念图启动")
    
    return Orchestrator(field, landscape, cg, learner)


def create_orchestrator_with_sequential(field, landscape, learner=None) -> Orchestrator:
    """创建调度器并加载序列臂有向字对 + SQLite 查询加速 + DualExtractor"""
    orch = create_orchestrator(field, landscape, learner)
    orch._load_directed_pairs()

    # ★ SQLite 查询加速层
    try:
        from loongpearl.core.concept_graph_sqlite import ConceptGraphSQLite
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(_project_root, 'data', 'models', 'concept_graph.db')
        orch._cgdb = ConceptGraphSQLite(db_path)
        orch._cgdb.create_tables()
        n = orch._cgdb.count_triples()
        log.info(f"  SQLite加速: {n}条三元组索引就绪")
    except Exception as e:
        log.warning(f"  SQLite加速初始化失败: {e}")
        orch._cgdb = None

    # ★ DualExtractor: 正则+LLM双重知识提取
    try:
        from loongpearl.learning.dual_extractor import DualExtractor
        orch._dual_extractor = DualExtractor()
        log.info("  DualExtractor: 正则+LLM双重提取就绪")
    except Exception as e:
        log.warning(f"  DualExtractor初始化失败: {e}")
        orch._dual_extractor = None

    return orch
