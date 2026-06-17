#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠调度器 (Orchestrator) — 8引擎咬合中枢
═══════════════════════════════════════════════════

统一管理 8 个引擎的生命周期、数据流和反馈闭环。
替代散装的 monkey-patch，确保每个引擎的输出成为下一个引擎的输入。

数据流:
  Harvester(外部采集) → 概念图 → ContraResolver(消解冲突)
                                         ↓
  用户查询 → SemParser → TaskPlanner → CG查询 → FuzzyGraph(D-S置信度)
                                         ↓
  MultiLang(跨语言路由) ←→ EnergyDecoder(渲染输出)


守护循环中的引擎调度:
  每轮:    Harvester 从 Wikipedia 真采集 + SemParser 理解材料
  每5轮:   ContraResolver 检测消解 + FuzzyGraph D-S回写CG
  每10轮:  验证闭环 + TaskPlanner 规划下半场学习目标  
  每20轮:  MultiFormKG 跨格式验证 + 剪枝对齐
"""

import sys
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

log = logging.getLogger('orchestrator')


class Orchestrator:
    """
    龙珠调度器 — 8引擎的中央神经系统。

    所有权:
      - 所有引擎共享同一个概念图引用 (杜绝幽灵引用)
      - D-S 融合结果自动回写概念图置信度
      - 矛盾消解直接在概念图上生效
    """

    def __init__(self, field, landscape, concept_graph, learner=None):
        self.field = field
        self.landscape = landscape
        self.cg = concept_graph  # ← 单一引用，所有引擎共享
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

        self._init_time = time.time()
        self._round_stats = {
            'harvested_triples': 0,
            'conflicts_resolved': 0,
            'd_s_feedbacks': 0,
            'cross_lang_lookups': 0,
        }

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
            from loongpearl.core.energy_decoder import EnergyDecoder
            self._decoder = EnergyDecoder()
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

    # ═══════════════════════════════════════════════════════════════════
    # 对话全链路 (chat模式)
    # ═══════════════════════════════════════════════════════════════════

    def dialogue(self, query: str) -> Dict[str, Any]:
        """
        全栈对话 — 四层路由:
          1. 对话引擎 (社交/闲聊/上下文)
          2. 创意引擎 (诗词/接龙/叙事)
          3. 知识查询 (NLU→CG→NLG)
          4. 兜底

        Returns:
            {"output": "...", "debug": {...}, "type": "..."}
        """
        result = {"output": "", "debug": {}, "type": "unknown"}

        # ── 路由1: 对话引擎 (社交 + 闲聊 + 上下文) ──
        conv_result = self.conversation.respond(query)
        if conv_result["type"] in ("social", "chitchat"):
            result["output"] = conv_result["output"]
            result["type"] = conv_result["type"]
            return result

        # 对话引擎可能增强了查询（上下文继承、代词消解）
        if conv_result.get("enhanced_query") and conv_result["enhanced_query"] != query:
            query = conv_result["enhanced_query"]
            result["debug"]["context_enhanced"] = True

        # ── 路由2: 创意引擎 (诗词/接龙/叙事) ──
        creative_output = self.creative.handle(query)
        if creative_output:
            result["output"] = creative_output
            result["type"] = "creative"
            result["debug"]["creative_type"] = (
                "poetry" if "《" in creative_output else
                "idiom_chain" if "→" in creative_output else
                "narrative"
            )
            # 更新对话状态
            self.conversation.state.add_turn(query, creative_output)
            return result

        # ── 路由3: 知识查询 (原有全栈) ──
        result["type"] = "knowledge"

        # 🔑 用户查询中的概念 → 自动汇入知识获取管线
        for concept in frame.concepts[:5]:
            if concept and len(concept) >= 2:
                self.pipeline.feed_user_concept(concept, context=query)

        # 语言检测 → 跨语言路由
        lang = self.multilang.detect_language(query)
        result['debug']['lang'] = lang

        if lang != 'zh':
            # 英文查询 → 映射到中文概念
            cids = self.multilang.map_to_concepts(query, lang=lang)
            if cids:
                zh_name = self.multilang.get_concept_name(cids[0], 'zh')
                if zh_name:
                    result['debug']['cross_lang'] = f"{query} → {zh_name}"
                    self._round_stats['cross_lang_lookups'] += 1
                    query = zh_name  # 替换为中文概念名继续处理

        # Step 1: NLU — 解义器
        frame = self.sem_parser.parse(query)
        result['debug']['frame'] = {
            'type': frame.question_type.name if frame.question_type else '陈述',
            'intent': frame.intent.name if frame.intent else 'N/A',
            'subject': frame.subject,
            'object': frame.object,
            'concepts': frame.concepts,
        }

        # Step 2: 规划 — 策应器
        plan = self.planner.plan(query)
        result['debug']['plan'] = {
            'type': plan.query_type,
            'tasks': len(plan.tasks),
        }

        # Step 3: 执行 — 概念图查询
        exec_results = self.planner.execute(plan)
        result['debug']['exec'] = {
            'results': len(exec_results.get('results', {})),
        }

        # Step 4: 跨格式查询 — 万象格
        for concept in frame.concepts[:3]:
            cross = self.mkg.reason_across_forms(concept)
            if any(v for v in cross.values()):
                result['debug']['cross_form'] = {
                    c: bool(v) for c, v in cross.items()
                }
                break

        # Step 5: 模糊格 — D-S 置信度验证 + 回写概念图
        if frame.intent and frame.intent.name in ('CHECK_TRUTH', 'FIND_PATH', 'DEFINE'):
            for concept in frame.concepts[:3]:
                if concept in self.cg.triples:
                    for rel, obj, conf, src in self.cg.triples[concept][:5]:
                        self.fuzzy.add_evidence(concept, rel, obj, 
                                                source=src, mass=conf)

        # D-S 融合后回写概念图置信度
        self._sync_fuzzy_to_cg()

        # Step 6: NLG — 化能器渲染
        render_input = self._build_render_input(frame, exec_results, query)
        result['output'] = self.decoder.render(render_input)

        return result

    def _build_render_input(self, frame, exec_results, query) -> Dict:
        """从执行结果构造化能器输入"""
        agg = exec_results.get('aggregated', {})

        if agg.get('comparison'):
            return {
                "render_type": "compare",
                "compare_subjects": frame.concepts[:2],
                "facts": agg['comparison'].get('common', []) +
                         agg['comparison'].get('only_a', []) +
                         agg['comparison'].get('only_b', []),
            }
        elif agg.get('paths'):
            paths = agg['paths']
            edge_list = [{"rel": "RELATED", "confidence": 0.5}
                         for _ in range(len(paths[0]) - 1)] if paths else []
            return {
                "render_type": "explain_path",
                "subject": frame.subject or query,
                "path": paths[0] if paths else [query],
                "edges": edge_list,
            }
        elif agg.get('facts'):
            return {
                "render_type": "list_related",
                "subject": frame.subject or query,
                "facts": agg['facts'],
            }
        elif agg.get('table'):
            return {
                "render_type": "table",
                "facts": agg['table'],
            }

        return {
            "render_type": "fact_statement",
            "facts": [
                {"relation": "RELATED", "object": c,
                 "subject": frame.subject or query}
                for c in frame.concepts[:3]
            ],
        }

    # ═══════════════════════════════════════════════════════════════════
    # D-S 反馈闭环 — 模糊格 → 概念图
    # ═══════════════════════════════════════════════════════════════════

    def _sync_fuzzy_to_cg(self):
        """将 D-S 融合后的置信度回写到概念图"""
        count = 0
        for (s, r, o), bpa in self.fuzzy._bpas.items():
            if bpa.combined_mass > 0:
                # 更新概念图中的置信度
                if s in self.cg.triples:
                    for i, (rel, obj, conf, src) in enumerate(self.cg.triples[s]):
                        if rel == r and obj == o:
                            # 用 D-S 融合后的置信度替换原始置信度
                            new_triple = (rel, obj, bpa.combined_mass, 
                                         f"{src}+DS")
                            self.cg.triples[s][i] = new_triple
                            count += 1
        if count:
            self._round_stats['d_s_feedbacks'] += count
            log.debug(f"  🔄 D-S回写: {count} 条置信度更新")

    # ═══════════════════════════════════════════════════════════════════
    # 守护循环 — 引擎调度
    # ═══════════════════════════════════════════════════════════════════

    def daemon_tick(self, round_num: int) -> Dict[str, Any]:
        """
        守护循环的一轮调度。

        核心变更: 用 KnowledgePipeline 替代旧的分散采集。
        
        调度表:
          每轮:    pipeline.tick() — 检测需求+多源采集+精炼注入
          每5轮:   ContraResolver + FuzzyGraph D-S回写
          每10轮:  TaskPlanner 学习目标规划
          每20轮:  MultiFormKG 跨格式验证
        """
        tick_report = {}

        # ── 每轮: 知识获取管线 (检测→采集→精炼→注入) ──
        if round_num % 2 == 0:
            try:
                pipe_result = self.pipeline.tick(max_demands=5, max_acquire=3)
                if pipe_result['acquired'] > 0:
                    tick_report['pipeline'] = pipe_result
            except Exception as e:
                log.debug(f"  管线异常: {e}")

        # ── 每5轮: 矛盾检测消解 + 模糊格闭环 ──
        if round_num % 5 == 0:
            tick_report['contra'] = self._run_contra_safe()
            tick_report['fuzzy'] = self._run_fuzzy_feedback()

        # ── 每10轮: 策应器规划学习目标 ──
        if round_num % 10 == 0:
            tick_report['plan'] = self._plan_learning_targets()

        # ── 每20轮: 万象格跨格式验证 ──
        if round_num % 20 == 0:
            tick_report['multiform'] = self._validate_multiform()

        return tick_report

    # ── 万象收: 真 Wikipedia 采集 ──
    def _harvest_from_wikipedia(self) -> Dict:
        """从 Wikipedia 采集与概念图相关的条目"""
        # 从概念图中选高频、高置信度的节点作为采集目标
        candidates = []
        for s in list(self.cg.triples.keys())[:200]:
            deg = len(self.cg.triples.get(s, []))
            if deg >= 3:  # 度≥3的节点才有足够上下文
                candidates.append(s)

        if not candidates:
            return {"collected": 0, "reason": "no candidates"}

        # 选取前5个不同的概念作为 Wikipedia 标题
        import random
        titles = random.sample(candidates, min(5, len(candidates)))

        try:
            total = self.harvester.harvest_wikipedia(
                titles=titles,
                max_per_page=30
            )
            result = {"collected": total, "titles": titles}

            # 🔑 关键: 采集后立即用解义器理解新文本
            if total > 0:
                self._understand_new_knowledge(titles)
                result['understood'] = True

            return result
        except Exception as e:
            return {"collected": 0, "error": str(e)}

    def _understand_new_knowledge(self, titles: List[str]):
        """用解义器理解新采集的知识"""
        for title in titles[:3]:
            try:
                frame = self.sem_parser.parse(title)
                # 将解析出的概念注入概念图
                for c in frame.concepts:
                    if c not in self.cg.triples and len(c) >= 2:
                        self.cg.add_triple(c, "RELATED", title[:20], 
                                          confidence=0.3, source="harvest+nlu")
            except Exception:
                pass

    # ── 矛盾解: 安全消解 ──
    def _run_contra_safe(self) -> Dict:
        """安全矛盾检测消解 — 带保护锁"""
        conflicts = self.contra.detect_all()
        if not conflicts:
            return {"detected": 0, "resolved": 0}

        resolved = 0
        for c in conflicts:
            # 🔒 安全锁: 高置信度+多证据的冲突不自动消解
            safe_to_purge = True
            for s, r, o, conf in c.involved_triples:
                evidence_count = 0
                if s in self.cg.triples:
                    evidence_count = sum(1 for rel, obj, _, _ in self.cg.triples[s]
                                        if rel == r and obj == o)
                # 置信度 > 0.7 且 证据数 > 1 → 不purge，标记争议
                if conf > 0.7 and evidence_count > 1:
                    safe_to_purge = False
                    break

            if safe_to_purge:
                self.contra.resolve(c, strategy="confidence_based")
                resolved += 1
            else:
                self.contra.resolve(c, strategy="keep_both")

        # 只 purge 安全消解的
        purged = self.contra.purge_resolved() if resolved > 0 else 0
        self._round_stats['conflicts_resolved'] += resolved

        return {
            "detected": len(conflicts),
            "resolved": resolved,
            "kept_as_disputed": len(conflicts) - resolved,
            "purged": purged,
        }

    # ── 模糊格: D-S 反馈闭环 ──
    def _run_fuzzy_feedback(self) -> Dict:
        """运行 D-S 证据融合并回写概念图"""
        # 对低置信度三元组添加证据
        added = 0
        for s in list(self.cg.triples.keys())[:1000]:
            for rel, obj, conf, src in self.cg.triples.get(s, []):
                if 0.2 < conf < 0.7:  # 中等置信度才值得重评估
                    self.fuzzy.add_evidence(s, rel, obj, source=src, mass=conf)
                    added += 1
                    if added >= 200:
                        break
            if added >= 200:
                break

        # D-S 融合后回写
        self._sync_fuzzy_to_cg()

        fg_stats = self.fuzzy.stats()
        return {
            "evidences_added": added,
            "propositions": fg_stats['propositions'],
            "d_s_feedbacks": self._round_stats['d_s_feedbacks'],
        }

    # ── 策应器: 学习目标规划 ──
    def _plan_learning_targets(self) -> Dict:
        """用策应器规划下一阶段的学习目标"""
        # 找连接稀疏但度高的节点（枢纽但冷门）
        low_degree_nodes = []
        for s in list(self.cg.triples.keys())[:5000]:
            deg = len(self.cg.triples.get(s, []))
            if deg >= 5 and deg <= 20:
                low_degree_nodes.append(s)

        if not low_degree_nodes:
            return {"targets": 0}

        # 用策应器规划"对这些概念需要补充什么"
        import random
        targets = random.sample(low_degree_nodes, min(5, len(low_degree_nodes)))
        plans = []
        for t in targets:
            try:
                plan = self.planner.plan(f"什么是{t}")
                plans.append({
                    "concept": t,
                    "query_type": plan.query_type,
                    "subtasks": len(plan.tasks),
                })
            except Exception:
                pass

        return {
            "targets": len(targets),
            "plans": plans[:3],
        }

    # ── 万象格: 跨格式验证 ──
    def _validate_multiform(self) -> Dict:
        """验证万象格数据与概念图的一致性"""
        s = self.mkg.stats()

        # 检查: 万象格中的过程步骤是否在概念图中有对应节点
        missing = 0
        for name, process in self.mkg.processes.items():
            for step in process.steps:
                if step.action not in self.cg.triples:
                    missing += 1

        # 检查: 时间线事件是否在概念图中
        timeline_missing = 0
        for name, tl in self.mkg.timelines.items():
            for ev in tl.events:
                if ev.name not in self.cg.triples:
                    timeline_missing += 1

        return {
            "total_form_knowledge": s['total_form_knowledge'],
            "process_steps_missing": missing,
            "timeline_events_missing": timeline_missing,
        }

    # ═══════════════════════════════════════════════════════════════════
    # 状态报告
    # ═══════════════════════════════════════════════════════════════════

    def status_report(self) -> str:
        """生成人类可读的状态报告"""
        cg_stats = self.cg.stats() if hasattr(self.cg, 'stats') else {}
        fg_stats = self.fuzzy.stats() if self._fuzzy else {}
        mkg_stats = self.mkg.stats() if self._mkg else {}
        ml_stats = self.multilang.stats() if self._multilang else {}

        lines = [
            "═══ 龙珠调度器状态 ═══",
            f"运行时间: {time.time() - self._init_time:.0f}s",
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
            f"  语言覆盖: {ml_stats.get('language_coverage', {})}",
            "",
            "📈 本轮统计:",
            f"  采集: {self._round_stats['harvested_triples']}",
            f"  冲突消解: {self._round_stats['conflicts_resolved']}",
            f"  D-S回写: {self._round_stats['d_s_feedbacks']}",
            f"  跨语言查: {self._round_stats['cross_lang_lookups']}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 工厂函数 — 从 loong_main 模型创建调度器
# ═══════════════════════════════════════════════════════════════════════

def create_orchestrator(field, landscape, learner=None) -> Orchestrator:
    """从已加载的模型创建一个调度器实例"""
    from loongpearl.core.concept_graph import ConceptGraph

    # 项目根目录 (orchestrator.py 在 loongpearl/core/ 下，上溯两级)
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CONCEPT_GRAPH_BASE = os.path.join(_project_root, 'data', 'models', 'concept_graph')
    cg = ConceptGraph(field, landscape)
    if os.path.exists(CONCEPT_GRAPH_BASE + '.json'):
        try:
            cg.load(CONCEPT_GRAPH_BASE)
        except Exception as e:
            log.warning(f"概念图加载失败: {e}")

    return Orchestrator(field, landscape, cg, learner)
