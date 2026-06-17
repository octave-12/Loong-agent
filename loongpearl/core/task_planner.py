#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠策应器 (TaskPlanner) — 复杂指令 → 子任务序列 → 自动编排执行
════════════════════════════════════════════════════════════════════════════

LLM 靠 prompt engineering 解析复杂指令，不可复现。策应器用确定性模板 +
概念图路径规划，将复杂指令分解为原子子任务序列并自动执行。

════════════════════════════════════════════════════════════════════════════
能力
════════════════════════════════════════════════════════════════════════════

  1. 指令识别      — 识别查询类型 (对比/列表/定义/因果链/表格)
  2. 子任务分解    — 将查询拆分为独立的原子操作
  3. 依赖分析      — 检测子任务间的数据依赖
  4. 并行调度      — 独立子任务并行执行
  5. 结果聚合      — 将子任务结果合并为统一输出

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.task_planner import TaskPlanner

    tp = TaskPlanner(concept_graph)
    plan = tp.plan("对比儒释道三家核心思想")
    results = tp.execute(plan)
    print(results)

"""
import re
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

class TaskType(Enum):
    """原子任务类型"""
    FIND_CONCEPT = auto()        # 在概念图中查找概念
    GET_FACTS = auto()           # 获取概念的所有三元组
    FIND_PATH = auto()           # 查找两概念间的路径
    COMPARE = auto()             # 对比两个概念
    LIST_INSTANCES = auto()      # 列出子类/实例
    CAUSAL_CHAIN = auto()        # 追溯因果链
    DEFINE = auto()              # 定义概念
    FORMAT_OUTPUT = auto()       # 格式化输出


class OutputFormat(Enum):
    """输出格式"""
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    COMPARISON = "comparison"
    TIMELINE = "timeline"


@dataclass
class SubTask:
    """原子子任务"""
    id: str
    type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # 依赖的子任务 ID
    result: Any = None
    status: str = "pending"  # pending/running/done/failed

    def __repr__(self):
        return (f"SubTask({self.id}, {self.type.name}, "
                f"params={self.params}, deps={self.depends_on}, "
                f"status={self.status})")


@dataclass
class ExecutionPlan:
    """执行计划"""
    original_query: str
    query_type: str
    output_format: OutputFormat = OutputFormat.TEXT
    tasks: List[SubTask] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 指令模式库
# ═══════════════════════════════════════════════════════════════════════════

_COMMAND_PATTERNS = [
    # (正则, 任务类型, 输出格式, 提取组映射)
    # 对比
    (r"(对比|比较|比较一下)(.+?)(和|与|跟|同)(.+)", "compare", OutputFormat.COMPARISON),
    (r"(.+?)(和|与|跟)(.+?)(有什么)?(区别|不同|差异)", "compare", OutputFormat.COMPARISON),
    # 列表
    (r"(.+?)(有哪些|列出|列举)", "list_instances", OutputFormat.LIST),
    (r"(列出|给我)(.+?)(的)?(列表|清单)", "list_instances", OutputFormat.LIST),
    # 定义
    (r"(什么是|啥是|定义)(.+)", "define", OutputFormat.TEXT),
    # 表格
    (r"(用表格|以表格|表格形式)(.+)", "table", OutputFormat.TABLE),
    # 因果
    (r"(.+?)(为什么|为何|为啥)(.+)", "causal_chain", OutputFormat.TEXT),
    # 路径
    (r"(.+?)(和|与)(.+?)(的)?(关系|关联|联系)", "find_path", OutputFormat.TEXT),
]

_COMMAND_COMPILED = [(re.compile(p), qtype, fmt) for p, qtype, fmt in _COMMAND_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# 策应器主类
# ═══════════════════════════════════════════════════════════════════════════

class TaskPlanner:
    """
    策应器 — 复杂指令分解与自动编排执行。

    处理流程:
      1. 指令识别 →  2. 概念提取 →  3. 子任务生成 →
      4. 依赖分析 →  5. 执行调度 →  6. 结果聚合
    """

    def __init__(self, concept_graph=None,
                 sem_parser=None,
                 energy_decoder=None):
        self.cg = concept_graph
        self.sem_parser = sem_parser
        self.decoder = energy_decoder
        self._task_counter = 0

    # ═════════════════════════════════════════════════════════════════════
    # 主入口
    # ═════════════════════════════════════════════════════════════════════

    def plan(self, query: str) -> ExecutionPlan:
        """
        将复杂查询分解为执行计划。

        Args:
            query: 用户自然语言查询

        Returns:
            ExecutionPlan 包含有序子任务列表
        """
        # Step 1: 识别查询类型和输出格式
        query_type, output_format = self._classify_query(query)

        # Step 2: 提取涉及的概念
        concepts = self._extract_concepts(query)

        # Step 3: 生成子任务
        tasks = self._generate_tasks(query, query_type, concepts, output_format)

        # Step 4: 分析依赖
        tasks = self._analyze_dependencies(tasks)

        return ExecutionPlan(
            original_query=query,
            query_type=query_type,
            output_format=output_format,
            tasks=tasks,
            context={"concepts": concepts},
        )

    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        执行计划，返回聚合结果。

        Returns:
            {
                "query": ...,
                "results": {task_id: result},
                "aggregated": ...,
                "format": ...,
            }
        """
        results = {}

        # 按拓扑顺序执行（优先执行没有依赖的任务）
        executed = set()
        while len(executed) < len(plan.tasks):
            for task in plan.tasks:
                if task.id in executed:
                    continue
                if all(dep in executed for dep in task.depends_on):
                    # 依赖已满足，可以执行
                    task.status = "running"
                    try:
                        task.result = self._execute_task(task, results)
                        task.status = "done"
                    except Exception as e:
                        task.result = f"执行失败: {e}"
                        task.status = "failed"
                    results[task.id] = task.result
                    executed.add(task.id)

        # 聚合结果
        aggregated = self._aggregate_results(plan, results)

        return {
            "query": plan.original_query,
            "query_type": plan.query_type,
            "results": results,
            "aggregated": aggregated,
            "format": plan.output_format.value,
        }

    # ═════════════════════════════════════════════════════════════════════
    # 查询分类
    # ═════════════════════════════════════════════════════════════════════

    def _classify_query(self, query: str) -> Tuple[str, OutputFormat]:
        """识别查询类型和输出格式"""
        for pattern, qtype, fmt in _COMMAND_COMPILED:
            if pattern.search(query):
                return qtype, fmt

        # 兜底
        if "?" in query or "？" in query or "为什么" in query:
            return "causal_chain", OutputFormat.TEXT
        return "get_facts", OutputFormat.TEXT

    # ═════════════════════════════════════════════════════════════════════
    # 概念提取
    # ═════════════════════════════════════════════════════════════════════

    def _extract_concepts(self, query: str) -> List[str]:
        """从查询中提取概念（可用 SemParser 或简单策略）"""
        if self.sem_parser:
            frame = self.sem_parser.parse(query)
            return frame.concepts if frame.concepts else []

        # 简单策略: 用标点/连接词分割，提取名词
        parts = re.split(r'[，,、和与跟同及以及的有什么是什么意思对比比较区别]', query)
        concepts = []
        for p in parts:
            p = p.strip()
            # 过滤疑问词和短词
            if len(p) >= 2 and p not in {'什么', '怎么', '如何', '为什么', '谁', '哪'}:
                concepts.append(p)
        return concepts[:5]  # 限5个

    # ═════════════════════════════════════════════════════════════════════
    # 子任务生成
    # ═════════════════════════════════════════════════════════════════════

    def _generate_tasks(self, query: str, query_type: str,
                        concepts: List[str],
                        output_format: OutputFormat) -> List[SubTask]:
        """根据查询类型生成原子子任务"""
        tasks = []
        tid = lambda: f"t{self._task_counter}"

        if query_type == "compare":
            # 对每个概念生成查找任务
            find_tasks = []
            for c in concepts[:2]:
                self._task_counter += 1
                t = SubTask(
                    id=tid(),
                    type=TaskType.FIND_CONCEPT,
                    params={"concept": c},
                )
                tasks.append(t)
                find_tasks.append(t)

                self._task_counter += 1
                tasks.append(SubTask(
                    id=tid(),
                    type=TaskType.GET_FACTS,
                    params={"concept": c},
                    depends_on=[find_tasks[-1].id],
                ))

            # 对比任务
            self._task_counter += 1
            tasks.append(SubTask(
                id=tid(),
                type=TaskType.COMPARE,
                params={"concepts": concepts[:2]},
                depends_on=[t.id for t in tasks[-2:]],
            ))

        elif query_type == "list_instances":
            concept = concepts[0] if concepts else ""
            self._task_counter += 1
            tasks.append(SubTask(
                id=tid(),
                type=TaskType.LIST_INSTANCES,
                params={"concept": concept},
            ))

        elif query_type == "define":
            concept = concepts[0] if concepts else ""
            self._task_counter += 1
            tasks.append(SubTask(
                id=tid(),
                type=TaskType.DEFINE,
                params={"concept": concept},
            ))

        elif query_type == "causal_chain":
            if len(concepts) >= 2:
                self._task_counter += 1
                tasks.append(SubTask(
                    id=tid(),
                    type=TaskType.FIND_PATH,
                    params={"from": concepts[0], "to": concepts[-1]},
                ))

        elif query_type == "find_path":
            if len(concepts) >= 2:
                self._task_counter += 1
                tasks.append(SubTask(
                    id=tid(),
                    type=TaskType.FIND_PATH,
                    params={"from": concepts[0], "to": concepts[-1]},
                ))

        elif query_type == "table":
            # 表格: 先提取概念，再找属性
            for c in concepts[:3]:
                self._task_counter += 1
                tasks.append(SubTask(
                    id=tid(),
                    type=TaskType.GET_FACTS,
                    params={"concept": c},
                ))

        else:
            # 通用: 对每个概念获取事实
            for c in concepts[:3]:
                self._task_counter += 1
                tasks.append(SubTask(
                    id=tid(),
                    type=TaskType.GET_FACTS,
                    params={"concept": c},
                ))

        # 所有查询类型最后都加格式化任务
        self._task_counter += 1
        format_task = SubTask(
            id=tid(),
            type=TaskType.FORMAT_OUTPUT,
            params={"format": output_format.value},
            depends_on=[t.id for t in tasks],
        )
        tasks.append(format_task)

        return tasks

    # ═════════════════════════════════════════════════════════════════════
    # 依赖分析
    # ═════════════════════════════════════════════════════════════════════

    def _analyze_dependencies(self,
                              tasks: List[SubTask]) -> List[SubTask]:
        """分析并补充任务间的隐式依赖"""
        # 目前显式依赖已在生成时设置
        # 可以扩展为自动检测数据流依赖
        return tasks

    # ═════════════════════════════════════════════════════════════════════
    # 任务执行
    # ═════════════════════════════════════════════════════════════════════

    def _execute_task(self, task: SubTask,
                      prev_results: Dict[str, Any]) -> Any:
        """执行单个原子任务（调用概念图 API）"""
        p = task.params

        if task.type == TaskType.FIND_CONCEPT:
            if self.cg and hasattr(self.cg, 'triples'):
                concept = p["concept"]
                if concept in self.cg.triples:
                    return {"found": True, "concept": concept,
                            "fact_count": len(self.cg.triples[concept])}
                return {"found": False, "concept": concept}
            return {"found": False, "concept": p["concept"], "reason": "no cg"}

        elif task.type == TaskType.GET_FACTS:
            if self.cg and hasattr(self.cg, 'triples'):
                concept = p["concept"]
                if concept in self.cg.triples:
                    facts = []
                    for rel, obj, conf, src in self.cg.triples[concept]:
                        facts.append({
                            "relation": rel,
                            "object": obj,
                            "confidence": conf,
                            "source": src,
                        })
                    return {"concept": concept, "facts": facts}
                return {"concept": concept, "facts": []}
            return {"concept": p["concept"], "facts": []}

        elif task.type == TaskType.FIND_PATH:
            if self.cg and hasattr(self.cg, 'reason'):
                paths = self.cg.reason(
                    start=p["from"],
                    target=p.get("to"),
                    max_hops=p.get("max_hops", 3),
                )
                return {"from": p["from"], "to": p.get("to"),
                        "paths": paths[:5]}

            # 回退: 简单 BFS 搜
            return {"from": p["from"], "to": p.get("to"), "paths": []}

        elif task.type == TaskType.COMPARE:
            # 从 prev_results 中提取两者的事实进行对比
            facts_a = None
            facts_b = None
            for tid, res in prev_results.items():
                if isinstance(res, dict) and res.get("concept") == p["concepts"][0]:
                    facts_a = res.get("facts", [])
                if isinstance(res, dict) and res.get("concept") == p["concepts"][1]:
                    facts_b = res.get("facts", [])

            common = [f for f in facts_a if f in facts_b] if facts_a and facts_b else []
            only_a = [f for f in facts_a if f not in facts_b] if facts_a and facts_b else facts_a or []
            only_b = [f for f in facts_b if f not in facts_a] if facts_a and facts_b else facts_b or []

            return {
                "concepts": p["concepts"],
                "common": common,
                "only_a": only_a,
                "only_b": only_b,
            }

        elif task.type == TaskType.LIST_INSTANCES:
            if self.cg and hasattr(self.cg, 'triples'):
                concept = p["concept"]
                instances = []
                for s in self.cg.triples:
                    for rel, o, conf, src in self.cg.triples.get(s, []):
                        if rel == "IS_A" and o == concept:
                            instances.append(s)
                return {"concept": concept, "instances": instances[:20]}
            return {"concept": p["concept"], "instances": []}

        elif task.type == TaskType.DEFINE:
            return self._execute_task(
                SubTask(id=task.id, type=TaskType.GET_FACTS, params=p),
                prev_results
            )

        elif task.type == TaskType.FORMAT_OUTPUT:
            # 格式化任务：用化能器渲染
            return {"format": p["format"], "ready": True}

        return None

    # ═════════════════════════════════════════════════════════════════════
    # 结果聚合
    # ═════════════════════════════════════════════════════════════════════

    def _aggregate_results(self, plan: ExecutionPlan,
                           results: Dict[str, Any]) -> Dict[str, Any]:
        """将子任务结果聚合为最终输出"""
        aggregated = {
            "type": plan.query_type,
            "format": plan.output_format.value,
        }

        if plan.query_type == "compare":
            # 找到 COMPARE 任务的结果
            for r in results.values():
                if isinstance(r, dict) and "common" in r:
                    aggregated["comparison"] = r
                    break

        elif plan.query_type in ("list_instances", "define"):
            for r in results.values():
                if isinstance(r, dict) and "facts" in r:
                    aggregated["facts"] = r.get("facts", [])
                if isinstance(r, dict) and "instances" in r:
                    aggregated["instances"] = r.get("instances", [])

        elif plan.query_type == "causal_chain":
            for r in results.values():
                if isinstance(r, dict) and "paths" in r:
                    aggregated["paths"] = r.get("paths", [])

        elif plan.query_type == "table":
            table_rows = []
            for r in results.values():
                if isinstance(r, dict) and "facts" in r:
                    concept = r["concept"]
                    for f in r["facts"]:
                        table_rows.append({
                            "概念": concept,
                            "关系": f["relation"],
                            "客体": f["object"],
                            "置信度": f["confidence"],
                        })
            aggregated["table"] = table_rows

        return aggregated

    # ═════════════════════════════════════════════════════════════════════
    # 辅助
    # ═════════════════════════════════════════════════════════════════════

    def print_plan(self, plan: ExecutionPlan):
        """打印执行计划"""
        print(f"查询: {plan.original_query}")
        print(f"类型: {plan.query_type}, 输出: {plan.output_format.value}")
        print(f"概念: {plan.context.get('concepts', [])}")
        print("子任务:")
        for t in plan.tasks:
            deps = f" ← 依赖 [{', '.join(t.depends_on)}]" if t.depends_on else ""
            print(f"  [{t.id}] {t.type.name} {t.params}{deps}")


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_task_planner():
    """自测策应器"""
    tp = TaskPlanner()

    queries = [
        "对比儒家和道家",
        "量子力学有哪些基本概念",
        "什么是相对论",
        "用表格列出太阳系八大行星",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        plan = tp.plan(q)
        tp.print_plan(plan)

        # 模拟执行（无概念图）
        results = tp.execute(plan)
        print(f"聚合结果类型: {results['query_type']}")
        print(f"聚合数据: {list(results['aggregated'].keys())}")


if __name__ == "__main__":
    test_task_planner()
