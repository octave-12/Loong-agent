#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠万象格 (MultiFormKG) — 四种超越三元组的知识结构
════════════════════════════════════════════════════════════════════════════

三元组 (主体, 关系, 客体) 是知识的基本单元，但不够。人类知识大量以
过程、条件、反事实、时序形式存在。万象格在概念图上新增四类知识格：

  1. 过程格 (Process)      — 有顺序的操作/事件序列
  2. 条件格 (Conditional)  — IF-THEN 规则，带门控
  3. 反事实格 (Counterfactual) — 要不是A就B，带负置信度边
  4. 时序格 (Temporal)     — 时间线上的因果序

════════════════════════════════════════════════════════════════════════════
设计原则
════════════════════════════════════════════════════════════════════════════

每种格都是图的扩展，可以在概念图上做统一推理:
  - 过程格 = 带顺序的有向路径
  - 条件格 = 带门控的有向边
  - 反事实格 = 负置信度的有向边（表示"如果A没发生"）
  - 时序格 = 带时间戳的节点，边自带先后关系

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.multiform_kg import MultiFormKG

    mkg = MultiFormKG(concept_graph)

    # 添加过程
    mkg.add_process("煎鸡蛋", [
        ("倒油", "锅"), ("打鸡蛋", "碗"), ("倒入", "锅"),
    ])

    # 添加条件规则
    mkg.add_conditional(
        condition={"temperature": ">100°C", "pressure": "standard"},
        consequent=("水", "→", "沸腾"),
        confidence=1.0
    )

    # 添加反事实
    mkg.add_counterfactual(
        "瓦特改良蒸汽机", "工业革命",
        confidence=0.4,
        narrative="如果不是瓦特改良了蒸汽机，工业革命可能推迟50年"
    )

    # 添加时间线
    mkg.add_timeline(
        "中国朝代",
        [("夏朝", -2070), ("商朝", -1600), ("周朝", -1046)]
    )

"""
import json
import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessStep:
    """过程格中的一个步骤"""
    index: int                     # 步骤序号 (0-based)
    action: str                    # 动作描述
    target: Optional[str] = None   # 作用对象
    agent: Optional[str] = None    # 执行者
    tool: Optional[str] = None     # 使用的工具
    precondition: Optional[str] = None   # 前置条件
    expected_result: Optional[str] = None  # 预期结果
    duration: Optional[float] = None  # 预估时长（秒）


@dataclass
class Process:
    """过程格 — 有顺序的操作/事件序列"""
    name: str                      # 过程名称（如"煎鸡蛋"）
    steps: List[ProcessStep]       # 有序步骤列表
    domain: str = ""               # 所属领域
    total_steps: int = 0           # 总步数
    reversible: bool = False       # 是否可逆

    def __post_init__(self):
        self.total_steps = len(self.steps)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [{
                "index": s.index,
                "action": s.action,
                "target": s.target,
                "agent": s.agent,
                "tool": s.tool,
                "precondition": s.precondition,
                "expected_result": s.expected_result,
                "duration": s.duration,
            } for s in self.steps],
            "domain": self.domain,
            "total_steps": self.total_steps,
            "reversible": self.reversible,
        }


@dataclass
class Conditional:
    """条件格 — IF-THEN 规则"""
    name: str                      # 规则名
    condition: Dict[str, Any]      # 条件（变量→值/范围）
    subject: str                   # 主体
    consequent: str                # 结果
    relation: str = "CAUSE"        # 关系类型
    confidence: float = 1.0        # 置信度
    else_consequent: Optional[str] = None  # ELSE 分支
    domain: str = ""               # 领域

    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """
        评估条件是否满足。
        context: {"temperature": 120, "pressure": "standard", ...}
        """
        for var, expected in self.condition.items():
            if var not in context:
                return None  # 未知
            actual = context[var]
            if not self._check_condition(actual, expected):
                return self.else_consequent if self.else_consequent else None
        return self.consequent

    def _check_condition(self, actual: Any, expected: Any) -> bool:
        """检查条件项"""
        if isinstance(expected, str) and isinstance(actual, (int, float)):
            # ">100" 格式的条件
            if expected.startswith(">"):
                return actual > float(expected[1:])
            if expected.startswith("<"):
                return actual < float(expected[1:])
            if expected.startswith(">="):
                return actual >= float(expected[2:])
            if expected.startswith("<="):
                return actual <= float(expected[2:])
            if expected.startswith("=="):
                return actual == float(expected[2:])
            if expected.startswith("!="):
                return actual != float(expected[2:])
        return actual == expected


@dataclass
class Counterfactual:
    """反事实格 — "要不是A就B"的知识"""
    subject: str                   # 被假设移除的事件
    object: str                    # 可能不会发生的结果
    confidence: float = 0.3        # 反事实置信度（通常较低）
    narrative: str = ""            # 反事实叙述
    domain: str = ""               # 领域
    evidence: List[str] = field(default_factory=list)  # 支持证据

    def to_triple(self) -> Tuple[str, str, str, float]:
        """转化为带负置信度标记的三元组"""
        return (self.subject, "PREVENTS", self.object, -self.confidence)


@dataclass
class TimelineEvent:
    """时序格中的事件"""
    name: str                      # 事件名
    start_year: float              # 开始时间（可为负数表示公元前）
    end_year: Optional[float] = None  # 结束时间
    description: str = ""          # 描述
    category: str = ""             # 分类标签


@dataclass
class Timeline:
    """时序格 — 时间线"""
    name: str                      # 时间线名称
    events: List[TimelineEvent]    # 事件列表（按时间排序）
    domain: str = ""

    def sort(self):
        """按时间排序"""
        self.events.sort(key=lambda e: e.start_year)

    def find_contemporaries(self, year: float, tolerance: float = 50) -> List[TimelineEvent]:
        """找到同一时期的事件"""
        return [e for e in self.events
                if abs(e.start_year - year) <= tolerance]

    def find_before(self, event_name: str) -> List[TimelineEvent]:
        """找到在某个事件之前的所有事件"""
        for e in self.events:
            if e.name == event_name:
                idx = self.events.index(e)
                return self.events[:idx]
        return []

    def find_after(self, event_name: str) -> List[TimelineEvent]:
        """找到在某个事件之后的所有事件"""
        for e in self.events:
            if e.name == event_name:
                idx = self.events.index(e)
                return self.events[idx + 1:]
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 万象格主类
# ═══════════════════════════════════════════════════════════════════════════

class MultiFormKG:
    """
    万象格 — 在概念图之上构建四种高级知识结构。

    属性:
        processes:       Dict[name, Process]    过程格
        conditionals:    Dict[name, Conditional]  条件格
        counterfactuals: Dict[id, Counterfactual] 反事实格
        timelines:       Dict[name, Timeline]   时序格
        concept_graph:   关联的概念图实例
    """

    def __init__(self, concept_graph=None):
        self.cg = concept_graph
        self.processes: Dict[str, Process] = {}
        self.conditionals: Dict[str, Conditional] = {}
        self.counterfactuals: Dict[str, Counterfactual] = {}
        self.timelines: Dict[str, Timeline] = {}

    # ═════════════════════════════════════════════════════════════════════
    # 过程格操作
    # ═════════════════════════════════════════════════════════════════════

    def add_process(self, name: str, steps: List[Tuple[str, str]],
                    domain: str = "", reversible: bool = False) -> Process:
        """
        添加一个过程。

        Args:
            name: 过程名称（如"煎鸡蛋"）
            steps: [(动作, 目标), ...] 有序步骤
            domain: 所属领域
            reversible: 是否可逆

        Returns:
            创建的 Process 实例
        """
        process_steps = []
        for i, (action, target) in enumerate(steps):
            process_steps.append(ProcessStep(
                index=i,
                action=action,
                target=target,
            ))

        process = Process(
            name=name,
            steps=process_steps,
            domain=domain,
            reversible=reversible,
        )
        self.processes[name] = process

        # 同时注入概念图：步骤间的 FOLLOWS 关系
        if self.cg and len(process_steps) >= 2:
            for i in range(len(process_steps) - 1):
                s_name = f"{name}_step{i}"
                o_name = f"{name}_step{i+1}"
                self.cg.add_triple(s_name, "FOLLOWS", o_name, confidence=0.95)

        return process

    def get_process(self, name: str) -> Optional[Process]:
        """获取过程"""
        return self.processes.get(name)

    def verify_process(self, name: str,
                       steps_taken: List[str]) -> Tuple[bool, List[str]]:
        """
        验证执行步骤是否正确。

        Returns:
            (是否完全匹配, 不匹配的步骤列表)
        """
        process = self.processes.get(name)
        if not process:
            return False, [f"未知过程 '{name}'"]

        mismatches = []
        for i, (actual_action, expected_step) in enumerate(
            zip(steps_taken, process.steps)
        ):
            if actual_action != expected_step.action:
                mismatches.append(
                    f"第{i+1}步: 期望'{expected_step.action}'，实际'{actual_action}'"
                )

        if len(steps_taken) != len(process.steps):
            mismatches.append(
                f"步骤数不匹配: 期望{len(process.steps)}步，实际{len(steps_taken)}步"
            )

        return len(mismatches) == 0, mismatches

    def list_processes(self, domain: str = "") -> List[str]:
        """列出所有过程（可按领域过滤）"""
        if domain:
            return [name for name, p in self.processes.items() if p.domain == domain]
        return list(self.processes.keys())

    # ═════════════════════════════════════════════════════════════════════
    # 条件格操作
    # ═════════════════════════════════════════════════════════════════════

    def add_conditional(self, name: str = None, condition: Dict[str, Any] = None,
                        consequent: Tuple[str, str, str] = None,
                        confidence: float = 1.0, else_branch: Optional[str] = None,
                        domain: str = "") -> Conditional:
        """
        添加一个条件规则。

        Args:
            name: 规则名（自动生成）
            condition: {变量: 预期值} 条件字典
            consequent: (主体, 关系, 结果) 三元组形式
            confidence: 置信度
            else_branch: ELSE 分支的结果描述
            domain: 领域

        例:
            add_conditional(
                condition={"temperature": ">100", "pressure": "standard"},
                consequent=("水", "→", "沸腾"),
                confidence=1.0
            )
        """
        if name is None:
            name = f"rule_{len(self.conditionals):04d}"

        subject, _, obj = consequent

        cond = Conditional(
            name=name,
            condition=condition,
            subject=subject,
            consequent=obj,
            relation="CAUSE",
            confidence=confidence,
            else_consequent=else_branch,
            domain=domain,
        )
        self.conditionals[name] = cond

        # 注入概念图：条件作为带门控的边
        if self.cg:
            # 标记这是一个条件规则
            self.cg.add_triple(subject, "CAUSE", obj, confidence=confidence)

        return cond

    def evaluate_condition(self, name: str,
                           context: Dict[str, Any]) -> Optional[str]:
        """评估条件规则"""
        cond = self.conditionals.get(name)
        if not cond:
            return None
        return cond.evaluate(context)

    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, str]:
        """在给定上下文中评估所有条件规则"""
        results = {}
        for name, cond in self.conditionals.items():
            result = cond.evaluate(context)
            if result:
                results[name] = result
        return results

    # ═════════════════════════════════════════════════════════════════════
    # 反事实格操作
    # ═════════════════════════════════════════════════════════════════════

    def add_counterfactual(self, subject: str, obj: str,
                           confidence: float = 0.3,
                           narrative: str = "",
                           domain: str = "",
                           evidence: List[str] = None) -> Counterfactual:
        """
        添加一个反事实关系。

        Args:
            subject: 被假设移除的事件
            obj: 如果前者没发生就不会发生的结果
            confidence: 反事实置信度（通常 ≤ 0.5）
            narrative: 反事实叙述
            domain: 领域
            evidence: 支持证据列表

        例:
            add_counterfactual(
                "瓦特改良蒸汽机", "工业革命",
                confidence=0.4,
                narrative="如果不是瓦特改良了蒸汽机，工业革命可能推迟50年"
            )
        """
        cf_id = f"cf_{subject}_{obj}".replace(" ", "_")
        cf = Counterfactual(
            subject=subject,
            object=obj,
            confidence=confidence,
            narrative=narrative,
            domain=domain,
            evidence=evidence or [],
        )
        self.counterfactuals[cf_id] = cf

        # 注入概念图：负置信度的 PREVENTS 边
        if self.cg:
            self.cg.add_triple(subject, "PREVENTS", obj, confidence=-confidence)

        return cf

    def query_counterfactual(self, event: str) -> List[Counterfactual]:
        """查询与某事件相关的所有反事实"""
        related = []
        for cf in self.counterfactuals.values():
            if cf.subject == event or cf.object == event:
                related.append(cf)
        return related

    def list_counterfactuals(self, domain: str = "") -> List[Counterfactual]:
        """列出所有反事实"""
        all_cfs = list(self.counterfactuals.values())
        if domain:
            return [cf for cf in all_cfs if cf.domain == domain]
        return all_cfs

    # ═════════════════════════════════════════════════════════════════════
    # 时序格操作
    # ═════════════════════════════════════════════════════════════════════

    def add_timeline(self, name: str,
                     events: List[Tuple[str, float]],
                     domain: str = "") -> Timeline:
        """
        添加一条时间线。

        Args:
            name: 时间线名称（如"中国朝代"）
            events: [(事件名, 开始年份), ...] 年份可为负（BCE）
            domain: 领域

        例:
            add_timeline("中国朝代", [
                ("夏朝", -2070), ("商朝", -1600), ("周朝", -1046),
                ("秦朝", -221), ("汉朝", -202),
            ])
        """
        timeline_events = []
        for ev_name, year in events:
            timeline_events.append(TimelineEvent(
                name=ev_name,
                start_year=year,
                category=name,
            ))

        timeline = Timeline(
            name=name,
            events=timeline_events,
            domain=domain,
        )
        timeline.sort()
        self.timelines[name] = timeline

        # 注入概念图：时序上的 FOLLOWS 关系
        if self.cg and len(timeline_events) >= 2:
            for i in range(len(timeline_events) - 1):
                self.cg.add_triple(
                    timeline_events[i].name,
                    "FOLLOWS",
                    timeline_events[i + 1].name,
                    confidence=0.95,
                )

        return timeline

    def add_timeline_event(self, timeline_name: str,
                           event: TimelineEvent):
        """向已有时间线追加事件"""
        if timeline_name not in self.timelines:
            self.timelines[timeline_name] = Timeline(
                name=timeline_name,
                events=[],
            )
        self.timelines[timeline_name].events.append(event)
        self.timelines[timeline_name].sort()

    def get_timeline(self, name: str) -> Optional[Timeline]:
        """获取时间线"""
        return self.timelines.get(name)

    def what_happened_in(self, year: float,
                         tolerance: float = 50) -> Dict[str, List[TimelineEvent]]:
        """
        查询某一年附近发生了什么。

        Returns:
            {timeline_name: [events]}
        """
        results = {}
        for name, tl in self.timelines.items():
            events = tl.find_contemporaries(year, tolerance)
            if events:
                results[name] = events
        return results

    def what_came_before(self, event_name: str) -> Dict[str, List[TimelineEvent]]:
        """查询某事件之前的所有事件"""
        results = {}
        for name, tl in self.timelines.items():
            before = tl.find_before(event_name)
            if before:
                results[name] = before
        return results

    def what_came_after(self, event_name: str) -> Dict[str, List[TimelineEvent]]:
        """查询某事件之后的所有事件"""
        results = {}
        for name, tl in self.timelines.items():
            after = tl.find_after(event_name)
            if after:
                results[name] = after
        return results

    # ═════════════════════════════════════════════════════════════════════
    # 跨格推理
    # ═════════════════════════════════════════════════════════════════════

    def reason_across_forms(self, query: str) -> Dict[str, Any]:
        """
        跨格式推理：在一个查询中同时利用四种格。

        例: "秦朝之后发生了什么" → 查时序格
           "如果不修长城会怎样" → 查反事实格
           "怎么制作豆浆" → 查过程格
           "如果温度超过100度" → 查条件格
        """
        results = {
            "processes": [],
            "conditionals": [],
            "counterfactuals": [],
            "timeline_events": [],
        }

        # 尝试匹配过程
        if query in self.processes:
            p = self.processes[query]
            results["processes"] = [
                f"第{i+1}步: {s.action} → {s.target}"
                for i, s in enumerate(p.steps)
            ]

        # 尝试匹配条件规则
        for name, cond in self.conditionals.items():
            if cond.subject in query or cond.consequent in query:
                conditions_str = ", ".join(
                    f"{k}{v}" for k, v in cond.condition.items()
                )
                results["conditionals"].append(
                    f"当{conditions_str}时，{cond.subject}会{cond.consequent}"
                )

        # 尝试匹配反事实
        for cf in self.counterfactuals.values():
            if cf.subject in query or cf.object in query:
                results["counterfactuals"].append(
                    cf.narrative or f"要不是{cf.subject}，就不会有{cf.object}"
                )

        # 尝试匹配时序事件
        query_lower = query.lower()
        for name, tl in self.timelines.items():
            for ev in tl.events:
                if ev.name in query:
                    after = tl.find_after(ev.name)
                    if after:
                        results["timeline_events"].append({
                            "query_event": ev.name,
                            "time": ev.start_year,
                            "after": [e.name for e in after[:5]],
                        })
                    before = tl.find_before(ev.name)
                    if before:
                        results["timeline_events"][-1]["before"] = [
                            e.name for e in before[-3:]
                        ]

        return results

    # ═════════════════════════════════════════════════════════════════════
    # 持久化
    # ═════════════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存万象格到 JSON"""
        data = {
            "processes": {name: p.to_dict() for name, p in self.processes.items()},
            "conditionals": {name: {
                "condition": c.condition,
                "subject": c.subject,
                "consequent": c.consequent,
                "relation": c.relation,
                "confidence": c.confidence,
                "else_consequent": c.else_consequent,
                "domain": c.domain,
            } for name, c in self.conditionals.items()},
            "counterfactuals": {
                cf_id: {
                    "subject": cf.subject,
                    "object": cf.object,
                    "confidence": cf.confidence,
                    "narrative": cf.narrative,
                    "domain": cf.domain,
                    "evidence": cf.evidence,
                } for cf_id, cf in self.counterfactuals.items()
            },
            "timelines": {name: {
                "events": [{
                    "name": e.name,
                    "start_year": e.start_year,
                    "end_year": e.end_year,
                    "description": e.description,
                    "category": e.category,
                } for e in tl.events],
                "domain": tl.domain,
            } for name, tl in self.timelines.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[万象格] 已保存至 {path}")

    def load(self, path: str):
        """从 JSON 加载万象格"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载过程
        for name, p_data in data.get("processes", {}).items():
            steps = []
            for s in p_data["steps"]:
                steps.append(ProcessStep(
                    index=s["index"],
                    action=s["action"],
                    target=s.get("target"),
                    agent=s.get("agent"),
                    tool=s.get("tool"),
                    precondition=s.get("precondition"),
                    expected_result=s.get("expected_result"),
                    duration=s.get("duration"),
                ))
            self.processes[name] = Process(
                name=name,
                steps=steps,
                domain=p_data.get("domain", ""),
                reversible=p_data.get("reversible", False),
            )

        # 加载条件
        for name, c_data in data.get("conditionals", {}).items():
            self.conditionals[name] = Conditional(
                name=name,
                condition=c_data["condition"],
                subject=c_data["subject"],
                consequent=c_data["consequent"],
                relation=c_data.get("relation", "CAUSE"),
                confidence=c_data.get("confidence", 1.0),
                else_consequent=c_data.get("else_consequent"),
                domain=c_data.get("domain", ""),
            )

        # 加载反事实
        for cf_id, cf_data in data.get("counterfactuals", {}).items():
            self.counterfactuals[cf_id] = Counterfactual(
                subject=cf_data["subject"],
                object=cf_data["object"],
                confidence=cf_data.get("confidence", 0.3),
                narrative=cf_data.get("narrative", ""),
                domain=cf_data.get("domain", ""),
                evidence=cf_data.get("evidence", []),
            )

        # 加载时间线
        for name, tl_data in data.get("timelines", {}).items():
            events = []
            for e in tl_data["events"]:
                events.append(TimelineEvent(
                    name=e["name"],
                    start_year=e["start_year"],
                    end_year=e.get("end_year"),
                    description=e.get("description", ""),
                    category=e.get("category", ""),
                ))
            self.timelines[name] = Timeline(
                name=name,
                events=events,
                domain=tl_data.get("domain", ""),
            )

        print(f"[万象格] 已加载: {len(self.processes)}过程 "
              f"{len(self.conditionals)}条件 {len(self.counterfactuals)}反事实 "
              f"{len(self.timelines)}时间线")

    # ═════════════════════════════════════════════════════════════════════
    # 统计
    # ═════════════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        total_steps = sum(p.total_steps for p in self.processes.values())
        total_events = sum(len(tl.events) for tl in self.timelines.values())
        return {
            "processes": len(self.processes),
            "total_steps": total_steps,
            "conditionals": len(self.conditionals),
            "counterfactuals": len(self.counterfactuals),
            "timelines": len(self.timelines),
            "total_timeline_events": total_events,
            "total_form_knowledge": (
                len(self.processes) + len(self.conditionals) +
                len(self.counterfactuals) + len(self.timelines)
            ),
        }

    def print_stats(self):
        """打印统计"""
        s = self.stats()
        print(f"═══ 万象格统计 ═══")
        print(f"  过程格:     {s['processes']:>6} 个过程 (共{s['total_steps']}步)")
        print(f"  条件格:     {s['conditionals']:>6} 条规则")
        print(f"  反事实格:   {s['counterfactuals']:>6} 条")
        print(f"  时序格:     {s['timelines']:>6} 条时间线 (共{s['total_timeline_events']}事件)")
        print(f"  总计:       {s['total_form_knowledge']:>6} 条非三元组知识")


# ═══════════════════════════════════════════════════════════════════════════
# 种子数据注入 — 预置常用知识
# ═══════════════════════════════════════════════════════════════════════════

def seed_multiform_kg(mkg: MultiFormKG):
    """注入种子知识到万象格"""

    # ── 过程格种子 ──
    mkg.add_process("煎鸡蛋", [
        ("准备", "鸡蛋"),
        ("热油", "锅"),
        ("打入", "鸡蛋"),
        ("翻炒", "鸡蛋"),
        ("调味", "盐"),
        ("装盘", "成品"),
    ], domain="烹饪")

    mkg.add_process("冲咖啡", [
        ("研磨", "咖啡豆"),
        ("放入", "滤纸"),
        ("加热", "水"),
        ("冲泡", "咖啡粉"),
        ("等待", "萃取"),
    ], domain="饮食")

    mkg.add_process("科学方法", [
        ("观察", "现象"),
        ("提出", "假设"),
        ("设计", "实验"),
        ("收集", "数据"),
        ("分析", "结果"),
        ("得出结论", ""),
        ("发表", "论文"),
    ], domain="科学")

    mkg.add_process("软件开发", [
        ("需求", "分析"),
        ("系统", "设计"),
        ("编码", "实现"),
        ("测试", "验证"),
        ("部署", "上线"),
        ("维护", "更新"),
    ], domain="计算机")

    # ── 条件格种子 ──
    mkg.add_conditional(
        condition={"temperature": ">100", "pressure": "standard"},
        consequent=("水", "→", "沸腾"),
        confidence=1.0,
        else_branch="水保持液态",
        domain="物理",
    )

    mkg.add_conditional(
        condition={"temperature": "<0"},
        consequent=("水", "→", "结冰"),
        confidence=1.0,
        else_branch="水保持液态",
        domain="物理",
    )

    mkg.add_conditional(
        condition={"pH": "<7"},
        consequent=("溶液", "→", "酸性"),
        confidence=1.0,
        else_branch="非酸性",
        domain="化学",
    )

    mkg.add_conditional(
        condition={"pH": ">7"},
        consequent=("溶液", "→", "碱性"),
        confidence=1.0,
        domain="化学",
    )

    # ── 反事实格种子 ──
    mkg.add_counterfactual(
        "瓦特改良蒸汽机", "工业革命",
        confidence=0.35,
        narrative="如果不是瓦特在18世纪改良了蒸汽机，工业革命可能不会以同样的速度展开，"
                  "机器大生产时代将推迟数十年。",
        domain="历史",
    )

    mkg.add_counterfactual(
        "爱因斯坦提出相对论", "核能利用",
        confidence=0.25,
        narrative="如果没有爱因斯坦的质能方程E=mc²，人类对核能的理解可能晚几十年。"
                  "不过其他物理学家最终也可能推导出等价的理论。",
        domain="物理",
    )

    mkg.add_counterfactual(
        "秦始皇统一文字", "中国文化统一",
        confidence=0.55,
        narrative="如果不是秦始皇推行'书同文'政策，中国各地方言差异可能导致文字分裂，"
                  "中华文明的大一统传统可能不存在。",
        domain="历史",
    )

    # ── 时序格种子 ──
    mkg.add_timeline("中国朝代", [
        ("夏朝", -2070),
        ("商朝", -1600),
        ("周朝", -1046),
        ("秦朝", -221),
        ("汉朝", -202),
        ("三国", 220),
        ("晋朝", 265),
        ("南北朝", 420),
        ("隋朝", 581),
        ("唐朝", 618),
        ("宋朝", 960),
        ("元朝", 1271),
        ("明朝", 1368),
        ("清朝", 1644),
        ("中华民国", 1912),
        ("中华人民共和国", 1949),
    ], domain="历史")

    mkg.add_timeline("物理学重大发现", [
        ("牛顿定律", 1687),
        ("电磁学", 1820),
        ("热力学", 1850),
        ("麦克斯韦方程", 1865),
        ("相对论", 1905),
        ("量子力学", 1925),
        ("核裂变", 1938),
        ("晶体管", 1947),
        ("激光", 1960),
        ("希格斯玻色子", 2012),
    ], domain="科学")

    mkg.add_timeline("计算机发展", [
        ("图灵机概念", 1936),
        ("ENIAC", 1946),
        ("晶体管计算机", 1955),
        ("集成电路", 1958),
        ("微处理器", 1971),
        ("个人电脑", 1975),
        ("互联网", 1983),
        ("万维网", 1990),
        ("智能手机", 2007),
        ("深度学习爆发", 2012),
        ("大语言模型", 2022),
    ], domain="计算机")

    print(f"[万象格] 种子注入完成: {mkg.stats()}")


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_multiform_kg():
    """自测万象格所有功能"""
    mkg = MultiFormKG()

    # 注入种子
    seed_multiform_kg(mkg)
    mkg.print_stats()

    print("\n" + "=" * 60)
    print("1. 过程格 — 查询'科学方法'")
    p = mkg.get_process("科学方法")
    for s in p.steps:
        print(f"  第{s.index+1}步: {s.action} → {s.target}")

    print("\n2. 过程格 — 验证步骤")
    ok, errors = mkg.verify_process("科学方法",
        ["观察", "提出", "设计", "收集", "分析", "得出结论", "发表"])
    print(f"  正确: {ok}, 错误: {errors}")

    print("\n3. 条件格 — 评估规则")
    result = mkg.evaluate_all({"temperature": 120, "pressure": "standard", "pH": 3})
    for name, r in result.items():
        print(f"  {name}: {r}")

    print("\n4. 条件格 — 评估(不满足条件)")
    result = mkg.evaluate_all({"temperature": 25})
    print(f"  结果: {result if result else '(无匹配规则)'}")

    print("\n5. 反事实格 — 查询反事实")
    cfs = mkg.query_counterfactual("工业革命")
    for cf in cfs:
        print(f"  {cf.narrative[:80]}...")

    print("\n6. 时序格 — 唐朝之后")
    after = mkg.what_came_after("唐朝")
    for tl_name, events in after.items():
        print(f"  [{tl_name}]")
        for e in events[:5]:
            print(f"    {e.name} ({e.start_year})")

    print("\n7. 时序格 — 1900年左右发生的事")
    happened = mkg.what_happened_in(1900, tolerance=60)
    for tl_name, events in happened.items():
        print(f"  [{tl_name}]")
        for e in events:
            print(f"    {e.name} ({e.start_year})")

    print("\n8. 跨格推理 — '秦朝' ")
    results = mkg.reason_across_forms("秦朝")
    for key, val in results.items():
        if val:
            print(f"  [{key}]")
            if isinstance(val, list):
                for v in val[:3]:
                    print(f"    {v}")

    print("\n9. 跨格推理 — '科学方法' ")
    results = mkg.reason_across_forms("科学方法")
    for key, val in results.items():
        if val:
            print(f"  [{key}]")
            for v in val[:5]:
                print(f"    {v}")

    # 测试持久化
    import tempfile
    import os
    tmp = os.path.join(tempfile.gettempdir(), "multiform_test.json")
    mkg.save(tmp)
    mkg2 = MultiFormKG()
    mkg2.load(tmp)
    print(f"\n10. 持久化: 保存→加载 = {mkg2.stats()}")
    os.remove(tmp)


if __name__ == "__main__":
    test_multiform_kg()
