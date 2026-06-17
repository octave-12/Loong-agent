#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠化能器 (EnergyDecoder) — 结构化知识 → 自然语言
════════════════════════════════════════════════════════════════════════════

龙珠的 NLG 后端。将概念图推理结果（三元组、路径、对比）转化为自然中文。
全链路确定性计算：模板森林 + 能量排序 + 衔接词选择。

设计哲学:
  - 事实型生成 → 模板填充，100% 确定，绝不幻觉
  - 推理型生成 → 图路径展开 + 因果链衔接
  - 创意型生成 → 放弃，这不是龙珠的战场

════════════════════════════════════════════════════════════════════════════
核心能力
════════════════════════════════════════════════════════════════════════════

  1. 模板森林      每种关系携带 5-10 个自然语言表达模板
  2. 能量排序      用金字塔深层能量评分选出最连贯的表达序列
  3. 格式适配      支持陈述/解释/对比/列表/表格等多种输出格式
  4. 上下文衔接    关联词衔接 (因果/转折/并列/递进)
  5. 证据附注      可选附上每条事实的证据来源和置信度

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.energy_decoder import EnergyDecoder

    decoder = EnergyDecoder()
    text = decoder.render({
        "type": "explain_path",
        "subject": "量子纠缠",
        "path": ["量子纠缠", "量子态", "测量", "波函数坍缩", "信息传递"],
        "edges": [
            {"rel": "IS_A", "confidence": 0.95},
            {"rel": "CAUSE", "confidence": 0.87},
        ]
    })
    # → "量子纠缠是一种量子态。当量子态被测量时，会导致波函数坍缩，因此不能用于信息传递。"

"""
import re
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# 关系表达模板森林
# ═══════════════════════════════════════════════════════════════════════════
# 每种关系携带 5-10 个自然语言模板，{s} = 主体, {o} = 客体
# 模板按正式程度分级：formal(学术)、neutral(通用)、casual(口语)

_RELATION_TEMPLATES = {
    "IS_A": {
        "formal": [
            "{s}是{o}的一种表现形式。",
            "{s}属于{o}范畴。",
            "{s}是{o}的子类。",
            "{s}可归类为{o}。",
        ],
        "neutral": [
            "{s}是一种{o}。",
            "{s}属于{o}。",
            "从分类上看，{s}是{o}。",
            "{s}是{o}的其中一类。",
        ],
        "casual": [
            "{s}说白了就是{o}。",
            "{s}算是一种{o}。",
        ],
    },
    "PART_OF": {
        "formal": [
            "{s}是{o}的组成部分。",
            "{s}构成{o}的基本单元。",
            "{o}由{s}构成。",
            "{s}作为{o}的子系统而存在。",
        ],
        "neutral": [
            "{s}是{o}的一部分。",
            "{o}包含{s}。",
            "{s}组成了{o}。",
            "{o}里面有{s}。",
        ],
        "casual": [
            "{o}里面有{s}。",
            "{s}是{o}里的东西。",
        ],
    },
    "HAS": {
        "formal": [
            "{s}具备{o}的特征。",
            "{s}拥有{o}属性。",
            "{s}具有{o}。",
        ],
        "neutral": [
            "{s}有{o}。",
            "{s}具备{o}。",
            "{s}含有{o}。",
        ],
        "casual": [
            "{s}带有{o}。",
            "{s}有{o}这个特点。",
        ],
    },
    "CAUSE": {
        "formal": [
            "{s}导致了{o}。",
            "{s}引发{o}。",
            "{o}由{s}所致。",
            "{s}是{o}的成因。",
            "{o}的发生归因于{s}。",
        ],
        "neutral": [
            "{s}造成{o}。",
            "{s}会引起{o}。",
            "因为{s}，所以{o}。",
            "{s}使得{o}发生。",
        ],
        "casual": [
            "有了{s}，才会有{o}。",
            "{s}带来了{o}。",
        ],
    },
    "OPPOSITE": {
        "formal": [
            "{s}与{o}构成对立统一关系。",
            "{s}和{o}互为反面。",
            "{s}与{o}在属性上相反。",
        ],
        "neutral": [
            "{s}和{o}是相反的。",
            "{s}与{o}相对。",
            "{s}是{o}的对立面。",
        ],
        "casual": [
            "{s}跟{o}对着干。",
            "{s}和{o}完全相反。",
        ],
    },
    "RELATED": {
        "formal": [
            "{s}与{o}在概念上相关联。",
            "{s}和{o}之间存在语义联系。",
            "{s}与{o}具有相关性。",
        ],
        "neutral": [
            "{s}和{o}有关联。",
            "{s}与{o}相关。",
            "{s}和{o}有关系。",
        ],
        "casual": [
            "{s}和{o}有关系。",
            "{s}跟{o}挂钩。",
        ],
    },
    # 扩展关系（预留给万象格和关系挖掘）
    "ENABLES": {
        "formal": [
            "{s}使得{o}成为可能。",
            "{s}为{o}提供了条件。",
        ],
        "neutral": [
            "{s}可以实现{o}。",
            "有了{s}才能{o}。",
        ],
        "casual": [
            "{s}让{o}能行。",
        ],
    },
    "PREVENTS": {
        "formal": [
            "{s}抑制了{o}的发生。",
            "{s}阻碍{o}。",
        ],
        "neutral": [
            "{s}防止{o}。",
            "{s}使得{o}不能发生。",
        ],
        "casual": [
            "{s}不让{o}发生。",
        ],
    },
    "FOLLOWS": {
        "formal": [
            "{s}随后发生{o}。",
            "{s}之后是{o}。",
        ],
        "neutral": [
            "{s}之后是{o}。",
            "在{s}之后出现了{o}。",
        ],
        "casual": [
            "{s}完了就是{o}。",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# 衔接词库
# ═══════════════════════════════════════════════════════════════════════════

_CONNECTORS = {
    "causal_chain": ["因此", "所以", "于是", "由此可见", "这意味着", "进而", "从而"],
    "contrast":     ["然而", "但是", "不过", "另一方面", "相反地", "与此相对"],
    "addition":     ["此外", "另外", "同时", "并且", "不仅如此", "再者"],
    "progression":  ["进一步说", "更具体地说", "具体而言", "也就是说", "换句话说"],
    "sequence":     ["首先", "其次", "最后", "第一步", "接下来", "之后"],
    "conclusion":   ["综上所述", "总的来说", "总而言之", "归根结底"],
}


# ═══════════════════════════════════════════════════════════════════════════
# 渲染输入数据结构
# ═══════════════════════════════════════════════════════════════════════════

class RenderType(Enum):
    EXPLAIN_PATH = "explain_path"           # 解释推理路径
    DEFINE = "define"                       # 定义概念
    COMPARE = "compare"                     # 对比概念
    LIST_RELATED = "list_related"           # 列出关联概念
    FACT_STATEMENT = "fact_statement"       # 单条事实陈述
    CAUSAL_CHAIN = "causal_chain"           # 因果链展开
    TABLE = "table"                         # 表格输出


@dataclass
class EdgeInfo:
    """路径边信息"""
    rel: str
    confidence: float = 0.5
    source: str = ""


@dataclass
class RenderInput:
    """化能器的标准化输入"""
    render_type: str  # one of RenderType
    subject: Optional[str] = None
    path: List[str] = field(default_factory=list)
    edges: List[EdgeInfo] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    compare_subjects: List[str] = field(default_factory=list)
    energy_scores: List[float] = field(default_factory=list)
    style: str = "neutral"  # formal/neutral/casual
    show_evidence: bool = False
    max_sentences: int = 8


# ═══════════════════════════════════════════════════════════════════════════
# 化能器主类
# ═══════════════════════════════════════════════════════════════════════════

class EnergyDecoder:
    """
    化能器 — 将结构化知识渲染为自然中文。

    核心流程:
      1. 识别渲染类型 (路径解释/定义/对比/事实列表)
      2. 为每条知识边选择最佳表达模板
      3. 用能量评分决定句子顺序
      4. 插入衔接词，编织成连贯文本
      5. 可选附上证据来源
    """

    def __init__(self, energy_pyramid=None):
        """
        Args:
            energy_pyramid: 可选的能量金字塔实例，用于句子级能量评分排序
        """
        self.pyramid = energy_pyramid
        self._template_cache = {}  # 缓存已选模板避免重复
        self._used_templates = set()

    def render(self, input_data: Union[RenderInput, Dict]) -> str:
        """
        主入口：将结构化输入渲染为自然语言文本。

        Args:
            input_data: RenderInput 实例或等价字典

        Returns:
            自然语言中文文本
        """
        if isinstance(input_data, dict):
            input_data = self._dict_to_input(input_data)

        self._reset_template_cache()

        render_type = input_data.render_type

        if render_type == "explain_path":
            return self._render_explain_path(input_data)
        elif render_type == "define":
            return self._render_define(input_data)
        elif render_type == "compare":
            return self._render_compare(input_data)
        elif render_type == "list_related":
            return self._render_list_related(input_data)
        elif render_type == "causal_chain":
            return self._render_causal_chain(input_data)
        elif render_type == "fact_statement":
            return self._render_fact_statement(input_data)
        elif render_type == "table":
            return self._render_table(input_data)
        else:
            return self._render_generic(input_data)

    def _dict_to_input(self, d: Dict) -> RenderInput:
        edges = [EdgeInfo(**e) if isinstance(e, dict) else e for e in d.get("edges", [])]
        return RenderInput(
            render_type=d.get("render_type", d.get("type", "fact_statement")),
            subject=d.get("subject"),
            path=d.get("path", []),
            edges=edges,
            facts=d.get("facts", []),
            compare_subjects=d.get("compare_subjects", []),
            energy_scores=d.get("energy_scores", []),
            style=d.get("style", "neutral"),
            show_evidence=d.get("show_evidence", False),
            max_sentences=d.get("max_sentences", 8),
        )

    def _reset_template_cache(self):
        self._template_cache = {}
        self._used_templates = set()

    # ═════════════════════════════════════════════════════════════════════
    # 模板选择引擎
    # ═════════════════════════════════════════════════════════════════════

    def _pick_template(self, relation: str, style: str = "neutral",
                       subject: str = "", obj: str = "") -> str:
        """
        从模板森林中为给定关系选择最佳模板。

        选择策略:
          1. 按指定风格查找
          2. 风格不可用时回退到 neutral
          3. 优先选择未使用过的模板（避免重复）
          4. 避免选择不适合主体/客体长度的模板
        """
        templates = _RELATION_TEMPLATES.get(relation, {})
        if not templates:
            # 未知关系：用 RELATED 兜底
            templates = _RELATION_TEMPLATES.get("RELATED", {})

        # 风格优先级: 指定 → neutral → formal → casual
        for style_candidate in [style, "neutral", "formal", "casual"]:
            pool = templates.get(style_candidate, [])
            if not pool:
                continue

            # 排除已使用过的模板
            fresh = [t for t in pool if t not in self._used_templates]
            if fresh:
                chosen = random.choice(fresh)
            else:
                chosen = random.choice(pool)

            self._used_templates.add(chosen)
            return chosen.format(s=subject, o=obj)

        # 完全兜底
        return f"{subject}与{obj}有关联。"

    def _pick_connector(self, relation_type: str) -> str:
        """根据关系类型选择衔接词"""
        if relation_type in _CONNECTORS:
            return random.choice(_CONNECTORS[relation_type])
        return ""

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 路径解释
    # ═════════════════════════════════════════════════════════════════════

    def _render_explain_path(self, inp: RenderInput) -> str:
        """
        将推理路径渲染为自然语言解释。

        例:
          路径: [量子纠缠, 量子态, 测量, 波函数坍缩, 信息传递]
          边:   [IS_A, CAUSE, CAUSE]
          输出: 量子纠缠是一种量子态。量子态被测量时会导致波函数坍缩。
                波函数坍缩意味着信息无法传递。因此量子纠缠不能用于超光速通信。
        """
        if not inp.path or len(inp.path) < 2:
            return self._render_define(inp)

        sentences = []
        path = inp.path[:inp.max_sentences + 1]  # 限制长度
        edges = inp.edges[:inp.max_sentences]

        # 1. 路径展开：每相邻两节点+边 → 一句话
        for i in range(len(path) - 1):
            s_node = path[i]
            o_node = path[i + 1]
            rel = edges[i].rel if i < len(edges) else "RELATED"
            conf = edges[i].confidence if i < len(edges) else 0.5

            sentence = self._pick_template(rel, inp.style, s_node, o_node)

            # 可选：附上置信度
            if inp.show_evidence and conf < 0.9:
                sentence += f"（置信度: {conf:.0%}）"

            sentences.append(sentence)

        # 2. 因果链衔接：当边是 CAUSE 时，插入"因此""所以"
        connected = []
        for i, s in enumerate(sentences):
            if i > 0 and i - 1 < len(edges):
                prev_rel = edges[i - 1].rel
                if prev_rel in ("CAUSE", "PREVENTS"):
                    connector = self._pick_connector("causal_chain")
                    s = f"{connector}，{s[0].lower()}{s[1:]}" if s else s
                elif prev_rel == "OPPOSITE":
                    connector = self._pick_connector("contrast")
                    s = f"{connector}，{s[0].lower()}{s[1:]}" if s else s
            connected.append(s)

        return "".join(connected)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 定义
    # ═════════════════════════════════════════════════════════════════════

    def _render_define(self, inp: RenderInput) -> str:
        """渲染概念定义"""
        subject = inp.subject or (inp.path[0] if inp.path else "该概念")
        sentences = []

        # 主定义句
        if inp.edges and inp.edges[0].rel == "IS_A":
            sentences.append(
                self._pick_template("IS_A", inp.style, subject,
                                    inp.path[1] if len(inp.path) > 1 else "某类")
            )
        else:
            sentences.append(f"{subject}是一个概念。")

        # 附加属性
        for i, fact in enumerate(inp.facts[:inp.max_sentences]):
            rel = fact.get("relation", "RELATED")
            obj = fact.get("object", "")
            if obj:
                sentences.append(self._pick_template(rel, inp.style, subject, obj))

        return " ".join(sentences)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 对比
    # ═════════════════════════════════════════════════════════════════════

    def _render_compare(self, inp: RenderInput) -> str:
        """渲染概念对比"""
        subjects = inp.compare_subjects or inp.path[:2]
        if len(subjects) < 2:
            return self._render_define(inp)

        a, b = subjects[0], subjects[1]
        sentences = [f"{a}和{b}的对比分析如下："]

        # 相同点
        common = [f for f in inp.facts if f.get("type") == "common"]
        diff = [f for f in inp.facts if f.get("type") == "difference"]

        if common:
            sentences.append("共同点：")
            for f in common[:inp.max_sentences // 2]:
                sentences.append(f"- {f.get('description', '')}")

        if diff:
            sentences.append("不同点：")
            for f in diff[:inp.max_sentences // 2]:
                sentences.append(f"- {f.get('description', '')}")

        return "\n".join(sentences)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 关联列表
    # ═════════════════════════════════════════════════════════════════════

    def _render_list_related(self, inp: RenderInput) -> str:
        """渲染关联概念列表"""
        subject = inp.subject or "该概念"
        if not inp.facts:
            return f"未找到与{subject}直接关联的概念。"

        sentences = [f"与{subject}相关的概念包括："]
        for i, fact in enumerate(inp.facts[:inp.max_sentences]):
            rel = fact.get("relation", "RELATED")
            obj = fact.get("object", "")
            conf = fact.get("confidence", 0.5)
            if obj:
                s = self._pick_template(rel, inp.style, subject, obj)
                if inp.show_evidence and conf < 0.9:
                    s += f"（置信度: {conf:.0%}）"
                sentences.append(f"{i+1}. {s}")

        return "\n".join(sentences)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 因果链
    # ═════════════════════════════════════════════════════════════════════

    def _render_causal_chain(self, inp: RenderInput) -> str:
        """渲染因果推理链"""
        if not inp.path or len(inp.path) < 2:
            return "因果链不完整，无法生成解释。"

        # 重用路径解释逻辑但强化因果连接词
        sentences = []
        path = inp.path[:inp.max_sentences + 1]

        # 开头
        sentences.append(f"关于为什么{path[-1]}，推理如下：")

        for i in range(len(path) - 1):
            s_node = path[i]
            o_node = path[i + 1]
            rel = inp.edges[i].rel if i < len(inp.edges) else "CAUSE"

            if i == 0:
                sentences.append(f"首先，{s_node}存在。")
            sentences.append(
                self._pick_template(rel, inp.style, s_node, o_node) + "。"
            )

        # 结尾
        connector = self._pick_connector("conclusion")
        sentences.append(
            f"{connector}，{path[0]}最终导致了{path[-1]}。"
        )

        return "".join(sentences)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 事实陈述
    # ═════════════════════════════════════════════════════════════════════

    def _render_fact_statement(self, inp: RenderInput) -> str:
        """渲染单个事实陈述"""
        facts = inp.facts
        if not facts:
            return "无可用事实。"

        sentences = []
        for fact in facts[:inp.max_sentences]:
            rel = fact.get("relation", "RELATED")
            subj = fact.get("subject", inp.subject or "")
            obj = fact.get("object", "")
            if subj and obj:
                sentences.append(self._pick_template(rel, inp.style, subj, obj))

        return " ".join(sentences)

    # ═════════════════════════════════════════════════════════════════════
    # 渲染器 — 表格
    # ═════════════════════════════════════════════════════════════════════

    def _render_table(self, inp: RenderInput) -> str:
        """渲染为 Markdown 表格"""
        facts = inp.facts
        if not facts:
            return "无数据可制表。"

        # 收集所有键
        keys = set()
        for f in facts:
            keys.update(f.keys())
        keys = sorted(keys)

        # 表头
        header = "| " + " | ".join(keys) + " |"
        sep = "|" + "|".join(["---" for _ in keys]) + "|"

        # 数据行
        rows = []
        for f in facts[:inp.max_sentences]:
            row = "| " + " | ".join(str(f.get(k, "")) for k in keys) + " |"
            rows.append(row)

        return "\n".join([header, sep] + rows)

    # ═════════════════════════════════════════════════════════════════════
    # 通用渲染
    # ═════════════════════════════════════════════════════════════════════

    def _render_generic(self, inp: RenderInput) -> str:
        """通用渲染：根据可用数据选择最佳渲染方式"""
        if inp.path and len(inp.path) >= 2:
            return self._render_explain_path(inp)
        if inp.compare_subjects and len(inp.compare_subjects) >= 2:
            return self._render_compare(inp)
        if inp.facts:
            if len(inp.facts) > 3:
                return self._render_list_related(inp)
            return self._render_fact_statement(inp)
        return self._render_define(inp)

    # ═════════════════════════════════════════════════════════════════════
    # 批量渲染
    # ═════════════════════════════════════════════════════════════════════

    def render_batch(self, inputs: List[Dict]) -> List[str]:
        """批量渲染"""
        return [self.render(inp) for inp in inputs]


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════

_decoder_instance = None

def get_decoder() -> EnergyDecoder:
    """获取全局单例化能器"""
    global _decoder_instance
    if _decoder_instance is None:
        _decoder_instance = EnergyDecoder()
    return _decoder_instance


def render_knowledge(data: Dict) -> str:
    """快捷渲染：字典输入 → 自然语言输出"""
    return get_decoder().render(data)


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_energy_decoder():
    """自测 — 验证化能器各种渲染模式"""
    decoder = EnergyDecoder()

    print("=" * 60)
    print("1. 路径解释")
    print("=" * 60)
    result = decoder.render({
        "render_type": "explain_path",
        "subject": "量子纠缠",
        "path": ["量子纠缠", "量子态", "测量", "波函数坍缩", "信息传递"],
        "edges": [
            {"rel": "IS_A", "confidence": 0.95},
            {"rel": "CAUSE", "confidence": 0.87},
            {"rel": "CAUSE", "confidence": 0.82},
        ],
        "style": "neutral",
    })
    print(result)

    print("\n" + "=" * 60)
    print("2. 因果链解释")
    print("=" * 60)
    result = decoder.render({
        "render_type": "causal_chain",
        "path": ["温度升高", "冰融化", "海平面上升", "沿海城市淹没"],
        "edges": [
            {"rel": "CAUSE", "confidence": 0.99},
            {"rel": "CAUSE", "confidence": 0.92},
            {"rel": "CAUSE", "confidence": 0.78},
        ],
    })
    print(result)

    print("\n" + "=" * 60)
    print("3. 概念对比")
    print("=" * 60)
    result = decoder.render({
        "render_type": "compare",
        "compare_subjects": ["儒家", "道家"],
        "facts": [
            {"type": "common", "description": "都产生于先秦时期"},
            {"type": "common", "description": "都对中国文化有深远影响"},
            {"type": "difference", "description": "儒家强调入世，道家主张出世"},
            {"type": "difference", "description": "儒家重礼教，道家尚自然"},
        ],
    })
    print(result)

    print("\n" + "=" * 60)
    print("4. 关联列表（带证据）")
    print("=" * 60)
    result = decoder.render({
        "render_type": "list_related",
        "subject": "电子",
        "facts": [
            {"relation": "PART_OF", "object": "原子", "confidence": 0.99},
            {"relation": "HAS", "object": "负电荷", "confidence": 0.98},
            {"relation": "RELATED", "object": "量子力学", "confidence": 0.85},
            {"relation": "PART_OF", "object": "电子云", "confidence": 0.72},
        ],
        "show_evidence": True,
    })
    print(result)

    print("\n" + "=" * 60)
    print("5. 表格")
    print("=" * 60)
    result = decoder.render({
        "render_type": "table",
        "facts": [
            {"概念": "电子", "质量": "9.1e-31 kg", "电荷": "-1", "发现者": "汤姆逊"},
            {"概念": "质子", "质量": "1.67e-27 kg", "电荷": "+1", "发现者": "卢瑟福"},
            {"概念": "中子", "质量": "1.67e-27 kg", "电荷": "0", "发现者": "查德威克"},
        ],
    })
    print(result)

    print("\n" + "=" * 60)
    print("6. 风格对比 (formal vs casual)")
    print("=" * 60)
    for style in ["formal", "neutral", "casual"]:
        result = decoder.render({
            "render_type": "explain_path",
            "subject": "细胞",
            "path": ["细胞", "组织", "器官"],
            "edges": [{"rel": "PART_OF", "confidence": 0.95}],
            "style": style,
        })
        print(f"  [{style}] {result}")


if __name__ == "__main__":
    test_energy_decoder()
