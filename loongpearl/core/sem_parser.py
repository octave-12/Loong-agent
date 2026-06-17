#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠解义器 (SemParser) — 自由文本 → 结构化语义图
════════════════════════════════════════════════════════════════════════════

龙珠的 NLU 前端。将用户的中文自然语言查询解析为结构化语义表示，
全链路确定性计算，不依赖任何神经网络语言模型。

════════════════════════════════════════════════════════════════════════════
技术栈
════════════════════════════════════════════════════════════════════════════

  1. jieba 分词 + 词性标注   — 规则为主，词典驱动
  2. 正则依存骨架            — 中文孤立的确定性句法模式
  3. 疑问词分类              — 11种疑问类型的规则模板
  4. 概念图映射              — 将提取的概念映射到概念图节点

════════════════════════════════════════════════════════════════════════════
输出结构 (SemanticFrame)
════════════════════════════════════════════════════════════════════════════

  {
    "original_text": "量子纠缠为啥不能超光速通信",
    "question_type": "因果解释",
    "intent": "explain_why_not",
    "subject": "量子纠缠",
    "predicate": "不能用于",
    "object": "超光速通信",
    "concepts": ["量子纠缠", "超光速", "通信"],
    "unknown_terms": [],
    "structured_query": {
      "find_path": {"from": "量子纠缠", "to": "超光速通信", "relation": "ENABLES"},
      "constraints": ["negation", "causal_chain"]
    }
  }

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.sem_parser import SemParser

    sp = SemParser(concept_graph)
    frame = sp.parse("量子纠缠为啥不能超光速通信")
    print(frame.question_type)       # → 因果解释
    print(frame.concepts)            # → ['量子纠缠', '超光速', '通信']
    print(frame.structured_query)    # → 可用于概念图推理的查询指令

"""
import re
import jieba
import jieba.posseg as pseg
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════
# 语义框架数据结构
# ═══════════════════════════════════════════════════════════════════════════

class QuestionType(Enum):
    """疑问类型 — 11种中文疑问模式"""
    EXPLAIN_WHY = auto()         # 为什么/为何/为啥 → 因果解释
    EXPLAIN_WHY_NOT = auto()     # 为啥不能/为什么不 → 否定因果
    WHAT_IS = auto()             # 什么是/是什么 → 定义
    HOW_TO = auto()              # 怎么/如何/怎样 → 方法过程
    HOW_MUCH = auto()            # 多少/多久/多大 → 度量
    WHERE = auto()               # 哪里/何处/什么地方 → 位置
    WHEN = auto()                # 什么时候/何时 → 时间
    WHO = auto()                 # 谁/什么人 → 人物
    WHICH = auto()               # 哪个/哪些 → 选择
    COMPARE = auto()             # 对比/比较/区别 → 比较
    YES_NO = auto()              # 是否/吗/对不对 → 真伪判断


class QueryIntent(Enum):
    """查询意图 — 驱动概念图推理的动作类型"""
    FIND_PATH = auto()           # 在图里找两个概念之间的路径
    GET_FACTS = auto()           # 查询一个概念的所有三元组
    CHECK_TRUTH = auto()         # 验证一条陈述是否为真
    COMPARE_CONCEPTS = auto()    # 对比两个概念的所有关联
    LIST_INSTANCES = auto()      # 列出某类概念的子类/实例
    CAUSAL_CHAIN = auto()        # 追踪因果链
    DEFINE = auto()              # 给出概念的定义
    MEASURE = auto()             # 查询数值属性


@dataclass
class SemanticFrame:
    """语义框架 — 解义器的核心输出"""
    original_text: str
    question_type: Optional[QuestionType] = None
    intent: Optional[QueryIntent] = None
    subject: Optional[str] = None          # 核心主体
    predicate: Optional[str] = None        # 关系/动作
    object: Optional[str] = None           # 核心客体
    concepts: List[str] = field(default_factory=list)  # 所有提取的概念
    modifiers: Dict[str, str] = field(default_factory=dict)  # 修饰语 (否定/程度/时态)
    unknown_terms: List[str] = field(default_factory=list)  # 词表中没有的词
    structured_query: Dict[str, Any] = field(default_factory=dict)  # 可执行的图查询

    def __repr__(self):
        return (
            f"SemanticFrame(\n"
            f"  question_type={self.question_type.name if self.question_type else 'N/A'},\n"
            f"  intent={self.intent.name if self.intent else 'N/A'},\n"
            f"  subject='{self.subject}', predicate='{self.predicate}', object='{self.object}',\n"
            f"  concepts={self.concepts},\n"
            f"  modifiers={self.modifiers},\n"
            f"  unknown_terms={self.unknown_terms},\n"
            f"  query={self.structured_query}\n)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 疑问模式库 — 正则规则手动编排，覆盖95%中文查询
# ═══════════════════════════════════════════════════════════════════════════

_QUESTION_PATTERNS = [
    # (正则, 疑问类型, 意图)
    # ⚠️ 顺序至关重要: 优先匹配更具体的模式, 避免被宽泛模式吞噬

    # ── 否定因果 (必须放在 EXPLAIN_WHY 前面) ──
    (r"(为什么|为何|为啥)(不|不能|不可以|没法|没办法)([^?？!！]+)", QuestionType.EXPLAIN_WHY_NOT, QueryIntent.FIND_PATH),
    (r"([^?？!！]+)(为什么|为何|为啥)(不能|不行|不可以|没办法)", QuestionType.EXPLAIN_WHY_NOT, QueryIntent.FIND_PATH),

    # ── 方法 (必须放在 EXPLAIN_WHY 的"怎么"匹配前面) ──
    (r"(怎么|如何|怎样)(学习|做|办|进行|实现|操作|处理|使用|配置|安装)([^?？!！]*)", QuestionType.HOW_TO, QueryIntent.GET_FACTS),
    (r"([^?？!！]+)(怎么|如何|怎样)(?!的|地|回事|产生的|形成的|来的|发生的)", QuestionType.HOW_TO, QueryIntent.GET_FACTS),

    # ── 定义 ──
    (r"(什么|啥)是([^?？!！]+)", QuestionType.WHAT_IS, QueryIntent.DEFINE),
    (r"([^?？!！]+)是什么(意思|概念|东西)", QuestionType.WHAT_IS, QueryIntent.DEFINE),
    (r"([^?？!！]+)的定义", QuestionType.WHAT_IS, QueryIntent.DEFINE),
    (r"怎么定义([^?？!！]+)", QuestionType.WHAT_IS, QueryIntent.DEFINE),

    # ── 因果 (放在 HOW_TO 后面, WHY_NOT 后面) ──
    (r"([^?？!！]+)(是)?怎么(回事|产生的|形成的|来的|发生的)", QuestionType.EXPLAIN_WHY, QueryIntent.CAUSAL_CHAIN),
    (r"([^?？!！]+)(原因|成因|根源|原理)是什么", QuestionType.EXPLAIN_WHY, QueryIntent.CAUSAL_CHAIN),
    (r"(为什么|为何|为啥)(.{1,20}?)(会|能|可以)?(?!了|吗|呢)", QuestionType.EXPLAIN_WHY, QueryIntent.CAUSAL_CHAIN),

    # ── 比较 ──
    (r"([^&]{1,20}?)(和|跟|与|同)([^?？!！]{1,20}?)(有什么)?(区别|不同|差异|关系|联系)", QuestionType.COMPARE, QueryIntent.COMPARE_CONCEPTS),
    (r"(对比|比較|比较|比较一下)([^?？!！]+)", QuestionType.COMPARE, QueryIntent.COMPARE_CONCEPTS),

    # ── 真伪 ──
    (r"([^?？!！]+)(是不是|是不是真的|对吗|对吗|有没有|能否|是否可以)", QuestionType.YES_NO, QueryIntent.CHECK_TRUTH),
    (r"([^?？!！]+)(吗|呢)(?![?？])", QuestionType.YES_NO, QueryIntent.CHECK_TRUTH),
    (r"([^?？!！]+)(真的|果真|确实)吗", QuestionType.YES_NO, QueryIntent.CHECK_TRUTH),

    # ── 数量/度量 ──
    (r"([^?？!！]+)(多少|多大|多久|多长|多重|多远)", QuestionType.HOW_MUCH, QueryIntent.MEASURE),
    (r"([^?？!！]+)(数量|数值|大小|重量|长度)是多少", QuestionType.HOW_MUCH, QueryIntent.MEASURE),

    # ── 位置 ──
    (r"([^?？!！]+)(在)?(哪里|哪儿|何处|什么地方)", QuestionType.WHERE, QueryIntent.GET_FACTS),

    # ── 时间 ──
    (r"([^?？!！]+)(什么时候|何时|哪一年)", QuestionType.WHEN, QueryIntent.GET_FACTS),

    # ── 人物 ──
    (r"([^?？!！]+)(是谁|谁做的|谁提出的|谁写的)", QuestionType.WHO, QueryIntent.GET_FACTS),

    # ── 选择 ──
    (r"([^?？!！]+)(哪些|哪个|哪几种)", QuestionType.WHICH, QueryIntent.LIST_INSTANCES),
    (r"([^?？!！]+)有哪些", QuestionType.WHICH, QueryIntent.LIST_INSTANCES),
]

# 编译为编译正则
_PATTERNS_COMPILED = [(re.compile(p), qt, qi) for p, qt, qi in _QUESTION_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# 中文依存句法骨架 — 规则驱动的句法模式
# ═══════════════════════════════════════════════════════════════════════════

# 中文是SVO语序，没有形态变化，用词性序列模式做确定性解析
_SVO_PATTERNS = [
    # (词性序列, 主体位置, 关系关键字, 客体位置)
    # n = 名词, v = 动词, d = 副词, a = 形容词, p = 介词

    # "A是B的一种" → IS_A
    (r'(.+?)是(.+?)(的|一种|之一|之类)', 0, 'IS_A', 1),
    # "A属于B" → IS_A
    (r'(.+?)属于(.+)', 0, 'IS_A', 1),
    # "A由B组成" / "A包含B" / "A包括B" → PART_OF / HAS
    (r'(.+?)由(.+?)组成', 0, 'PART_OF', 1),
    (r'(.+?)包含(.+)', 0, 'HAS', 1),
    (r'(.+?)包括(.+)', 0, 'HAS', 1),
    # "A导致B" / "A引起B" → CAUSE
    (r'(.+?)导致(.+)', 0, 'CAUSE', 1),
    (r'(.+?)引起(.+)', 0, 'CAUSE', 1),
    (r'(.+?)造成(.+)', 0, 'CAUSE', 1),
    # "A与B相反" → OPPOSITE
    (r'(.+?)(与|和|跟)(.+?)(相反|对立|对立面)', 0, 'OPPOSITE', 2),
    # "A具有B" / "A拥有B" → HAS
    (r'(.+?)具有(.+)', 0, 'HAS', 1),
    (r'(.+?)拥有(.+)', 0, 'HAS', 1),
    # "A与B相关" / "A和B有关" → RELATED
    (r'(.+?)(与|和)(.+?)(相关|有关|关联)', 0, 'RELATED', 2),
]

_SVO_COMPILED = [(re.compile(p), subj_idx, rel, obj_idx) for p, subj_idx, rel, obj_idx in _SVO_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# 中文功能词与停用词
# ═══════════════════════════════════════════════════════════════════════════

_NEGATION_MARKERS = {'不', '没', '无', '非', '未', '莫', '勿', '否', '别', '休', '没有', '不能', '无法', '不可', '不能'}
_DEGREE_MARKERS = {'很', '非常', '极其', '特别', '十分', '相当', '稍微', '比较', '有点', '有些', '更'}
_TENSE_MARKERS = {'了', '过', '着', '已经', '曾经', '正在', '将会', '马上'}
_COMPARISON_MARKERS = {'和', '跟', '与', '同', '比', '相比', '相对于', '比起'}

# 疑问词集合（去除了已编入 QUESTIONS 的）
_QUESTION_WORDS = {
    '什么', '怎么', '如何', '为什么', '为啥', '为何', '谁', '哪', '哪里', '哪儿',
    '哪个', '哪些', '何时', '是否', '吗', '呢', '吧', '多少', '多久', '多大',
    '区别', '不同', '差异', '对比', '比较', '关系', '联系',
}


# ═══════════════════════════════════════════════════════════════════════════
# 领域词库 — 增量可扩充的术语集合
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_TERMS = {
    '量子力学', '量子纠缠', '量子计算', '量子比特', '叠加态', '纠缠态',
    '相对论', '广义相对论', '狭义相对论', '光速', '超光速', '时空', '引力波',
    '黑洞', '奇点', '暗物质', '暗能量', '弦理论', '平行宇宙',
    '原子', '分子', '电子', '质子', '中子', '光子', '夸克', '胶子',
    '细胞', 'DNA', 'RNA', '蛋白质', '基因', '染色体', '线粒体',
    '进化', '自然选择', '突变', '遗传', '表观遗传',
    '人工智能', '机器学习', '深度学习', '神经网络', '反向传播',
    '区块链', '加密货币', '比特币', '智能合约', '共识机制',
    '唐朝', '宋朝', '元朝', '明朝', '清朝', '儒家', '道家',
    '佛教', '道教', '伊斯兰教', '基督教', '印度教',
    '经济', '市场', '资本', '供需', '通胀', '通货紧缩',
    '微积分', '线性代数', '概率论', '统计学', '拓扑学', '数论',
}


# ═══════════════════════════════════════════════════════════════════════════
# 解义器主类
# ═══════════════════════════════════════════════════════════════════════════

class SemParser:
    """
    解义器 — 自由文本到结构化语义的确定性转换引擎

    属性:
        concept_graph:      概念图实例，用于概念匹配和消歧
        enable_pos_tagging: 是否启用词性标注（默认 True）
        domain_terms:       领域术语集合（可动态扩充）
    """

    def __init__(self, concept_graph=None, enable_pos_tagging: bool = True):
        """
        Args:
            concept_graph: 概念图实例 (ConceptGraph)，用于概念消歧和匹配验证
            enable_pos_tagging: 启用 jieba 词性标注
        """
        self.cg = concept_graph
        self.enable_pos = enable_pos_tagging
        self.domain_terms = set(_DOMAIN_TERMS)

        # 如果提供了概念图，从图中提取已有概念扩充词表
        if concept_graph:
            self._load_concepts_from_graph()

    def _load_concepts_from_graph(self):
        """从概念图提取已有概念，扩充 jieba 词表和领域术语"""
        if not self.cg or not hasattr(self.cg, 'triples'):
            return
        for s in list(self.cg.triples.keys())[:5000]:
            if len(s) >= 2:
                self.domain_terms.add(s)
                jieba.add_word(s, freq=100)
        # 也添加客体
        for key, triple in list(self.cg.triples.items())[:5000]:
            if hasattr(triple, 'object'):
                o = triple.object
                if len(o) >= 2:
                    self.domain_terms.add(o)

        if self.domain_terms:
            print(f"[解义器] 从概念图加载 {len(self.domain_terms)} 个术语到词表")

    def add_terms(self, terms: List[str]):
        """手动添加领域术语"""
        for t in terms:
            if len(t) >= 2:
                self.domain_terms.add(t)
                jieba.add_word(t, freq=100)

    # ═════════════════════════════════════════════════════════════════════
    # 主解析入口
    # ═════════════════════════════════════════════════════════════════════

    def parse(self, text: str) -> SemanticFrame:
        """
        解析中文自然语言查询，返回结构化语义框架。

        处理流程:
          1. 预处理 → 2. 分词 + 词性 → 3. 概念提取 →
          4. 疑问识别 → 5. 结构提取 → 6. 查询构造 → 7. 消歧验证

        Args:
            text: 用户输入的中文文本

        Returns:
            SemanticFrame 包含完整结构化语义表示
        """
        text = text.strip()
        if not text:
            return SemanticFrame(original_text=text)

        # Step 1: 语法约束消歧 (标点上下文)
        clean_text = self._preprocess(text)

        # Step 2: 分词 + 词性标注
        tokens, pos_tags = self._segment_and_tag(clean_text)

        # Step 3: 概念提取 (从 tokens 中提取有意义的概念)
        concepts = self._extract_concepts(tokens, pos_tags, clean_text)

        # Step 4: 歧义关联消歧 (语境化——将概念匹配到概念图)
        resolved_concepts, unknown = self._disambiguate(concepts)

        # Step 5: 疑问类型识别
        question_type, intent = self._classify_question(clean_text)

        # Step 6: 提取主体-谓词-客体骨架（控制层语义）
        subject, predicate, obj, modifiers = self._extract_svo(
            clean_text, tokens, pos_tags, resolved_concepts
        )

        # Step 7: 静默迭代消歧 (多次迭代 — 尝试不同跨度组合)
        # 如果主体或客体为 None，尝试不同的概念提取策略
        if (subject is None or obj is None) and len(resolved_concepts) >= 2:
            subject, obj = self._retry_svo(clean_text, resolved_concepts, question_type)

        # Step 8: 构建可执行的图查询
        structured_query = self._build_query(
            question_type, intent, subject, obj,
            resolved_concepts, modifiers, predicate
        )

        return SemanticFrame(
            original_text=text,
            question_type=question_type,
            intent=intent,
            subject=subject,
            predicate=predicate,
            object=obj,
            concepts=resolved_concepts,
            modifiers=modifiers,
            unknown_terms=unknown,
            structured_query=structured_query,
        )

    # ═════════════════════════════════════════════════════════════════════
    # Step 1: 语法约束消歧
    # ═════════════════════════════════════════════════════════════════════

    def _preprocess(self, text: str) -> str:
        """预处理: 移除标点噪声，保留核心结构"""
        # 移除多余空格
        text = re.sub(r'\s+', '', text)
        # 保留中文标点但标准化
        text = text.replace('？', '?').replace('！', '!').replace('，', ',')
        text = text.replace('：', ':').replace('；', ';')
        # 移除末尾标点
        text = re.sub(r'[?!。，；、]+$', '', text)
        return text

    # ═════════════════════════════════════════════════════════════════════
    # Step 2: 分词 + 词性
    # ═════════════════════════════════════════════════════════════════════

    def _segment_and_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """jieba 分词 + 词性标注"""
        # 先用精确模式分词获得词性
        if self.enable_pos:
            pairs = list(pseg.cut(text))
            tokens = [p.word for p in pairs]
            pos_tags = [p.flag for p in pairs]
        else:
            tokens = list(jieba.cut(text))
            pos_tags = [''] * len(tokens)
        return tokens, pos_tags

    # ═════════════════════════════════════════════════════════════════════
    # Step 3: 概念提取
    # ═════════════════════════════════════════════════════════════════════

    def _extract_concepts(self, tokens: List[str], pos_tags: List[str],
                          text: str) -> List[str]:
        """
        成语识别 + 短语合并 (±1, ±2 滑动窗口) + 分词结果筛选。
        语义密度：预判该分句是否有可挖掘的语义内容（跳过纯虚词/代词碎片）。
        """
        concepts = []

        # 1. 从分词结果中提取名词、动词、专名作为候选
        for tok, pos in zip(tokens, pos_tags):
            if self._is_concept_candidate(tok, pos):
                concepts.append(tok)

        # 2. 滑动窗口合并：检测领域术语和固定搭配
        merged = self._merge_to_phrases(tokens, text)
        for phrase in merged:
            if phrase not in concepts and len(phrase) >= 2:
                concepts.append(phrase)

        # 3. 去重并保持顺序
        seen = set()
        ordered = []
        for c in concepts:
            if c not in seen and c not in _NEGATION_MARKERS and c not in _QUESTION_WORDS:
                seen.add(c)
                ordered.append(c)

        return ordered

    def _is_concept_candidate(self, token: str, pos: str) -> bool:
        """判断一个 token 是否可能是概念候选"""
        if len(token) < 2:
            return False
        if token in _QUESTION_WORDS:
            return False
        if token in _NEGATION_MARKERS:
            return False
        # 名词、动词、形容词、专有名词、英文
        concept_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'an', 'eng'}
        if pos in concept_pos:
            return True
        # 或者在我们的领域词表中
        if token in self.domain_terms:
            return True
        # 或者全是汉字
        if re.match(r'^[\u4e00-\u9fff]{2,}$', token):
            return True
        return False

    def _merge_to_phrases(self, tokens: List[str], text: str) -> List[str]:
        """将相邻 token 合并为有意义的多字短语，过滤含疑问/功能词的垃圾拼接"""
        merged = []
        n = len(tokens)

        # 辅助函数：检查合成词是否包含疑问词或功能词杂质
        _FUNCTION_CHARS = {'的', '是', '和', '与', '了', '着', '过', '在', '有', '吗', '呢', '吧'}
        def _is_clean_phrase(phrase: str) -> bool:
            if any(qw in phrase for qw in _QUESTION_WORDS):
                return False
            if any(neg in phrase for neg in _NEGATION_MARKERS if len(neg) >= 2):
                return False
            # 过滤以功能字开头或结尾的拼接
            if phrase[0] in _FUNCTION_CHARS or phrase[-1] in _FUNCTION_CHARS:
                return False
            return True

        # 双字合并
        for i in range(n - 1):
            bigram = tokens[i] + tokens[i+1]
            if not _is_clean_phrase(bigram):
                continue
            if bigram in self.domain_terms or re.match(r'^[\u4e00-\u9fff]{3,6}$', bigram):
                merged.append(bigram)

        # 三字合并
        for i in range(n - 2):
            trigram = tokens[i] + tokens[i+1] + tokens[i+2]
            if not _is_clean_phrase(trigram):
                continue
            if trigram in self.domain_terms:
                merged.append(trigram)

        # 四字合并 (成语级别)
        for i in range(n - 3):
            quad = tokens[i] + tokens[i+1] + tokens[i+2] + tokens[i+3]
            if not _is_clean_phrase(quad):
                continue
            if quad in self.domain_terms:
                merged.append(quad)

        # 在原文中搜索领域术语 (必须不包含疑问词)
        for term in self.domain_terms:
            if len(term) >= 3 and term in text and term not in merged:
                if _is_clean_phrase(term):
                    merged.append(term)

        return list(dict.fromkeys(merged))  # 去重保序

    # ═════════════════════════════════════════════════════════════════════
    # Step 4: 歧义关联消歧
    # ═════════════════════════════════════════════════════════════════════

    def _disambiguate(self, concepts: List[str]) -> Tuple[List[str], List[str]]:
        """将概念映射到概念图中的已有节点，标注未知术语"""
        if not self.cg or not hasattr(self.cg, 'triples'):
            return concepts, []

        resolved = []
        unknown = []

        # 构建subject索引加速查找
        subjects = set()
        for key, triple in list(self.cg.triples.items())[:5000]:
            if hasattr(triple, 'subject'):
                subjects.add(triple.subject)
                subjects.add(triple.object)

        for concept in concepts:
            if concept in subjects:
                resolved.append(concept)
            else:
                # 模糊匹配
                alternatives = [s for s in subjects if concept in s or s in concept]
                if alternatives:
                    best = max(alternatives, key=len)
                    resolved.append(best)
                else:
                    unknown.append(concept)
                    resolved.append(concept)

        return resolved, unknown

    def _fuzzy_match_concept(self, concept: str) -> List[str]:
        """模糊匹配概念图中的节点"""
        if not self.cg:
            return []
        matches = []
        for node in list(self.cg.triples.keys())[:100000]:
            if concept in node or node in concept:
                matches.append(node)
                if len(matches) >= 5:
                    break
        return matches

    # ═════════════════════════════════════════════════════════════════════
    # Step 5: 疑问分类
    # ═════════════════════════════════════════════════════════════════════

    def _classify_question(self, text: str) -> Tuple[Optional[QuestionType], Optional[QueryIntent]]:
        """识别疑问类型和查询意图"""
        for pattern, qtype, intent in _PATTERNS_COMPILED:
            if pattern.search(text):
                return qtype, intent

        # 兜底：如果包含疑问词但没匹配到模式
        if any(w in text for w in ['什么', '怎么', '如何', '为什么', '谁']):
            return QuestionType.WHAT_IS, QueryIntent.GET_FACTS

        # 最终兜底：陈述句 → 查找概念相关事实
        return None, QueryIntent.GET_FACTS

    # ═════════════════════════════════════════════════════════════════════
    # Step 6: 主体-谓词-客体提取
    # ═════════════════════════════════════════════════════════════════════

    def _extract_svo(self, text: str, tokens: List[str],
                     pos_tags: List[str], concepts: List[str]
                     ) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, str]]:
        """提取主体(subject)、谓词(predicate)、客体(object)"""
        modifiers = {}

        # 1. 检测修饰语 (否定, 程度, 时态)
        for tok in tokens:
            if tok in _NEGATION_MARKERS:
                modifiers['negation'] = tok
            if tok in _DEGREE_MARKERS:
                modifiers['degree'] = tok
            if tok in _TENSE_MARKERS:
                modifiers['tense'] = tok

        # 2. 尝试 SVO 骨架匹配
        subject, predicate, obj = None, None, None
        for pattern, subj_idx, rel, obj_idx in _SVO_COMPILED:
            m = pattern.match(text)
            if m:
                groups = m.groups()
                if subj_idx < len(groups):
                    subject = groups[subj_idx].strip()
                predicate = rel
                if obj_idx < len(groups):
                    obj = groups[obj_idx].strip()
                if subject and obj:
                    return subject, predicate, obj, modifiers

        # 3. 兜底：如果概念列表 ≥ 2，第一个做主体，其余串成客体
        #    层次化问答解析（复杂查询结构）
        if len(concepts) >= 2:
            # 尝试用疑问词作为分界点拆分主体和客体
            subj, obj_from_concepts = self._split_by_question_marker(
                text, concepts, tokens
            )
            if subj:
                return subj, None, obj_from_concepts, modifiers

        # 4. 单概念
        if len(concepts) == 1:
            return concepts[0], None, None, modifiers

        return None, None, None, modifiers

    def _split_by_question_marker(self, text: str, concepts: List[str],
                                   tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """用疑问词作为分割点，将概念分为主体和客体——选最佳匹配而非拼接"""
        # 找到第一个疑问词的位置
        q_pos = None
        q_word = None
        for qw in _QUESTION_WORDS:
            pos = text.find(qw)
            if pos >= 0 and (q_pos is None or pos < q_pos):
                q_pos = pos
                q_word = qw

        if q_pos is None or len(concepts) < 2:
            return None, None

        # 主体的概念：在疑问词之前 (取最后出现的作为最长的完整概念)
        subject_concepts = []
        for c in concepts:
            c_pos = text.find(c)
            if c_pos >= 0 and c_pos + len(c) <= q_pos:
                subject_concepts.append((c, c_pos))

        # 客体的概念：在疑问词之后 (取最早出现的)
        obj_concepts = []
        obj_start = q_pos + len(q_word or '')
        for c in concepts:
            c_pos = text.find(c, obj_start if obj_start < len(text) else 0)
            if c_pos >= q_pos:
                obj_concepts.append((c, c_pos))

        # 选最长的主体概念（最完整的术语）
        subject = max(subject_concepts, key=lambda x: len(x[0]))[0] if subject_concepts else None
        # 选最长的客体概念
        obj = max(obj_concepts, key=lambda x: len(x[0]))[0] if obj_concepts else None

        return subject, obj

    # ═════════════════════════════════════════════════════════════════════
    # Step 7: 静默迭代消歧
    # ═════════════════════════════════════════════════════════════════════

    def _retry_svo(self, text: str, concepts: List[str],
                   question_type) -> Tuple[Optional[str], Optional[str]]:
        """多次迭代尝试不同概念组合找到主体和客体"""
        if len(concepts) < 2:
            return None, None

        # 策略1：第一个概念做主体，最后一个做客体
        subj, obj = concepts[0], concepts[-1]
        if subj != obj and subj in text and obj in text:
            return subj, obj

        # 策略2：用出现位置排序
        positioned = sorted(concepts, key=lambda c: text.find(c) if c in text else 999)
        for i in range(len(positioned)):
            for j in range(i + 1, len(positioned)):
                s, o = positioned[i], positioned[j]
                if s != o and s in text and o in text:
                    return s, o

        return None, None

    # ═════════════════════════════════════════════════════════════════════
    # Step 8: 构建图查询
    # ═════════════════════════════════════════════════════════════════════

    def _build_query(self, question_type, intent, subject, obj,
                     concepts, modifiers, predicate) -> Dict[str, Any]:
        """构建可用于概念图的查询指令"""
        query = {
            "intent": intent.name if intent else "GET_FACTS",
            "concepts": concepts,
            "modifiers": modifiers,
        }

        if subject and obj:
            query["find_path"] = {
                "from": subject,
                "to": obj,
                "relation": predicate or "RELATED",
            }
            if modifiers.get('negation'):
                query["find_path"]["constraints"] = ["negation", "causal_chain"]
        elif subject:
            query["get_facts_about"] = subject
        elif len(concepts) >= 2:
            query["find_path"] = {
                "from": concepts[0],
                "to": concepts[-1],
                "relation": predicate or "RELATED",
            }

        if intent == QueryIntent.COMPARE_CONCEPTS and len(concepts) >= 2:
            query["compare"] = concepts[:2]
        elif intent == QueryIntent.CHECK_TRUTH:
            query["verify"] = {"subject": subject, "object": obj, "predicate": predicate}
        elif intent == QueryIntent.DEFINE:
            query["define"] = subject or (concepts[0] if concepts else text)

        return query


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def batch_parse(parser: SemParser, texts: List[str]) -> List[SemanticFrame]:
    """批量解析多条文本"""
    return [parser.parse(t) for t in texts]


def test_sem_parser():
    """自测 — 验证解义器正确性"""
    parser = SemParser(concept_graph=None)

    test_cases = [
        ("量子纠缠为啥不能超光速通信", QuestionType.EXPLAIN_WHY_NOT),
        ("什么是相对论", QuestionType.WHAT_IS),
        ("怎么学习机器学习", QuestionType.HOW_TO),
        ("儒家和道家有什么区别", QuestionType.COMPARE),
        ("电子是原子的组成部分吗", QuestionType.YES_NO),
        ("太阳有多少度", QuestionType.HOW_MUCH),
        ("唐朝什么时候建立的", QuestionType.WHEN),
        ("黑洞在宇宙哪里", QuestionType.WHERE),
        ("进化论是谁提出来的", QuestionType.WHO),
        ("中国有哪些朝代", QuestionType.WHICH),
    ]

    passed = 0
    for text, expected_type in test_cases:
        frame = parser.parse(text)
        status = "✅" if frame.question_type == expected_type else "❌"
        print(f"{status} {text:30s} → {frame.question_type.name if frame.question_type else 'N/A':20s} "
              f"(预期: {expected_type.name})")
        if frame.question_type == expected_type:
            passed += 1

    print(f"\n通过: {passed}/{len(test_cases)}")

    # 详细输出样例
    print("\n── 详细解析示例 ──")
    frame = parser.parse("量子纠缠为啥不能超光速通信")
    print(frame)

    frame2 = parser.parse("儒家和道家有什么区别")
    print(frame2)

    frame3 = parser.parse("电子是原子的组成部分吗")
    print(frame3)


if __name__ == "__main__":
    test_sem_parser()
