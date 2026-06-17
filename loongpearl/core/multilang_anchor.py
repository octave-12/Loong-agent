#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠万语锚 (MultiLang) — 跨语言锚点与概念对齐
════════════════════════════════════════════════════════════════════════════

当前94K汉字锚点覆盖CJK统一表意文字，但概念空间应该是语言无关的。
万语锚实现跨语言扩展：

  第一阶段: 日韩越汉字 — CJK统一表意文字内，只需加语言标注
  第二阶段: 英文词锚点 — 用英文嵌入模型生成锚点，对齐到同一概念空间
  第三阶段: 概念层映射 — 概念节点是语言无关的，多余语言入口映射到同一节点

════════════════════════════════════════════════════════════════════════════
核心设计
════════════════════════════════════════════════════════════════════════════

  概念ID (语言无关)
      ├── zh: "量子纠缠"   ← 从字场组合的嵌入
      ├── en: "quantum entanglement"  ← 从英文嵌入模型生成
      ├── ja: "量子もつれ"
      └── ko: "양자 얽힘"

  查询时:
    中文查询 → 中文锚点 → 概念ID
    英文查询 → 英文锚点 → 同一概念ID

════════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════════

    from loongpearl.core.multilang import MultiLangAnchor

    mla = MultiLangAnchor(concept_graph, zichang)

    # 添加英文词嵌入
    mla.add_english_anchor("quantum", embedding)

    # 跨语言概念映射
    concepts = mla.map_to_concepts("quantum entanglement", lang="en")

    # 反向查找
    zh_name = mla.get_concept_name("quantum_mechanics", lang="zh")

"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LanguageAnchor:
    """单语言锚点"""
    text: str                     # 该语言中的表达式
    lang: str                     # 语言代码 (zh/en/ja/ko/...)
    embedding: Optional[np.ndarray] = None  # 嵌入向量
    concept_id: Optional[str] = None  # 映射到的概念ID
    is_hanzi_based: bool = False  # 是否基于汉字的组合嵌入

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "lang": self.lang,
            "concept_id": self.concept_id,
            "is_hanzi_based": self.is_hanzi_based,
        }


@dataclass
class ConceptEntry:
    """跨语言概念条目"""
    concept_id: str               # 语言无关的概念ID
    names: Dict[str, List[str]] = field(default_factory=dict)  # lang → [名称列表]
    primary_lang: str = "zh"      # 主语言
    domain: str = ""              # 领域

    def get_name(self, lang: str = "zh") -> Optional[str]:
        """获取指定语言的名称"""
        names = self.names.get(lang, [])
        return names[0] if names else None

    def add_name(self, lang: str, name: str):
        """添加一个语言的名称"""
        if lang not in self.names:
            self.names[lang] = []
        if name not in self.names[lang]:
            self.names[lang].append(name)


# ═══════════════════════════════════════════════════════════════════════════
# 预置跨语言概念映射
# ═══════════════════════════════════════════════════════════════════════════

_SEED_CROSSLINGUAL_MAP = {
    # 物理学
    "quantum_mechanics": {
        "zh": ["量子力学"],
        "en": ["quantum mechanics"],
        "domain": "物理",
    },
    "relativity": {
        "zh": ["相对论"],
        "en": ["relativity", "theory of relativity"],
        "domain": "物理",
    },
    "electron": {
        "zh": ["电子"],
        "en": ["electron"],
        "domain": "物理",
    },
    "atom": {
        "zh": ["原子"],
        "en": ["atom"],
        "domain": "物理",
    },
    "photon": {
        "zh": ["光子"],
        "en": ["photon"],
        "domain": "物理",
    },
    "gravity": {
        "zh": ["引力", "重力"],
        "en": ["gravity", "gravitation"],
        "domain": "物理",
    },
    "entropy": {
        "zh": ["熵"],
        "en": ["entropy"],
        "domain": "物理",
    },

    # 生物学
    "cell": {
        "zh": ["细胞"],
        "en": ["cell"],
        "domain": "生物",
    },
    "dna": {
        "zh": ["DNA", "脱氧核糖核酸"],
        "en": ["DNA", "deoxyribonucleic acid"],
        "domain": "生物",
    },
    "evolution": {
        "zh": ["进化", "演化"],
        "en": ["evolution"],
        "domain": "生物",
    },
    "gene": {
        "zh": ["基因"],
        "en": ["gene"],
        "domain": "生物",
    },

    # 计算机
    "artificial_intelligence": {
        "zh": ["人工智能"],
        "en": ["artificial intelligence", "AI"],
        "domain": "计算机",
    },
    "machine_learning": {
        "zh": ["机器学习"],
        "en": ["machine learning"],
        "domain": "计算机",
    },
    "neural_network": {
        "zh": ["神经网络"],
        "en": ["neural network"],
        "domain": "计算机",
    },
    "algorithm": {
        "zh": ["算法"],
        "en": ["algorithm"],
        "domain": "计算机",
    },

    # 数学
    "calculus": {
        "zh": ["微积分"],
        "en": ["calculus"],
        "domain": "数学",
    },
    "linear_algebra": {
        "zh": ["线性代数"],
        "en": ["linear algebra"],
        "domain": "数学",
    },
    "probability": {
        "zh": ["概率", "概率论"],
        "en": ["probability", "probability theory"],
        "domain": "数学",
    },

    # 化学
    "molecule": {
        "zh": ["分子"],
        "en": ["molecule"],
        "domain": "化学",
    },
    "catalyst": {
        "zh": ["催化剂"],
        "en": ["catalyst"],
        "domain": "化学",
    },
    "oxidation": {
        "zh": ["氧化"],
        "en": ["oxidation"],
        "domain": "化学",
    },

    # 哲学
    "confucianism": {
        "zh": ["儒家", "儒学", "儒家思想"],
        "en": ["Confucianism"],
        "domain": "哲学",
    },
    "taoism": {
        "zh": ["道家", "道教", "道家思想"],
        "en": ["Taoism", "Daoism"],
        "domain": "哲学",
    },
    "buddhism": {
        "zh": ["佛教", "佛学"],
        "en": ["Buddhism"],
        "domain": "哲学",
    },
    "dialectics": {
        "zh": ["辩证法"],
        "en": ["dialectics"],
        "domain": "哲学",
    },

    # 经济
    "inflation": {
        "zh": ["通货膨胀", "通胀"],
        "en": ["inflation"],
        "domain": "经济",
    },
    "supply_demand": {
        "zh": ["供需", "供求关系"],
        "en": ["supply and demand"],
        "domain": "经济",
    },
    "gdp": {
        "zh": ["GDP", "国内生产总值"],
        "en": ["GDP", "gross domestic product"],
        "domain": "经济",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 万语锚主类
# ═══════════════════════════════════════════════════════════════════════════

class MultiLangAnchor:
    """
    万语锚 — 跨语言概念映射与查询。

    三层架构:
      Layer 1: 语言锚点 (text + embedding)
      Layer 2: 概念ID (语言无关)
      Layer 3: 概念图节点 (与知识图谱集成)
    """

    def __init__(self, concept_graph=None, zichang=None):
        self.cg = concept_graph
        self.zichang = zichang

        # 概念注册表: concept_id → ConceptEntry
        self.concepts: Dict[str, ConceptEntry] = {}
        # 名称反向索引: (lang, text) → concept_id
        self.name_index: Dict[Tuple[str, str], str] = {}
        # 英文锚点: text → embedding
        self.english_anchors: Dict[str, np.ndarray] = {}

        # 加载种子映射
        self._seed_crosslingual_map()

    # ═════════════════════════════════════════════════════════════════════
    # 概念注册
    # ═════════════════════════════════════════════════════════════════════

    def register_concept(self, concept_id: str, names: Dict[str, List[str]],
                         primary_lang: str = "zh", domain: str = ""):
        """注册一个跨语言概念"""
        entry = ConceptEntry(
            concept_id=concept_id,
            names=names,
            primary_lang=primary_lang,
            domain=domain,
        )
        self.concepts[concept_id] = entry

        # 更新反向索引
        for lang, name_list in names.items():
            for name in name_list:
                self.name_index[(lang, name)] = concept_id

    def add_name(self, concept_id: str, lang: str, name: str):
        """为已有概念添加一个语言的名称"""
        if concept_id in self.concepts:
            self.concepts[concept_id].add_name(lang, name)
            self.name_index[(lang, name)] = concept_id

    # ═════════════════════════════════════════════════════════════════════
    # 英文锚点
    # ═════════════════════════════════════════════════════════════════════

    def add_english_anchor(self, text: str, embedding: np.ndarray):
        """添加英文词嵌入锚点"""
        self.english_anchors[text] = embedding

    def add_english_anchors_batch(self,
                                   texts: List[str],
                                   embeddings: List[np.ndarray]):
        """批量添加英文锚点"""
        for text, emb in zip(texts, embeddings):
            self.english_anchors[text] = emb

    def get_english_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取英文词的嵌入"""
        # 先查精确匹配
        if text in self.english_anchors:
            return self.english_anchors[text]

        # 尝试小写匹配
        text_lower = text.lower()
        for key, emb in self.english_anchors.items():
            if key.lower() == text_lower:
                return emb

        return None

    # ═════════════════════════════════════════════════════════════════════
    # 跨语言查询
    # ═════════════════════════════════════════════════════════════════════

    def map_to_concepts(self, text: str, lang: str = "zh") -> List[str]:
        """
        将任意语言的文本映射到概念ID。

        Args:
            text: 查询文本
            lang: 语言代码

        Returns:
            匹配的概念ID列表（按相关度排序）
        """
        concept_ids = []

        # 1. 精确名称匹配
        if (lang, text) in self.name_index:
            concept_ids.append(self.name_index[(lang, text)])

        # 2. 模糊匹配: 文本包含概念名或概念名包含文本
        for (l, name), cid in self.name_index.items():
            if l == lang:
                if text in name or name in text:
                    if cid not in concept_ids:
                        concept_ids.append(cid)

        # 3. 如果在概念图中，尝试概念节点匹配
        if self.cg and hasattr(self.cg, 'triples') and lang == "zh":
            if text in self.cg.triples:
                cid = f"zh_{text}"
                if cid not in concept_ids:
                    concept_ids.append(cid)

        return concept_ids

    def get_concept_name(self, concept_id: str, lang: str = "zh") -> Optional[str]:
        """获取概念在指定语言中的名称"""
        entry = self.concepts.get(concept_id)
        if entry:
            return entry.get_name(lang)
        return None

    def get_all_names(self, concept_id: str) -> Dict[str, List[str]]:
        """获取概念在所有语言中的名称"""
        entry = self.concepts.get(concept_id)
        return entry.names if entry else {}

    def translate(self, concept_id: str, target_lang: str) -> Optional[str]:
        """'翻译' — 将概念ID转为目标语言名称"""
        return self.get_concept_name(concept_id, target_lang)

    # ═════════════════════════════════════════════════════════════════════
    # 嵌入级跨语言对齐
    # ═════════════════════════════════════════════════════════════════════

    def align_english_to_chinese(self,
                                  en_text: str,
                                  en_embedding: np.ndarray,
                                  top_k: int = 5) -> List[Tuple[str, float]]:
        """
        用嵌入相似度将英文词对齐到中文汉字锚点。

        这需要字场(zichang)提供中文嵌入查询能力。
        """
        if not self.zichang or not hasattr(self.zichang, 'anchors'):
            return []

        # 用字场的 find_nearest 找最近的中文锚点
        import torch
        query_tensor = torch.from_numpy(en_embedding).float()

        if hasattr(self.zichang, 'find_nearest'):
            indices, chars, scores = self.zichang.find_nearest(
                query_tensor, k=min(top_k, 100)
            )
            return list(zip(chars, scores.tolist()))

        # 手动余弦相似度
        anchors = self.zichang.anchors
        if anchors is not None:
            with torch.no_grad():
                query_norm = query_tensor / query_tensor.norm()
                anchors_norm = anchors / anchors.norm(dim=1, keepdim=True)
                sims = torch.mm(query_norm.unsqueeze(0), anchors_norm.T).squeeze()
                top_vals, top_idx = torch.topk(sims, min(top_k, len(sims)))
                return [(self.zichang.hanzi_list[i], v.item())
                        for i, v in zip(top_idx, top_vals)]

        return []

    # ═════════════════════════════════════════════════════════════════════
    # 语言检测
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言"""
        # 检测是否包含汉字
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh"
        # 检测日文假名
        if any('\u3040' <= c <= '\u309f' for c in text) or \
           any('\u30a0' <= c <= '\u30ff' for c in text):
            return "ja"
        # 检测韩文
        if any('\uac00' <= c <= '\ud7af' for c in text):
            return "ko"
        # 默认英文/拉丁
        return "en"

    # ═════════════════════════════════════════════════════════════════════
    # 种子映射加载
    # ═════════════════════════════════════════════════════════════════════

    def _seed_crosslingual_map(self):
        """加载预置的跨语言概念映射"""
        for concept_id, data in _SEED_CROSSLINGUAL_MAP.items():
            names = {k: v for k, v in data.items() if k != "domain"}
            self.register_concept(
                concept_id=concept_id,
                names=names,
                primary_lang="zh",
                domain=data.get("domain", ""),
            )

    # ═════════════════════════════════════════════════════════════════════
    # 持久化
    # ═════════════════════════════════════════════════════════════════════

    def save(self, path: str):
        """保存万语锚"""
        data = {
            "concepts": {},
            "english_anchor_count": len(self.english_anchors),
        }
        for cid, entry in self.concepts.items():
            data["concepts"][cid] = {
                "names": entry.names,
                "primary_lang": entry.primary_lang,
                "domain": entry.domain,
            }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载万语锚"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for cid, cdata in data.get("concepts", {}).items():
            self.register_concept(
                concept_id=cid,
                names=cdata["names"],
                primary_lang=cdata.get("primary_lang", "zh"),
                domain=cdata.get("domain", ""),
            )

    # ═════════════════════════════════════════════════════════════════════
    # 统计
    # ═════════════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        lang_counts = defaultdict(int)
        for entry in self.concepts.values():
            for lang in entry.names:
                lang_counts[lang] += 1

        return {
            "total_concepts": len(self.concepts),
            "language_coverage": dict(lang_counts),
            "english_anchors": len(self.english_anchors),
            "name_index_entries": len(self.name_index),
        }

    def print_stats(self):
        """打印统计"""
        s = self.stats()
        print(f"═══ 万语锚统计 ═══")
        print(f"  跨语言概念: {s['total_concepts']:>6}")
        for lang, count in s['language_coverage'].items():
            print(f"    {lang}: {count} 个名称")
        print(f"  英文锚点:   {s['english_anchors']:>6}")
        print(f"  反向索引:   {s['name_index_entries']:>6}")


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_multilang():
    """自测万语锚"""
    mla = MultiLangAnchor()
    mla.print_stats()

    print("\n" + "=" * 60)
    print("1. 语言检测")
    tests = ["量子力学", "quantum mechanics", "量子もつれ", "양자역학"]
    for t in tests:
        print(f"  '{t}' → {mla.detect_language(t)}")

    print("\n2. 中文→概念ID")
    cids = mla.map_to_concepts("量子力学")
    print(f"  '量子力学' → {cids}")
    for cid in cids:
        all_names = mla.get_all_names(cid)
        print(f"    所有名称: {all_names}")

    print("\n3. 概念ID→英文名 (翻译)")
    en_name = mla.translate("quantum_mechanics", "en")
    print(f"  quantum_mechanics → en: {en_name}")
    zh_name = mla.translate("quantum_mechanics", "zh")
    print(f"  quantum_mechanics → zh: {zh_name}")

    print("\n4. 英文→概念ID")
    cids = mla.map_to_concepts("artificial intelligence", lang="en")
    print(f"  'artificial intelligence' → {cids}")

    print("\n5. 模糊匹配")
    cids = mla.map_to_concepts("进化论")  # 会匹配到 "进化"
    print(f"  '进化论' → {cids}")

    print("\n6. 未注册概念")
    cids = mla.map_to_concepts("暗物质", lang="zh")  # 不在预置表里
    print(f"  '暗物质' → {cids} (预期空)")

    # 持久化
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "multilang_test.json")
    mla.save(tmp)
    mla2 = MultiLangAnchor()
    mla2.load(tmp)
    print(f"\n7. 持久化: {mla.stats()['total_concepts']}概念 → "
          f"加载 {mla2.stats()['total_concepts']}概念")
    os.remove(tmp)


if __name__ == "__main__":
    test_multilang()
