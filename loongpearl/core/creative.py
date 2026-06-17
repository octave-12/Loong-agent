#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠创意引擎 (Creative) — 约束性创作：格律诗词 + 成语接龙 + 事实叙事
════════════════════════════════════════════════════════════════════════════

不做开放创作（小说/散文）——那需要随机解码器，违背确定性设计。
只做有规则约束的创作形式：
  1. 格律诗词   — 五言/七言，押韵规则，平仄检查，词能量引导选字
  2. 成语接龙   — 字场嵌入相似度 + 尾字匹配
  3. 事实叙事   — 概念图路径 → 故事化渲染
  4. 对联       — 对仗规则 + 平仄检查

════════════════════════════════════════════════════════════════════════════
核心原理
════════════════════════════════════════════════════════════════════════════

格律诗词 = 规则约束 × 字场能量评分：
  对于每个位置，根据格律规则（平仄、押韵）筛选候选字，
  然后用能量景观对候选字排序，选能量最低（最自然）的字。

事实叙事 = 概念图路径 → 时序展开：
  取概念图中的一条路径，用化能器的模板渲染成叙事文。

════════════════════════════════════════════════════════════════════════════
"""

import re
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# 中文音韵数据
# ═══════════════════════════════════════════════════════════════════════════

# 平声字（阴平+阳平，现代汉语一声+二声）
_PING_SHENG = set(
    "春江花朝秋月夜人天风中云山光明清水流长高深红香金"
    "银楼台亭阁门桥城关河湖海洋田园林兰梅竹菊松桃李"
    "龙凰麒麟鹏鸾鸳鸯鸿鹄牛羊驼名字辞诗书文章言辞"
    "中华神州炎黄天地乾坤阴阳方圆东西南北来归逢迎"
    "耕耘锄犁收割丰登和平安昌繁荣兴隆团圆欢欣"
)

# 仄声字（上声+去声，三声+四声）
_ZE_SHENG = set(
    "去梦断路远尽老古旧往事刻漏岁月日暮夜晚雪雨露"
    "草木叶落果满翠绿紫碧玉锦锦绣画笔墨纸砚剑戟"
    "虎豹象马犬鹤燕雀鸟凤影动静起坐立卧步履走跳"
    "万丈百千亿兆里外上下左右前后表里内外远近"
    "快乐喜悦愤怒怨恨爱恨喜怒哀乐好坏美丑善恶"
)

# 韵母分组 (简化 — 常用押韵组)
_RHYME_GROUPS = {
    "ang": {"江", "光", "霜", "香", "长", "芳", "阳", "茫", "黄", "王", 
            "方", "堂", "康", "扬", "藏", "旁", "章", "强", "梁", "凉"},
    "an":  {"天", "山", "关", "寒", "安", "然", "烟", "前", "年", "闲",
            "间", "边", "眠", "泉", "篇", "弦", "缘", "言", "颜", "鲜"},
    "eng": {"风", "空", "红", "东", "灯", "峰", "横", "声", "程", "征",
            "更", "鹏", "腾", "城", "明", "星", "生", "平", "情", "青"},
    "i":   {"西", "凄", "低", "奇", "稀", "依", "归", "飞", "枝", "知",
            "诗", "时", "池", "期", "离", "衣", "机", "溪", "丝", "思"},
    "u":   {"湖", "珠", "无", "孤", "初", "途", "书", "如", "苏", "壶",
            "图", "都", "奴", "租", "呼", "夫", "居", "虚", "鱼", "余"},
    "ou":  {"楼", "愁", "秋", "流", "游", "留", "舟", "头", "收", "忧",
            "求", "谋", "浮", "眸", "州", "牛", "柔", "休", "钩", "偷"},
    "ai":  {"来", "台", "开", "白", "才", "怀", "苔", "徊", "猜", "哀",
            "垓", "霾", "钗", "骸", "栽", "腮", "牌", "埋", "柴", "胎"},
}

# 所有押韵字的并集
_ALL_RHYME_CHARS = set()
for chars in _RHYME_GROUPS.values():
    _ALL_RHYME_CHARS.update(chars)


# ═══════════════════════════════════════════════════════════════════════════
# 诗词创作核心
# ═══════════════════════════════════════════════════════════════════════════

class PoetryEngine:
    """
    格律诗词引擎 — 规则约束 + 字场能量引导 + n-gram共现。
    
    填字策略:
      1. 从韵脚开始反向填充 (押韵字优先)
      2. 每步选与上下文共现频率最高的字
      3. 平仄冲突时在备选中用字场嵌入选语义最佳
    """

    def __init__(self, field=None, concept_graph=None):
        self.field = field
        self.cg = concept_graph
        self.hanzi_to_idx = field._char_to_idx if field and hasattr(field, '_char_to_idx') else {}
        # n-gram 共现表: (prev_char, next_char) → frequency
        self._bigram_freq = {}
        self._build_bigram_table()

    def _build_bigram_table(self):
        """从概念图 JSON 直接提取字对共现（绕过 ConceptGraph 类，秒级加载）"""
        # 1. 注入诗意种子
        self._seed_poetic_bigrams()

        # 2. 尝试从概念图 JSON 直接提取字对
        import os, json
        # 定位概念图文件 (orchestrator 在 loongpearl/core/ 下)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cg_path = os.path.join(_root, 'data', 'models', 'concept_graph.json')

        if not os.path.exists(cg_path):
            # 尝试相对路径
            for candidate in ['data/models/concept_graph.json',
                             '../data/models/concept_graph.json']:
                if os.path.exists(candidate):
                    cg_path = candidate
                    break

        if not os.path.exists(cg_path):
            return  # 只能用种子

        try:
            with open(cg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 从三元组的 s(ubject) 和 o(bject) 中提取多字概念名
            count = 0
            for t in data.get('triples', []):
                for key in ('s', 'o'):
                    name = t.get(key, '')
                    if isinstance(name, str):
                        for i in range(len(name) - 1):
                            a, b = name[i], name[i + 1]
                            if '\u4e00' <= a <= '\u9fff' and '\u4e00' <= b <= '\u9fff':
                                # 概念图中的共现加权较高
                                self._bigram_freq[(a, b)] = self._bigram_freq.get((a, b), 0) + 1
                                count += 1
        except Exception:
            pass  # 加载失败也不影响，用种子即可

    def _seed_poetic_bigrams(self):
        """注入经典诗意字对"""
        poetic_pairs = [
            # 春
            ("春", "风"), ("春", "花"), ("春", "雨"), ("春", "水"),
            ("春", "光"), ("春", "色"), ("春", "草"), ("春", "意"),
            ("春", "江"), ("春", "山"), ("春", "日"), ("春", "天"),
            # 秋
            ("秋", "风"), ("秋", "月"), ("秋", "水"), ("秋", "色"),
            ("秋", "叶"), ("秋", "雨"), ("秋", "霜"), ("秋", "意"),
            # 月
            ("明", "月"), ("秋", "月"), ("江", "月"), ("山", "月"),
            ("夜", "月"), ("冷", "月"), ("残", "月"), ("晓", "月"),
            # 花
            ("桃", "花"), ("梅", "花"), ("菊", "花"), ("落", "花"),
            ("飞", "花"), ("春", "花"), ("红", "花"), ("寒", "花"),
            # 山水
            ("青", "山"), ("江", "水"), ("流", "水"), ("绿", "水"),
            ("高", "山"), ("远", "山"), ("深", "山"), ("空", "山"),
            # 天
            ("青", "天"), ("苍", "天"), ("长", "天"), ("碧", "天"),
            # 人
            ("故", "人"), ("佳", "人"), ("离", "人"), ("行", "人"),
            # 风
            ("春", "风"), ("秋", "风"), ("东", "风"), ("寒", "风"),
            ("清", "风"), ("凉", "风"), ("金", "风"), ("朔", "风"),
            # 雨
            ("烟", "雨"), ("风", "雨"), ("细", "雨"), ("夜", "雨"),
            ("暮", "雨"), ("寒", "雨"), ("微", "雨"),
            # 雪
            ("白", "雪"), ("飞", "雪"), ("冬", "雪"), ("残", "雪"),
            # 鸟
            ("飞", "鸟"), ("归", "鸟"), ("孤", "鸟"), ("啼", "鸟"),
            # 其他诗意搭配
            ("孤", "舟"), ("长", "安"), ("洛", "阳"), ("长", "江"),
            ("黄", "河"), ("夕", "阳"), ("落", "日"), ("烟", "波"),
            ("故", "乡"), ("千", "里"), ("万", "里"), ("天", "涯"),
        ]
        for a, b in poetic_pairs:
            self._bigram_freq[(a, b)] = self._bigram_freq.get((a, b), 0) + 3  # 加权

    def _fill_line(self, pattern: List[str], theme_chars: List[str],
                   fixed_last: str = None) -> str:
        """
        智能填字：从韵脚反向填充，用n-gram共现选最佳前驱字。
        
        算法:
          1. 固定最后一个字(韵脚)
          2. 从右往左，每个位置:
             a. 候选字 = theme_chars中符合平仄的
             b. 排序: 与右边邻字的共现频率 → 与主题的嵌入距离
             c. 选最高分
        """
        n = len(pattern)
        chars = [None] * n

        # Step 1: 韵脚
        if fixed_last:
            chars[-1] = fixed_last
        else:
            # 从 theme_chars 中选符合平仄的
            tone_req = pattern[-1]
            candidates = [c for c in theme_chars 
                         if self._check_tone(c, tone_req)]
            chars[-1] = random.choice(candidates) if candidates else self._random_tone(tone_req)

        # Step 2: 反向填充
        for i in range(n - 2, -1, -1):
            right_char = chars[i + 1]
            tone_req = pattern[i]
            
            # 候选: theme_chars中符合平仄 + 与右边字有共现的
            candidates = []
            for c in theme_chars:
                if c == right_char:
                    continue
                if self._check_tone(c, tone_req):
                    freq = self._bigram_freq.get((c, right_char), 0)
                    candidates.append((c, freq))
            
            if candidates:
                # 按共现频率排序，加一点随机性
                candidates.sort(key=lambda x: -x[1])
                # 在前5个中随机选（避免总是选最高频导致重复）
                top = candidates[:max(5, len(candidates)//3)]
                chars[i] = random.choice(top)[0]
            else:
                # 兜底: 用字场嵌入找与右边字最相似的
                chars[i] = self._best_semantic_fit(right_char, tone_req, theme_chars)

        return "".join(chars)

    def _check_tone(self, char: str, tone: str) -> bool:
        if tone == "ping":
            return char in _PING_SHENG
        elif tone == "ze":
            return char in _ZE_SHENG
        return True

    def _random_tone(self, tone: str) -> str:
        pool = _PING_SHENG if tone == "ping" else _ZE_SHENG
        return random.choice(list(pool))

    def _best_semantic_fit(self, right_char: str, tone_req: str,
                           theme_chars: List[str]) -> str:
        """用字场嵌入找语义最匹配的候选字"""
        if not self.field or not self.hanzi_to_idx:
            return self._random_tone(tone_req)

        if right_char not in self.hanzi_to_idx:
            return self._random_tone(tone_req)

        import torch
        right_emb = self.field.anchors[self.hanzi_to_idx[right_char]]

        candidates = [c for c in theme_chars 
                     if c in self.hanzi_to_idx and self._check_tone(c, tone_req)]
        if not candidates:
            # 扩大搜索范围
            pool = _PING_SHENG if tone_req == "ping" else _ZE_SHENG
            candidates = [c for c in pool if c in self.hanzi_to_idx]

        if not candidates:
            return self._random_tone(tone_req)

        best_char = candidates[0]
        best_dist = float('inf')
        with torch.no_grad():
            for c in candidates[:50]:
                c_emb = self.field.anchors[self.hanzi_to_idx[c]]
                dist = 1 - torch.cosine_similarity(
                    right_emb.unsqueeze(0), c_emb.unsqueeze(0), dim=1
                ).item()
                if dist < best_dist:
                    best_dist = dist
                    best_char = c

        return best_char

    def compose(self, theme: str = "春天",
                format: str = "五言绝句",
                rhyme_group: str = None) -> str:
        """
        根据主题和格式创作诗词。

        Args:
            theme: 主题 (如"春天"、"离别"、"山水")
            format: "五言绝句" | "七言绝句" | "五言律诗"
            rhyme_group: 押韵组 (如"ang"、"an")，不指定则根据主题自动选择

        Returns:
            完整的诗词文本
        """
        if format == "五言绝句":
            return self._compose_wuyan_jueju(theme, rhyme_group)
        elif format == "七言绝句":
            return self._compose_qiyan_jueju(theme, rhyme_group)
        elif format == "五言律诗":
            return self._compose_wuyan_lvshi(theme, rhyme_group)
        else:
            return self._compose_wuyan_jueju(theme, rhyme_group)

    def _compose_wuyan_jueju(self, theme: str, rhyme_group: str = None) -> str:
        """
        五言绝句: 5字×4句，第2、4句押韵。
        平仄格式 (首句不入韵):
          仄仄平平仄
          平平仄仄平 (押韵)
          平平平仄仄
          仄仄仄平平 (押韵)
        """
        # 1. 从概念图中获取主题相关字
        theme_chars = self._get_theme_chars(theme, count=30)

        # 2. 选择押韵组
        if rhyme_group is None:
            rhyme_group = self._pick_rhyme_group(theme)

        # 3. 选押韵字 (第二句和第四句的最后一个字)
        rhyme_chars = list(_RHYME_GROUPS.get(rhyme_group, _RHYME_GROUPS["ang"]))
        rhyme_char_2 = random.choice([c for c in rhyme_chars if c in _PING_SHENG])
        rhyme_char_4 = random.choice([c for c in rhyme_chars 
                                      if c in _PING_SHENG and c != rhyme_char_2])

        # 4. 按格律填字
        # 句1: 仄仄平平仄
        line1 = self._fill_line(["ze", "ze", "ping", "ping", "ze"], theme_chars)
        # 句2: 平平仄仄平 (末字=rhyme_char_2)
        line2 = self._fill_line(["ping", "ping", "ze", "ze", "ping"], theme_chars,
                                fixed_last=rhyme_char_2)
        # 句3: 平平平仄仄
        line3 = self._fill_line(["ping", "ping", "ping", "ze", "ze"], theme_chars)
        # 句4: 仄仄仄平平 (末字=rhyme_char_4)
        line4 = self._fill_line(["ze", "ze", "ze", "ping", "ping"], theme_chars,
                                fixed_last=rhyme_char_4)

        return f"《{theme}》\n{line1}\n{line2}\n{line3}\n{line4}"

    def _compose_qiyan_jueju(self, theme: str, rhyme_group: str = None) -> str:
        """七言绝句: 7字×4句"""
        theme_chars = self._get_theme_chars(theme, count=40)
        if rhyme_group is None:
            rhyme_group = self._pick_rhyme_group(theme)

        rhyme_chars = list(_RHYME_GROUPS.get(rhyme_group, _RHYME_GROUPS["ang"]))
        rhyme_char_2 = random.choice([c for c in rhyme_chars if c in _PING_SHENG])
        rhyme_char_4 = random.choice([c for c in rhyme_chars
                                      if c in _PING_SHENG and c != rhyme_char_2])

        # 七言格律 (首句不入韵):
        # 平平仄仄平平仄
        # 仄仄平平仄仄平 (押韵)
        # 仄仄平平平仄仄
        # 平平仄仄仄平平 (押韵)
        line1 = self._fill_line(
            ["ping","ping","ze","ze","ping","ping","ze"], theme_chars)
        line2 = self._fill_line(
            ["ze","ze","ping","ping","ze","ze","ping"], theme_chars,
            fixed_last=rhyme_char_2)
        line3 = self._fill_line(
            ["ze","ze","ping","ping","ping","ze","ze"], theme_chars)
        line4 = self._fill_line(
            ["ping","ping","ze","ze","ze","ping","ping"], theme_chars,
            fixed_last=rhyme_char_4)

        return f"《{theme}》\n{line1}\n{line2}\n{line3}\n{line4}"

    def _compose_wuyan_lvshi(self, theme: str, rhyme_group: str = None) -> str:
        """五言律诗: 5字×8句"""
        theme_chars = self._get_theme_chars(theme, count=50)
        if rhyme_group is None:
            rhyme_group = self._pick_rhyme_group(theme)

        rhyme_chars = list(_RHYME_GROUPS.get(rhyme_group, _RHYME_GROUPS["ang"]))
        # 律诗: 第2,4,6,8句押韵
        r2 = random.choice([c for c in rhyme_chars if c in _PING_SHENG])
        r4 = random.choice([c for c in rhyme_chars if c in _PING_SHENG and c != r2])
        r6 = random.choice([c for c in rhyme_chars if c in _PING_SHENG and c not in (r2,r4)])
        r8 = random.choice([c for c in rhyme_chars if c in _PING_SHENG and c not in (r2,r4,r6)])

        lines = [
            self._fill_line(["ze","ze","ping","ping","ze"], theme_chars),
            self._fill_line(["ping","ping","ze","ze","ping"], theme_chars, fixed_last=r2),
            self._fill_line(["ping","ping","ping","ze","ze"], theme_chars),
            self._fill_line(["ze","ze","ze","ping","ping"], theme_chars, fixed_last=r4),
            self._fill_line(["ze","ze","ping","ping","ze"], theme_chars),
            self._fill_line(["ping","ping","ze","ze","ping"], theme_chars, fixed_last=r6),
            self._fill_line(["ping","ping","ping","ze","ze"], theme_chars),
            self._fill_line(["ze","ze","ze","ping","ping"], theme_chars, fixed_last=r8),
        ]
        return f"《{theme}》\n" + "\n".join(lines)

    def _fill_line(self, pattern: List[str], theme_chars: List[str],
                   fixed_last: str = None) -> str:
        """按平仄模式填充一行"""
        chars = []
        for i, tone in enumerate(pattern):
            if i == len(pattern) - 1 and fixed_last:
                chars.append(fixed_last)
            else:
                if tone == "ping":
                    candidates = [c for c in theme_chars if c in _PING_SHENG]
                else:
                    candidates = [c for c in theme_chars if c in _ZE_SHENG]
                if candidates:
                    chars.append(random.choice(candidates))
                else:
                    # 兜底: 从整个平/仄字库中选
                    pool = _PING_SHENG if tone == "ping" else _ZE_SHENG
                    chars.append(random.choice(list(pool)))
        return "".join(chars)

    def _get_theme_chars(self, theme: str, count: int = 30) -> List[str]:
        """从字场嵌入空间获取与主题语义相关的汉字"""
        chars = list(theme.replace(" ", ""))
        
        # 用字场嵌入找语义最接近的汉字
        if self.field and hasattr(self.field, 'anchors') and self.hanzi_to_idx:
            import torch
            # 将主题字映射到嵌入空间
            theme_indices = []
            for c in theme:
                if c in self.hanzi_to_idx:
                    theme_indices.append(self.hanzi_to_idx[c])
            
            if theme_indices:
                with torch.no_grad():
                    # 主题字的平均嵌入
                    theme_emb = sum(self.field.anchors[i] for i in theme_indices) / len(theme_indices)
                    # 在整个字场中找最近邻
                    similarities = torch.cosine_similarity(
                        theme_emb.unsqueeze(0), self.field.anchors, dim=1
                    )
                    # 取top-k个最相似的字（排除主题字本身）
                    top_k = min(count * 3, len(similarities))
                    top_indices = torch.topk(similarities, top_k).indices.tolist()
                    
                    for idx in top_indices:
                        c = self.field.hanzi_list[idx]
                        if c not in chars and '\u4e00' <= c <= '\u9fff':
                            chars.append(c)
                            if len(chars) >= count:
                                break

        # 如果字场不可用，从概念图补充
        if len(chars) < count and self.cg and hasattr(self.cg, 'triples'):
            related_concepts = []
            for s in list(self.cg.triples.keys())[:10000]:
                if any(c in s for c in theme):
                    related_concepts.append(s)
            
            for concept in related_concepts[:30]:
                for c in concept:
                    if c not in chars and '\u4e00' <= c <= '\u9fff':
                        chars.append(c)
                        if len(chars) >= count:
                            break
                if len(chars) >= count:
                    break

        # 最后兜底: 常用诗意字
        poetic_chars = "春风花草香山水明月光云雾雨雪天地人花鸟梦楼台亭阁"
        for c in poetic_chars:
            if c not in chars:
                chars.append(c)
                if len(chars) >= count:
                    break

        return chars[:count]

    def _pick_rhyme_group(self, theme: str) -> str:
        """根据主题选择合适的押韵组"""
        # 简单映射
        mappings = {
            "春": "ang", "花": "ang", "江": "ang", "光": "ang",
            "天": "an",  "山": "an",  "烟": "an",
            "风": "eng", "梦": "eng", "灯": "eng",
            "思": "i",   "离": "i",   "期": "i",
            "秋": "ou",  "愁": "ou",  "舟": "ou",
        }
        for key, group in mappings.items():
            if key in theme:
                return group
        return "ang"  # 默认


# ═══════════════════════════════════════════════════════════════════════════
# 成语接龙
# ═══════════════════════════════════════════════════════════════════════════

class IdiomChain:
    """成语接龙 — 尾字匹配 + 能量排序"""

    def __init__(self, field=None, concept_graph=None):
        self.field = field
        self.cg = concept_graph
        self._idioms = self._load_idioms()

    def _load_idioms(self) -> List[str]:
        """从概念图加载成语"""
        idioms = []
        if self.cg and hasattr(self.cg, 'triples'):
            for s in self.cg.triples:
                if len(s) == 4 and re.match(r'^[\u4e00-\u9fff]{4}$', s):
                    idioms.append(s)

        if not idioms:
            # 兜底成语库
            idioms = [
                "一心一意", "意气风发", "发愤图强", "强词夺理", "理直气壮",
                "壮志凌云", "云开雾散", "散兵游勇", "勇往直前", "前所未有",
                "有目共睹", "睹物思人", "人山人海", "海阔天空", "空前绝后",
                "后来居上", "上行下效", "效犬马力", "力挽狂澜", "澜翻絮涌",
                "画龙点睛", "睛目不凡", "凡夫俗子", "子虚乌有", "有口皆碑",
            ]
        return idioms

    def chain(self, start: str, max_length: int = 10) -> List[str]:
        """
        从给定成语开始接龙。

        Args:
            start: 起始成语
            max_length: 最大链长

        Returns:
            成语链列表
        """
        if len(start) != 4:
            # 尝试从概念图中找一个以此字开头的成语
            start_char = start[0] if start else "一"
            for idiom in self._idioms:
                if idiom.startswith(start_char):
                    start = idiom
                    break
            else:
                return [f"(未找到以'{start_char}'开头的成语)"]

        chain = [start]
        used = {start}

        for _ in range(max_length - 1):
            last_char = chain[-1][-1]
            candidates = [i for i in self._idioms 
                         if i.startswith(last_char) and i not in used]

            if not candidates:
                break

            # 用能量评分选最佳（如果有field）
            if self.field and hasattr(self.field, 'anchors'):
                best = self._energy_rank(candidates, chain)
            else:
                best = random.choice(candidates)

            chain.append(best)
            used.add(best)

        return chain

    def _energy_rank(self, candidates: List[str], chain: List[str]) -> str:
        """用能量景观对候选成语排序"""
        # 简化: 选一个平凡的
        if len(chain) >= 2:
            prev = chain[-2]
        else:
            prev = chain[-1]

        best = candidates[0]
        best_score = -1

        import torch
        for cand in candidates[:20]:
            # 简单启发: 选与上文有共同字的
            score = sum(1 for c in cand if c in prev)
            if score > best_score:
                best_score = score
                best = cand

        return best


# ═══════════════════════════════════════════════════════════════════════════
# 事实叙事
# ═══════════════════════════════════════════════════════════════════════════

class FactualNarrative:
    """事实叙事 — 概念图路径 → 故事化渲染"""

    def __init__(self, concept_graph=None, decoder=None):
        self.cg = concept_graph
        self.decoder = decoder

    def narrate(self, start_concept: str, style: str = "neutral") -> str:
        """
        将概念图路径渲染为叙事文。

        Args:
            start_concept: 起始概念
            style: "neutral" | "story" | "educational"

        Returns:
            叙事文本
        """
        if not self.cg or start_concept not in self.cg.triples:
            return f"关于{start_concept}，我暂时没有足够的素材来叙事。"

        # 1. 从概念图提取路径
        paths = self._extract_paths(start_concept, max_hops=4)

        # 2. 渲染为叙事
        if style == "story":
            return self._render_as_story(start_concept, paths)
        elif style == "educational":
            return self._render_as_educational(start_concept, paths)
        else:
            return self._render_neutral(start_concept, paths)

    def _extract_paths(self, start: str, max_hops: int = 4) -> List[List[Tuple[str, str, str]]]:
        """从概念图提取多条路径 — 双向搜索（正向+反向）"""
        paths = []
        visited = {start}

        def dfs(current: str, path: List[Tuple[str, str, str]], depth: int):
            if depth >= max_hops or len(paths) >= 5:
                return

            # 正向: current → object
            if current in self.cg.triples:
                triples = self.cg.triples[current]
                sorted_triples = sorted(triples, key=lambda x: -x[2])[:3]
                for rel, obj, conf, src in sorted_triples:
                    if obj not in visited and len(obj) >= 2:
                        visited.add(obj)
                        new_path = path + [(current, rel, obj)]
                        paths.append(list(new_path))
                        dfs(obj, new_path, depth + 1)

            # 反向: subject → current (current作为客体)
            for s in list(self.cg.triples.keys())[:20000]:
                if s in visited:
                    continue
                for rel, obj, conf, src in self.cg.triples.get(s, []):
                    if obj == current and len(s) >= 2 and s not in visited:
                        visited.add(s)
                        new_path = path + [(s, rel, current)]
                        paths.append(list(new_path))
                        dfs(s, new_path, depth + 1)
                        break  # 只取第一条反向边

        dfs(start, [], 0)
        return paths

    def _render_as_story(self, start: str,
                         paths: List[List[Tuple[str, str, str]]]) -> str:
        """渲染为故事风格"""
        if not paths:
            return f"在知识的宇宙中，{start}静静地存在着，等待被发现。"

        best_path = max(paths, key=len)
        sentences = [f"很久以前，{start}诞生了。"]

        for s, r, o in best_path:
            if r == "IS_A":
                sentences.append(f"它是{o}的一种。")
            elif r == "PART_OF":
                sentences.append(f"它成为了{o}的组成部分。")
            elif r == "CAUSE":
                sentences.append(f"它的存在导致了{o}。")
            elif r == "HAS":
                sentences.append(f"它拥有{o}。")
            elif r == "RELATED":
                sentences.append(f"它与{o}有着千丝万缕的联系。")
            elif r == "OPPOSITE":
                sentences.append(f"它和{o}形成了鲜明的对比。")

        return " ".join(sentences)

    def _render_as_educational(self, start: str,
                               paths: List[List[Tuple[str, str, str]]]) -> str:
        """渲染为教学风格"""
        sentences = [f"我们来学习'{start}'这个概念。"]
        added = set()

        for path in paths[:3]:
            for s, r, o in path:
                key = (s, r, o)
                if key in added:
                    continue
                added.add(key)
                if r == "IS_A":
                    sentences.append(f"从分类角度看，{s}属于{o}。")
                elif r == "PART_OF":
                    sentences.append(f"{s}是{o}的组成部分。")
                elif r == "CAUSE":
                    sentences.append(f"{s}是{o}的原因。")
                elif r == "HAS":
                    sentences.append(f"{s}具备{o}的特征。")
                elif r == "RELATED":
                    sentences.append(f"{s}与{o}密切相关。")

        sentences.append(f"以上就是关于{start}的基本知识。")
        return "\n".join(sentences)

    def _render_neutral(self, start: str,
                        paths: List[List[Tuple[str, str, str]]]) -> str:
        """中性叙事"""
        sentences = [f"关于{start}："]
        seen = set()
        for path in paths[:3]:
            for s, r, o in path:
                if (s, r, o) not in seen:
                    seen.add((s, r, o))
                    sentences.append(f"• {s} {r} {o}")
        return "\n".join(sentences)


# ═══════════════════════════════════════════════════════════════════════════
# 创意引擎统一入口
# ═══════════════════════════════════════════════════════════════════════════

class CreativeEngine:
    """创意引擎 — 约束性创作的统一入口"""

    def __init__(self, field=None, concept_graph=None, decoder=None):
        self.poetry = PoetryEngine(field, concept_graph)
        self.idiom_chain = IdiomChain(field, concept_graph)
        self.narrative = FactualNarrative(concept_graph, decoder)
        self.field = field
        self.cg = concept_graph

    def handle(self, query: str) -> Optional[str]:
        """
        判断是否为创作请求，如果是则处理。

        支持:
          - "写一首关于XX的诗" / "作诗" / "赋诗一首"
          - "成语接龙：XX" / "XX开头的成语接龙"
          - "讲一个关于XX的故事" / "介绍一下XX的故事"
        """
        # 诗词
        m = re.search(r'(写|作|赋|来)(一)?首?(关于|描写)?(.{1,10})?(的)?(诗|诗词|绝句|律诗)', query)
        if m:
            theme = m.group(4) or m.group(6) or "春天"
            theme = theme.strip()
            fmt = "五言绝句"
            if "七言" in query or "七绝" in query:
                fmt = "七言绝句"
            elif "律诗" in query or "八句" in query:
                fmt = "五言律诗"
            return self.poetry.compose(theme=theme, format=fmt)

        # 成语接龙
        m = re.search(r'(成语接龙|词语接龙).(.{1,4})', query)
        if m or re.search(r'(.{1,4})开头的成语接龙', query):
            start = m.group(2) if m else query[:4]
            chain = self.idiom_chain.chain(start, max_length=8)
            return " → ".join(chain)

        # 事实叙事
        m = re.search(r'(讲|说|介绍|聊聊)(一(个|下))?(关于)?(.{1,10})的(故事|来历|起源)', query)
        if m:
            concept = m.group(5)
            style = "story" if "故事" in query else "educational"
            return self.narrative.narrate(concept, style=style)

        return None  # 不是创作请求


# ═══════════════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════════════

def test_creative():
    pe = PoetryEngine()
    print("=== 五言绝句 ===")
    print(pe.compose("春天", "五言绝句"))
    print("\n=== 七言绝句 ===")
    print(pe.compose("明月", "七言绝句"))
    
    ic = IdiomChain()
    print("\n=== 成语接龙 ===")
    print(" → ".join(ic.chain("画龙点睛", 8)))
    
    ce = CreativeEngine()
    print("\n=== 创意路由 ===")
    for q in ["写一首关于月亮的诗", "一心一意开头的成语接龙", "介绍一下量子的故事"]:
        result = ce.handle(q)
        print(f"\n查询: {q}")
        print(f"结果: {result[:100] if result else 'N/A'}...")


if __name__ == "__main__":
    test_creative()
