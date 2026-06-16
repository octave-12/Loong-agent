#!/usr/bin/env python3
"""
龙珠学语课程 (baby_curriculum.py)
=================================
按人类语言习得顺序，分八个阶段循序渐进:

  阶段1: 单字     — 认字、记字形、懂字义
  阶段2: 组合字   — 偏旁部首、会意形声
  阶段3: 词语     — 双字词、多字词、成语
  阶段4: 成句     — 主谓宾、修饰、语序
  阶段5: 段落     — 句间衔接、逻辑连贯
  阶段6: 文章     — 篇章结构、论点论据
  阶段7: 古诗词   — 格律、意象、对仗
  阶段8: 文言文   — 之乎者也、典故用典

每个阶段：
  - 只使用当前阶段及之前阶段的知识
  - 有明确的学习目标和检测标准
  - 学完一个阶段才能进入下一个
"""

import sys, os, json, re, random, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Loong-pearl/ 项目根


# ====================================================================
# 课程数据定义
# ====================================================================

@dataclass
class StageProgress:
    """一个阶段的学习进度"""
    name: str
    level: int           # 1-8
    total_items: int     # 本阶段总学习量
    learned: int = 0     # 已学
    mastered: int = 0    # 已掌握 (检测通过)
    target_mastery: float = 0.8  # 掌握率目标


# ====================================================================
# 阶段1: 单字 — 从高频到低频逐字认读
# ====================================================================

class Stage1_SingleChars:
    """
    单字学习：按字频从高到低，逐个认识汉字。
    
    数据源:
      - 字频统计 (基于语料库)
      - 汉字释义 (Unihan + Decompose)
      - 笔画、部首信息
    
    学习方式:
      - 每次展示一个字
      - 给出: 字形、读音、释义、笔画数、部首
      - 检测: 看字说义、听义写字
    """
    
    def __init__(self, decompose: dict, unihan: dict):
        self.decompose = decompose
        self.unihan = unihan
        self._web_lookup = None  # 懒加载, 避免每次初始化都创建网络会话
        
        # 字频数据 (常用字按频率排列, 基于现代汉语语料库)
        # 数据来源: 现代汉语字频统计
        self._init_freq_list()
    
    def _init_freq_list(self):
        """构建按频率排序的字表 (从 GB2312 一级汉字文件加载)"""
        gb_path = os.path.join(BASE, 'data/wordlists/gb2312_level1.txt')
        
        if os.path.exists(gb_path):
            all_chars = open(gb_path, encoding='utf-8').read().strip()
        else:
            # 兜底: 硬编码常用字
            all_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严首底液官德随病苏失尔死讲配女黄推显谈罪神艺呢席含企望密批营项防举球英氧势告李台落木帮轮破亚师围注远字材排供河态封另施减树溶怎止案言士均武固叶鱼波视仅费紧爱左章早朝害续轻服试食充兵源判护司足某练差致板田降黑犯负击范继兴似余坚曲输修故城夫够送笔船占右财吃富春职觉汉画功巴跟虽杂飞检吸助升阳互初创抗考投坏策古径换未跑留钢曾端责站简述钱副尽帝射草冲承独令限"

        chars = list(all_chars)
        
        # 按字频分三档
        n = len(chars)
        self.high_freq = chars[:min(500, n)]      # 前500高频
        self.mid_freq = chars[500:min(1500, n)]   # 501-1500次高频
        self.low_freq = chars[1500:n]             # 1501+ 低频
        
        self.all_chars = set(chars)
    
    def get_char_info(self, char: str) -> dict:
        """获取一个字的完整信息"""
        info = {
            'char': char,
            'definition': '',
            'radical': '',
            'components': [],
            'pinyin': '',
        }
        
        if char in self.decompose:
            d = self.decompose[char]
            info['definition'] = d.get('definition', '')
            info['radical'] = d.get('radical', '')
            info['components'] = d.get('components', [])
            info['pinyin'] = d.get('pinyin', [''])[0] if d.get('pinyin') else ''
        
        if char in self.unihan and not info['definition']:
            info['definition'] = self.unihan[char].get('definition', '')
            info['pinyin'] = info['pinyin'] or self.unihan[char].get('mandarin', '')
        
        # 🌐 联网兜底: 本地缺释义或缺拼音 → 上网学
        if not info['definition'] or not info['pinyin']:
            if self._web_lookup is None:
                from loongpearl.web.lookup import CharWebLookup
                self._web_lookup = CharWebLookup()
            web_info = self._web_lookup.learn(char)
            if web_info.get('definition') and not info['definition']:
                info['definition'] = web_info['definition']
            if web_info.get('pinyin') and not info['pinyin']:
                info['pinyin'] = web_info['pinyin']
        
        return info
    
    def learn_batch(self, start: int = 0, count: int = 20) -> List[dict]:
        """按频率顺序学习一批字 (跨高频→中频→低频)"""
        all_chars = self.high_freq + self.mid_freq + self.low_freq
        chars = all_chars[start:start + count]
        return [self.get_char_info(ch) for ch in chars]


# ====================================================================
# 阶段2: 组合字 — 偏旁部首、造字法
# ====================================================================

class Stage2_CompoundChars:
    """
    组合字学习：理解汉字的结构。
    
    学习内容:
      - 偏旁部首的意义
      - 会意字: 日+月=明, 人+木=休
      - 形声字: 氵(水)+可=河, 木+几=机
      - 部件组合规律
    
    学习方式:
      - 从已学单字中找有部件拆解的字
      - 教婴儿：这个字由哪几块拼成？每块什么意思？
    """
    
    # 常见偏旁部首及其含义
    RADICAL_MEANINGS = {
        '氵': '水', '讠': '言', '亻': '人', '扌': '手',
        '忄': '心', '纟': '丝', '犭': '犬', '灬': '火',
        '钅': '金', '饣': '食', '衤': '衣', '礻': '示',
        '冫': '冰', '刂': '刀', '阝': '邑/阜', '艹': '草',
        '辶': '走', '宀': '屋', '广': '房屋', '疒': '病',
        '罒': '网', '皿': '器皿', '目': '眼睛', '田': '田地',
        '禾': '谷物', '米': '米', '竹': '竹子', '糸': '丝线',
        '贝': '贝壳/钱', '车': '车', '马': '马', '鱼': '鱼',
        '鸟': '鸟', '虫': '虫', '木': '木', '石': '石',
        '王': '玉', '月': '肉/月', '日': '太阳', '口': '嘴',
    }
    
    def __init__(self, decompose: dict):
        self.decompose = decompose
    
    def find_compound_chars(self, known_chars: set, limit: int = 100) -> List[dict]:
        """从已学字中找可拆解的字"""
        results = []
        for char in known_chars:
            if char in self.decompose:
                d = self.decompose[char]
                comps = d.get('components', [])
                if len(comps) >= 2:
                    results.append({
                        'char': char,
                        'components': comps,
                        'definition': d.get('definition', ''),
                        'etymology': d.get('etymology', ''),
                        'radical': d.get('radical', ''),
                    })
                    if len(results) >= limit:
                        break
        return results
    
    def explain_char(self, char_info: dict) -> str:
        """用婴儿能懂的方式解释一个字的结构"""
        char = char_info['char']
        comps = char_info.get('components', [])
        defn = char_info.get('definition', '')
        
        if not comps:
            return f"{char}（{defn[:20]}）是一个独体字。"
        
        # 解释每个部件
        parts = []
        for c in comps:
            c_info = {'char': c, 'definition': ''}
            if c in self.decompose:
                c_info['definition'] = self.decompose[c].get('definition', '')
            
            # 如果是常见偏旁，用中文解释
            if c in self.RADICAL_MEANINGS:
                meaning = self.RADICAL_MEANINGS[c]
                parts.append(f"{c}（{meaning}旁）")
            elif c_info['definition']:
                parts.append(f"{c}（{c_info['definition'][:10]}）")
            else:
                parts.append(c)
        
        return f"{char} = {' + '.join(parts)}"


# ====================================================================
# 阶段3: 词语 — 双字词、多字词、成语
# ====================================================================

class Stage3_Words:
    """
    词语学习：从单字组合成有意义的词。
    
    学习内容:
      - 双字词 (最常见)
      - 三字词、四字成语
      - 词义、搭配、用法
    
    学习方式:
      - 从CC-CEDICT提取由已学单字组成的词
      - 先学构词能力强的字组成的词
      - 成语额外标注出处和典故
    """
    
    def __init__(self, cedict: dict):
        self.cedict = cedict
    
    def find_words_from_chars(self, known_chars: set, limit: int = 200) -> List[dict]:
        """从已学字中找可组合的词"""
        results = []
        for word, info in self.cedict.items():
            if len(word) < 2:
                continue
            # 词中所有字都已知
            if all(ch in known_chars for ch in word):
                results.append({
                    'word': word,
                    'pinyin': info.get('pinyin', ''),
                    'definitions': info.get('definitions', []),
                    'length': len(word),
                })
                if len(results) >= limit:
                    break
        return results
    
    def find_chengyu(self, known_chars: set, limit: int = 50) -> List[dict]:
        """找四字成语"""
        results = []
        for word, info in self.cedict.items():
            if len(word) == 4 and all(ch in known_chars for ch in word):
                defs = info.get('definitions', [])
                # 成语通常释义中包含"idiom"
                is_idiom = any('idiom' in d.lower() or 'proverb' in d.lower() for d in defs)
                if is_idiom or len(word) == 4:
                    results.append({
                        'word': word,
                        'pinyin': info.get('pinyin', ''),
                        'definitions': defs,
                    })
                    if len(results) >= limit:
                        break
        return results


# ====================================================================
# 阶段4: 成句 — 从词到句
# ====================================================================

class Stage4_Sentences:
    """
    成句学习：把词语串成完整的句子。
    
    学习内容:
      - 基本句式: 主谓宾、是字句、有字句
      - 修饰语: 的、地、得
      - 时态: 了、着、过
      - 疑问: 吗、呢、什么、怎么
    
    学习方式:
      - 从简单句开始 (2-3词组成)
      - 逐步加修饰语
      - 婴儿自己尝试造句，错了纠正
    """
    
    # 基本句式模板
    SENTENCE_PATTERNS = {
        '是字句': '{A}是{B}。',        # 龙是神兽。
        '有字句': '{A}有{B}。',         # 龙有鳞。
        '主谓': '{A}{V}。',             # 龙飞。
        '主谓宾': '{A}{V}{B}。',        # 龙爱水。
        '形容词': '{A}很{adj}。',       # 龙很大。
        '疑问吗': '{sentence}吗？',      # 龙飞吗？
        '疑问什么': '什么是{A}？',       # 什么是龙？
        '修饰的': '{adj}的{A}',          # 大的龙
        '修饰地': '{adv}地{V}',          # 快快地飞
        '了': '{A}{V}了。',             # 龙飞了。
    }
    
    def __init__(self):
        pass
    
    def make_simple_sentence(self, subject: str, verb: str = None, obj: str = None) -> str:
        """用已知的词造简单句"""
        if verb and obj:
            return f"{subject}{verb}{obj}。"
        elif verb:
            return f"{subject}{verb}。"
        else:
            return f"{subject}很大。"
    
    def make_shì_sentence(self, a: str, b: str) -> str:
        """是字句"""
        return f"{a}是{b}。"
    
    def make_question(self, topic: str) -> str:
        """对某主题提问"""
        return f"什么是{topic}？"
    
    def make_modified(self, adj: str, noun: str) -> str:
        """加修饰语"""
        return f"{adj}的{noun}"


# ====================================================================
# 阶段5-8 占位 (后续实现)
# ====================================================================

class Stage5_Paragraphs:
    """段落: 句间衔接、转折、因果"""
    pass

class Stage6_Articles:
    """文章: 篇章结构、论点论据"""
    pass

class Stage7_Poetry:
    """古诗词: 格律、意象、对仗"""
    pass

class Stage8_Classical:
    """文言文: 之乎者也、典故"""
    pass


# ====================================================================
# 课程管理器 — 统一调度八个阶段
# ====================================================================

class BabyCurriculum:
    """
    龙珠婴儿的学语课程。
    
    用法:
        baby = BabyCurriculum()
        baby.start_stage(1)          # 开始学单字
        baby.learn_next_batch(20)    # 学下一批
        baby.test_current()          # 检测当前阶段掌握程度
        baby.advance_if_ready()      # 如果掌握够了就升阶段
    """
    
    STAGES = {
        1: '单字',
        2: '组合字', 
        3: '词语',
        4: '成句',
        5: '段落',
        6: '文章',
        7: '古诗词',
        8: '文言文',
    }
    
    def __init__(self):
        # 加载数据
        self.decompose = self._load_json('data/dicts/dict_decompose.json')
        self.unihan = self._load_json('data/dicts/dict_unihan.json')
        self.cedict = self._load_json('data/dicts/cedict_parsed.json')
        
        # 初始化各阶段
        self.stage1 = Stage1_SingleChars(self.decompose, self.unihan)
        self.stage2 = Stage2_CompoundChars(self.decompose)
        self.stage3 = Stage3_Words(self.cedict)
        self.stage4 = Stage4_Sentences()
        
        # 学习状态
        self.current_stage = 1
        self.known_chars = set()      # 已学单字
        self.known_words = set()      # 已学词语
        self.mastered_chars = set()   # 掌握单字 (检测通过)
        
        # 课程进度
        self.char_index = 0           # 单字学到第几个
        self.stage_progress = {}      # {stage_num: StageProgress}
        
        # 可恢复
        self.save_path = os.path.join(BASE, 'data/runtime/baby_progress.json')
        self._load_progress()
        
        # 发声器 (懒加载)
        self._voice = None
    
    @property
    def voice(self):
        """婴儿的发声器官 (按需加载)"""
        if self._voice is None:
            from loongpearl.voice.baby_voice import BabyVoice
            self._voice = BabyVoice()
        return self._voice
    
    def say_char(self, char: str) -> str:
        """让婴儿读一个字 → 返回 WAV 文件路径"""
        info = self.stage1.get_char_info(char)
        pinyin = info.get('pinyin', '')
        if not pinyin:
            return ''
        return self.voice.say(pinyin)
    
    def _load_json(self, filename: str) -> dict:
        path = os.path.join(BASE, filename)
        if os.path.exists(path):
            return json.load(open(path, encoding='utf-8'))
        return {}
    
    def _load_progress(self):
        if os.path.exists(self.save_path):
            data = json.load(open(self.save_path, encoding='utf-8'))
            self.current_stage = data.get('stage', 1)
            self.known_chars = set(data.get('known_chars', []))
            self.known_words = set(data.get('known_words', []))
            self.mastered_chars = set(data.get('mastered_chars', []))
            self.char_index = data.get('char_index', 0)
    
    def save_progress(self):
        json.dump({
            'stage': self.current_stage,
            'known_chars': list(self.known_chars),
            'known_words': list(self.known_words),
            'mastered_chars': list(self.mastered_chars),
            'char_index': self.char_index,
        }, open(self.save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    # ── 阶段1: 单字学习 ────────────────────────
    
    def start_stage(self, stage: int):
        """开始或切换到某阶段"""
        self.current_stage = stage
        print(f"\n🌟 龙珠进入第{stage}阶段: {self.STAGES[stage]}")
        
        if stage == 1:
            total = len(self.stage1.high_freq)
            print(f"   待学单字: {total} 个 (高频→低频)")
            print(f"   已学: {len(self.known_chars)} 个")
        elif stage == 2:
            compounds = self.stage2.find_compound_chars(self.known_chars)
            print(f"   可拆解的字: {len(compounds)} 个")
        elif stage == 3:
            words = self.stage3.find_words_from_chars(self.known_chars, limit=99999)
            print(f"   可组成的词: {len(words)} 个")
    
    def learn_next_batch(self, count: int = 10) -> List[dict]:
        """
        学下一批内容 (根据当前阶段)
        
        Returns: 本批学习的内容列表
        """
        stage = self.current_stage
        
        if stage == 1:
            return self._learn_chars(count)
        elif stage == 2:
            return self._learn_compounds(count)
        elif stage == 3:
            return self._learn_words(count)
        elif stage == 4:
            return self._learn_sentences(count)
        else:
            print(f"阶段{stage}尚未实现")
            return []
    
    def _learn_chars(self, count: int) -> List[dict]:
        """学单字"""
        batch = self.stage1.learn_batch(self.char_index, count)
        for item in batch:
            self.known_chars.add(item['char'])
        self.char_index += len(batch)
        return batch
    
    def _learn_compounds(self, count: int) -> List[dict]:
        """学组合字"""
        unlearned = self.known_chars - self.mastered_chars
        compounds = self.stage2.find_compound_chars(unlearned, limit=count)
        return compounds
    
    def _learn_words(self, count: int) -> List[dict]:
        """学词语"""
        words = self.stage3.find_words_from_chars(self.known_chars, limit=count)
        for w in words:
            self.known_words.add(w['word'])
        return words
    
    def _learn_sentences(self, count: int) -> List[str]:
        """造句练习"""
        sentences = []
        # 从已知词中随机造句
        words_list = list(self.known_words)[:50]
        chars_list = list(self.known_chars)[:30]
        
        for _ in range(count):
            pattern = random.choice(['是', '有', '修饰', '疑问'])
            if pattern == '是' and len(chars_list) >= 2:
                a, b = random.sample(chars_list, 2)
                sentences.append(self.stage4.make_shì_sentence(a, b))
            elif pattern == '有' and len(chars_list) >= 2:
                a, b = random.sample(chars_list, 2)
                sentences.append(f"{a}有{b}。")
            elif pattern == '修饰' and len(chars_list) >= 2:
                a, b = random.sample(chars_list, 2)
                sentences.append(f"{a}的{b}")
            elif pattern == '疑问':
                a = random.choice(chars_list)
                sentences.append(self.stage4.make_question(a))
        
        return sentences
    
    def test_current(self) -> dict:
        """检测当前阶段的掌握程度"""
        if self.current_stage == 1:
            # 随机抽10个已学字，查释义
            sample = random.sample(list(self.known_chars), min(10, len(self.known_chars)))
            results = []
            for ch in sample:
                info = self.stage1.get_char_info(ch)
                results.append({
                    'char': ch,
                    'has_def': bool(info['definition']),
                    'has_comp': bool(info['components']),
                    'def_length': len(info['definition']),
                })
            
            with_def = sum(1 for r in results if r['has_def'])
            return {
                'stage': 1,
                'sampled': len(sample),
                'with_definition': with_def,
                'coverage': with_def / len(sample) if sample else 0,
            }
        
        return {'stage': self.current_stage, 'note': '检测逻辑待实现'}
    
    def advance_if_ready(self) -> bool:
        """如果当前阶段掌握够了，升到下一阶段"""
        test = self.test_current()
        
        if self.current_stage == 1:
            # 至少学了500字且释义覆盖率>70%
            if len(self.known_chars) >= 500 and test.get('coverage', 0) >= 0.7:
                self.current_stage = 2
                print(f"\n🎉 单字阶段完成! 已识 {len(self.known_chars)} 字")
                print(f"   进入第2阶段: 组合字")
                self.save_progress()
                return True
        
        return False
    
    # ── 状态查询 ────────────────────────────────
    
    def status(self) -> str:
        """返回当前学习状态"""
        lines = [
            f"╔{'═'*48}╗",
            f"║  龙珠学语进度{' '*33}║",
            f"╠{'═'*48}╣",
        ]
        for i in range(1, 9):
            marker = '👶' if i == self.current_stage else ('✅' if i < self.current_stage else '  ')
            lines.append(f"║ {marker} 阶段{i}: {self.STAGES[i]:<6} {'◀' if i == self.current_stage else '  '}{' '*29}║")
        
        lines.append(f"╠{'═'*48}╣")
        lines.append(f"║  已学单字: {len(self.known_chars):>5} 个{' '*25}║")
        lines.append(f"║  已学词语: {len(self.known_words):>5} 个{' '*25}║")
        lines.append(f"║  掌握单字: {len(self.mastered_chars):>5} 个{' '*25}║")
        lines.append(f"╚{'═'*48}╝")
        return '\n'.join(lines)


# ====================================================================
# 演示
# ====================================================================

if __name__ == "__main__":
    baby = BabyCurriculum()
    
    print(baby.status())
    
    # ===== 阶段1: 学单字 =====
    print("\n" + "=" * 60)
    print("  阶段1: 认字 — 像婴儿一样一个一个认")
    print("=" * 60)
    
    # 学前30个字
    batch = baby.learn_next_batch(30)
    for i, item in enumerate(batch):
        ch = item['char']
        defn = item['definition'][:30] if item['definition'] else '???'
        comps = item.get('components', [])
        comp_str = ' + '.join(comps) if comps else '独体'
        print(f"  {i+1:3d}. {ch}  [{comp_str}]  {defn}")
    
    baby.save_progress()
    
    # ===== 阶段2: 组合字 =====
    baby.start_stage(2)
    print("\n  学组合字 — 看看这些字由什么拼成:")
    compounds = baby.learn_next_batch(10)
    for item in compounds:
        explanation = baby.stage2.explain_char(item)
        print(f"  {explanation}")
    
    # ===== 阶段3: 词语 =====
    baby.start_stage(3)
    print("\n  学词语 — 把这些字拼成词:")
    words = baby.learn_next_batch(15)
    for w in words:
        defs = w['definitions'][:2]
        print(f"  {w['word']} [{w['pinyin']}] {'; '.join(defs)[:40]}")
    
    # ===== 阶段4: 成句 =====
    baby.start_stage(4)
    print("\n  造句 — 试试把词串起来:")
    sentences = baby.learn_next_batch(10)
    for s in sentences:
        print(f"  {s}")
    
    # ===== 检测 =====
    test = baby.test_current()
    print(f"\n  检测: {json.dumps(test, ensure_ascii=False)}")
    
    baby.save_progress()
    print(baby.status())
