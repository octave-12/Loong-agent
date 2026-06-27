#!/usr/bin/env python3
"""从 Wikipedia 文本中提取因果/时序/空间/属性关系，注入 concept_graph.db"""
import sys, os, re, time, sqlite3, logging

PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')
WIKI_DB = os.path.join(PROJECT, 'data', 'wikipedia', 'zhwiki.db')

# ═══════════════════════════════════════════════════════════════
# 关系提取模式
# ═══════════════════════════════════════════════════════════════

PATTERNS = {
    # ── 因果 ──
    "CAUSE": [
        (r'因为(.{2,20})[，,]?所以(.{2,30})', 0.65),       # 因为X所以Y → X-CAUSE->Y
        (r'由于(.{2,20})[，,]?(.{2,30})', 0.55),            # 由于X，Y
        (r'(.{2,20})导致(.{2,30})', 0.70),                  # X导致Y
        (r'(.{2,20})引起(.{2,30})', 0.65),                  # X引起Y
        (r'(.{2,20})造成(.{2,30})', 0.65),                  # X造成Y
        (r'(.{2,20})引发(.{2,30})', 0.60),                  # X引发Y
        (r'(.{2,20})[是]?(.{2,30})的原因', 0.75),           # X是Y的原因
        (r'(.{2,20})[是]?(.{2,30})的结果', 0.70),           # X是Y的结果
    ],
    # ── 时序 ──
    "BEFORE": [
        (r'(.{2,20})(?:之?)后[,，]?(.{2,30})', 0.55),       # X之后Y
        (r'(.{2,20})先于(.{2,30})', 0.65),                  # X先于Y
        (r'(.{2,20})(?:之?)前[,，]?(.{2,30})', 0.55),       # X之前Y
        (r'先(.{2,15})[，,]?再(.{2,15})', 0.60),           # 先X再Y
    ],
    # ── 部分-整体 ──
    "PART_OF": [
        (r'(.{2,20})是(.{2,20})的(?:组成|构成|一部分|分支)', 0.70),  # X是Y的组成
        (r'(.{2,20})属于(.{2,20})的(?:范畴|一部分)', 0.65),        # X属于Y范畴
        (r'(.{2,20})[，,]?(.{2,20})的(?:一种|一个分支)', 0.65),    # X，Y的一种
    ],
    "HAS": [
        (r'(.{2,20})(?:包括|包含|由.{0,3}组成[：:]?)(.{2,60})', 0.60),  # X包括Y1、Y2
        (r'(.{2,20})主要(?:有|分为)(.{2,60})', 0.55),                   # X主要有Y
    ],
    # ── 空间/位置 ──
    "LOCATED_IN": [
        (r'(.{2,20})位于(.{2,20})', 0.75),                  # X位于Y
        (r'(.{2,20})地处(.{2,20})', 0.70),                  # X地处Y
        (r'(.{2,20})在(.{2,20})[东西南北中]部', 0.60),      # X在Y东部
    ],
    # ── 属性 ──
    "HAS_PROPERTY": [
        (r'(.{2,15})的([熔沸凝固]点|密度|质量|速度|温度|长度|高度)是?[约大约]?(\d+[\.\d]*[°℃%]?[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]*)', 0.70),
        (r'(.{2,15})的(颜色|形状|大小|重量)为(.{2,15})', 0.60),
    ],
    # ── 同义 ──
    "SYNONYM": [
        (r'(.{2,15})[也又称也叫亦称作俗称]?(?:称|叫|名)(.{2,15})', 0.55),
        (r'(.{2,15})[和与](.{2,15})(?:同义|意思相同|含义相同)', 0.65),
    ],
    # ── 材料构成 ──
    "MADE_OF": [
        (r'(.{2,15})[由用](.{2,15})(?:制成|做成|打造|铸造)', 0.60),
        (r'(.{2,15})的材料是(.{2,15})', 0.70),
    ],
}

def clean_text(t):
    """清理提取文本"""
    t = re.sub(r'[［［\[]\d+[］］\]]', '', t)  # 去引用标记[1]
    t = re.sub(r'{{[^}]+}}', '', t)           # 去wiki模板
    t = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', t)  # wikilink
    return t.strip()

def split_multi_items(text):
    """分割多项目（Y1、Y2、Y3）"""
    items = re.split(r'[、，,;；和及与]', text)
    return [i.strip()[:30] for i in items if len(i.strip()) >= 2]

def is_valid_concept(s):
    """检查是否为有效概念"""
    if not s or len(s) < 2 or len(s) > 30:
        return False
    # 不能全是标点/数字
    if re.match(r'^[\d\.\s，。；：、！？""''（）《》\-,:;!?\'\"()\[\]{}]+$', s):
        return False
    return True

def insert_triple(conn, s, r, o, confidence, source="wikipedia_extract"):
    try:
        cur = conn.execute("SELECT id, c FROM triples WHERE s=? AND r=? AND o=?", (s, r, o))
        existing = cur.fetchone()
        if existing:
            if confidence > existing[1]:
                conn.execute("UPDATE triples SET c=? WHERE id=?", (confidence, existing[0]))
                return "updated"
            return "duplicate"
        conn.execute(
            "INSERT INTO triples (s, r, o, c, src, ev) VALUES (?, ?, ?, ?, ?, '1')",
            (s, r, o, confidence, source)
        )
        return "new"
    except Exception:
        return "error"

# ═══════════════════════════════════════════════════════════════
def extract_from_wikipedia():
    log.info("=" * 60)
    log.info("🔍 从 Wikipedia 挖掘因果/时序/空间/属性关系")
    log.info("=" * 60)
    
    conn_cg = sqlite3.connect(CG_DB)
    conn_wiki = sqlite3.connect(WIKI_DB)
    
    # 获取有Wikipedia文章的中文概念
    titles = conn_wiki.execute("""
        SELECT title FROM articles 
        WHERE title NOT LIKE '%/%' 
        AND length(title) >= 2 
        AND title GLOB '[一-龥]*'
        AND char_count > 500
        LIMIT 50000
    """).fetchall()
    
    log.info(f"   候选文章: {len(titles)} 篇")
    
    stats = {rel: {"new": 0, "dup": 0, "updated": 0} for rel in PATTERNS}
    total_pages = 0
    
    t0 = time.time()
    for (title,) in titles:
        row = conn_wiki.execute("SELECT text FROM articles WHERE title=?", (title,)).fetchone()
        if not row or not row[0]:
            continue
        
        text = clean_text(row[0][:8000])  # 前8000字符
        total_pages += 1
        
        for rel, patterns in PATTERNS.items():
            for pattern, base_conf in patterns:
                for m in re.finditer(pattern, text):
                    groups = m.groups()
                    if len(groups) >= 2:
                        s, o = groups[0], groups[1]
                        if is_valid_concept(s) and is_valid_concept(o) and s != o:
                            # 特殊处理HAS: 分割多项目
                            if rel == "HAS" and len(o) > 30:
                                items = split_multi_items(o)
                            else:
                                items = [o]
                            
                            for item in items:
                                if not is_valid_concept(item) or item == s:
                                    continue
                                result = insert_triple(conn_cg, s, rel, item, base_conf)
                                stats[rel][result] = stats[rel].get(result, 0) + 1
        
        if total_pages % 5000 == 0:
            conn_cg.commit()
            elapsed = time.time() - t0
            new_total = sum(s["new"] for s in stats.values())
            log.info(f"  已处理 {total_pages}/{len(titles)} 篇 | 新增={new_total} | {elapsed:.0f}s")
    
    conn_cg.commit()
    conn_cg.close()
    conn_wiki.close()
    
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"📊 提取统计 ({total_pages} 篇, {elapsed:.0f}s)")
    log.info(f"{'='*60}")
    for rel, s in sorted(stats.items()):
        if s["new"] + s["updated"] > 0:
            log.info(f"  {rel:15s}  new={s['new']:5d}  upd={s['updated']:3d}  dup={s['dup']:5d}")
    
    total_new = sum(s["new"] for s in stats.values())
    log.info(f"\n  总新增: {total_new}")
    return total_new

if __name__ == '__main__':
    extract_from_wikipedia()
