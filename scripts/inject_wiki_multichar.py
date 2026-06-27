#!/usr/bin/env python3
"""Wikipedia 知识注入 — 直接注入多字科学概念（低覆盖优先）"""
import sys, os, time, sqlite3, re, logging
PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

from loongpearl.core.wiki_lookup import WikipediaLookup

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')
WIKI_DB = os.path.join(PROJECT, 'data', 'wikipedia', 'zhwiki.db')

# ── 关系类型 ──
REL_IS_A    = "IS_A"
REL_RELATED = "RELATED"
REL_PART_OF = "PART_OF"
REL_HAS     = "HAS"

def insert_triple(conn, s, r, o, confidence, source="wikipedia_extract"):
    """注入一条三元组，跳过重复"""
    key = f"{s}|{r}|{o}"
    cur = conn.execute("SELECT id, c, ev FROM triples WHERE s=? AND r=? AND o=?", (s, r, o))
    existing = cur.fetchone()
    if existing:
        # 已存在：提升置信度
        old_c = existing[1]
        if confidence > old_c:
            conn.execute("UPDATE triples SET c=?, ev=CAST(ev AS INTEGER)+1 WHERE id=?",
                        (confidence, existing[0]))
            return "updated"
        conn.execute("UPDATE triples SET ev=CAST(ev AS INTEGER)+1 WHERE id=?", (existing[0],))
        return "duplicate"
    conn.execute(
        "INSERT INTO triples (s, r, o, c, src, ev) VALUES (?, ?, ?, ?, ?, '1')",
        (s, r, o, confidence, source)
    )
    return "new"

def extract_triples_from_text(text, concept):
    """从Wikipedia文本提取IS_A/PART_OF/HAS三元组"""
    triples = []
    
    # 模式1: X是一种/类Y
    for m in re.finditer(rf'{re.escape(concept)}是[一]?[种个类门项]([^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+?)(?:[，。；,\.;]|$)', text):
        obj = m.group(1).strip()[:30]
        if len(obj) >= 2:
            triples.append((concept, REL_IS_A, obj, 0.70))
    
    # 模式2: X属于Y
    for m in re.finditer(rf'{re.escape(concept)}属于([^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+?)(?:[，。；,\.;]|$)', text):
        obj = m.group(1).strip()[:30]
        if len(obj) >= 2:
            triples.append((concept, REL_IS_A, obj, 0.75))
    
    # 模式3: X是Y的(组成部分/一种)
    for m in re.finditer(rf'{re.escape(concept)}是([^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+?)的(?:组成|构成|部分)', text):
        obj = m.group(1).strip()[:30]
        if len(obj) >= 2:
            triples.append((concept, REL_PART_OF, obj, 0.65))
    
    # 模式4: X包含/包括Y
    for m in re.finditer(rf'{re.escape(concept)}(?:包含|包括|由.*组成[：:]?)([^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+?)(?:[，。；,\.;]|$)', text):
        parts = re.split(r'[、，,和及与]', m.group(1))
        for part in parts:
            part = part.strip()[:20]
            if len(part) >= 2 and part != concept:
                triples.append((concept, REL_HAS, part, 0.55))
    
    # 模式5: X与Y (关联)
    for m in re.finditer(rf'{re.escape(concept)}[与和及]([^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+?)(?:[，。；,\.;密切相关有关联]|$)', text):
        obj = m.group(1).strip()[:20]
        if 2 <= len(obj) <= 20 and obj != concept:
            triples.append((concept, REL_RELATED, obj, 0.40))
    
    return triples[:30]  # 每篇最多30条

def get_low_coverage_concepts(conn, min_chars=2, limit=200):
    """获取低覆盖多字概念（三元组数<5）"""
    query = """
    SELECT s, cnt FROM (
        SELECT s, count(*) as cnt FROM triples 
        WHERE length(s) >= ? 
        GROUP BY s 
        HAVING cnt < 5
        ORDER BY cnt ASC
    )
    """
    rows = conn.execute(query, (min_chars,)).fetchall()
    return rows[:limit]

# ═══════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("📖 Wikipedia 知识注入 — 多字概念（低覆盖优先）")
    log.info("=" * 60)
    
    conn = sqlite3.connect(CG_DB)
    wiki = WikipediaLookup(WIKI_DB)
    
    # 获取低覆盖多字概念
    concepts = get_low_coverage_concepts(conn, min_chars=2, limit=200)
    log.info(f"   低覆盖多字概念: {len(concepts)} 个")
    
    # 过滤 — 找有Wikipedia文章的概念
    candidates = []
    for concept, cnt in concepts:
        # 跳过停用词和单字
        if len(concept) < 2:
            continue
        articles = wiki.search_articles(concept, limit=3)
        if articles:
            candidates.append((concept, cnt, articles))
    
    candidates = sorted(candidates, key=lambda x: x[1])[:100]  # 最需要的前100
    
    log.info(f"   有Wiki文章: {len(candidates)} 个（取前100）")
    log.info(f"   Top-10: {', '.join(c[0] for c in candidates[:10])}")
    
    total_new = 0
    total_updated = 0
    total_dup = 0
    t0 = time.time()
    
    for i, (concept, existing_cnt, articles) in enumerate(candidates):
        try:
            article = wiki.get_article(concept)
            if not article:
                continue
            
            text = article.get('text', '')[:10000]  # 前10000字符
            triples = extract_triples_from_text(text, concept)
            
            added = 0
            for s, r, o, c in triples:
                result = insert_triple(conn, s, r, o, c)
                if result == "new":
                    added += 1
                    total_new += 1
                elif result == "updated":
                    total_updated += 1
                else:
                    total_dup += 1
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(candidates) - i - 1) / rate if rate > 0 else 0
                log.info(f"  [{i+1}/{len(candidates)}] {concept} "
                         f"现有={existing_cnt} 新增={added} | "
                         f"{rate:.1f}con/s ETA={eta:.0f}s "
                         f"累计new={total_new} upd={total_updated} dup={total_dup}")
        except Exception as e:
            log.debug(f"  [{i+1}] {concept}: {e}")
    
    conn.commit()
    elapsed = time.time() - t0
    
    # 新总数
    total = conn.execute("SELECT count(*) FROM triples").fetchone()[0]
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 注入统计")
    log.info(f"{'='*60}")
    log.info(f"   处理概念:     {len(candidates)}")
    log.info(f"   新增三元组:   {total_new}")
    log.info(f"   更新置信度:   {total_updated}")
    log.info(f"   跳过重复:     {total_dup}")
    log.info(f"   总耗时:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"   新总数:       {total}")
    
    conn.close()

if __name__ == '__main__':
    main()
