#!/usr/bin/env python3
"""从 Baidu Baike JSON 提取知识注入 concept_graph.db（仅注入已有概念）"""
import sys, os, re, json, time, sqlite3, logging

PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')

def insert_triple(conn, s, r, o, confidence, source="baidu_baike"):
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

# 因果/时序/部分-整体提取模式（与extract_relations.py相同）
PATTERNS = {
    "CAUSE": [
        (r'因为(.{2,20})[，,]?所以(.{2,30})', 0.65),
        (r'由于(.{2,20})[，,]?(.{2,30})', 0.55),
        (r'(.{2,20})导致(.{2,30})', 0.70),
        (r'(.{2,20})引起(.{2,30})', 0.65),
        (r'(.{2,20})是(.{2,30})的原因', 0.75),
        (r'(.{2,20})是(.{2,30})的结果', 0.70),
    ],
    "BEFORE": [
        (r'(.{2,20})(?:之?)后[,，]?(.{2,30})', 0.55),
    ],
    "PART_OF": [
        (r'(.{2,20})是(.{2,20})的(?:组成|构成|一部分|分支)', 0.70),
        (r'(.{2,20})属于(.{2,20})的(?:范畴|一部分)', 0.65),
    ],
    "LOCATED_IN": [
        (r'(.{2,20})位于(.{2,20})', 0.75),
        (r'(.{2,20})地处(.{2,20})', 0.70),
    ],
    "HAS_PROPERTY": [
        (r'(.{2,15})的([熔沸凝固]点|密度|质量|速度|温度)是?[约大约]?(\d+[\.\d]*[°℃%]?[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]*)', 0.70),
    ],
}

def clean_text(t):
    t = re.sub(r'[［［\[]\d+[］］\]]', '', t)
    t = re.sub(r'<[^>]+>', '', t)
    return t.strip()

def is_valid_concept(s):
    if not s or len(s) < 2 or len(s) > 30:
        return False
    if re.match(r'^[\d\.\s，。；：、！？""''（）《》\-,:;!?\'\"()\[\]{}]+$', s):
        return False
    return True

def extract_from_text(text, title, conn, stats):
    text = clean_text(text)
    
    # 从文本中提取关系
    for rel, patterns in PATTERNS.items():
        for pattern, base_conf in patterns:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                if len(groups) >= 2:
                    s, o = groups[0].strip(), groups[1].strip()
                    if s == title or s in title or title in s:
                        if is_valid_concept(o) and o != title:
                            result = insert_triple(conn, title, rel, o, base_conf)
                            stats[result] = stats.get(result, 0) + 1

def inject_baidu():
    log.info("=" * 60)
    log.info("📖 注入 Baidu Baike 知识")
    log.info("=" * 60)
    
    conn = sqlite3.connect(CG_DB)
    
    # 获取概念图中所有多字中文概念
    existing_concepts = set()
    for (concept,) in conn.execute(
        "SELECT DISTINCT s FROM triples WHERE length(s) >= 2 AND s GLOB '[一-龥]*'"
    ):
        existing_concepts.add(concept)
    log.info(f"   已有概念: {len(existing_concepts):,} 个")
    
    baike_path = hf_hub_download('lars1234/baidu-baike-dataset', filename='563w_baidubaike.json', repo_type='dataset')
    
    stats = {"new": 0, "dup": 0, "updated": 0, "processed": 0, "matched": 0}
    t0 = time.time()
    
    with open(baike_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            title = item.get('title', '')
            if title not in existing_concepts:
                continue
            
            stats["matched"] += 1
            
            # 1. DEFINED_AS from summary
            summary = item.get('summary', '')
            if summary and len(summary) >= 10:
                # 取第一句作为定义
                first_sent = re.split(r'[。；\n]', summary)[0].strip()[:100]
                if len(first_sent) >= 8:
                    # 清理：去掉"XX是指""XX是"前缀
                    first_sent = re.sub(r'^' + re.escape(title) + r'[是指为]?', '', first_sent).strip()
                    if len(first_sent) >= 4:
                        result = insert_triple(conn, title, "DEFINED_AS", first_sent, 0.70, "baidu_baike")
                        stats[result] = stats.get(result, 0) + 1
            
            # 2. IS_A from tags
            tags = item.get('tags', [])
            for tag in tags:
                if tag != title and len(tag) >= 2:
                    result = insert_triple(conn, title, "IS_A", tag, 0.55, "baidu_baike")
                    stats[result] = stats.get(result, 0) + 1
            
            # 3. HAS from section titles
            sections = item.get('sections', [])
            for sec in sections:
                sec_title = sec.get('title', '')
                if sec_title and len(sec_title) >= 2 and sec_title != title:
                    result = insert_triple(conn, title, "HAS", sec_title, 0.50, "baidu_baike")
                    stats[result] = stats.get(result, 0) + 1
            
            # 4. 从 summary+sections 提取因果关系
            full_text = summary or ''
            for sec in sections:
                full_text += ' ' + (sec.get('content', '') or '')
            if full_text:
                extract_from_text(full_text[:5000], title, conn, stats)
            
            stats["processed"] += 1
            
            if stats["matched"] % 10000 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = stats["matched"] / elapsed if elapsed > 0 else 0
                eta = (len(existing_concepts) - stats["matched"]) / rate if rate > 0 else 0
                log.info(f"  匹配 {stats['matched']:,} | new={stats['new']:6d} dup={stats['dup']:6d} | {rate:.0f}con/s ETA={eta/60:.0f}min")
    
    conn.commit()
    elapsed = time.time() - t0
    
    new_total = conn.execute("SELECT count(*) FROM triples").fetchone()[0]
    conn.close()
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 Baidu Baike 注入统计")
    log.info(f"{'='*60}")
    log.info(f"   匹配概念: {stats['matched']:,}")
    log.info(f"   新增: {stats['new']}  更新: {stats['updated']}  重复: {stats['dup']}")
    log.info(f"   新总数: {new_total:,}")
    log.info(f"   耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

if __name__ == '__main__':
    inject_baidu()
