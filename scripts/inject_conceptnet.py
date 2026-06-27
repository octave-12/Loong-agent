#!/usr/bin/env python3
"""从 ConceptNet 5 注入中文常识关系到 concept_graph.db"""
import sys, os, re, time, sqlite3, logging
import pyarrow.parquet as pq

PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')

# ConceptNet → 龙珠关系映射
REL_MAP = {
    '/r/Causes':          ('CAUSE', 0.65),
    '/r/CapableOf':       ('HAS_CAPABILITY', 0.55),
    '/r/AtLocation':      ('LOCATED_IN', 0.60),
    '/r/DerivedFrom':     ('DERIVED_FROM', 0.55),
    '/r/Antonym':         ('ANTONYM', 0.70),
    '/r/PartOf':          ('PART_OF', 0.65),
    '/r/HasProperty':     ('HAS_PROPERTY', 0.55),
    '/r/HasSubevent':     ('HAS_SUBEVENT', 0.50),
    '/r/HasPrerequisite': ('HAS_PREREQUISITE', 0.55),
    '/r/MadeOf':          ('MADE_OF', 0.60),
    '/r/UsedFor':         ('USED_FOR', 0.55),
    '/r/Synonym':         ('SYNONYM', 0.65),
    '/r/IsA':             ('IS_A', 0.60),
    '/r/RelatedTo':       ('RELATED', 0.40),
    '/r/FormOf':          ('RELATED', 0.45),
}

def extract_concept(uri):
    """从 /c/zh/概念 提取中文概念"""
    if '/c/zh/' in uri:
        concept = uri.split('/c/zh/')[-1]
        if '/' in concept:
            concept = concept.split('/')[0]
        return concept
    return None

def is_good_concept(c):
    """过滤：2-20字中文，非纯标点/数字"""
    if not c or len(c) < 2 or len(c) > 20:
        return False
    if not re.match(r'^[\u4e00-\u9fff\w]+$', c):
        return False
    # 跳过纯英文/数字
    if re.match(r'^[a-zA-Z0-9_]+$', c):
        return False
    return True

def insert_triple(conn, s, r, o, confidence, source="conceptnet"):
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

def inject_conceptnet():
    log.info("=" * 60)
    log.info("🌐 注入 ConceptNet 中文常识关系")
    log.info("=" * 60)
    
    conn = sqlite3.connect(CG_DB)
    stats = {"new": 0, "dup": 0, "updated": 0, "rows": 0, "zh_rows": 0}
    rel_stats = {}
    
    t0 = time.time()
    for shard_idx in range(23):
        fname = f'conceptnet5/train-{shard_idx:05d}-of-00023.parquet'
        path = hf_hub_download('conceptnet5/conceptnet5', filename=fname, repo_type='dataset')
        df = pq.read_table(path).to_pandas()
        zh = df[df['lang'] == 'zh']
        
        stats["rows"] += len(df)
        stats["zh_rows"] += len(zh)
        
        for _, row in zh.iterrows():
            rel = row['rel']
            if rel not in REL_MAP:
                continue
            
            new_rel, base_conf = REL_MAP[rel]
            c1 = extract_concept(row['arg1'])
            c2 = extract_concept(row['arg2'])
            
            if not is_good_concept(c1) or not is_good_concept(c2):
                continue
            if c1 == c2:
                continue
            
            # 用 weight 微调置信度
            weight = row.get('weight', 1.0)
            conf = min(base_conf + (weight - 1.0) * 0.1, 0.95)
            
            result = insert_triple(conn, c1, new_rel, c2, conf)
            stats[result] = stats.get(result, 0) + 1
            rel_stats[new_rel] = rel_stats.get(new_rel, {"new": 0, "dup": 0})
            rel_stats[new_rel][result] = rel_stats[new_rel].get(result, 0) + 1
        
        conn.commit()
        elapsed = time.time() - t0
        log.info(f"  [{shard_idx+1}/23] new={stats['new']:6d} dup={stats['dup']:6d} | {elapsed:.0f}s")
    
    conn.close()
    elapsed = time.time() - t0
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 ConceptNet 注入统计")
    log.info(f"{'='*60}")
    log.info(f"   扫描行数: {stats['rows']:,}  |  中文: {stats['zh_rows']:,}")
    log.info(f"   新增: {stats['new']}  |  更新: {stats['updated']}  |  重复: {stats['dup']}")
    log.info(f"   耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"\n   关系分布:")
    for rel, s in sorted(rel_stats.items(), key=lambda x: x[1]['new'], reverse=True):
        log.info(f"     {rel:20s}  new={s['new']:6d}  dup={s['dup']:6d}")

if __name__ == '__main__':
    inject_conceptnet()
