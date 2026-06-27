#!/usr/bin/env python3
"""Baidu Baike 全量注入 — 临时表批量去重版"""
import sys, os, re, json, time, sqlite3, logging
from itertools import islice

PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')
BATCH_SIZE = 200000  # 每批20万行JSON → 约10万条三元组

def is_valid_title(t):
    if not t or len(t) < 2 or len(t) > 30:
        return False
    if not re.search(r'[\u4e00-\u9fff]', t):
        return False
    return True

def inject_baidu_fast():
    log.info("=" * 60)
    log.info("📖 Baidu Baike 全量注入 (临时表批量去重)")
    log.info("=" * 60)
    
    conn = sqlite3.connect(CG_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-2000000")
    conn.execute("PRAGMA temp_store=MEMORY")
    
    # 创建临时表（无约束，快速写入）
    conn.execute("DROP TABLE IF EXISTS _baidu_batch")
    conn.execute("""
        CREATE TEMP TABLE _baidu_batch (
            s TEXT, r TEXT, o TEXT, c REAL, src TEXT
        )
    """)
    
    baike_path = hf_hub_download('lars1234/baidu-baike-dataset', filename='563w_baidubaike.json', repo_type='dataset')
    
    stats = {"processed": 0, "skipped": 0, "total_inserted": 0}
    t0 = time.time()
    t0_batch = time.time()
    
    with open(baike_path, 'r', encoding='utf-8') as f:
        batch = []
        
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            title = item.get('title', '')
            if not is_valid_title(title):
                stats["skipped"] += 1
                continue
            
            summary = item.get('summary', '') or ''
            
            # DEFINED_AS
            if len(summary) >= 10:
                first = summary.split('。')[0].split('；')[0].strip()[:120]
                if first.startswith(title):
                    first = re.sub(r'^' + re.escape(title) + r'[是指为]?', '', first).strip()
                if len(first) >= 4:
                    batch.append((title, "DEFINED_AS", first, 0.70, "baidu_baike"))
            
            # IS_A from tags
            for tag in item.get('tags', []):
                if tag != title and len(tag) >= 2:
                    batch.append((title, "IS_A", tag, 0.55, "baidu_baike"))
            
            # HAS from sections
            for sec in item.get('sections', []):
                sec_title = sec.get('title', '')
                if sec_title and len(sec_title) >= 2 and sec_title != title:
                    batch.append((title, "HAS", sec_title, 0.50, "baidu_baike"))
            
            stats["processed"] += 1
            
            # 积累到阈值→批量写入
            if len(batch) >= BATCH_SIZE:
                _flush(conn, batch, stats)
                batch.clear()
                
                elapsed = time.time() - t0
                rate = stats["processed"] / elapsed if elapsed > 0 else 0
                remaining = 5630000 - stats["processed"]
                eta = remaining / rate if rate > 0 else 0
                log.info(f"  {stats['processed']/10000:.0f}万 | 写入{stats['total_inserted']/10000:.0f}万条 | {rate:.0f}行/s | {elapsed:.0f}s | ETA={eta/60:.0f}min")
        
        # 处理剩余
        if batch:
            _flush(conn, batch, stats)
    
    elapsed = time.time() - t0
    new_total = conn.execute("SELECT count(*) FROM triples").fetchone()[0]
    
    conn.execute("DROP TABLE IF EXISTS _baidu_batch")
    conn.close()
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 完成")
    log.info(f"   处理: {stats['processed']:,}  跳过: {stats['skipped']:,}")
    log.info(f"   注入: {stats['total_inserted']:,}")
    log.info(f"   总数: {new_total:,}")
    log.info(f"   耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

def _flush(conn, batch, stats):
    """写入临时表 → 去重合并到主表"""
    if not batch:
        return
    
    # 写入临时表（无约束，极快）
    conn.execute("DELETE FROM _baidu_batch")
    conn.executemany("INSERT INTO _baidu_batch VALUES (?, ?, ?, ?, ?)", batch)
    
    # 从临时表去重合并到主表（一次SQL完成）
    conn.execute("""
        INSERT OR IGNORE INTO triples (s, r, o, c, src, ev)
        SELECT DISTINCT s, r, o, MAX(c), src, '1'
        FROM _baidu_batch
        GROUP BY s, r, o
    """)
    conn.commit()
    stats["total_inserted"] += conn.total_changes

if __name__ == '__main__':
    inject_baidu_fast()
