#!/usr/bin/env python3
"""从 SQLite 重建 concept_graph.json — 流式写入，避免 OOM"""
import json, sqlite3, os, time

DB = "/mnt/d/soso/projects/Loong-agent/data/models/concept_graph.db"
JSON_PATH = "/mnt/d/soso/projects/Loong-agent/data/models/concept_graph.json"
BATCH = 100000

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
print(f"三元组总数: {total:,}")

# 备份旧 JSON
if os.path.exists(JSON_PATH):
    bak = JSON_PATH + f".bak.{int(time.time())}"
    os.rename(JSON_PATH, bak)
    print(f"旧 JSON 备份: {bak}")

t0 = time.time()
written = 0

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    f.write('{\n  "total_triples": ')
    f.write(str(total))
    f.write(',\n  "total_inferred": 0,\n  "aliases": {},\n  "triples": [\n')

    cursor = conn.execute("SELECT s, r, o, c, src, ev FROM triples ORDER BY id")
    first = True
    
    while True:
        rows = cursor.fetchmany(BATCH)
        if not rows:
            break
        
        for row in rows:
            if not first:
                f.write(',\n')
            first = False
            
            # 手动序列化，比 json.dumps 更快
            s = json.dumps(row['s'], ensure_ascii=False)
            r = json.dumps(row['r'], ensure_ascii=False)
            o = json.dumps(row['o'], ensure_ascii=False)
            c = row['c']
            src = json.dumps(row['src'] or '', ensure_ascii=False)
            ev_raw = row['ev'] or '1'
            # 防御非JSON数据（Python repr、列表等）
            try:
                ev = json.dumps(ev_raw, ensure_ascii=False)
            except (TypeError, ValueError):
                ev = '"1"'
            
            f.write(f'    {{"s":{s},"r":{r},"o":{o},"c":{c},"src":{src},"ev":{ev}}}')
        
        written += len(rows)
        elapsed = time.time() - t0
        rate = written / elapsed if elapsed > 0 else 0
        pct = written / total * 100
        print(f"\r  {written/10000:.0f}万/{total/10000:.1f}万 ({pct:.1f}%) | {rate:.0f}条/s | {elapsed:.0f}s", end='')
    
    f.write('\n  ]\n}\n')

conn.close()
elapsed = time.time() - t0
size_mb = os.path.getsize(JSON_PATH) / (1024*1024)
print(f"\n完成: {written:,} 条 | {size_mb:.0f}MB | {elapsed:.0f}s")
