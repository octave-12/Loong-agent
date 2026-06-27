#!/usr/bin/env python3
"""从 Wiktionary 和 Wikibooks XML dump 提取知识注入 concept_graph.db"""
import sys, os, re, time, sqlite3, bz2, logging
from collections import defaultdict
from xml.etree import cElementTree as ET

PROJECT = "/mnt/d/soso/projects/Loong-agent"
sys.path.insert(0, PROJECT)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

CG_DB = os.path.join(PROJECT, 'data', 'models', 'concept_graph.db')
WIKTIONARY_PATH = os.path.join(PROJECT, 'data', 'zhwiktionary.xml.bz2')
WIKIBOOKS_PATH = os.path.join(PROJECT, 'data', 'zhwikibooks.xml.bz2')

def insert_triple(conn, s, r, o, confidence, source):
    """注入三元组"""
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

def parse_wiktionary(conn):
    """从 Wiktionary XML 提取词语定义"""
    log.info("=" * 60)
    log.info("📖 注入 Wiktionary 词语定义")
    log.info("=" * 60)
    
    stats = {"new": 0, "dup": 0, "updated": 0, "pages": 0}
    
    with bz2.open(WIKTIONARY_PATH, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('end',))
        for event, elem in context:
            if elem.tag != '{http://www.mediawiki.org/xml/export-0.11/}page':
                continue
            
            ns = '{http://www.mediawiki.org/xml/export-0.11/}'
            title = elem.find(f'{ns}title')
            revision = elem.find(f'{ns}revision')
            if title is None or revision is None:
                elem.clear()
                continue
            
            title = title.text
            text_elem = revision.find(f'{ns}text')
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue
            
            text = text_elem.text
            stats["pages"] += 1
            
            # 只处理中文字词（2-8字）
            if not re.match(r'^[\u4e00-\u9fff]{2,8}$', title):
                elem.clear()
                continue
            
            # 1. 提取 ==漢語== / ==汉语== 下的定义
            chinese_section = re.search(r'==[汉漢]语==(.*?)(?=\n==[^=]|\Z)', text, re.DOTALL)
            if not chinese_section:
                elem.clear()
                continue
            
            section = chinese_section.group(1)
            
            # 提取词性+释义: # {{名词}} 释义文本
            defs = re.findall(r'#\s*(?:{{[^}]+}})?\s*(.+?)(?:\n|$)', section)
            for d in defs[:3]:  # 最多3条定义
                d = re.sub(r'{{[^}]+}}', '', d).strip()
                d = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', d)  # wikilink
                if len(d) >= 4 and len(d) <= 80:
                    result = insert_triple(conn, title, "DEFINED_AS", d, 0.65, "wiktionary")
                    stats[result] = stats.get(result, 0) + 1
            
            # 2. 提取分类关系 ([[Category:xxx]])
            categories = re.findall(r'\[\[Category:(.+?)\]\]', text)
            for cat in categories[:5]:
                cat = cat.strip()
                if len(cat) >= 2:
                    result = insert_triple(conn, title, "IS_A", cat, 0.50, "wiktionary")
                    stats[result] = stats.get(result, 0) + 1
            
            # 3. 提取相关词 (===相关词=== 下的 [[xxx]])
            related = re.search(r'===相关词===(.*?)(?=\n===|\Z)', section, re.DOTALL)
            if related:
                links = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', related.group(1))
                for link in links[:10]:
                    link = link.strip()
                    if re.match(r'^[\u4e00-\u9fff]{2,8}$', link) and link != title:
                        result = insert_triple(conn, title, "RELATED", link, 0.40, "wiktionary")
                        stats[result] = stats.get(result, 0) + 1
            
            if stats["pages"] % 5000 == 0:
                conn.commit()
                log.info(f"  已处理 {stats['pages']} 页 | new={stats['new']} dup={stats['dup']} upd={stats['updated']}")
            
            elem.clear()
    
    conn.commit()
    elapsed = time.time() - t0
    log.info(f"完成: {stats['pages']} 页, new={stats['new']}, dup={stats['dup']}, upd={stats['updated']}")
    return stats

def parse_wikibooks(conn):
    """从 Wikibooks XML 提取教科书结构化知识"""
    log.info("=" * 60)
    log.info("📚 注入 Wikibooks 教科书知识")
    log.info("=" * 60)
    
    stats = {"new": 0, "dup": 0, "updated": 0, "pages": 0}
    
    with bz2.open(WIKIBOOKS_PATH, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('end',))
        for event, elem in context:
            if elem.tag != '{http://www.mediawiki.org/xml/export-0.11/}page':
                continue
            
            ns = '{http://www.mediawiki.org/xml/export-0.11/}'
            title = elem.find(f'{ns}title')
            revision = elem.find(f'{ns}revision')
            if title is None or revision is None:
                elem.clear()
                continue
            
            title = title.text
            text_elem = revision.find(f'{ns}text')
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue
            
            text = text_elem.text
            stats["pages"] += 1
            
            # 跳过非内容页
            if '/' in title and not title.startswith(('数学', '物理', '化学', '生物', '地理', '历史')):
                elem.clear()
                continue
            
            # 提取章节标题作为分类
            headings = re.findall(r'={2,4}\s*(.+?)\s*={2,4}', text)
            parent = title
            for h in headings[:20]:
                h = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', h).strip()
                if len(h) >= 2 and h != parent:
                    result = insert_triple(conn, parent, "HAS", h, 0.45, "wikibooks")
                    stats[result] = stats.get(result, 0) + 1
            
            # 提取 [[链接]] 中的术语
            links = re.findall(r'\[\[([^\]|:#]+)(?:\|[^\]]+)?\]\]', text)
            for link in links[:15]:
                link = link.strip()
                if re.match(r'^[\u4e00-\u9fff\w]{2,20}$', link) and link != title:
                    result = insert_triple(conn, title, "RELATED", link, 0.35, "wikibooks")
                    stats[result] = stats.get(result, 0) + 1
            
            if stats["pages"] % 1000 == 0:
                conn.commit()
                log.info(f"  已处理 {stats['pages']} 页 | new={stats['new']} dup={stats['dup']}")
            
            elem.clear()
    
    conn.commit()
    elapsed = time.time() - t0
    log.info(f"完成: {stats['pages']} 页, new={stats['new']}, dup={stats['dup']}, upd={stats['updated']}")
    return stats

t0 = time.time()
conn = sqlite3.connect(CG_DB)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

base_total = conn.execute("SELECT count(*) FROM triples").fetchone()[0]
log.info(f"概念图基线: {base_total} 三元组")

wiktionary_stats = parse_wiktionary(conn)
wikibooks_stats = parse_wikibooks(conn)

new_total = conn.execute("SELECT count(*) FROM triples").fetchone()[0]
conn.close()

elapsed = time.time() - t0
log.info(f"\n{'='*60}")
log.info(f"📊 总结")
log.info(f"{'='*60}")
log.info(f"   Wiktionary:  {wiktionary_stats}")
log.info(f"   Wikibooks:   {wikibooks_stats}")
log.info(f"   总新增:      {new_total - base_total}")
log.info(f"   总耗时:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
