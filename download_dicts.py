#!/usr/bin/env python3
"""
龙珠多源知识网点下载器（download_dicts.py）
============================================
下载并解析:
  1. Make Me a Hanzi (9574字) — 字形部件拆解 + 拼音 + 释义 + 字源
  2. Unihan (94117字) — 部首 + 笔画 + 读音 + 定义
  3. CC-CEDICT (12万条) — 汉英词典

输出:
  - dict_decompose.json    -> 字形拆解图 {char: {radical, components, pinyin, def}}
  - dict_unihan.json       -> Unicode全量字典 {char: {radical, strokes, mandarin, def}}
  - dict_cedict.json       -> 汉英词典（备用）
"""

import sys, os, re, json, zipfile, io, time
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, ".dict_cache")
os.makedirs(CACHE, exist_ok=True)

def download(url, filename, mirrors=None):
    """下载文件并缓存，支持多镜像降级"""
    path = os.path.join(CACHE, filename)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"  [缓存] {filename}")
        return open(path, 'rb').read()
    
    urls = [url] + (mirrors or [])
    for i, u in enumerate(urls):
        try:
            print(f"  [下载{'镜像'+str(i) if i>0 else ''}] {u[:80]}...")
            resp = requests.get(u, timeout=120)
            resp.raise_for_status()
            with open(path, 'wb') as f:
                f.write(resp.content)
            return resp.content
        except Exception as e:
            print(f"    ✗ {e}")
            if i == len(urls) - 1:
                raise
    raise RuntimeError("所有镜像均失败")


# ====================================================================
# 1. Make Me a Hanzi — 字形部件拆解
# ====================================================================

def parse_makemeahanzi() -> dict:
    """
    解析 Make Me a Hanzi 字典。
    
    每行一个 JSON:
      {"character": "明", "definition": "...", "pinyin": ["míng"],
       "decomposition": "⿰日月", "radical": "日", "etymology": {...}}
    
    返回: {char: {radical, components, pinyin, definition, etymology_type}}
    """
    print("\n" + "="*50)
    print("1/3 Make Me a Hanzi — 字形部件拆解")
    print("="*50)
    
    url = "https://raw.githubusercontent.com/skishore/makemeahanzi/master/dictionary.txt"
    mirrors = [
        "https://cdn.jsdelivr.net/gh/skishore/makemeahanzi@master/dictionary.txt",
        "https://ghproxy.com/https://raw.githubusercontent.com/skishore/makemeahanzi/master/dictionary.txt",
    ]
    data = download(url, "makemeahanzi.txt", mirrors=mirrors)
    
    result = {}
    for line in data.decode('utf-8').split('\n'):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        ch = d.get('character', '')
        decomp = d.get('decomposition', '')
        radical = d.get('radical', '')
        
        # 从拆解字符串中提取部件 (如 "⿰日月" → ["日", "月"])
        # IDS (Ideographic Description Sequence): ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻
        components = []
        if decomp and decomp != '？':
            # 去掉 IDS 操作符，保留汉字部件
            ids_ops = set('⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻')
            for c in decomp:
                if c not in ids_ops and '\u4e00' <= c <= '\u9fff':
                    components.append(c)
        
        if ch:
            result[ch] = {
                'radical': radical,
                'components': components,
                'pinyin': d.get('pinyin', []),
                'definition': d.get('definition', ''),
                'etymology': d.get('etymology', {}).get('type', ''),
                'decomposition_raw': decomp,
            }
    
    # 统计
    n_with_decomp = sum(1 for v in result.values() if v['components'])
    n_with_def = sum(1 for v in result.values() if v['definition'])
    print(f"  解析: {len(result)} 字")
    print(f"    有部件拆解: {n_with_decomp}")
    print(f"    有英文释义: {n_with_def}")
    
    # 示例
    for ch in ['明', '想', '龍']:
        if ch in result:
            r = result[ch]
            print(f"    {ch} = {r['decomposition_raw']} → {r['components']} [{r['radical']}]")
    
    out = os.path.join(BASE, "dict_decompose.json")
    json.dump(result, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"  保存: {out} ({os.path.getsize(out)/1024:.0f} KB)")
    return result


# ====================================================================
# 2. Unihan — Unicode 官方全量汉字数据库
# ====================================================================

def parse_unihan() -> dict:
    """
    解析 Unihan 数据库 (Unicode 官方, 覆盖全部 CJK 字符).
    
    格式:
      U+4E00	kDefinition	bright; light; brilliant
      U+4E00	kMandarin	míng
      U+4E00	kRSUnicode	72.0  (部首.剩余笔画)
      U+4E00	kTotalStrokes	8
    
    返回: {char: {definition, mandarin, radical, strokes}}
    """
    print("\n" + "="*50)
    print("2/3 Unihan — Unicode 全量汉字数据")
    print("="*50)
    
    url = "https://www.unicode.org/Public/UNIDATA/Unihan.zip"
    data = download(url, "Unihan.zip")
    
    result = {}
    
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # 查找主数据文件
        for name in zf.namelist():
            if 'Unihan_Readings.txt' in name:
                with zf.open(name) as f:
                    for line in f:
                        line = line.decode('utf-8', errors='replace')
                        if line.startswith('#') or line.startswith('\n'):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                        
                        codepoint, field, value = parts[0], parts[1], parts[2]
                        # 只关心 CJK 统一汉字区
                        if not codepoint.startswith('U+'):
                            continue
                        try:
                            cp = int(codepoint[2:], 16)
                        except ValueError:
                            continue
                        
                        char = chr(cp)
                        
                        if char not in result:
                            result[char] = {}
                        
                        if field == 'kDefinition':
                            result[char]['definition'] = value
                        elif field == 'kMandarin':
                            result[char]['mandarin'] = value
                        elif field == 'kTotalStrokes':
                            result[char]['strokes'] = int(value.split()[0]) if value else 0
                        elif field == 'kRSUnicode':
                            # kRSUnicode format: "radical.additional_strokes"
                            parts_rs = value.split('.')
                            if len(parts_rs) >= 2:
                                try:
                                    radical_idx = int(parts_rs[0])
                                    # 部首索引转实际字符
                                    result[char]['radical_idx'] = radical_idx
                                except ValueError:
                                    pass
    
    # 只保留有至少一个字段的
    result = {k: v for k, v in result.items() if v}
    
    n_with_def = sum(1 for v in result.values() if v.get('definition'))
    n_with_mandarin = sum(1 for v in result.values() if v.get('mandarin'))
    print(f"  解析: {len(result)} 字")
    print(f"    有定义: {n_with_def}")
    print(f"    有读音: {n_with_mandarin}")
    
    # 示例
    for ch in ['一', '龍', '明']:
        if ch in result:
            print(f"    {ch}: {result[ch]}")
    
    out = os.path.join(BASE, "dict_unihan.json")
    json.dump(result, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"  保存: {out} ({os.path.getsize(out)/1024:.0f} KB)")
    return result


# ====================================================================
# 3. CC-CEDICT — 汉英词典（可选）
# ====================================================================

def parse_cedict() -> dict:
    """
    解析 CC-CEDICT 汉英词典。
    
    格式:
      傳統 传统 [chuan2 tong3] /tradition/...
    
    返回: {traditional: {simplified, pinyin, definitions}}
    """
    print("\n" + "="*50)
    print("3/3 CC-CEDICT — 汉英词典")
    print("="*50)
    
    url = "https://raw.githubusercontent.com/edolstra/cc-cedict/master/cedict_ts.u8"
    mirrors = [
        "https://ghproxy.com/https://raw.githubusercontent.com/edolstra/cc-cedict/master/cedict_ts.u8",
        "https://mirror.ghproxy.com/https://raw.githubusercontent.com/edolstra/cc-cedict/master/cedict_ts.u8",
    ]
    data = download(url, "cedict_ts.u8", mirrors=mirrors)
    
    result = {}
    for line in data.decode('utf-8').split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        
        # 格式: 傳統 传统 [chuan2 tong3] /definition1/definition2/
        match = re.match(r'(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/', line)
        if not match:
            continue
        
        trad, simp, pinyin_raw, defs_raw = match.groups()
        pinyin = pinyin_raw.replace(' ','')
        definitions = [d.strip() for d in defs_raw.split('/') if d.strip()]
        
        # 只保存单字条目
        if len(trad) == 1 and len(simp) == 1:
            result[trad] = {
                'simplified': simp,
                'pinyin': pinyin,
                'definitions': definitions,
            }
    
    print(f"  解析: {len(result)} 单字条目")
    
    # 统计
    total_defs = sum(len(v['definitions']) for v in result.values())
    print(f"  平均 {total_defs/max(len(result),1):.1f} 释义/字")
    
    out = os.path.join(BASE, "dict_cedict.json")
    json.dump(result, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"  保存: {out} ({os.path.getsize(out)/1024:.0f} KB)")
    return result


# ====================================================================
# 主流程
# ====================================================================

def main():
    t0 = time.time()
    print("🐉 龙珠多源知识网点下载器")
    print("="*50)
    
    decompose = parse_makemeahanzi()
    unihan = parse_unihan()
    cedict = parse_cedict()
    
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"✅ 全部完成 ({elapsed:.1f}s)")
    print(f"  dict_decompose.json  — {len(decompose)} 字 (字形拆解)")
    print(f"  dict_unihan.json     — {len(unihan)} 字 (Unicode全量)")
    print(f"  dict_cedict.json     — {len(cedict)} 字 (汉英词典)")
    print(f"\n下一步: python enrich_landscape.py 将多源知识注入能量景观")

if __name__ == "__main__":
    main()
