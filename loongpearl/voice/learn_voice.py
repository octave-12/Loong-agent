#!/usr/bin/env python3
"""
重新分析 1105 音节 — 只学 F0 曲线 + 时长
共振峰用学术标准值 (本来就准)
"""
import numpy as np, wave, struct, json, os, sys, time, requests

CDN = 'https://hanyu-word-pinyin-short.cdn.bcebos.com'
REF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ref_audio')
PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'runtime', 'voice_params.json')
os.makedirs(REF, exist_ok=True)

def read_wav(path):
    with wave.open(path,'r') as w:
        n=w.getnframes(); data=w.readframes(n); sr=w.getframerate()
    return np.array(struct.unpack(f'{n}h',data),dtype=np.float64)/32768, sr

def extract_f0_detail(sig, sr):
    """逐帧提取F0, 返回 {mean, min, max, contour}"""
    frame_ms, hop_ms = 20, 10
    fn = int(sr*frame_ms/1000); hn = int(sr*hop_ms/1000)
    nf = max(1, (len(sig)-fn)//hn + 1)
    f0s = []
    for i in range(nf):
        start = i*hn; frame = sig[start:start+fn]
        if len(frame)<50: continue
        best_lag, best_corr = 0, 0
        for lag in range(int(sr/500), min(int(sr/60), len(frame)//2)):
            c = abs(np.dot(frame[:-lag], frame[lag:]))
            if c > best_corr: best_corr=c; best_lag=lag
        if best_corr > 0 and best_lag > 0:
            f = sr/best_lag
            if 60 < f < 500: f0s.append(f)
    
    f0a = np.array(f0s) if f0s else np.array([200])
    f0c = f0a[(f0a>60)&(f0a<500)]
    if len(f0c) < 2: f0c = np.array([200])
    
    return {
        'f0_mean': round(float(np.mean(f0c)), 1),
        'f0_min': round(float(np.min(f0c)), 1),
        'f0_max': round(float(np.max(f0c)), 1),
        'f0_std': round(float(np.std(f0c)), 1),
    }

def analyze_file(py_num):
    """分析一个音节文件"""
    wav_path = os.path.join(REF, f'{py_num}.wav')
    if not os.path.exists(wav_path):
        return None
    
    sig, sr = read_wav(wav_path)
    dur = len(sig) / sr
    if dur < 0.1: return {'dur': round(dur,3), 'f0_mean': 200, 'f0_min': 150, 'f0_max': 250, 'f0_std': 20}
    
    # Skip consonant portion for F0 (first 60ms)
    v_start = int(0.06 * sr)
    vowel = sig[v_start:] if len(sig) > v_start else sig
    
    f0_info = extract_f0_detail(vowel, sr)
    f0_info['dur'] = round(dur, 3)
    return f0_info

# ── Main ──
sys.path.insert(0, '/mnt/d/soso/projects/Loong-agent/Loong-pearl')
from loongpearl.learning.curriculum import BabyCurriculum

TONE_M = {'ā':'a1','á':'a2','ǎ':'a3','à':'a4','ē':'e1','é':'e2','ě':'e3','è':'e4',
          'ī':'i1','í':'i2','ǐ':'i3','ì':'i4','ō':'o1','ó':'o2','ǒ':'o3','ò':'o4',
          'ū':'u1','ú':'u2','ǔ':'u3','ù':'u4','ǖ':'v1','ǘ':'v2','ǚ':'v3','ǜ':'v4'}

def py_to_num(py):
    p=py.strip().lower()
    if p[-1].isdigit(): return p
    r=''; t='0'
    for c in p:
        if c in TONE_M: m=TONE_M[c]; r+=m[0]; t=m[1]
        else: r+=c
    return r+t

print("收集唯一拼音...")
baby = BabyCurriculum()
unique_py = set()
for ch in baby.known_chars:
    info = baby.stage1.get_char_info(ch)
    py = info.get('pinyin','')
    if py: unique_py.add(py)
py_nums = sorted(set(py_to_num(p) for p in unique_py))
print(f"唯一音节: {len(py_nums)} 个")

# 加载已有
params = json.load(open(PARAMS)) if os.path.exists(PARAMS) else {}
print(f"已有参数: {len(params)} 个")

# 下载缺失的 + 重新分析所有
session = requests.Session()
session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://hanyu.baidu.com/'})

new_count = 0
for i, py_num in enumerate(py_nums):
    mp3_path = os.path.join(REF, f'{py_num}.mp3')
    wav_path = os.path.join(REF, f'{py_num}.wav')
    
    # 下载
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 500:
        try:
            r = session.get(f'{CDN}/{py_num}.mp3', timeout=8)
            if r.status_code==200 and len(r.content)>500:
                with open(mp3_path,'wb') as f: f.write(r.content)
            else: continue
        except: continue
        time.sleep(0.12)
    
    # 转WAV
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
        os.system(f'ffmpeg -y -i "{mp3_path}" -ar 16000 -ac 1 "{wav_path}" 2>/dev/null')
    
    if not os.path.exists(wav_path): continue
    
    # 分析
    try:
        info = analyze_file(py_num)
        if info:
            params[py_num] = info
            new_count += 1
            if new_count % 100 == 0:
                print(f"  已分析 {new_count} 个...")
    except Exception as e:
        print(f"  ⚠ {py_num}: {e}")

# 保存
with open(PARAMS, 'w') as f:
    json.dump(params, f, ensure_ascii=False, indent=1)
print(f"\n完成! {len(params)} 个参数 (新增/更新 {new_count})")

# 打印几个样本验证
for k in ['hua4','long2','dian3','mao1','ai4']:
    print(f"  {k}: {params.get(k, '?')}")
