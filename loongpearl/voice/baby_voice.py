#!/usr/bin/env python3
"""
龙珠婴儿发声器 v10 (baby_voice.py)
====================================
改进:
  1. 10次谐波 (v9只有3次) → 声音更丰满
  2. 加F3第三共振峰 → 韵母区分度更高  
  3. 共振峰带宽模拟 → 不再纯正弦, 更自然
  4. 微气息噪声 → 像真婴儿
  5. 参数学自真声 (F0曲线+时长), 共振峰用学术值
  
实时合成, 不存文件, 纯内存运算.
"""

import numpy as np, json, os, io, wave as wavlib
from loongpearl.data_config import DATA_ROOT, MODEL_DIR, DICT_DIR, RUNTIME_DIR

BASE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(BASE, 'data/runtime/voice_params.json')

# ── 学术标准共振峰 (带F3) ──
VOWEL_F = {
    'a':(900,1250,2600),'o':(550,900,2550),'e':(550,1200,2600),
    'i':(320,2400,3100),'u':(350,700,2450),'v':(320,2100,2750),
    'ü':(320,2100,2750),'er':(550,1450,2500),
    'ai':(850,1300,350,2300,2700),'ei':(550,2000,350,2200,2650),
    'ao':(850,1100,550,850,2550),'ou':(550,950,350,750,2500),
    'ia':(320,2350,900,1250,2800),'ie':(320,2250,550,1200,2750),
    'ua':(350,750,900,1250,2550),'uo':(350,750,550,900,2500),
    'ue':(350,750,550,1200,2550),'ve':(320,2050,550,1200,2700),
    'an':(850,1350,600,1600,2600),'en':(550,1600,450,1800,2550),
    'in':(320,2300,350,2000,2750),'un':(350,750,450,1600,2500),
    'ang':(850,1200,700,1100,2500),'eng':(550,1150,500,1200,2550),
    'ing':(320,2300,350,2200,2700),'ong':(500,850,450,800,2450),
    'iong':(350,2200,450,800,2600),'ian':(350,2300,600,1600,2750),
    'uan':(350,750,600,1600,2550),'uang':(350,750,700,1100,2500),
}

TONE_M = {'ā':'a1','á':'a2','ǎ':'a3','à':'a4','ē':'e1','é':'e2','ě':'e3','è':'e4',
          'ī':'i1','í':'i2','ǐ':'i3','ì':'i4','ō':'o1','ó':'o2','ǒ':'o3','ò':'o4',
          'ū':'u1','ú':'u2','ǔ':'u3','ù':'u4','ǖ':'v1','ǘ':'v2','ǚ':'v3','ǜ':'v4'}

_LEARNED = {}
def _load():
    global _LEARNED
    if _LEARNED: return
    if os.path.exists(PARAMS_FILE):
        _LEARNED = json.load(open(PARAMS_FILE))

def _parse(pinyin):
    p = pinyin.strip().lower()
    tone = 0
    if p[-1].isdigit(): tone=int(p[-1]); p=p[:-1]
    else:
        for ch in p:
            if ch in 'āēīōūǖ': tone=1; break
            if ch in 'áéíóúǘ': tone=2; break
            if ch in 'ǎěǐǒǔǚ': tone=3; break
            if ch in 'àèìòùǜ': tone=4; break
    r = {'ā':'a','á':'a','ǎ':'a','à':'a','ē':'e','é':'e','ě':'e','è':'e',
         'ī':'i','í':'i','ǐ':'i','ì':'i','ō':'o','ó':'o','ǒ':'o','ò':'o',
         'ū':'u','ú':'u','ǔ':'u','ù':'u','ǖ':'v','ǘ':'v','ǚ':'v','ǜ':'v'}
    clean = ''.join(r.get(c,c) for c in p)
    initials = ['zh','ch','sh','b','p','m','f','d','t','n','l',
                'g','k','h','j','q','x','r','z','c','s','y','w']
    init, final = '', clean
    for i in sorted(initials, key=len, reverse=True):
        if clean.startswith(i): init=i; final=clean[len(i):]; break
    return init, final, tone

def py_to_num(py):
    p=py.strip().lower()
    if p[-1].isdigit(): return p
    r=''; t='0'
    for c in p:
        if c in TONE_M: m=TONE_M[c]; r+=m[0]; t=m[1]
        else: r+=c
    return r+t

BABY_SHIFT = 1.25
BABY_JITTER = 0.01

class BabyVoice:
    """v10 — 10次谐波 + F3共振峰 + 气息感"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        _load()
    
    def _baby_f0(self, tone, n):
        t = np.linspace(0, 1, n)
        ref = None
        for k, v in _LEARNED.items():
            if k.endswith(str(tone)): ref=v; break
        
        base = (ref['f0_mean'] if ref else 220) * BABY_SHIFT
        
        if tone == 1:    f0 = np.full(n, base * 1.08)
        elif tone == 2:  f0 = base*0.72 + base*0.45 * t**0.5
        elif tone == 3:
            f0 = np.zeros(n)
            for i in range(n):
                f0[i] = base*0.85 - base*0.38*(t[i]/0.4) if t[i]<0.4 else base*0.47 + base*0.55*((t[i]-0.4)/0.6)
        elif tone == 4:  f0 = base*1.12 - base*0.62 * t**0.4
        else:            f0 = np.full(n, base*0.8)
        
        j = 1.0 + BABY_JITTER * np.random.randn(n)
        j = np.convolve(j, np.ones(20)/20, mode='same')
        return f0 * j
    
    def _get_formants(self, final, n):
        vf = VOWEL_F.get(final)
        if not vf:
            for k in sorted(VOWEL_F.keys(), key=len, reverse=True):
                if final.startswith(k): vf=VOWEL_F[k]; break
        if not vf: vf=(600,1400,2600)
        
        r = np.linspace(0, 1, n)
        if len(vf) == 3:
            return np.full(n,vf[0])*1.03, np.full(n,vf[1])*1.01, np.full(n,vf[2])
        elif len(vf) == 5:
            f1s,f2s,f1e,f2e,f3 = vf
            return f1s+(f1e-f1s)*r*1.03, f2s+(f2e-f2s)*r*1.01, np.full(n,f3)
        else:
            return np.full(n,600), np.full(n,1400), np.full(n,2600)
    
    def _resonator(self, sig, fc, bw):
        """二阶谐振器 — 模拟声道共振 (带带宽)"""
        sr = self.sr
        y = np.zeros_like(sig)
        if len(sig) < 2: return sig
        y[0] = sig[0]; y[1] = sig[1]
        
        if hasattr(fc, '__len__'):
            for n in range(2, len(sig)):
                r = np.exp(-np.pi*bw/sr); th = 2*np.pi*fc[n]/sr
                y[n] = (1-r)*sig[n] + 2*r*np.cos(th)*y[n-1] - r*r*y[n-2]
        else:
            r = np.exp(-np.pi*bw/sr); th = 2*np.pi*fc/sr
            a1 = 2*r*np.cos(th); a2 = -r*r
            for n in range(2, len(sig)):
                y[n] = (1-r)*sig[n] + a1*y[n-1] + a2*y[n-2]
        return y
    
    def synthesize(self, pinyin: str, duration: float = None) -> np.ndarray:
        init, final, tone = _parse(pinyin)
        sr = self.sr
        
        # 从学习参数取时长
        if duration is None:
            ref = _LEARNED.get(py_to_num(pinyin), {})
            duration = ref.get('dur', 0.75)
        
        init_n = int(0.04 * sr) if init else 0
        vowel_n = int(sr * duration) - init_n
        n = init_n + vowel_n
        
        # ── 丰富谐波声门源 (10次谐波) ──
        f0 = self._baby_f0(tone, vowel_n)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        glot = np.sin(phase) * 1.0
        glot += np.sin(2*phase) * 0.75
        glot += np.sin(3*phase) * 0.50
        glot += np.sin(4*phase) * 0.30
        glot += np.sin(5*phase) * 0.18
        glot += np.sin(6*phase) * 0.10
        glot += np.sin(7*phase) * 0.06
        glot += np.sin(8*phase) * 0.04
        glot += np.sin(9*phase) * 0.02
        glot += np.sin(10*phase) * 0.01
        glot /= 3.0
        
        # ── 共振峰滤波 (带真实带宽) ──
        f1a, f2a, f3a = self._get_formants(final, vowel_n)
        vowel = self._resonator(glot, f1a, 80)
        vowel = self._resonator(vowel, f2a, 120)
        vowel = self._resonator(vowel, f3a, 180)
        
        # (无气息噪声 — 婴儿声带虽弱但不漏气)
        
        # 包络
        atk=int(0.025*sr); rel=int(0.06*sr)
        if vowel_n>atk+rel:
            vowel[:atk]*=np.linspace(0.1,1,atk)
            vowel[-rel:]*=np.linspace(1,0,rel)
        
        out = np.zeros(n, dtype=np.float32)
        out[init_n:] = vowel
        
        # ── 辅音: 极柔处理 (避免爆音) ──
        if init:
            fi = {'m':280,'n':300,'l':360}.get(init, 400)
            cons = np.sin(2*np.pi*fi*np.arange(init_n)/sr) * 0.03
            # 辅音自身渐起渐落
            c_atk = min(int(0.01*sr), init_n//2)
            c_rel = min(int(0.008*sr), init_n//3)
            if c_atk>0: cons[:c_atk] *= np.linspace(0,1,c_atk)
            if c_rel>0: cons[-c_rel:] *= np.linspace(1,0.3,c_rel)
            out[:init_n] = cons
            
            # 辅音→元音柔过渡
            cross=min(int(0.035*sr), init_n, vowel_n)
            if cross>0:
                out[init_n-cross:init_n] *= np.linspace(1,0,cross)
                out[init_n:init_n+cross] *= np.linspace(0.05,1,cross)
        
        peak=np.max(np.abs(out))
        if peak>0: out=out/peak*0.85
        return out.astype(np.float32)
    
    def say(self, pinyin, duration=None):
        return self.synthesize(pinyin, duration)
    
    def say_sentence(self, pinyins, gap=0.18):
        parts=[]
        for item in pinyins:
            py=item[0] if isinstance(item,list) else item
            dur=item[1] if isinstance(item,tuple) else None
            if py: 
                parts.append(self.synthesize(py,dur))
                parts.append(np.zeros(int(gap*self.sr),dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(100,dtype=np.float32)
    
    def to_wav(self, a, p):
        i16=(a*32767).astype(np.int16)
        with wavlib.open(p,'w') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(self.sr);w.writeframes(i16.tobytes())
        return p


if __name__ == "__main__":
    v = BabyVoice()
    print(f"👶 v10 — {len(_LEARNED)} 参数, 10次谐波+F3+共振峰带宽")
    import os as _os
    out = _os.path.join(BASE, 'baby_sounds')
    _os.makedirs(out,exist_ok=True)
    for ch,py in [('画','huà'),('龙','lóng'),('点','diǎn'),('睛','jīng'),('猫','māo')]:
        a=v.say(py); p=v.to_wav(a,f'{out}/v10_{ch}.wav')
        print(f"  {ch} [{py}] {len(a)}样本 → MEDIA:{p}")
    a=v.say_sentence(['huà','lóng','diǎn','jīng']); p=v.to_wav(a,f'{out}/v10_画龙点睛.wav')
    print(f"  成语 → MEDIA:{p}")
