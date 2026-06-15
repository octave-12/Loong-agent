#!/usr/bin/env python3
"""龙珠批量播种器 v2 - 已验证批量 JSON 可行, 5x 加速"""
import sys, os, json, time, argparse, re
import requests, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape
from loongpearl_learner import DragonBallLearner

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "energy_landscape_1024d.pt")
CKPT = os.path.join(BASE, "batch_seed_checkpoint.json")
MODEL = "deepseek-r1:7b"
API = "http://localhost:11434/api/generate"

class BatchSeederV2:
    def __init__(self, zc, ls, lr, bs=5):
        self.zc, self.ls, self.lr, self.bs = zc, ls, lr, bs
        self.seeded, self.pairs = set(), set()
        self.imp, self.fail_c, self.fail_b = 0, 0, 0

    def _call(self, chars):
        joined = ",".join(chars)
        prompt = (
            f'For [{joined}], give 3 related Chinese characters each.\n'
            f'Output ONLY JSON: {{"字":[{{"hanzi":"字","relation":"rel"}}]}}\n'
            f'Hanzi values must be single Chinese chars.'
        )
        try:
            r = requests.post(API, json={"model":MODEL,"prompt":prompt,"stream":False,
                "options":{"temperature":0.6,"num_predict":4000}}, timeout=180)
            if r.status_code != 200: return {}
            return self._parse(r.json().get("response",""))
        except Exception as e:
            if self.fail_b < 3: print(f"  [err] batch req: {e}")
            return {}

    def _parse(self, text):
        if not text: return {}
        # 去 think 块
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # 找 JSON 对象
        s, e = text.find('{'), text.rfind('}')+1
        if s < 0 or e <= s: return {}
        try: data = json.loads(text[s:e])
        except Exception:
            if self.fail_b < 3: print(f"  [parse] JSON fail: {text[s:e][:200]}")
            return {}
        if not isinstance(data, dict): return {}
        # 规范化
        result = {}
        for ch, items in data.items():
            if ch not in self.zc._char_to_idx: continue
            if not isinstance(items, list): continue
            norm = []
            for it in items:
                if isinstance(it, str):
                    if it in self.zc._char_to_idx:
                        norm.append({"hanzi":it,"relation":"?"})
                elif isinstance(it, dict):
                    h = it.get("hanzi","")
                    if isinstance(h,str):
                        # 可能 hanzi 是多字词，拆成单字
                        for c in h.strip():
                            if c in self.zc._char_to_idx:
                                norm.append({"hanzi":c,"relation":it.get("relation","?")})
            if norm: result[ch] = norm
        return result

    def _implant(self, ch, items, strength):
        idx = self.zc._char_to_idx[ch]; imp = 0
        for it in items:
            tgt = it.get("hanzi","")
            if not tgt or tgt not in self.zc._char_to_idx or tgt == ch: continue
            pair = tuple(sorted([ch, tgt]))
            if pair in self.pairs: continue
            try:
                r = self.lr.hebbian.update(self.zc.anchors[idx],
                    self.zc.anchors[self.zc._char_to_idx[tgt]], feedback=strength)
                if r.get("status") != "skipped": imp += 1; self.pairs.add(pair)
            except: pass
        return imp

    def run(self, chars, strength=0.5, dry=False):
        t0 = time.time()
        if not dry and os.path.exists(CKPT):
            ck = json.load(open(CKPT))
            self.seeded = set(ck.get("chars",[]))
            self.pairs = set(tuple(p) for p in ck.get("pairs",[]))
            self.imp = ck.get("implanted",0); self.fail_c = ck.get("failed",0)
            print(f"断点: {len(self.seeded)}字/{len(self.pairs)}对/{self.imp}植入")

        pending = [c for c in chars if c in self.zc._char_to_idx and c not in self.seeded]
        if not pending: print("全播完!"); return
        total = len(pending)
        est = self.bs * 4  # 保守估计
        print(f"\n批量播种: {total}字 batch={self.bs} est={est:.0f}字/min eta={total/est/60:.0f}h")
        if dry: print("DRY RUN")
        last = len(self.seeded)

        for i in range(0, total, self.bs):
            batch = pending[i:i+self.bs]
            data = self._call(batch)
            if not data:
                self.fail_b += 1
                for ch in batch: self.seeded.add(ch); self.fail_c += 1
            else:
                for ch in batch:
                    if ch in data and data[ch]:
                        if not dry: self.imp += self._implant(ch, data[ch], strength)
                        self.seeded.add(ch)
                    else:
                        self.seeded.add(ch); self.fail_c += 1

            n = len(self.seeded)
            if n - last >= 200:
                el = time.time()-t0; rate = (n-last+200)/max(el,1)*60
                print(f"  [{n}] rate={n/max(el,1)*60:.1f}/min imp={self.imp} fail={self.fail_c} "
                      f"eta={max(total-n,0)/max(rate/200,0.01)/60:.0f}h")
                if not dry:
                    self._save()
                    torch.save(self.ls.state_dict(), LANDSCAPE.replace(".pt","_auto.pt"))
                last = n

        if not dry:
            self._save(); self.ls.save(LANDSCAPE)
            auto = LANDSCAPE.replace(".pt","_auto.pt")
            if os.path.exists(auto): os.remove(auto)
        e = time.time()-t0; n = len(self.seeded)
        print(f"\n播:{n} 植:{self.imp} 对:{len(self.pairs)} 失字:{self.fail_c} 失批:{self.fail_b} {e/3600:.1f}h")

    def _save(self):
        json.dump({"chars":list(self.seeded),"pairs":[list(p) for p in self.pairs],
            "implanted":self.imp,"failed":self.fail_c}, open(CKPT,"w"), ensure_ascii=False, indent=2)

def get_remaining():
    data = torch.load(ZICHANG, map_location="cpu", weights_only=True)
    allc = data["hanzi_list"]
    seeded = set()
    if os.path.exists(CKPT): seeded.update(json.load(open(CKPT)).get("chars",[]))
    else:
        for f in ["seed_v7_result.json","seed_v7b_result.json","seed_v6_result.json","seed_final.json"]:
            p = os.path.join(BASE,f)
            if os.path.exists(p) and os.path.getsize(p)>100:
                d = json.load(open(p))
                if "chars" in d: seeded.update(d["chars"])
    # 优先 CJK 统一汉字区 (U+4E00-U+9FFF) — 模型认识
    common = [c for c in allc if 0x4E00 <= ord(c) <= 0x9FFF and c not in seeded]
    # 生僻扩展区 — 模型大概率不认识，先跳过
    rare = [c for c in allc if c not in seeded and not (0x4E00 <= ord(c) <= 0x9FFF)]
    print(f"CJK统一区剩余:{len(common)} | 扩展区剩余:{len(rare)}")
    return common + rare  # 常用在前，生僻在后

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars","-n",type=int,default=0)
    ap.add_argument("--batch-size","-b",type=int,default=5)
    ap.add_argument("--strength","-s",type=float,default=0.5)
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--reset",action="store_true")
    args = ap.parse_args()
    if args.reset and os.path.exists(CKPT): os.remove(CKPT)
    zc = HanziAnchorField.load(ZICHANG)
    ls = EnergyLandscape.load(LANDSCAPE); ls.eval()
    lr = DragonBallLearner(landscape=ls, anchor_field=zc, hebbian_lr=0.001)
    chars = get_remaining()
    print(f"全量:94117 已播:{94117-len(chars)} 剩余:{len(chars)}")
    if args.chars>0: chars = chars[:args.chars]
    s = BatchSeederV2(zc,ls,lr,bs=args.batch_size)
    s.run(chars, strength=args.strength, dry=args.dry_run)

if __name__ == "__main__": main()
