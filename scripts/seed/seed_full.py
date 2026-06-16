#!/usr/bin/env python3
"""龙珠全量播种 - 逐字模式（已验证）+ 断点续传, 目标 90362 剩余字"""
import sys, os, json, time, argparse, re
import requests, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zichang import HanziAnchorField
from energy_landscape import EnergyLandscape
from loongpearl_learner import DragonBallLearner

BASE = os.path.dirname(os.path.abspath(__file__))
ZICHANG = os.path.join(BASE, "zichang_94117_1024d.pt")
LANDSCAPE = os.path.join(BASE, "energy_landscape_1024d.pt")
CKPT = os.path.join(BASE, "seed_full_checkpoint.json")
MODEL = "deepseek-r1:7b"
API = "http://localhost:11434/api/generate"

# v7 验证过的 prompt（单字）
PROMPT = 'List 3 Chinese characters semantically related to "{ch}". Output ONLY: [{{"hanzi":"char","relation":"type"}}]'

class FullSeeder:
    def __init__(self, zc, ls, lr):
        self.zc, self.ls, self.lr = zc, ls, lr
        self.seeded, self.pairs = set(), set()
        self.imp, self.fail = 0, 0

    def _call(self, ch):
        try:
            r = requests.post(API, json={
                "model": MODEL,
                "prompt": PROMPT.format(ch=ch),
                "stream": False,
                "options": {"temperature": 0.6, "num_predict": 3000},
            }, timeout=120)
            if r.status_code != 200: return []
            return self._parse(r.json().get("response", ""))
        except: return []

    def _parse(self, text):
        if not text: return []
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        for m in re.finditer(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL):
            r = self._norm(m.group(1))
            if r: return r
        s, e = text.find('['), text.rfind(']')+1
        if s >= 0 and e > s:
            r = self._norm(text[s:e])
            if r: return r
        return []

    def _norm(self, js):
        try: data = json.loads(js)
        except: return []
        if not isinstance(data, list): return []
        res = []
        for item in data:
            if isinstance(item, str):
                if item in self.zc._char_to_idx:
                    res.append({"hanzi": item, "relation": "?"})
            elif isinstance(item, dict):
                h = item.get("hanzi", "")
                if isinstance(h, str) and h.strip() in self.zc._char_to_idx:
                    res.append({"hanzi": h.strip(), "relation": item.get("relation", "?")})
        return res

    def _implant(self, ch, items, strength):
        idx = self.zc._char_to_idx[ch]; imp = 0
        for it in items:
            tgt = it.get("hanzi", "")
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
            self.seeded = set(ck.get("chars", []))
            self.pairs = set(tuple(p) for p in ck.get("pairs", []))
            self.imp = ck.get("implanted", 0); self.fail = ck.get("failed", 0)
            print(f"断点: {len(self.seeded)}字/{len(self.pairs)}对/{self.imp}植入")

        pending = [c for c in chars if c in self.zc._char_to_idx and c not in self.seeded]
        if not pending: print("全播完!"); return
        total = len(pending)
        print(f"\n全量播种: {total}字 | 预估~6字/min | eta~{total/6/60:.0f}h")
        last = len(self.seeded)

        for i, ch in enumerate(pending):
            items = self._call(ch)
            if items:
                if not dry: self.imp += self._implant(ch, items, strength)
                self.seeded.add(ch)
            else:
                self.seeded.add(ch); self.fail += 1

            n = len(self.seeded)
            if n - last >= 100:
                el = time.time()-t0; rate = (n-last+100)/max(el,1)*60
                eta = (total - (n-last+100)) / max(rate/100, 0.01)
                print(f"  [{n}/{total+len(self.seeded)-n}] "
                      f"rate={n/max(el,1)*60:.1f}/min imp={self.imp} "
                      f"eta={eta/60:.0f}h{eta%60:.0f}m")
                if not dry:
                    self._save()
                    torch.save(self.ls.state_dict(), LANDSCAPE.replace(".pt", "_auto.pt"))
                last = n

        if not dry:
            self._save(); self.ls.save(LANDSCAPE)
            auto = LANDSCAPE.replace(".pt", "_auto.pt")
            if os.path.exists(auto): os.remove(auto)
        e = time.time()-t0; n = len(self.seeded)
        print(f"\n播:{n} 植:{self.imp} 对:{len(self.pairs)} 失:{self.fail} {e/3600:.1f}h")

    def _save(self):
        json.dump({"chars": list(self.seeded), "pairs": [list(p) for p in self.pairs],
            "implanted": self.imp, "failed": self.fail}, open(CKPT, "w"),
            ensure_ascii=False, indent=2)

def get_remaining():
    data = torch.load(ZICHANG, map_location="cpu", weights_only=True)
    allc = data["hanzi_list"]
    seeded = set()
    if os.path.exists(CKPT): seeded.update(json.load(open(CKPT)).get("chars", []))
    else:
        for f in ["seed_v7_result.json","seed_v7b_result.json","seed_v6_result.json","seed_final.json"]:
            p = os.path.join(BASE, f)
            if os.path.exists(p) and os.path.getsize(p) > 100:
                d = json.load(open(p))
                if "chars" in d: seeded.update(d["chars"])
    return [c for c in allc if c not in seeded]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", "-n", type=int, default=0, help="播种字数(0=全量)")
    ap.add_argument("--strength", "-s", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset and os.path.exists(CKPT): os.remove(CKPT)
    zc = HanziAnchorField.load(ZICHANG)
    ls = EnergyLandscape.load(LANDSCAPE); ls.eval()
    lr = DragonBallLearner(landscape=ls, anchor_field=zc, hebbian_lr=0.001)
    chars = get_remaining()
    print(f"全量:{94117} 已播:{94117-len(chars)} 剩余:{len(chars)}")
    if args.chars > 0: chars = chars[:args.chars]
    s = FullSeeder(zc, ls, lr)
    s.run(chars, strength=args.strength, dry=args.dry_run)

if __name__ == "__main__": main()
