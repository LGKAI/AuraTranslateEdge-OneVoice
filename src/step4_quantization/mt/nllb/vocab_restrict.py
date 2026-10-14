# -*- coding: utf-8 -*-
import json, sys
from transformers import AutoTokenizer
VI="àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
def cls(t):
    s=t.replace("▁","")
    if not s: return "empty"
    if all(ord(c)<128 for c in s): return "ascii"
    if all(ord(c)<128 or c.lower() in VI or c in "’‘“”–—…" for c in s): return "latin_vi"
    if all(0xAC00<=ord(c)<=0xD7A3 or 0x1100<=ord(c)<=0x11FF or 0x3130<=ord(c)<=0x318F or ord(c)<128 for c in s): return "hangul"
    if all(0x4E00<=ord(c)<=0x9FFF or 0x3000<=ord(c)<=0x303F or 0xFF00<=ord(c)<=0xFFEF or ord(c)<128 for c in s): return "cjk"
    return "other"
def build(max_short=3, all_scripts=False):
    tok=AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",use_fast=False); V=tok.get_vocab(); inv={i:t for t,i in V.items()}
    seen=set(json.load(open("work/keep_ids.json"))["keep"]); keep=set(seen)|set(tok.all_special_ids)
    for i,t in inv.items():
        s=t.replace("▁","")
        if cls(t) in ("ascii","latin_vi","hangul","cjk") and (all_scripts or len(s)<=max_short): keep.add(i)
    return tok,V,inv,sorted(keep)
class Restricted:
    """Segment with the original SentencePiece, then re-split any piece outside `keep` into the longest kept sub-pieces (down to characters)."""
    def __init__(self, tok, V, keep):
        self.tok, self.V, self.keep = tok, V, set(keep); self.pieces = {t for t, i in V.items() if i in self.keep}
    def _split(self, p):
        out, i = [], 0
        while i < len(p):
            for j in range(len(p), i, -1):
                if p[i:j] in self.pieces: out.append(p[i:j]); i = j; break
            else: out.append("<unk>"); i += 1
        return out
    def ids(self, text, src_lang):
        pieces = []
        for p in self.tok.tokenize(text):
            pieces += [p] if p in self.pieces else self._split(p)
        return [self.V[src_lang]] + [self.V.get(p, 3) for p in pieces] + [2]
def restrict(tok, inv, keep): return Restricted(tok, tok.get_vocab(), keep)
if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tok,V,inv,keep=build(); print("kept",len(keep))
    full=AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",use_fast=False)
    test=json.load(open("work/fleurs_test.json",encoding="utf-8")); K=set(keep)
    R=restrict(tok,inv,keep)
    for nl,ss in test.items():
        full.src_lang=nl; a=b=miss=0; unk=0
        for s in ss:
            x=R.ids(s,nl); y=full(s)["input_ids"]; a+=len(x); b+=len(y); miss+=sum(1 for i in x if i not in K); unk+=x.count(3)
        print(f"{nl}: tokens full {b} -> restricted {a} ({100*(a/b-1):+.1f}%) | outside keep: {miss} | unk: {unk}")
    json.dump({"keep":keep},open("work/keep2.json","w"))
