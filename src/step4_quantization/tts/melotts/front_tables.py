# -*- coding: utf-8 -*-
"""Build every static table the NPU text front needs (MeloTTS-ZH): token space (chars+punct+specials), default readings, override tables.
Token index space is shared by the g2p context keys AND the BERT embedding rows (so BERT embedding = one-hot @ table, no int arithmetic)."""
import sys, re, pickle, collections, numpy as np
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer
from melo.api import TTS
from melo.text.cleaner import clean_text
from melo.text.symbols import punctuation as _PUN

def build(minsup=3, purity=0.9):
    tts = TTS(language="ZH", device="cpu"); sid = tts.symbol_to_id
    tk = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large"); V = tk.get_vocab()
    big = pickle.load(open("work/g2p_big.pkl", "rb")); F = pickle.load(open("work/g2p_dataset.pkl", "rb"))
    train = big[:-20000] + F["train"]
    PUN = list(_PUN)                                  # ['!', '?', '…', ',', '.', "'", '-']
    hz = [c for c in V if len(c) == 1 and "一" <= c <= "龥"]
    toks = ["[CLS]", "[SEP]", "[PAD]", "^", "$"] + PUN + hz     # token index space
    idx = {t: i for i, t in enumerate(toks)}
    cnt = collections.defaultdict(collections.Counter)
    for s in train:
        for ch, ph, tn in s: cnt[ch][(ph, tn)] += 1
    def default(ch):
        if ch in cnt: return cnt[ch].most_common(1)[0][0]
        try: n, p, t, w = clean_text(ch, "ZH"); return (tuple(p[1:-1]), tuple(t[1:-1]))
        except Exception: return None
    D = {}
    for ch in toks[5:]:
        d = default(ch)
        if d and 1 <= len(d[0]) <= 2: D[ch] = d
    hz_ok = [c for c in hz if c in D]; toks2 = toks[:5] + PUN + hz_ok; idx = {t: i for i, t in enumerate(toks2)}
    print("tokens", len(toks2), "| hanzi with readings", len(hz_ok), "| missing", len(hz) - len(hz_ok), flush=True)
    bt = lambda c: (D[c][1][-1] if c in D and D[c][1] else 0)
    poly = {c for c, v in cnt.items() if c in D and len(v) > 1 and ((sum(v.values()) - v.most_common(1)[0][1]) >= 0.003 * sum(v.values()) or (c in D and D[c][1] and D[c][1][-1] in (3, 1) and (sum(v.values()) - v.most_common(1)[0][1]) >= 2))}
    ORDER = ["tri", "quad_l", "quad_r", "tone3", "tone_r", "bil", "bir"]
    CENTER = {"tri": 1, "bil": 1, "bir": 0, "tone3": 1, "quad_l": 2, "quad_r": 0, "tone_r": 0}
    def feats(s, i):
        g = lambda j, dflt: s[j][0] if 0 <= j < len(s) else dflt
        p, c, n, pp, nn = g(i-1, "^"), s[i][0], g(i+1, "$"), g(i-2, "^"), g(i+2, "$")
        tp = bt(p) if i > 0 else -1; tn_ = bt(n) if i + 1 < len(s) else -1
        return {"tri": (p, c, n), "bil": (p, c), "bir": (c, n), "tone3": (("T", tp), c, ("T", tn_)), "quad_l": (pp, p, c), "quad_r": (c, n, nn), "tone_r": (c, ("T", tn_))}
    T = {k: collections.defaultdict(collections.Counter) for k in ORDER}
    for s in train:
        for i, (ch, ph, tn) in enumerate(s):
            if ch in poly:
                f = feats(s, i)
                for k in ORDER: T[k][f[k]][(ph, tn)] += 1
    over = {k: {} for k in ORDER}
    for k in ORDER:
        for key, cn in T[k].items():
            tot = sum(cn.values()); out, c = cn.most_common(1)[0]
            cen = key[CENTER[k]]
            if tot >= minsup and c / tot >= purity and out != D.get(cen) and out[0] and len(out[0]) <= 2 and all(isinstance(x, tuple) or x in idx for x in key): over[k][key] = out
    print("override entries:", {k: len(v) for k, v in over.items()}, flush=True)
    # phone id / tone for the outputs
    ph_id = lambda p: sid[p]
    pickle.dump({"toks": toks2, "idx": idx, "D": D, "over": over, "order": ORDER, "center": CENTER, "sid": sid, "pun": PUN, "poly": poly, "pad_sym": sid["_"]}, open("work/front_tables.pkl", "wb"))
    return toks2

if __name__ == "__main__": build()
