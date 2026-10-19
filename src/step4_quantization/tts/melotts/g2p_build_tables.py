# -*- coding: utf-8 -*-
"""Build + evaluate static context tables for the MeloTTS-ZH g2p, trained on ~466k OPUS sentences + FLEURS train/dev.
Only chars with a real choice (>=2 outputs, minority >= 0.3%) get context overrides; everything else is one default entry."""
import sys, pickle, collections, json
sys.stdout.reconfigure(encoding="utf-8")
big = pickle.load(open("work/g2p_big.pkl", "rb")); F = pickle.load(open("work/g2p_dataset.pkl", "rb"))
hold_o = big[-20000:]; train = big[:-20000] + F["train"]; tests = {"OPUS held-out": hold_o, "FLEURS test": F["test"]}
cnt = collections.defaultdict(collections.Counter)
for s in train:
    for ch, ph, tn in s: cnt[ch][(ph, tn)] += 1
default = {c: v.most_common(1)[0][0] for c, v in cnt.items()}
poly = {c for c, v in cnt.items() if len(v) > 1 and (sum(v.values()) - v.most_common(1)[0][1]) >= 0.003 * sum(v.values())}
print("chars", len(cnt), "| chars with a real choice:", len(poly), flush=True)
bt = lambda c: (default[c][1][-1] if c in default and default[c][1] else 0)
def feats(s, i):
    p = s[i-1][0] if i > 0 else "^"; n = s[i+1][0] if i + 1 < len(s) else "$"; c = s[i][0]
    pp = s[i-2][0] if i > 1 else "^"; nn = s[i+2][0] if i + 2 < len(s) else "$"
    return {"tri": (p, c, n), "bil": (p, c), "bir": (c, n), "tone3": (bt(p) if i > 0 else -1, c, bt(n) if i + 1 < len(s) else -1),
            "quad_l": (pp, p, c), "quad_r": (c, n, nn), "tone_r": (c, bt(n) if i + 1 < len(s) else -1)}
ORDER = ["tri", "quad_l", "quad_r", "tone3", "tone_r", "bil", "bir"]
T = {k: collections.defaultdict(collections.Counter) for k in ORDER}
for s in train:
    for i, (ch, ph, tn) in enumerate(s):
        if ch in poly:
            f = feats(s, i)
            for k in ORDER: T[k][f[k]][(ph, tn)] += 1
CENTER = {"tri": 1, "bil": 1, "bir": 0, "tone3": 1, "quad_l": 2, "quad_r": 0, "tone_r": 0}
def compact(minsup, purity):
    keep = {}
    for k in ORDER:
        d = {}
        for key, cn in T[k].items():
            tot = sum(cn.values()); out, c = cn.most_common(1)[0]
            if tot >= minsup and c / tot >= purity and out != default[key[CENTER[k]]]: d[key] = out     # only real deviations from the default reading
        keep[k] = d
    return keep
def predict(s, i, tabs):
    ch = s[i][0]
    if ch in poly:
        f = feats(s, i)
        for k in ORDER:
            if f[k] in tabs[k]: return tabs[k][f[k]]
    return default.get(ch)
for minsup, purity in ((3, 0.9), (5, 0.9), (20, 0.9), (100, 0.9)):
    tabs = compact(minsup, purity); n = sum(len(v) for v in tabs.values())
    n2 = n; res = []
    for name, S in tests.items():
        ok = tot = 0; se = 0
        for s in S:
            good = True
            for i, (ch, ph, tn) in enumerate(s):
                tot += 1; g = predict(s, i, tabs) == (ph, tn); ok += g; good &= g
            se += good
        res.append(f"{name}: char {100*ok/tot:.2f}% sent {100*se/len(S):.0f}%")
    print(f"minsup {minsup:3d} purity {purity}: override entries {n2:7d} | " + " | ".join(res), flush=True)
pickle.dump({"default": default, "poly": poly}, open("work/g2p_tables_base.pkl", "wb"))
