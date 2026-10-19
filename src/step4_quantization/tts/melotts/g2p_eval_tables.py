import sys, pickle, collections, re
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
from melo.text.cleaner import clean_text
D = pickle.load(open("work/g2p_dataset.pkl", "rb")); tr, te = D["train"], D["test"]
out = lambda r: (r[1], r[2])
# 1) default per char: majority in train; chars unseen in train -> context-free MeloTTS g2p of the single char (this is what a full-vocabulary table would hold)
cnt = collections.defaultdict(collections.Counter)
for s in tr:
    for ch, ph, tn in s: cnt[ch][(ph, tn)] += 1
def default(ch):
    if ch in cnt: return cnt[ch].most_common(1)[0][0]
    try:
        n, p, t, w = clean_text(ch, "ZH"); return (tuple(p[1:-1]), tuple(t[1:-1]))
    except Exception: return None
def ctx_table(keyf, minc=1):
    T = collections.defaultdict(collections.Counter)
    for s in tr:
        for i, (ch, ph, tn) in enumerate(s): T[keyf(s, i)][(ph, tn)] += 1
    return T
pad = lambda s, i: (s[i-1][0] if i > 0 else "^", s[i][0], s[i+1][0] if i + 1 < len(s) else "$")
tri = ctx_table(lambda s, i: pad(s, i)); bil = ctx_table(lambda s, i: pad(s, i)[:2]); bir = ctx_table(lambda s, i: pad(s, i)[1:])
def predict(s, i, mode):
    p, c, n = pad(s, i)
    if mode >= 3 and (p, c, n) in tri: return tri[(p, c, n)].most_common(1)[0][0]
    if mode >= 2:
        for T, k in ((bil, (p, c)), (bir, (c, n))):
            if k in T and len(T[k]) == 1: return T[k].most_common(1)[0][0]
    return default(c)
for mode, name in ((1, "char only"), (2, "char + bigram context"), (3, "char + bigram + trigram")):
    ok = tot = 0; wrong = collections.Counter()
    for s in te:
        for i, (ch, ph, tn) in enumerate(s):
            tot += 1; pr = predict(s, i, mode)
            if pr == (ph, tn): ok += 1
            else: wrong[ch] += 1
    print(f"{name:26s} per-char exact {100*ok/tot:.2f}%  ({tot-ok} wrong of {tot}) | sentences fully exact: ", end="")
    se = sum(1 for s in te if all(predict(s, i, mode) == (ph, tn) for i, (ch, ph, tn) in enumerate(s))); print(f"{se}/{len(te)}")
    if mode == 3: print("most frequent wrong chars:", wrong.most_common(14))
poly = {c: len(v) for c, v in cnt.items() if len(v) > 1}; print("chars with >1 output in train:", len(poly), "of", len(cnt), "| e.g.", sorted(poly.items(), key=lambda x: -x[1])[:10])
