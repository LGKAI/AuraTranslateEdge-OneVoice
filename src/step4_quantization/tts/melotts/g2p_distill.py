# -*- coding: utf-8 -*-
"""Distil MeloTTS-ZH text front (jieba + pypinyin + tone sandhi + opencpop map) into static lookup tables.
Per-character output = (phone tuple, tone). Measure how well char-only / char+context tables reproduce it on held-out FLEURS sentences."""
import sys, re, json, collections, pickle
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
from huggingface_hub import hf_hub_download
from melo.text.cleaner import clean_text

def load(split):
    p = hf_hub_download("google/fleurs", f"data/cmn_hans_cn/{split}.tsv", repo_type="dataset")
    S = [l.rstrip("\n").split("\t")[2] for l in open(p, encoding="utf-8") if l.count("\t") > 3]
    return list(dict.fromkeys(S))          # FLEURS repeats each sentence for several speakers

def per_char(text):
    norm, phones, tones, w2p = clean_text(text, "ZH")
    assert len(w2p) == len(norm) + 2, (len(w2p), len(norm))
    out, off = [], 1                       # phones/tones begin with a leading '_' (w2p[0]==1)
    for ch, w in zip(norm, w2p[1:-1]):
        out.append((ch, tuple(phones[off:off + w]), tuple(tones[off:off + w]))); off += w
    return out

if __name__ == "__main__":
    tr, de, te = load("train"), load("dev"), load("test"); print("unique sentences", len(tr), len(de), len(te), flush=True)
    D = {}
    for name, S in (("train", tr + de), ("test", te)):
        rows = []
        for s in S:
            try: rows.append(per_char(s))
            except Exception as e: pass
        D[name] = rows; print(name, "sentences ok", len(rows), "chars", sum(len(r) for r in rows), flush=True)
    pickle.dump(D, open("work/g2p_dataset.pkl", "wb"))
