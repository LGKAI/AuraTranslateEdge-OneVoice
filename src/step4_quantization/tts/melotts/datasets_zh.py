# -*- coding: utf-8 -*-
"""Calibration inputs for MeloTTS-ZH: short Chinese clauses from FLEURS (cmn_hans_cn) -> same input dict as the graph."""
import re, sys, numpy as np
sys.path.insert(0, "MeloTTS")
from huggingface_hub import hf_hub_download
from melo import utils


def clauses(maxc=20, n=60):
    out = []
    for sp in ("train", "dev"):
        p = hf_hub_download("google/fleurs", f"data/cmn_hans_cn/{sp}.tsv", repo_type="dataset")
        for l in open(p, encoding="utf-8"):
            f = l.rstrip("\n").split("\t")
            if len(f) > 3:
                for c in re.split(r"[，。！？；、,.!?;]", f[2]):
                    c = c.strip()
                    if 6 <= len(c) <= maxc and re.fullmatch(r"[一-鿿]+", c): out.append(c)
    seen, res = set(), []
    for c in out:
        if c not in seen: seen.add(c); res.append(c)
    return res[:n]


def calib_inputs(tts, N, n=60):
    D = {k: [] for k in ["x", "tone", "lang", "bert", "x_len"]}
    for t in clauses(n=n):
        bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(t, "ZH", tts.hps, "cpu", tts.symbol_to_id); m = len(ph)
        if m > N: continue
        pad = lambda v: np.concatenate([v.numpy(), np.zeros(N - m)]).astype(np.int32)[None]
        bp = np.zeros((1, 1024, N), np.float32); bp[0, :, :m] = bert.numpy()
        for k, v in zip(D, [pad(ph), pad(tone), pad(lang), bp, np.array([m], np.int32)]): D[k].append(v)
    print("calibration samples:", len(D["x"]))
    return D
