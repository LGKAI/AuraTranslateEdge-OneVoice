# -*- coding: utf-8 -*-
"""Robustness of the vi / not-vi language identification: full clips, 3 s and 1.5 s crops, with babble noise at 5 dB.
  python tools/lid_robust.py --models models.dev.json --eval <eval dir> --n 15"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry
from s2s_edge.conversation import Engine, LanguageID
from s2s_edge.audio import load_audio, normalize_level
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--n", type=int, default=15); ap.add_argument("--out", default="logs/lid_robust.json"); ap.add_argument("--bias", type=float, default=0.0); a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); e = Engine(Registry(a.models, "cpu")); lid = LanguageID(e, bias=a.bias); av = e.asr("vi")
man = {l: json.load(open(os.path.join(a.eval, f"manifest_{l}.json"), encoding="utf-8")) for l in ("vi", "en", "zh", "ko")}
def mix(x, nz, snr): nz = np.resize(nz, len(x)); return (x + nz * np.sqrt(((x ** 2).mean() + 1e-12) / (((nz ** 2).mean() + 1e-12) * 10 ** (snr / 10)))).astype(np.float32)
res = {}
for name, crop, snr in (("full", None, None), ("3s", 3.0, None), ("1.5s", 1.5, None), ("3s+babble5dB", 3.0, 5.0)):
    ok = {l: [] for l in man}; pr = {l: [] for l in man}
    for l in man:
        for i, it in enumerate(man[l][-a.n:]):                                         # clips not used to fit the model (it was fitted on the first 30)
            w = load_audio(os.path.join(a.eval, it["path"])); s0 = int(1.0 * 16000) if crop else 0
            if crop: w = w[s0: s0 + int(crop * 16000)]
            if snr: w = mix(w, load_audio(os.path.join(a.eval, man["zh" if l != "zh" else "ko"][i]["path"]))[: len(w)] + load_audio(os.path.join(a.eval, man["en" if l != "en" else "ko"][i]["path"]))[: len(w)], snr)
            av(normalize_level(w)); p = lid.vi_prob(av.stats) if av.stats["nwords"] >= 1 else 0.0; pr[l].append(p); ok[l].append((p >= 0.5) == (l == "vi"))
    res[name] = {l: round(float(np.mean(v)), 3) for l, v in ok.items()}; print(name, "accuracy per language", res[name], "| vi prob mean %.2f, others max %.2f" % (np.mean(pr["vi"]), max(max(pr[l]) for l in ("en", "zh", "ko"))), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
