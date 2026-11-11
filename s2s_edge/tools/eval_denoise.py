# -*- coding: utf-8 -*-
"""Does the GTCRN front end help or hurt ASR? Source-side word error rate with and without enhancement on clean speech and on speech mixed with
synthetic noise at a given SNR (babble = three other clips, factory = 100/200 Hz hum + broadband machinery bursts).
  python tools/eval_denoise.py --models models.dev.json --eval ../../s2s_npu/eval --n 10 --snr 5 --out logs/eval_denoise.json"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jiwer
from s2s_edge.registry import Registry, ASSETS
from s2s_edge.audio import load_audio, normalize_level
from s2s_edge.denoise import Gtcrn

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--n", type=int, default=10); ap.add_argument("--snr", type=float, default=5.0)
ap.add_argument("--langs", default="vi,en"); ap.add_argument("--out", default="logs/eval_denoise.json"); a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
reg = Registry(a.models, "cpu"); den = Gtcrn(os.path.join(ASSETS, "gtcrn_dns3.onnx")); rng = np.random.default_rng(0)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()
man = {l: json.load(open(os.path.join(a.eval, f"manifest_{l}.json"), encoding="utf-8")) for l in ("vi", "en", "zh", "ko")}


def mix(x, noise, snr):
    noise = np.resize(noise, len(x)); px = (x ** 2).mean() + 1e-12; pn = (noise ** 2).mean() + 1e-12
    return (x + noise * np.sqrt(px / (pn * 10 ** (snr / 10)))).astype(np.float32)


def factory(n, sr=16000):
    t = np.arange(n) / sr; hum = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t); bur = rng.standard_normal(n) * (np.sin(2 * np.pi * 3 * t) > 0.6)
    return (hum + 1.2 * bur + 0.3 * rng.standard_normal(n)).astype(np.float32)


res = {}
for lang in a.langs.split(","):
    asr = reg.asr(lang); rows = {k: [] for k in ("clean", "clean+den", "babble", "babble+den", "factory", "factory+den")}
    for i, it in enumerate(man[lang][: a.n]):
        w = load_audio(os.path.join(a.eval, it["path"]))[: 12 * 16000]; ref = norm(it["refs"][lang])
        others = [load_audio(os.path.join(a.eval, m["path"]))[: 12 * 16000] for l in ("zh", "ko", "en" if lang == "vi" else "vi") for m in [man[l][i]]]
        bab = sum(np.resize(o, len(w)) for o in others)
        for tag, x in (("clean", w), ("babble", mix(w, bab, a.snr)), ("factory", mix(w, factory(len(w)), a.snr))):
            for d in (False, True):
                y = den(x) if d else x; hyp = norm(asr(normalize_level(y))); ref_w = norm(" ".join(ref.split()[: len(hyp.split()) + 40]))
                rows[tag + ("+den" if d else "")].append(jiwer.wer(norm(it["refs"][lang]), hyp))
        print(lang, i, {k: round(v[-1], 2) for k, v in rows.items()}, flush=True)
    res[lang] = {k: round(float(np.mean(v)), 3) for k, v in rows.items()}; print(lang, "mean WER", res[lang], flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(dict(snr=a.snr, n=a.n, wer=res), open(a.out, "w"), indent=1)
