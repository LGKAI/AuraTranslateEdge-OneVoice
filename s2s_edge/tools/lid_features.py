# -*- coding: utf-8 -*-
"""Collect per-clip language-identification features on the eval sets (CPU reference).
 vi-likelihood from the Vietnamese CTC model (confidence + share of decoded words that are known Vietnamese syllables) and the script of SenseVoice(auto) text.
 python tools/lid_features.py --models models.dev.json --eval <dir with manifest_*.json> --n 40 --out logs/lid_features.json"""
import argparse, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry, ASSETS
from s2s_edge.audio import load_audio
from s2s_edge.asr import _fbank, LID

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default="logs/lid_features.json")
ap.add_argument("--syl", default=r"C:\Users\Admin\Downloads\piper\g2p\g2p_tables.pkl"); a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
syl = set(pickle.load(open(a.syl, "rb"))[6].keys()); reg = Registry(a.models, "cpu"); zg = reg.graph("asr_vi"); asr_vi = reg.asr("vi"); sv = reg.asr("en")
rows = []
for lang in ("vi", "en", "zh", "ko"):
    for it in json.load(open(os.path.join(a.eval, f"manifest_{lang}.json"), encoding="utf-8"))[: a.n]:
        w = load_audio(os.path.join(a.eval, it["path"]))[: 14 * 16000]; f = _fbank(w); n = min(len(f), 1500); x = np.zeros((1, 1500, 80), np.float32); x[0, :n] = f[:n]
        lg, el = zg(x=x, x_lens=np.array([n], np.int64)); lg = np.asarray(lg).reshape(-1, 2000); real = int(np.asarray(el).reshape(-1)[0]); lg = lg[:real]
        e = np.exp(lg - lg.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True); am = p.argmax(-1); nb = am != 0
        text = asr_vi.sp.decode([int(t) for i, t in enumerate(am) if t != 0 and (i == 0 or t != am[i - 1])]).lower(); words = text.split()
        feat = dict(lang=lang, nb_ratio=float(nb.mean()), conf_nb=float(p.max(-1)[nb].mean()) if nb.any() else 0.0, conf_all=float(p.max(-1).mean()), blank_p=float(p[:, 0].mean()),
                    known=float(np.mean([wd in syl for wd in words])) if words else 0.0, nwords=len(words), dur=len(w) / 16000)
        sv.lang = "en"; b = np.zeros((1, 464000), np.float32); m = min(len(w), 464000); b[0, :m] = w[:m]
        o = sv.fe(wav=b, wav_len=np.array([m], np.int32)); fb = [np.asarray(q) for q in o]; big = [q for q in fb if q.size > 1000][0].reshape(1, 500, 560).astype(np.float32); sl = [q for q in fb if q.size == 1][0].reshape(1).astype(np.int32)
        for lid_name, lid in (("auto", 0),):
            ob = np.asarray(sv.enc(fbank=big, speech_lengths=sl, language=np.array([lid], np.int32), textnorm=np.array([14], np.int32))[0]).astype(np.int64).reshape(-1)
            feat["sv_auto"] = bytes(int(v) & 0xFF for v in ob).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
        rows.append(feat); print(lang, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in feat.items() if k != "sv_auto"}, feat["sv_auto"][:40], flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
