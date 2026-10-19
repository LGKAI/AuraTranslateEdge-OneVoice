# -*- coding: utf-8 -*-
"""CER of CPU variants on ~40 short Chinese clauses (FLEURS): original infer() vs static graph with different noise / seed / sdp settings."""
import sys, os, re, json, numpy as np, torch, onnxruntime as ort, jiwer
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from melo.api import TTS
from melo import utils
from melo_static import MeloStatic
from datasets_zh import clauses
N, Y = 96, 384
tts = TTS(language="ZH", device="cpu"); hps = tts.hps; m = tts.model
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
nc = lambda s: re.sub(r"[^\w]", "", s)
def cer_of(a, sr, t):
    x = resample_poly(np.asarray(a, np.float32), 16000, sr).astype(np.float32); n = min(len(x), 464000); b = np.zeros(464000, np.float32); b[:n] = x[:n]
    o = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([n], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    return (jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0), h
T = [c for c in clauses(maxc=18, n=80)][:40]; print("clauses", len(T), flush=True)
CFG = {"orig(sdp0.2,noise0.6,random)": None, "static sdp0 noise0.6 seed0": (0.6, 0), "static sdp0 noise0.6 seed1": (0.6, 1), "static sdp0 noise0.667 seed0": (0.667, 0), "static sdp0 noise0.3 seed0": (0.3, 0)}
S = {k: MeloStatic(m, N, Y, noise_scale=v[0], seed=v[1]).eval() for k, v in CFG.items() if v}
res = {k: [] for k in CFG}; bad = {k: [] for k in CFG}
for i, t in enumerate(T):
    bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(t, "ZH", hps, "cpu", tts.symbol_to_id); n = len(ph)
    if n > N: continue
    pad = lambda v: torch.cat([v, torch.zeros(N - n, dtype=v.dtype)])[None].int(); bp = torch.zeros(1, 1024, N); bp[0, :, :n] = bert
    with torch.no_grad():
        for k, v in CFG.items():
            if v is None:
                a = tts.tts_to_file(t, 1, None, speed=1.0, quiet=True); sr = hps.data.sampling_rate
            else:
                o, yl = S[k](pad(ph), pad(tone), pad(lang), bp, torch.tensor([n], dtype=torch.int32)); a = o[0, :int(yl[0]) * 512].numpy(); sr = 44100
            c, h = cer_of(a, sr, t); res[k].append(c)
            if c > 0: bad[k].append((t, h))
    print(i, flush=True)
for k in CFG: print(f"{k:34s} mean CER {np.mean(res[k]):.3f}  (n={len(res[k])}, sentences with errors {sum(1 for c in res[k] if c>0)})")
json.dump({"cer": res, "errors": bad}, open("work/variants.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
