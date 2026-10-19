# -*- coding: utf-8 -*-
"""Step 2 gate: fused ONE-graph (CPU ORT) vs the 4-model static chain with identical noise, + ASR round trip.
usage: python verify_fused.py <fused.onnx> N L"""
import sys, os, re, json
import numpy as np, onnxruntime as ort, jiwer, soundfile as sf
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS

FUSED, N, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); D = os.path.join(HERE, f"static_N{N}_L{L}")
tts = TTS(); m = tts.model; style = tts.get_voice_style(voice_name="M1")
cpu = ["CPUExecutionProvider"]
S = {k: ort.InferenceSession(os.path.join(D, f"{k}_static.onnx"), providers=cpu) for k in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]}
F = ort.InferenceSession(FUSED, providers=cpu)
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=cpu)
LID = {"zh": 3, "en": 4, "ko": 12}
SENT = json.load(open(os.path.join(HERE, "..", "data", "sentences.json"), encoding="utf-8"))
cos = lambda a, b: float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cp_of(text, lang):
    p = m.text_processor._preprocess_text(text, lang); v = [ord(c) for c in p]
    assert len(v) <= N, f"too long {len(v)}>{N}"
    a = np.zeros((1, N), np.int32); a[0, :len(v)] = v; return a, p


def chain(text, lang, seed=0):
    ids, mask = m.text_processor([text], lang); n = ids.shape[1]
    ids_p = np.zeros((1, N), np.int64); ids_p[:, :n] = ids
    mask_p = np.zeros((1, 1, N), np.float32); mask_p[:, :, :n] = 1
    dur = S["duration_predictor"].run(None, {"text_ids": ids_p, "style_dp": style.dp, "text_mask": mask_p})[0] / 1.05
    emb = S["text_encoder"].run(None, {"text_ids": ids_p, "style_ttl": style.ttl, "text_mask": mask_p})[0]
    lat = int(np.ceil(int(dur[0] * 44100) / 3072)); lm = np.zeros((1, 1, L), np.float32); lm[:, :, :lat] = 1
    x = np.random.RandomState(seed).randn(1, 144, L).astype(np.float32) * lm
    for s in range(5):
        x = S["vector_estimator"].run(None, {"noisy_latent": x, "text_emb": emb, "style_ttl": style.ttl, "text_mask": mask_p, "latent_mask": lm,
                                             "current_step": np.array([s], np.float32), "total_step": np.array([5], np.float32)})[0]
    return S["vocoder"].run(None, {"latent": x})[0].reshape(-1)[:lat * 3072], lat


def asr_text(w44, lang):
    w = resample_poly(w44.astype(np.float32), 16000, 44100).astype(np.float32); n = min(len(w), 464000); b = np.zeros(464000, np.float32); b[:n] = w[:n]
    o = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([n], np.int32), "language": np.array([LID[lang]], np.int32), "textnorm": np.array([14], np.int32)})
    return bytes(int(x) & 0xFF for x in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


nw = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()
nc = lambda s: re.sub(r"\s+", "", re.sub(r"[^\w\s]", "", s.lower()))
rows = []
for i, (lang, text) in enumerate(SENT):
    cp, prep = cp_of(text, lang)
    wav, lat = F.run(None, {"cp": cp, "style_dp": style.dp, "style_ttl": style.ttl})
    lat = int(lat[0]); wf = wav.reshape(-1)[:lat * 3072]
    wc, latc = chain(text, lang, 0)
    tail = float(np.abs(wav.reshape(-1)[lat * 3072:]).max()) if wav.size > lat * 3072 else 0.0
    h = asr_text(wf, lang)
    r = {"lang": lang, "lat_fused": lat, "lat_chain": latc, "cos": cos(wf[:min(len(wf), len(wc))], wc[:min(len(wf), len(wc))]) if lat == latc else None,
         "tail": tail, "hyp": h, "wer": jiwer.wer(nw(text), nw(h)) if nw(h) else 1.0, "cer": jiwer.cer(nc(text), nc(h)) if nc(h) else 1.0}
    rows.append(r); sf.write(os.path.join(HERE, "wav_cpu", f"{i:02d}_{lang}_fusedcpu.wav"), wf, 44100)
    print(f"[{i}] {lang} lat fused/chain {lat}/{latc} | wav cos {r['cos'] if r['cos'] is None else round(r['cos'],6)} | tail {tail:.1e} | WER {r['wer']:.2f} CER {r['cer']:.2f}", flush=True)
for lang in ("en", "ko"):
    xs = [r for r in rows if r["lang"] == lang]
    print(f"fused-CPU {lang}: WER {np.mean([r['wer'] for r in xs])*100:.1f}%  CER {np.mean([r['cer'] for r in xs])*100:.1f}%  (n={len(xs)})")
