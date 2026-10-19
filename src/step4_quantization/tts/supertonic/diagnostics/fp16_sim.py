# -*- coding: utf-8 -*-
"""CPU fp16 emulation per stage (onnxconverter_common float16, keep_io_types) to predict which Supertonic stage breaks on HTP fp16.
Each stage is fed the FP32 output of the previous stage, so errors are isolated per stage."""
import sys, os, numpy as np, onnx, onnxruntime as ort
from onnxconverter_common import float16
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS
N, L = 96, 150
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); D = os.path.join(HERE, f"static_N{N}_L{L}")
tts = TTS(); m = tts.model; style = tts.get_voice_style(voice_name="M1"); cpu = ["CPUExecutionProvider"]
def sess(name, f16):
    p = os.path.join(D, f"{name}_static.onnx")
    if f16:
        mm = float16.convert_float_to_float16(onnx.load(p), keep_io_types=True, disable_shape_infer=True)
        return ort.InferenceSession(mm.SerializeToString(), providers=cpu)
    return ort.InferenceSession(p, providers=cpu)
names = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
S32 = {k: sess(k, False) for k in names}; S16 = {k: sess(k, True) for k in names}
cos = lambda a, b: float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
for text, lang in [("The quick brown fox jumps over the lazy dog near the river bank.", "en"), ("안녕하세요, 오늘 날씨가 정말 좋네요.", "ko")]:
    ids, mask = m.text_processor([text], lang); n = ids.shape[1]
    ids_p = np.zeros((1, N), np.int64); ids_p[:, :n] = ids; mp = np.zeros((1, 1, N), np.float32); mp[:, :, :n] = 1
    f = {"text_ids": ids_p, "style_dp": style.dp, "text_mask": mp}
    d32 = S32["duration_predictor"].run(None, f)[0]; d16 = S16["duration_predictor"].run(None, f)[0]
    e32 = S32["text_encoder"].run(None, {"text_ids": ids_p, "style_ttl": style.ttl, "text_mask": mp})[0]
    e16 = S16["text_encoder"].run(None, {"text_ids": ids_p, "style_ttl": style.ttl, "text_mask": mp})[0]
    lat = int(np.ceil(int(d32[0] / 1.05 * 44100) / 3072)); lm = np.zeros((1, 1, L), np.float32); lm[:, :, :lat] = 1
    x = np.random.RandomState(0).randn(1, 144, L).astype(np.float32) * lm
    ve = lambda S, x, e, s: S["vector_estimator"].run(None, {"noisy_latent": x, "text_emb": e, "style_ttl": style.ttl, "text_mask": mp, "latent_mask": lm, "current_step": np.array([s], np.float32), "total_step": np.array([5], np.float32)})[0]
    v32 = ve(S32, x, e32, 0); v16 = ve(S16, x, e32, 0)           # one step, same fp32 inputs
    x32 = x.copy(); x16 = x.copy()
    for s in range(5): x32 = ve(S32, x32, e32, s); x16 = ve(S16, x16, e32, s)   # full ODE, fp32 cond
    w32 = S32["vocoder"].run(None, {"latent": x32})[0]; w16 = S16["vocoder"].run(None, {"latent": x32})[0]   # vocoder fp16 on fp32 latent
    print(f"[{lang}] dur fp32 {float(d32[0]):.4f} fp16 {float(d16[0]):.4f} | text_emb cos {cos(e32[:,:,:n], e16[:,:,:n]):.5f} | VE 1-step cos {cos(v32[:,:,:lat], v16[:,:,:lat]):.5f} | ODE5 latent cos {cos(x32[:,:,:lat], x16[:,:,:lat]):.5f} | vocoder(fp16) wav cos {cos(w32.reshape(-1)[:lat*3072], w16.reshape(-1)[:lat*3072]):.5f}")
    print(f"      ranges: text_emb max|.| {float(np.abs(e32).max()):.1f} | latent max|.| {float(np.abs(x32).max()):.1f} | VE out max {float(np.abs(v32).max()):.1f}")
