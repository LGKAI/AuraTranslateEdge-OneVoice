# -*- coding: utf-8 -*-
"""Step 1 check: static models (padded to N, latent L) vs original dynamic models, same inputs, same noise."""
import sys, os, collections
import numpy as np, onnx, onnxruntime as ort
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS

N, L = int(sys.argv[1]), int(sys.argv[2])
D = os.path.join((os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")), f"static_N{N}_L{L}")
tts = TTS(); m = tts.model
style = tts.get_voice_style(voice_name="M1")
cpu = ["CPUExecutionProvider"]
st = {k: ort.InferenceSession(os.path.join(D, f"{k}_static.onnx"), providers=cpu) for k in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]}
cos = lambda a, b: float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

for k in st:
    mm = onnx.load(os.path.join(D, f"{k}_static.onnx"))
    nd = collections.Counter(n.domain for n in mm.graph.node)
    print(f"{k:18s} node domains: {dict(nd)}")

for lang, text in [("en", "The quick brown fox jumps over the lazy dog near the river bank."),
                   ("ko", "안녕하세요, 오늘 날씨가 정말 좋네요.")]:
    ids, mask = m.text_processor([text], lang)
    n = ids.shape[1]
    ids_p = np.zeros((1, N), np.int64); ids_p[:, :n] = ids
    mask_p = np.zeros((1, 1, N), np.float32); mask_p[:, :, :n] = 1
    # reference (original dynamic models)
    dur_r = m.dp_ort.run(None, {"text_ids": ids, "style_dp": style.dp, "text_mask": mask})[0]
    emb_r = m.text_enc_ort.run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask})[0]
    # static
    dur_s = st["duration_predictor"].run(None, {"text_ids": ids_p, "style_dp": style.dp, "text_mask": mask_p})[0]
    emb_s = st["text_encoder"].run(None, {"text_ids": ids_p, "style_ttl": style.ttl, "text_mask": mask_p})[0]
    print(f"[{lang}] chars+tags={n} | dur ref {float(dur_r[0]):.4f}s static {float(dur_s[0]):.4f}s | text_emb cos (valid) {cos(emb_r[:, :, :n], emb_s[:, :, :n]):.6f}")
    # latent length from reference duration (speed 1.05)
    dur = dur_r / 1.05
    lat_len = int(np.ceil(int(dur[0] * 44100) / 3072)); assert lat_len <= L, (lat_len, L)
    rng = np.random.RandomState(0)
    noise = rng.randn(1, 144, L).astype(np.float32)
    lmask_p = np.zeros((1, 1, L), np.float32); lmask_p[:, :, :lat_len] = 1
    x_r = (noise[:, :, :lat_len] * 1.0).astype(np.float32); lmask_r = np.ones((1, 1, lat_len), np.float32)
    x_s = noise * lmask_p
    for step in range(5):
        cs = np.array([step], np.float32); ts = np.array([5], np.float32)
        x_r = m.vector_est_ort.run(None, {"noisy_latent": x_r, "text_emb": emb_r, "style_ttl": style.ttl, "text_mask": mask, "latent_mask": lmask_r, "current_step": cs, "total_step": ts})[0]
        x_s = st["vector_estimator"].run(None, {"noisy_latent": x_s, "text_emb": emb_s, "style_ttl": style.ttl, "text_mask": mask_p, "latent_mask": lmask_p, "current_step": cs, "total_step": ts})[0]
    print(f"      latent len {lat_len}/{L} | after 5 ODE steps latent cos (valid) {cos(x_r, x_s[:, :, :lat_len]):.6f}")
    w_r = m.vocoder_ort.run(None, {"latent": x_r})[0].reshape(-1)
    w_s = st["vocoder"].run(None, {"latent": x_s})[0].reshape(-1)
    nv = len(w_r)
    print(f"      wav samples ref {nv} static {len(w_s)} | wav cos (valid) {cos(w_r, w_s[:nv]):.6f} | tail energy beyond valid {float(np.abs(w_s[nv:]).max()) if len(w_s)>nv else 0:.3e}")
