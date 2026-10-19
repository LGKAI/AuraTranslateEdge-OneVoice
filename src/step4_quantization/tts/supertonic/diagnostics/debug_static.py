# -*- coding: utf-8 -*-
import sys, os, numpy as np, onnxruntime as ort
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS
N, L = int(sys.argv[1]), int(sys.argv[2])
D = os.path.join((os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")), f"static_N{N}_L{L}")
tts = TTS(); m = tts.model; style = tts.get_voice_style(voice_name="M1")
cpu = ["CPUExecutionProvider"]
S = {k: ort.InferenceSession(os.path.join(D, f"{k}_static.onnx"), providers=cpu) for k in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]}
cos = lambda a, b: float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
ids, mask = m.text_processor(["The quick brown fox jumps over the lazy dog near the river bank."], "en"); n = ids.shape[1]
emb_r = m.text_enc_ort.run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask})[0]
lat = 62
rng = np.random.RandomState(0); noise = rng.randn(1, 144, L).astype(np.float32)
mask_p = np.zeros((1, 1, N), np.float32); mask_p[:, :, :n] = 1
lm_p = np.zeros((1, 1, L), np.float32); lm_p[:, :, :lat] = 1
emb_p = np.zeros((1, 256, N), np.float32); emb_p[:, :, :n] = emb_r          # IDENTICAL conditioning, zero padded
x0 = noise[:, :, :lat]
cs, ts = np.array([0], np.float32), np.array([5], np.float32)
r = m.vector_est_ort.run(None, {"noisy_latent": x0, "text_emb": emb_r, "style_ttl": style.ttl, "text_mask": mask, "latent_mask": np.ones((1, 1, lat), np.float32), "current_step": cs, "total_step": ts})[0]
s = S["vector_estimator"].run(None, {"noisy_latent": noise * lm_p, "text_emb": emb_p, "style_ttl": style.ttl, "text_mask": mask_p, "latent_mask": lm_p, "current_step": cs, "total_step": ts})[0]
print("one VE step, identical inputs: cos valid", round(cos(r, s[:, :, :lat]), 6), "| max abs diff", float(np.abs(r - s[:, :, :lat]).max()))
# static VE with NO padding at all (L=lat would need different static); check dependence on padded-latent region instead
lm_full = np.ones((1, 1, L), np.float32)
s2 = S["vector_estimator"].run(None, {"noisy_latent": noise, "text_emb": emb_p, "style_ttl": style.ttl, "text_mask": mask_p, "latent_mask": lm_full, "current_step": cs, "total_step": ts})[0]
print("static VE, latent fully valid vs partially padded: cos on first", lat, "frames:", round(cos(s[:, :, :lat], s2[:, :, :lat]), 6))
# is the ORIGINAL model itself sensitive to trailing latent padding?
x_pad = noise * lm_p
r2 = m.vector_est_ort.run(None, {"noisy_latent": x_pad, "text_emb": emb_r, "style_ttl": style.ttl, "text_mask": mask, "latent_mask": lm_p, "current_step": cs, "total_step": ts})[0]
print("ORIGINAL VE, padded latent (len L) vs exact (len lat): cos", round(cos(r, r2[:, :, :lat]), 6))
emb_p2 = np.zeros((1, 256, N), np.float32); emb_p2[:, :, :n] = emb_r
mask_pad_orig = mask_p
r3 = m.vector_est_ort.run(None, {"noisy_latent": x0, "text_emb": emb_p2, "style_ttl": style.ttl, "text_mask": mask_pad_orig, "latent_mask": np.ones((1, 1, lat), np.float32), "current_step": cs, "total_step": ts})[0]
print("ORIGINAL VE, text padded to N vs exact: cos", round(cos(r, r3), 6))
