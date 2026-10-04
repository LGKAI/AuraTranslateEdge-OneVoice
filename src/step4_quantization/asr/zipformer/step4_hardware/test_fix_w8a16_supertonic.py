"""w8a16 generalization test #3: Supertonic 4 submodels (duration_predictor,
text_encoder, vector_estimator, vocoder). Same recipe that fixed NLLB
(0.18->0.9998) and partially fixed Zipformer (0.21->0.80): submit_quantize_job
weights=INT8/activations=INT16, compile on IQ-9075 EVK (Hexagon v73+
required), real inference vs local fp32. Supertonic has ZERO graph surgery
(step4.md SS4f) so any residual divergence here isolates a pure QNN/HTP
quantization effect, unclouded by any of our own graph edits.

Submits all 4 quantize jobs up front (they run independently on AI Hub),
rather than one at a time, to avoid serializing ~4x the wait time.
"""
import os
import json

import numpy as np
import onnxruntime as ort
import qai_hub as hub

from supertonic import TTS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
SRC_DIR = os.path.expanduser("~/.cache/supertonic3/onnx")
FIXED_TEXT_LEN = 80
FIXED_LATENT_LEN = 150
TOTAL_STEPS = 5

RECIPE = os.environ.get("RECIPE", "w8a16")
WDT = hub.QuantizeDtype.INT16 if RECIPE.startswith("w16") else hub.QuantizeDtype.INT8
ADT = hub.QuantizeDtype.INT16 if RECIPE.endswith("a16") else hub.QuantizeDtype.INT8


def pad_text(text_ids, text_mask, fixed_len):
    real_len = text_ids.shape[1]
    if real_len < fixed_len:
        ids = np.zeros((1, fixed_len), dtype=np.int64)
        ids[:, :real_len] = text_ids
        mask = np.zeros((1, 1, fixed_len), dtype=np.float32)
        mask[:, :, :real_len] = text_mask
    else:
        ids = text_ids[:, :fixed_len]
        mask = text_mask[:, :, :fixed_len]
    return ids, mask


def main():
    tts = TTS(auto_download=True)
    style = tts.get_voice_style("M1")
    text_processor = tts.model.text_processor

    so = ort.SessionOptions(); so.log_severity_level = 3
    dp_sess = ort.InferenceSession(os.path.join(SRC_DIR, "duration_predictor.onnx"), so, providers=["CPUExecutionProvider"])
    enc_sess = ort.InferenceSession(os.path.join(SRC_DIR, "text_encoder.onnx"), so, providers=["CPUExecutionProvider"])
    ve_sess = ort.InferenceSession(os.path.join(SRC_DIR, "vector_estimator.onnx"), so, providers=["CPUExecutionProvider"])
    voc_sess = ort.InferenceSession(os.path.join(SRC_DIR, "vocoder.onnx"), so, providers=["CPUExecutionProvider"])

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)
    eval_text = rows[0]["vi"]
    calib_texts = [r["vi"] for r in rows[1:5]]
    print(f"[st-w8a16] eval: '{eval_text}'  calib: {len(calib_texts)} sentences", flush=True)

    def build_dp_enc(text):
        raw_ids, raw_mask = text_processor([text], "vi")
        ids, mask = pad_text(raw_ids, raw_mask, FIXED_TEXT_LEN)
        return ids, mask

    eval_ids, eval_mask = build_dp_enc(eval_text)
    calib_ids_masks = [build_dp_enc(t) for t in calib_texts]

    # ---- local fp32 chain (eval) ----
    dp_inputs = {"text_ids": eval_ids, "style_dp": style.dp, "text_mask": eval_mask}
    duration_local = dp_sess.run(None, dp_inputs)[0]
    enc_inputs = {"text_ids": eval_ids, "style_ttl": style.ttl, "text_mask": eval_mask}
    text_emb_local = enc_sess.run(None, enc_inputs)[0]

    latent_dim = tts.model.ldim * tts.model.chunk_compress_factor
    chunk_size = tts.model.base_chunk_size * tts.model.chunk_compress_factor
    wav_len = float(duration_local[0]) * tts.model.sample_rate
    real_latent_len = min(FIXED_LATENT_LEN, int(np.ceil(wav_len / chunk_size)))
    rng = np.random.default_rng(0)
    latent_mask = np.zeros((1, 1, FIXED_LATENT_LEN), dtype=np.float32)
    latent_mask[:, :, :real_latent_len] = 1.0
    xt = (rng.standard_normal((1, latent_dim, FIXED_LATENT_LEN)).astype(np.float32) * latent_mask)

    ve_local_by_step = []
    for step in range(TOTAL_STEPS):
        cur = np.array([step], dtype=np.float32)
        tot = np.array([TOTAL_STEPS], dtype=np.float32)
        ve_inputs = {"noisy_latent": xt, "text_emb": text_emb_local, "style_ttl": style.ttl,
                     "latent_mask": latent_mask, "text_mask": eval_mask,
                     "current_step": cur, "total_step": tot}
        vt_local = ve_sess.run(None, ve_inputs)[0]
        ve_local_by_step.append((dict(ve_inputs), vt_local.copy()))
        xt = vt_local * latent_mask
    voc_local = voc_sess.run(None, {"latent": xt})[0]

    np.savez(os.path.join(ROOT, "outputs", "supertonic_w8a16_fp32ref.npz"),
             duration=duration_local, text_emb=text_emb_local, voc=voc_local,
             ve_last=ve_local_by_step[-1][1])
    print(f"[st-w8a16] local fp32 done. duration={duration_local.flatten()[:2]} "
          f"text_emb.shape={text_emb_local.shape} voc.shape={voc_local.shape}", flush=True)

    # ---- calibration data per submodel ----
    dp_calib = {"text_ids": [], "style_dp": [], "text_mask": []}
    enc_calib = {"text_ids": [], "style_ttl": [], "text_mask": []}
    for ids, mask in calib_ids_masks:
        dp_calib["text_ids"].append(ids); dp_calib["style_dp"].append(style.dp); dp_calib["text_mask"].append(mask)
        enc_calib["text_ids"].append(ids); enc_calib["style_ttl"].append(style.ttl); enc_calib["text_mask"].append(mask)

    ve_calib = {"noisy_latent": [], "text_emb": [], "style_ttl": [], "latent_mask": [],
                "text_mask": [], "current_step": [], "total_step": []}
    voc_calib = {"latent": []}
    for step, (inp, out) in enumerate(ve_local_by_step):
        ve_calib["noisy_latent"].append(inp["noisy_latent"]); ve_calib["text_emb"].append(inp["text_emb"])
        ve_calib["style_ttl"].append(inp["style_ttl"]); ve_calib["latent_mask"].append(inp["latent_mask"])
        ve_calib["text_mask"].append(inp["text_mask"]); ve_calib["current_step"].append(inp["current_step"])
        ve_calib["total_step"].append(inp["total_step"])
        voc_calib["latent"].append(out)

    MODELS = {
        "duration_predictor": (os.path.join(SRC_DIR, "duration_predictor.onnx"), dp_calib),
        "text_encoder": (os.path.join(SRC_DIR, "text_encoder.onnx"), enc_calib),
        "vector_estimator": (os.path.join(SRC_DIR, "vector_estimator.onnx"), ve_calib),
        "vocoder": (os.path.join(SRC_DIR, "vocoder.onnx"), voc_calib),
    }

    print(f"[st-w8a16] RECIPE={RECIPE} weights={WDT} activations={ADT}", flush=True)
    quant_jobs = {}
    for name, (path, calib) in MODELS.items():
        ds = hub.upload_dataset(calib, name=f"supertonic-{name}-calib-{RECIPE}")
        qjob = hub.submit_quantize_job(
            model=path, calibration_data=ds,
            weights_dtype=WDT, activations_dtype=ADT, name=f"supertonic-{name}-{RECIPE}-quant")
        quant_jobs[name] = qjob.job_id
        print(f"[st-w8a16] {name}: quant job {qjob.job_id}  {qjob.url}", flush=True)

    print("\n[st-w8a16] === ALL 4 QUANTIZE JOBS SUBMITTED ===", flush=True)
    for name, jid in quant_jobs.items():
        print(f"  {name}: {jid}", flush=True)


if __name__ == "__main__":
    main()
