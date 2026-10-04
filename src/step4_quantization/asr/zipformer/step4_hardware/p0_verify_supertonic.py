"""P0 -- numeric ground-truth check for all 4 already-"successfully compiled"
Supertonic submodels. Until now every Supertonic AI Hub job only measured
compile+profile (timing), never real output correctness (see step4.md SS4f).
Chains a REAL forward pass (duration_predictor -> text_encoder -> vector_estimator
xN -> vocoder) locally in fp32 (onnxruntime), capturing every real intermediate
tensor, then feeds the SAME real inputs to the compiled AI Hub target models
(submit_inference_job) and compares outputs via cosine similarity / max_abs_diff.
Supertonic has ZERO graph surgery (see SS4f) so any divergence here would
indicate a QNN-level issue unrelated to any of our own graph edits -- the
most diagnostic single test in the whole pipeline.
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
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

FIXED_TEXT_LEN = 80
FIXED_LATENT_LEN = 150
TOTAL_STEPS = 5

COMPILE_JOBS = {
    "duration_predictor": "jg9doe9m5",
    "text_encoder": "jgo8dqm1p",
    "vector_estimator": "jpxx0238p",
    "vocoder": "j5m89y47p",
}


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


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def run_hw(compile_job_id, inputs, device, name):
    target_model = hub.get_job(compile_job_id).get_target_model()
    print(f"  [{name}] submitting real inference job ...")
    job = hub.submit_inference_job(model=target_model, device=device, inputs=inputs, name=name)
    print(f"  [{name}] job: {job.job_id}  {job.url}")
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        raise RuntimeError(f"{name} FAILED: {status.message}")
    return job.download_output_data()


def main():
    device = hub.Device(DEVICE_NAME)
    tts = TTS(auto_download=True)
    style = tts.get_voice_style("M1")
    text_processor = tts.model.text_processor

    so = ort.SessionOptions()
    so.log_severity_level = 3
    dp_sess = ort.InferenceSession(os.path.join(SRC_DIR, "duration_predictor.onnx"), so, providers=["CPUExecutionProvider"])
    enc_sess = ort.InferenceSession(os.path.join(SRC_DIR, "text_encoder.onnx"), so, providers=["CPUExecutionProvider"])
    ve_sess = ort.InferenceSession(os.path.join(SRC_DIR, "vector_estimator.onnx"), so, providers=["CPUExecutionProvider"])
    voc_sess = ort.InferenceSession(os.path.join(SRC_DIR, "vocoder.onnx"), so, providers=["CPUExecutionProvider"])

    with open(MANIFEST, "r", encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    print(f"[p0_supertonic] real sentence: '{text}'")

    raw_ids, raw_mask = text_processor([text], "vi")
    ids, mask = pad_text(raw_ids, raw_mask, FIXED_TEXT_LEN)

    # --- 1. duration_predictor ---
    # compiled with --truncate_64bit_io: real hardware calls need int32 for
    # declared int64 graph inputs (text_ids), confirmed via a real "Cannot
    # assign data from unexpected type. Expected int32, got int64" error --
    # local onnxruntime keeps int64 (the graph's real declared dtype).
    ids_i32 = ids.astype(np.int32)
    dp_inputs = {"text_ids": ids, "style_dp": style.dp, "text_mask": mask}
    duration_local = dp_sess.run(None, dp_inputs)[0]
    hw_out = run_hw(COMPILE_JOBS["duration_predictor"],
                     {"text_ids": [ids_i32], "style_dp": [style.dp], "text_mask": [mask]},
                     device, "duration_predictor")
    duration_hw = np.array(hw_out["output_0"][0])
    print(f"[p0_supertonic] duration_predictor: local={duration_local.flatten()[:3]} "
          f"hw={duration_hw.flatten()[:3]} cos_sim={cos_sim(duration_local, duration_hw):.4f} "
          f"max_abs_diff={max_abs_diff(duration_local, duration_hw):.4e}")

    # --- 2. text_encoder ---
    enc_inputs = {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask}
    text_emb_local = enc_sess.run(None, enc_inputs)[0]
    hw_out = run_hw(COMPILE_JOBS["text_encoder"],
                     {"text_ids": [ids_i32], "style_ttl": [style.ttl], "text_mask": [mask]},
                     device, "text_encoder")
    text_emb_hw = np.array(hw_out["output_0"][0])
    print(f"[p0_supertonic] text_encoder: cos_sim={cos_sim(text_emb_local, text_emb_hw):.4f} "
          f"max_abs_diff={max_abs_diff(text_emb_local, text_emb_hw):.4e}")

    # --- 3. vector_estimator (real diffusion loop, LOCAL fp32 output used to
    #         drive the loop; hardware is tested per-step against the SAME
    #         local-driven xt so any per-step divergence is isolated, not
    #         compounded across steps) ---
    latent_dim = tts.model.ldim * tts.model.chunk_compress_factor
    chunk_size = tts.model.base_chunk_size * tts.model.chunk_compress_factor
    wav_len = float(duration_local[0]) * tts.model.sample_rate
    real_latent_len = min(FIXED_LATENT_LEN, int(np.ceil(wav_len / chunk_size)))
    print(f"[p0_supertonic] real_latent_len={real_latent_len} (budget {FIXED_LATENT_LEN})")

    rng = np.random.default_rng(0)
    latent_mask = np.zeros((1, 1, FIXED_LATENT_LEN), dtype=np.float32)
    latent_mask[:, :, :real_latent_len] = 1.0
    xt = (rng.standard_normal((1, latent_dim, FIXED_LATENT_LEN)).astype(np.float32) * latent_mask)

    ve_cos_sims = []
    for step in range(TOTAL_STEPS):
        cur = np.array([step], dtype=np.float32)
        tot = np.array([TOTAL_STEPS], dtype=np.float32)
        ve_inputs_local = {"noisy_latent": xt, "text_emb": text_emb_local, "style_ttl": style.ttl,
                            "latent_mask": latent_mask, "text_mask": mask,
                            "current_step": cur, "total_step": tot}
        vt_local = ve_sess.run(None, ve_inputs_local)[0]

        ve_inputs_hw = {"noisy_latent": [xt], "text_emb": [text_emb_local], "style_ttl": [style.ttl],
                         "latent_mask": [latent_mask], "text_mask": [mask],
                         "current_step": [cur], "total_step": [tot]}
        hw_out = run_hw(COMPILE_JOBS["vector_estimator"], ve_inputs_hw, device, f"vector_estimator-step{step}")
        vt_hw = np.array(hw_out["output_0"][0])

        cs = cos_sim(vt_local, vt_hw)
        ve_cos_sims.append(cs)
        print(f"[p0_supertonic] vector_estimator step {step}: cos_sim={cs:.4f} "
              f"max_abs_diff={max_abs_diff(vt_local, vt_hw):.4e}")

        # advance xt using the LOCAL fp32 result (ground truth), matching a real run
        xt = vt_local * latent_mask

    # --- 4. vocoder ---
    voc_local = voc_sess.run(None, {"latent": xt})[0]
    hw_out = run_hw(COMPILE_JOBS["vocoder"], {"latent": [xt]}, device, "vocoder")
    voc_hw = np.array(hw_out["output_0"][0])
    print(f"[p0_supertonic] vocoder: cos_sim={cos_sim(voc_local, voc_hw):.4f} "
          f"max_abs_diff={max_abs_diff(voc_local, voc_hw):.4e}")

    print("\n[p0_supertonic] === SUMMARY ===")
    print(f"  duration_predictor cos_sim: {cos_sim(duration_local, duration_hw):.4f}")
    print(f"  text_encoder       cos_sim: {cos_sim(text_emb_local, text_emb_hw):.4f}")
    print(f"  vector_estimator   cos_sim (mean over {TOTAL_STEPS} steps): {np.mean(ve_cos_sims):.4f}")
    print(f"  vocoder            cos_sim: {cos_sim(voc_local, voc_hw):.4f}")


if __name__ == "__main__":
    main()
