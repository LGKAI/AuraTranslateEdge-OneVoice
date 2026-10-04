"""Muui 1 -- decisive 1-line-change test: is NLLB's host-computed attention
mask bias value (-30000) blowing up the int8 quantization range for
/Where_1_output_0, since real attention logits only span roughly +/-10?
With --quantize_io, int8 has 256 levels; a [-30000, +10] range gives
~118 units/level, so the entire real logit range collapses into ~1
quantization bucket. Softmax then sees near-uniform logits -> attention
dies -> exactly the "variance-compressed near-random output" pattern
measured (hw std 0.117 vs local std 0.339, cos_sim 0.04-0.18).

Fix: use -100 instead of -30000. exp(-100) ~ 0 is still a fully effective
mask for softmax (real logits ~+/-10), but the quantization range shrinks
30000/100 = 300x, restoring resolution for the real logit values.
Everything else (graph, compile recipe, calibration size) is held IDENTICAL
to the ORIGINAL naive test that gave cos_sim=0.1776, to isolate this one
variable cleanly.
"""
import os
import json

import numpy as np
import onnxruntime as ort
import qai_hub as hub
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask.onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
FIXED_SEQ_LEN = 64
MODEL_ID = "facebook/nllb-200-distilled-600M"
MASK_VALUE = -100.0


def host_mask(mask, neg_bias=MASK_VALUE):
    inv = (1 - mask).astype(np.float32)
    bias = inv * neg_bias
    return np.broadcast_to(bias[:, None, None, :], (bias.shape[0], 1, bias.shape[1], bias.shape[1])).copy()


def build_calibration(tok):
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_sentences = [r["vi"] for r in manifest][:8]  # same 8 sentences as the ORIGINAL test

    input_ids_list, bias_list = [], []
    for text in vi_sentences:
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        input_ids_list.append(ids)
        bias_list.append(host_mask(mask))
        real_len = int(mask.sum())
        print(f"[nllb_m100] '{text[:35]}...' -> {real_len} real tokens (windowed to {FIXED_SEQ_LEN})")

    dataset = hub.upload_dataset(
        {"input_ids": input_ids_list, "/Where_1_output_0": bias_list},
        name="nllb_encoder_hostmask_m100_calibration",
    )
    print(f"[nllb_m100] calibration dataset id: {dataset.dataset_id}")
    return dataset.dataset_id


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="vie_Latn")
    device = hub.Device(DEVICE_NAME)
    calib_id = os.environ.get("M100_CALIB_ID") or build_calibration(tok)
    calib = hub.get_dataset(calib_id)

    input_specs = {
        "input_ids": ((1, FIXED_SEQ_LEN), "int64"),
        "/Where_1_output_0": ((1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN), "float32"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[nllb_m100] compile options: {options} (mask value={MASK_VALUE}, was -30000)")

    compile_job = hub.submit_compile_job(
        model=MODEL_PATH, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[nllb_m100] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[nllb_m100] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[nllb_m100] compile FAILED: {status.message}")
        return
    target_model = compile_job.get_target_model()
    print(f"[nllb_m100] compile SUCCESS, target_model_id={target_model.model_id}")

    # --- real-input diagnostic ---
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    print(f"[nllb_m100] test sentence: '{text}'")
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    bias = host_mask(mask)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    local_out = sess.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]
    print(f"[nllb_m100] local fp32 encoder_out shape={local_out.shape}")

    ids_i32 = ids.astype(np.int32)
    job = hub.submit_inference_job(
        model=target_model, device=device,
        inputs={"input_ids": [ids_i32], "/Where_1_output_0": [bias]}, name="nllb-m100-test")
    print(f"[nllb_m100] inference job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[nllb_m100] inference FAILED: {job.get_status().message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    print(f"[nllb_m100] hardware encoder_out shape={hw_out.shape}")

    print(f"\n[nllb_m100] === RESULT ===")
    print(f"[nllb_m100] cos_sim={cos_sim(local_out, hw_out):.4f}  max_abs_diff={max_abs_diff(local_out, hw_out):.4e}")
    print(f"[nllb_m100] local: mean={local_out.mean():.4f} std={local_out.std():.4f}")
    print(f"[nllb_m100] hw:    mean={hw_out.mean():.4f} std={hw_out.std():.4f}")
    cs = cos_sim(local_out, hw_out)
    if cs > 0.9:
        print("[nllb_m100] *** CONFIRMED: outlier mask value (-30000) was destroying the "
              "int8 quantization range -- -100 fixes it ***")
    else:
        print("[nllb_m100] still wrong -- outlier-range hypothesis insufficient")


if __name__ == "__main__":
    main()
