"""Step 4 -- decisive experiment: does the PROPER AI Hub quantization workflow
(submit_quantize_job with real AIMET AdaRound + 500-1000 calibration samples)
fix the NLLB encoder's ~0.18 cosine-similarity corruption, where the naive
path (submit_compile_job with --quantize_full_type int8 + only 4-8 calibration
sentences) failed badly?

Root cause hypothesis (from research, see conversation): AI Hub's own docs
recommend 500-1000 calibration samples for submit_quantize_job's AIMET-backed
PTQ (Cross-Layer Equalization, Bias Correction, AdaRound) -- we used 4-8
samples via the OTHER path (--quantize_full_type baked directly into compile,
a simpler/cruder quantizer) for every model in this project so far.

Uses 600 REAL Vietnamese sentences from the FLEURS dataset (already cached
locally, 2994 rows total) as calibration -- not synthetic, not the same 4-8
sentences reused everywhere else in this project.
"""
import os
import json

import numpy as np
import onnxruntime as ort
import qai_hub as hub
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNQUANTIZED_MODEL = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask_static.onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
FIXED_SEQ_LEN = 64
MODEL_ID = "facebook/nllb-200-distilled-600M"
N_CALIB = 600


def build_calibration(tok):
    print(f"[aimet_nllb] loading FLEURS vi_vn (real sentences, not synthetic) ...")
    ds = load_dataset("google/fleurs", "vi_vn", split="train")
    ds = ds.remove_columns(["audio"])
    texts = [r["transcription"] for r in ds][:N_CALIB]
    print(f"[aimet_nllb] using {len(texts)} REAL Vietnamese sentences for calibration")

    input_ids_list, bias_list = [], []
    for text in texts:
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        inv = (1 - mask).astype(np.float32)
        bias = np.broadcast_to((inv * -30000.0)[:, None, None, :], (1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)).copy()
        input_ids_list.append(ids)
        bias_list.append(bias)

    dataset = hub.upload_dataset(
        {"input_ids": input_ids_list, "/Where_1_output_0": bias_list},
        name=f"nllb_encoder_aimet_calibration_{N_CALIB}",
    )
    print(f"[aimet_nllb] calibration dataset id: {dataset.dataset_id} ({len(texts)} samples)")
    return dataset.dataset_id


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="vie_Latn")
    calib_id = os.environ.get("AIMET_CALIB_ID") or build_calibration(tok)
    calib = hub.get_dataset(calib_id)

    print(f"[aimet_nllb] model: {UNQUANTIZED_MODEL}")
    reuse_quantize_job = os.environ.get("AIMET_QUANTIZE_JOB_ID")
    if reuse_quantize_job:
        print(f"[aimet_nllb] reusing existing quantize job {reuse_quantize_job}")
        quantized_model = hub.get_job(reuse_quantize_job).get_target_model()
    else:
        print(f"[aimet_nllb] submitting AIMET quantize job (AdaRound, real AIMET quantizer) ...")
        quantize_job = hub.submit_quantize_job(
            model=UNQUANTIZED_MODEL,
            calibration_data=calib,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
            name="nllb_encoder_aimet_adaround",
        )
        print(f"[aimet_nllb] quantize job: {quantize_job.job_id}  {quantize_job.url}")
        quantize_job.wait()
        qstatus = quantize_job.get_status()
        print(f"[aimet_nllb] quantize status: {qstatus}")
        if qstatus.code != "SUCCESS":
            print(f"[aimet_nllb] quantize FAILED: {qstatus.message}. See {quantize_job.url}")
            return
        quantized_model = quantize_job.get_target_model()
    print(f"[aimet_nllb] quantize SUCCESS, quantized_model_id={quantized_model.model_id}")

    device = hub.Device(DEVICE_NAME)
    print(f"[aimet_nllb] compiling quantized QDQ model for {device.name} ...")
    # /Where_1_output_0 (our host-computed mask bias) is a float32 GRAPH INPUT,
    # not an internal activation the AIMET quantize job would have touched --
    # confirmed via a real "has a floating-point type which is not supported
    # by the targeted device" error (same class as the Zipformer fp16 finding)
    compile_job = hub.submit_compile_job(
        model=quantized_model, device=device,
        options="--target_runtime qnn_context_binary --truncate_64bit_io --quantize_io",
    )
    print(f"[aimet_nllb] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    cstatus = compile_job.get_status()
    print(f"[aimet_nllb] compile status: {cstatus}")
    if cstatus.code != "SUCCESS":
        print(f"[aimet_nllb] compile FAILED: {cstatus.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print(f"[aimet_nllb] compile SUCCESS, target_model_id={target_model.model_id}")

    # --- real-input diagnostic, same method as p0_verify_nllb.py ---
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    print(f"[aimet_nllb] test sentence: '{text}'")
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    inv = (1 - mask).astype(np.float32)
    bias = np.broadcast_to((inv * -30000.0)[:, None, None, :], (1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)).copy()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(UNQUANTIZED_MODEL, so, providers=["CPUExecutionProvider"])
    local_out = sess.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]
    print(f"[aimet_nllb] local fp32 encoder_out shape={local_out.shape}")

    ids_i32 = ids.astype(np.int32)
    job = hub.submit_inference_job(
        model=target_model, device=device,
        inputs={"input_ids": [ids_i32], "/Where_1_output_0": [bias]}, name="aimet-nllb-test")
    print(f"[aimet_nllb] inference job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[aimet_nllb] inference FAILED: {job.get_status().message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    print(f"[aimet_nllb] hardware encoder_out shape={hw_out.shape}")

    print(f"\n[aimet_nllb] === RESULT ===")
    print(f"[aimet_nllb] cos_sim={cos_sim(local_out, hw_out):.4f}  max_abs_diff={max_abs_diff(local_out, hw_out):.4e}")
    print(f"[aimet_nllb] local: mean={local_out.mean():.4f} std={local_out.std():.4f}")
    print(f"[aimet_nllb] hw:    mean={hw_out.mean():.4f} std={hw_out.std():.4f}")
    cs = cos_sim(local_out, hw_out)
    if cs > 0.9:
        print("[aimet_nllb] *** CONFIRMED: proper AIMET quantize workflow fixes the corruption ***")
    else:
        print("[aimet_nllb] still wrong -- calibration-starvation hypothesis insufficient, "
              "need deeper investigation (per-op sensitivity, mixed precision, etc.)")


if __name__ == "__main__":
    main()
