"""Deploy Zipformer mixed-precision candidate to Qualcomm AI Hub.

Candidate:
  - Start from a local static-int8 QDQ encoder.
  - Remove activation QDQ pairs under stages 0, 1, 4, 5 so those stages stay
    high precision locally.
  - Compile/profile/infer on Dragonwing IQ-9075 EVK.

This intentionally answers a concrete question: does the local block-level
mixed-precision idea map to a deployable QNN context binary, and if it does,
what are model size, latency, and hardware cosine?
"""
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import qai_hub as hub
import soundfile as sf

from diag_zipformer_block_mixed_precision import (
    ROOT,
    MODEL_PATH,
    build_quantized_model,
    remove_qdq_pairs_under_prefix,
    eval_model,
    cos_sim,
    pad,
    compute_fbank,
)


OUT_DIR = os.path.join(ROOT, "outputs", "zipformer-qnn")
MIXED_ONNX = os.path.join(OUT_DIR, "encoder_mixed_stage0145_qdq.onnx")
RESULT_JSON = os.path.join(ROOT, "outputs", "zipformer_mixed_stage0145_aihub.json")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
FIXED_FRAMES = 1500
KEEP_HIGH_PRECISION = ["/encoder/0", "/encoder/1", "/encoder/4", "/encoder/5"]


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {st.message[:180] if st.message else ''}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def build_inputs():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi = [r for r in manifest if r["lang"] == "vi"]
    calib_items = vi[:3]
    eval_item = min(vi, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    calib_inputs = []
    for it in calib_items:
        wav, sr = sf.read(os.path.join(ROOT, it["path"]), dtype="float32")
        x, x_lens, _ = pad(compute_fbank(wav, sr), FIXED_FRAMES)
        calib_inputs.append({"x": x.astype(np.float32), "x_lens": x_lens})

    wav, sr = sf.read(os.path.join(ROOT, eval_item["path"]), dtype="float32")
    eval_x, eval_lens, eval_real = pad(compute_fbank(wav, sr), FIXED_FRAMES)
    return calib_inputs, eval_item, eval_x, eval_lens, eval_real


def make_mixed_model(calib_inputs):
    tmp_int8 = os.path.join(OUT_DIR, "encoder_static_int8_for_mixed.onnx")
    build_quantized_model(calib_inputs, tmp_int8)
    model = onnx.load(tmp_int8)
    removed = 0
    for prefix in KEEP_HIGH_PRECISION:
        removed += remove_qdq_pairs_under_prefix(model, prefix)
    onnx.save(model, MIXED_ONNX)
    return tmp_int8, removed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    calib_inputs, eval_item, eval_x, eval_lens, eval_real = build_inputs()
    tmp_int8, removed_pairs = make_mixed_model(calib_inputs)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    fp_sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    fp = eval_model(fp_sess, eval_x, eval_lens)
    mixed_sess = ort.InferenceSession(MIXED_ONNX, so, providers=["CPUExecutionProvider"])
    mixed = eval_model(mixed_sess, eval_x, eval_lens)
    local_cos = cos_sim(fp, mixed)

    result = {
        "candidate": "mixed_stage0145",
        "device": DEVICE_NAME,
        "mixed_onnx": MIXED_ONNX,
        "source_fp32_onnx_size_bytes": os.path.getsize(MODEL_PATH),
        "mixed_onnx_size_bytes": os.path.getsize(MIXED_ONNX),
        "local_cos": float(local_cos),
        "removed_qdq_pairs": int(removed_pairs),
        "eval_path": eval_item["path"],
        "eval_real_frames": int(eval_real),
    }
    print(json.dumps(result, indent=2), flush=True)

    device = hub.Device(DEVICE_NAME)
    compile_job = hub.submit_compile_job(
        model=MIXED_ONNX,
        device=device,
        input_specs={
            "x": ((1, FIXED_FRAMES, 80), "float32"),
            "x_lens": ((1,), "int64"),
        },
        options="--target_runtime qnn_context_binary --quantize_io --truncate_64bit_io",
        name="zipformer-mixed-stage0145-compile",
    )
    result["compile_job_id"] = compile_job.job_id
    result["compile_url"] = compile_job.url
    st = poll(compile_job, "compile")
    result["compile_status"] = st.code
    result["compile_message"] = st.message
    if st.code != "SUCCESS":
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[done] compile failed; wrote {RESULT_JSON}")
        return

    target = compile_job.get_target_model()
    result["target_model_id"] = target.model_id

    profile_job = hub.submit_profile_job(model=target, device=device, name="zipformer-mixed-stage0145-profile")
    result["profile_job_id"] = profile_job.job_id
    result["profile_url"] = profile_job.url
    pst = poll(profile_job, "profile")
    result["profile_status"] = pst.code
    result["profile_message"] = pst.message
    if pst.code == "SUCCESS":
        profile = profile_job.download_profile()
        result["profile_execution_summary"] = profile.get("execution_summary", {})

    infer_job = hub.submit_inference_job(
        model=target,
        device=device,
        inputs={
            "x": [eval_x.astype(np.float32)],
            "x_lens": [eval_lens.astype(np.int32)],
        },
        name="zipformer-mixed-stage0145-infer",
    )
    result["inference_job_id"] = infer_job.job_id
    result["inference_url"] = infer_job.url
    ist = poll(infer_job, "inference")
    result["inference_status"] = ist.code
    result["inference_message"] = ist.message
    if ist.code == "SUCCESS":
        hw_data = infer_job.download_output_data()
        hw = np.array(hw_data["output_0"][0])
        hw_T = int(np.array(hw_data["output_1"][0]).reshape(-1)[0])
        hw = hw[:, :hw_T, :]
        T = min(fp.shape[1], hw.shape[1])
        result["hardware_cos"] = float(cos_sim(fp[:, :T, :], hw[:, :T, :]))
        result["hardware_valid_T"] = int(hw_T)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {RESULT_JSON}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
