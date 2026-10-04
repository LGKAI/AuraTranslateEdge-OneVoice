"""Muui 2 -- decisive discriminating test: compile Supertonic's text_encoder
(ZERO graph surgery -- pristine original ONNX, see step4.md SS4f) at fp16 on
Dragonwing IQ-9075 EVK (Hexagon v73, htp-supports-fp16:true) and check signal
preservation via the same cosine-similarity method used throughout.

This isolates whether the QNN converter/HTP execution path itself is clean
on an untouched graph: if fp16 gives cos_sim ~1.0 here (vs the 0.7339 seen
at int8 with only 4 calibration samples), that confirms the QNN pipeline is
fundamentally sound and all prior corruption traces back to OUR OWN graph
surgery + quantization choices (host-mask outlier values, insufficient
calibration, etc) -- not a QNN-level bug. If it's STILL degraded even on a
clean graph at fp16, that would point to something more fundamental in how
AI Hub/QNN handles this class of model, regardless of anything we've done.
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
MODEL_PATH = os.path.join(SRC_DIR, "text_encoder.onnx")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
FIXED_TEXT_LEN = 80


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


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[st_textenc_fp16] device: {device.name}")
    print(f"[st_textenc_fp16] model: {MODEL_PATH} (ZERO graph surgery, pristine original)")

    input_specs = {
        "text_ids": ((1, FIXED_TEXT_LEN), "int64"),
        "style_ttl": ((1, 256), "float32"),  # placeholder, corrected below from real style shape
        "text_mask": ((1, 1, FIXED_TEXT_LEN), "float32"),
    }

    tts = TTS(auto_download=True)
    style = tts.get_voice_style("M1")
    input_specs["style_ttl"] = (tuple(style.ttl.shape), "float32")
    print(f"[st_textenc_fp16] style.ttl shape: {style.ttl.shape}")

    reuse_job = os.environ.get("REUSE_COMPILE_JOB")
    if reuse_job:
        print(f"[st_textenc_fp16] reusing compile job {reuse_job}")
        target_model = hub.get_job(reuse_job).get_target_model()
    else:
        options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type float16 --quantize_io"
        print(f"[st_textenc_fp16] compile options: {options}")
        compile_job = hub.submit_compile_job(
            model=MODEL_PATH, device=device, input_specs=input_specs, options=options)
        print(f"[st_textenc_fp16] compile job: {compile_job.job_id}  {compile_job.url}")
        compile_job.wait()
        status = compile_job.get_status()
        print(f"[st_textenc_fp16] compile status: {status}")
        if status.code != "SUCCESS":
            print(f"[st_textenc_fp16] compile FAILED: {status.message}")
            return
        target_model = compile_job.get_target_model()
    print(f"[st_textenc_fp16] compile SUCCESS, target_model_id={target_model.model_id}")

    # --- real-input diagnostic ---
    text_processor = tts.model.text_processor
    with open(MANIFEST, encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    print(f"[st_textenc_fp16] test sentence: '{text}'")
    raw_ids, raw_mask = text_processor([text], "vi")
    ids, mask = pad_text(raw_ids, raw_mask, FIXED_TEXT_LEN)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    local_out = sess.run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask})[0]
    print(f"[st_textenc_fp16] local fp32 output shape={local_out.shape}")

    # compiled with --truncate_64bit_io: real hardware needs int32 for the
    # declared int64 text_ids input (confirmed via a real "Cannot assign
    # data from unexpected type. Expected int32, got int64" error)
    job = hub.submit_inference_job(
        model=target_model, device=device,
        inputs={"text_ids": [ids.astype(np.int32)], "style_ttl": [style.ttl], "text_mask": [mask]},
        name="st-textenc-fp16-test")
    print(f"[st_textenc_fp16] inference job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[st_textenc_fp16] inference FAILED: {job.get_status().message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    print(f"[st_textenc_fp16] hardware output shape={hw_out.shape}")

    cs = cos_sim(local_out, hw_out)
    print(f"\n[st_textenc_fp16] === RESULT ===")
    print(f"[st_textenc_fp16] cos_sim={cs:.4f}  max_abs_diff={max_abs_diff(local_out, hw_out):.4e}")
    print(f"[st_textenc_fp16] local: mean={local_out.mean():.4f} std={local_out.std():.4f}")
    print(f"[st_textenc_fp16] hw:    mean={hw_out.mean():.4f} std={hw_out.std():.4f}")
    if cs > 0.99:
        print("[st_textenc_fp16] *** QNN pipeline is CLEAN on an untouched graph -- "
              "all corruption traces back to OUR graph surgery / quantization choices ***")
    elif cs > 0.9:
        print("[st_textenc_fp16] much better than int8 (0.7339) but not perfect -- "
              "QNN pipeline mostly clean, some residual precision loss")
    else:
        print("[st_textenc_fp16] STILL degraded even on a clean graph at fp16 -- "
              "points to something more fundamental in the QNN pipeline itself")


if __name__ == "__main__":
    main()
