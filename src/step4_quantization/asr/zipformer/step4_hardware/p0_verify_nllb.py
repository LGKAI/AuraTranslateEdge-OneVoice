"""P0 -- numeric ground-truth check for the NLLB encoder (job j579o33lg,
step4.md SS4g). Until now only compile+profile (timing) was ever measured,
never real output correctness. Runs a real Vietnamese sentence through the
local fp32 (host-mask) ONNX graph and the real compiled hardware target,
compares via cosine similarity / max_abs_diff.
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
COMPILE_JOB = "j579o33lg"
FIXED_SEQ_LEN = 64
MODEL_ID = "facebook/nllb-200-distilled-600M"


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    print(f"[p0_nllb] real sentence: '{text}'")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="vie_Latn")
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    real_len = int(mask.sum())
    print(f"[p0_nllb] {real_len} real tokens (windowed to {FIXED_SEQ_LEN})")

    inv = (1 - mask).astype(np.float32)
    bias = np.broadcast_to((inv * -30000.0)[:, None, None, :], (1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)).copy()

    print(f"[p0_nllb] local fp32 run: {MODEL_PATH}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    local_out = sess.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]
    print(f"[p0_nllb] local encoder_out shape={local_out.shape}")

    device = hub.Device(DEVICE_NAME)
    target_model = hub.get_job(COMPILE_JOB).get_target_model()
    print(f"[p0_nllb] submitting REAL inference job on {device.name} ...")
    # compiled with --truncate_64bit_io: real hardware needs int32 for the
    # declared int64 input_ids (confirmed via a real "Cannot assign data
    # from unexpected type. Expected int32, got int64" error)
    ids_i32 = ids.astype(np.int32)
    job = hub.submit_inference_job(
        model=target_model, device=device,
        inputs={"input_ids": [ids_i32], "/Where_1_output_0": [bias]}, name="p0-nllb-encoder")
    print(f"[p0_nllb] job: {job.job_id}  {job.url}")
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        print(f"[p0_nllb] inference FAILED: {status.message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    print(f"[p0_nllb] hardware encoder_out shape={hw_out.shape}")

    print(f"\n[p0_nllb] === RESULT ===")
    print(f"[p0_nllb] cos_sim={cos_sim(local_out, hw_out):.4f}  max_abs_diff={max_abs_diff(local_out, hw_out):.4e}")
    print(f"[p0_nllb] local: mean={local_out.mean():.4f} std={local_out.std():.4f}")
    print(f"[p0_nllb] hw:    mean={hw_out.mean():.4f} std={hw_out.std():.4f}")


if __name__ == "__main__":
    main()
