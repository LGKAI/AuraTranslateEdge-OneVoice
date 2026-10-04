"""Step 4 -- compile + profile NLLB-200-distilled-600M's encoder on real
QCS6490 hardware. First real MT hardware attempt in this project (see
step4.md). The decoder (with-past) export came out ~5.8GB on disk
(likely untied embeddings x large 256k-token NLLB vocab inflating the
LM-head/embedding matrices) -- too large to practically upload/compile
in this pass, so this script covers the encoder only; the decoder is
left as documented future work.
"""
import os

import numpy as np
import qai_hub as hub
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
MODEL_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask.onnx")

FIXED_SEQ_LEN = 64
MODEL_ID = "facebook/nllb-200-distilled-600M"


def build_calibration():
    import json
    tok = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="vie_Latn")
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_sentences = [r["vi"] for r in manifest][:8]

    input_ids_list, bias_list = [], []
    for text in vi_sentences:
        enc = tok(text, return_tensors="np", padding="max_length",
                   truncation=True, max_length=FIXED_SEQ_LEN)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        input_ids_list.append(ids)
        # host-computed additive attention bias -- same formula as
        # fix_nllb_mask_hostcompute.py's host_mask(), -30000 (not -3.4e38,
        # see step4.md) verified bit-exact for real padding patterns.
        # attention_mask ITSELF is not fed to the graph -- it became an
        # orphaned input (zero remaining consumers) once the mask-
        # construction branch that used to consume it was removed, and
        # a real compile attempt caught the resulting mismatch directly:
        # "[QNN_CPU] Expected number of inputs for Graph is 2 instead 3
        # provided" -- fixed by dropping it from the graph's declared
        # inputs too (see fix_nllb_mask_hostcompute.py).
        inv = (1 - mask).astype(np.float32)
        bias = np.broadcast_to((inv * -30000.0)[:, None, None, :],
                                (1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)).copy()
        bias_list.append(bias)
        real_len = int(mask.sum())
        print(f"[nllb_calib] '{text[:35]}...' -> {real_len} real tokens (windowed to {FIXED_SEQ_LEN})")

    # key order matches model.graph.input's actual order (input_ids,
    # /Where_1_output_0) -- AI Hub matches calibration data to graph
    # inputs positionally, not by name (see step4.md SS4e/SS4g).
    dataset = hub.upload_dataset(
        {"input_ids": input_ids_list, "/Where_1_output_0": bias_list},
        name="nllb_encoder_hostmask_calibration_v2",
    )
    print(f"[nllb_calib] dataset id: {dataset.dataset_id}")
    return dataset.dataset_id


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[nllb_encoder] device: {device.name}")
    print(f"[nllb_encoder] model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.1f}MB proto "
          f"+ external data)")

    calib_id = os.environ.get("NLLB_CALIB_ID") or build_calibration()
    calib = hub.get_dataset(calib_id)

    input_specs = {
        "input_ids": ((1, FIXED_SEQ_LEN), "int64"),
        "/Where_1_output_0": ((1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN), "float32"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[nllb_encoder] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=MODEL_PATH, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[nllb_encoder] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[nllb_encoder] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[nllb_encoder] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[nllb_encoder] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[nllb_encoder] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[nllb_encoder] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[nllb_encoder] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[nllb_encoder] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
