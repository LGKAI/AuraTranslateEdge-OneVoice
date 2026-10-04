# -*- coding: utf-8 -*-
"""Deploy and Profile Zipformer Encoder on Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73).
Reuses uploaded model mmxjjx0rq and dataset d7zn4qp67.
Compile -> Profile -> Inference -> Save Results.
"""
import os, sys, time, json
import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = r"d:\ChuyenNganhAI\AuraTranslateEdge-OneVoice"
os.chdir(ROOT)

MODEL_ID = "mmxjjx0rq"
DATASET_ID = "d7zn4qp67"
RESULTS_DIR = os.path.join(ROOT, "src", "step4_quantization", "step1_asr", "zipformer", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
FIXED_FRAMES = 1500

def compute_fbank(wav, sr=16000):
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = sr
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sr, wav.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)

def poll(job, label, interval=20):
    print(f"[{label}] Job {job.job_id}: {job.url}")
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:120]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"):
            return st
        time.sleep(interval)

def main():
    print(f"=== Qualcomm AI Hub: Zipformer Deployment on {DEVICE_NAME} ===")
    device = hub.Device(DEVICE_NAME)
    print(f"Target Device: {device.name} (Attributes: {device.attributes})")

    model = hub.get_model(MODEL_ID)
    dataset = hub.get_dataset(DATASET_ID)
    print(f"Using Model: {model.model_id} ({getattr(model, 'name', '')})")
    print(f"Using Calibration Dataset: {dataset.dataset_id} ({getattr(dataset, 'name', '')})")

    # 1. Compile Job
    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64")
    }
    options = "--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type int16 --quantize_io"
    print(f"\nSubmitting Compile Job: options='{options}' ...")
    cjob = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options,
        calibration_data=dataset,
        name="Zipformer_Encoder_INT16_IQ9075_Compile"
    )
    cst = poll(cjob, "COMPILE")
    if cst.code != "SUCCESS":
        sys.exit(f"Compile FAILED: {cst.message}")

    target_model = cjob.get_target_model()

    # 2. Profile Job
    print(f"\nSubmitting Profile Job on {DEVICE_NAME} ...")
    pjob = hub.submit_profile_job(
        model=target_model,
        device=device,
        name="Zipformer_Encoder_INT16_IQ9075_Profile"
    )

    # 3. Load test inputs for inference
    manifest_path = os.path.join(ROOT, "data", "asr", "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]

    test_x, test_lens = [], []
    for item in vi_items:
        wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
        feats = compute_fbank(wav, sr)
        n = feats.shape[0]
        padded = np.zeros((1, FIXED_FRAMES, 80), dtype=np.float32)
        padded[0, :min(n, FIXED_FRAMES)] = feats[:min(n, FIXED_FRAMES)]
        test_x.append(padded)
        test_lens.append(np.array([min(n, FIXED_FRAMES)], dtype=np.int32))

    # 4. Silicon Inference Job
    print(f"\nSubmitting Hardware Silicon Inference Job ({len(test_x)} samples) ...")
    ijob = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={
            "x": test_x,
            "x_lens": test_lens
        },
        name="Zipformer_Encoder_INT16_IQ9075_Inference"
    )

    pst = poll(pjob, "PROFILE")
    ist = poll(ijob, "INFER")

    profile_summary = {}
    if pst.code == "SUCCESS":
        p_json = pjob.download_profile()
        es = p_json.get("execution_summary", {})
        profile_summary = {
            "estimated_inference_time_ms": round(es.get("estimated_inference_time", 0) / 1000.0, 2),
            "first_load_time_sec": round(es.get("first_load_time", 0) / 1e6, 3),
            "peak_memory_mb": round(es.get("estimated_inference_peak_memory", 0) / 1e6, 2),
            "compute_unit_summary": es.get("compute_unit_summary", {})
        }
        print("\n--- Profile Execution Summary ---")
        print(f"Latency: {profile_summary['estimated_inference_time_ms']} ms")
        print(f"Peak Memory: {profile_summary['peak_memory_mb']} MB")
        print(f"Compute Unit Breakdown: {profile_summary['compute_unit_summary']}")

    results = {
        "device": DEVICE_NAME,
        "model": "Zipformer Encoder (Surgically Patched for QNN HTP)",
        "model_id": MODEL_ID,
        "compile_job": cjob.job_id,
        "compile_url": cjob.url,
        "compile_status": cst.code,
        "profile_job": pjob.job_id,
        "profile_url": pjob.url,
        "profile_status": pst.code,
        "infer_job": ijob.job_id,
        "infer_url": ijob.url,
        "infer_status": ist.code,
        "hardware_profile": profile_summary
    }

    res_json_path = os.path.join(RESULTS_DIR, "hub_silicon_test_results.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {res_json_path}")
    print("Deployment finished successfully!")

if __name__ == "__main__":
    main()
