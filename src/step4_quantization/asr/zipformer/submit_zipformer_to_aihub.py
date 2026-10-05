# -*- coding: utf-8 -*-
"""Full pipeline submission for Zipformer-150M-CR-CTC Single Static DAG to Qualcomm AI Hub:
1. Upload calibration dataset
2. Quantize W16A16
3. Compile for Qualcomm Dragonwing IQ-9075 EVK (QNN DLC)
4. Profile on physical silicon (HTP Hexagon NPU v73)
5. Run silicon inference on all 15 test items (clean, snr5, snr0)
6. Evaluate WER and save results to results/hub_silicon_test_results.json and .csv
"""
import os, sys, time, json, re, csv
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"d:\ChuyenNganhAI\AuraTranslateEdge-OneVoice"
os.chdir(ROOT)

ONNX_PATH = os.path.join(ROOT, "outputs", "zip150_full_npu", "zip150_full_pipeline.onnx")
RESULTS_DIR = os.path.join(ROOT, "src", "step4_quantization", "asr", "zipformer", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
N_SAMPLES = 240240  # 15.015s @ 16kHz (exactly 1500 frames)

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()

def prep_wave(wav):
    n = min(wav.shape[0], N_SAMPLES)
    x = np.zeros((N_SAMPLES,), dtype=np.float32)
    x[:n] = wav[:n]
    n_frames = 1 + (n - 400) // 160
    return x, n_frames

def decode_output(byte_matrix, byte_len):
    out = bytearray()
    byte_matrix = np.asarray(byte_matrix).reshape(-1, 12)
    byte_len = np.asarray(byte_len).reshape(-1)
    for row, l in zip(byte_matrix, byte_len):
        l = int(l)
        if l <= 0:
            continue
        out += bytes(int(b) & 0xFF for b in row[:l])
    return out.decode("utf-8", errors="replace").strip()

def poll_job(job, label, interval=25):
    print(f"[{label}] Job {job.job_id} submitted: {job.url}")
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:120]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"):
            return st
        time.sleep(interval)

def main():
    print(f"=== Qualcomm AI Hub Deployment: Zipformer-150M-CR-CTC Single Static DAG ===")
    assert os.path.exists(ONNX_PATH), f"Model missing: {ONNX_PATH}"
    print(f"Model: {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")

    # 1. Load evaluation dataset
    manifest_path = os.path.join(ROOT, "data", "asr", "manifest.json")
    rows = [r for r in json.load(open(manifest_path, encoding="utf-8")) if r["lang"] == "vi"]
    items = []
    for tag in ("clean", "snr5", "snr0"):
        for r in rows:
            f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
            if os.path.exists(f):
                w, _ = sf.read(f, dtype="float32")
                items.append((tag, r["transcript"], w, os.path.basename(f)))
    print(f"Loaded {len(items)} evaluation samples across clean, snr5, snr0.")

    device = hub.Device(DEVICE_NAME)

    # 2. Compile directly to Native FP16 on Qualcomm NPU (No PTQ degradation, Zero-CPU architecture)
    print(f"\n--- Submitting Compile Job (Native FP16) for {DEVICE_NAME} ---")
    cjob = hub.submit_compile_job(
        model=ONNX_PATH,
        device=device,
        input_specs={
            "fb_raw_wave_flat": ((N_SAMPLES,), "float32"),
            "enc_x_lens": ((1,), "int64")
        },
        options="--target_runtime qnn_dlc --truncate_64bit_io",
        name="Zipformer150M_Full_FP16_Direct_Compile"
    )
    cst = poll_job(cjob, "COMPILE")
    if cst.code != "SUCCESS":
        sys.exit(f"Compile failed: {cst.message}")

    target_model = cjob.get_target_model()
    print(f"Target model ID: {target_model.model_id}")

    # 3. Profile on Hardware
    print(f"\n--- Submitting Profile Job on {DEVICE_NAME} ---")
    pjob = hub.submit_profile_job(
        model=target_model,
        device=device,
        name="Zipformer150M_Full_FP16_Profile"
    )

    # 4. Run Hardware Inference
    print(f"\n--- Submitting Silicon Inference Job (15 samples) ---")
    infer_inputs = {
        "fb_raw_wave_flat": [prep_wave(w)[0].astype(np.float32) for _, _, w, _ in items],
        "enc_x_lens": [np.array([prep_wave(w)[1]], np.int32) for _, _, w, _ in items]
    }
    ijob = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=infer_inputs,
        name="Zipformer150M_Full_FP16_Silicon_Inference"
    )

    pst = poll_job(pjob, "PROFILE")
    ist = poll_job(ijob, "INFER")

    # 7. Collect Profile Results
    profile_data = {}
    if pst.code == "SUCCESS":
        p_json = pjob.download_profile()
        es = p_json.get("execution_summary", {})
        profile_data = {
            "estimated_inference_time_ms": round(es.get("estimated_inference_time", 0) / 1000.0, 2),
            "first_load_time_sec": round(es.get("first_load_time", 0) / 1e6, 3),
            "peak_memory_mb": round(es.get("estimated_inference_peak_memory", 0) / 1e6, 2),
            "compute_unit_summary": es.get("compute_unit_summary", {})
        }
        print("\n--- Profile Summary ---")
        print(f"Latency: {profile_data['estimated_inference_time_ms']} ms")
        print(f"Peak Memory: {profile_data['peak_memory_mb']} MB")
        print(f"Compute Units: {profile_data['compute_unit_summary']}")

    # 8. Collect Inference Results
    inference_records = []
    if ist.code == "SUCCESS":
        out = ijob.download_output_data()
        bm_key = [k for k in out if 12 in np.asarray(out[k][0]).shape or np.asarray(out[k][0]).ndim > 1][0]
        bl_key = [k for k in out if k != bm_key][0]

        for (tag, ref, w, fname), bm, bl in zip(items, out[bm_key], out[bl_key]):
            hyp = decode_output(np.asarray(bm), np.asarray(bl))
            w_score = jiwer.wer(norm(ref), norm(hyp))
            inference_records.append({
                "file": fname,
                "tag": tag,
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(float(w_score), 4),
                "is_out_of_vocab": ("springboks" in ref)
            })

    # Summary WER
    wers_all = [r["wer"] for r in inference_records]
    wers_no_oov = [r["wer"] for r in inference_records if not r["is_out_of_vocab"]]
    by_cond = {}
    for tag in ("clean", "snr5", "snr0"):
        cond_wers = [r["wer"] for r in inference_records if r["tag"] == tag]
        cond_wers_no_oov = [r["wer"] for r in inference_records if r["tag"] == tag and not r["is_out_of_vocab"]]
        by_cond[tag] = {
            "all_wer": round(float(np.mean(cond_wers)) * 100, 2) if cond_wers else 0.0,
            "no_oov_wer": round(float(np.mean(cond_wers_no_oov)) * 100, 2) if cond_wers_no_oov else 0.0,
            "count": len(cond_wers)
        }

    overall_results = {
        "device": DEVICE_NAME,
        "quant_job": qjob.job_id,
        "compile_job": cjob.job_id,
        "profile_job": pjob.job_id,
        "infer_job": ijob.job_id,
        "quant_status": qst.code,
        "compile_status": cst.code,
        "profile_status": pst.code,
        "infer_status": ist.code,
        "hardware_profile": profile_data,
        "wer_summary": {
            "overall_all_wer": round(float(np.mean(wers_all)) * 100, 2) if wers_all else 0.0,
            "overall_no_oov_wer": round(float(np.mean(wers_no_oov)) * 100, 2) if wers_no_oov else 0.0,
            "by_condition": by_cond
        },
        "records": inference_records
    }

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "hub_silicon_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "hub_silicon_test_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tag", "wer", "is_out_of_vocab", "reference", "hypothesis"])
        for r in inference_records:
            writer.writerow([r["file"], r["tag"], r["wer"], r["is_out_of_vocab"], r["reference"], r["hypothesis"]])

    print("\n================ FINAL RESULTS ================")
    print(f"Quantize Job: {qjob.job_id} ({qjob.url})")
    print(f"Compile Job:  {cjob.job_id} ({cjob.url})")
    print(f"Profile Job:  {pjob.job_id} ({pjob.url})")
    print(f"Inference Job:{ijob.job_id} ({ijob.url})")
    print(f"\nHardware Latency: {profile_data.get('estimated_inference_time_ms', 'N/A')} ms")
    print(f"Peak Memory:      {profile_data.get('peak_memory_mb', 'N/A')} MB")
    print("\nWER by Condition (All 15 samples):")
    for tag, d in by_cond.items():
        print(f"  [{tag:5s}] All WER: {d['all_wer']:5.2f}% | No-OOV WER: {d['no_oov_wer']:5.2f}%")
    print(f"\nOverall: All WER = {overall_results['wer_summary']['overall_all_wer']:.2f}% | No-OOV WER = {overall_results['wer_summary']['overall_no_oov_wer']:.2f}%")
    print(f"\nSaved results to:\n- {json_path}\n- {csv_path}")

if __name__ == "__main__":
    main()
