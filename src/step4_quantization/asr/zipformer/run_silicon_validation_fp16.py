# -*- coding: utf-8 -*-
"""Run compilation monitoring, hardware profile, and 15-sample silicon inference
for the full 5-block Zipformer pipeline on Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73 HTP).
Zero-CPU architecture: NPU directly returns UTF-8 byte stream.
"""
import json, os, re, sys, time
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"d:\ChuyenNganhAI\AuraTranslateEdge-OneVoice"
os.chdir(ROOT)

COMPILE_JOB_ID = "jgd6x4oep"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
N_SAMPLES = 240240
RESULT_PATH = os.path.join(ROOT, "src", "step4_quantization", "asr", "zipformer", "results", "hub_silicon_test_results.json")
MANIFEST_PATH = os.path.join(ROOT, "data", "asr", "manifest.json")

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()

def prep_wave(wav):
    n = min(wav.shape[0], N_SAMPLES)
    x = np.zeros((N_SAMPLES,), dtype=np.float32)
    x[:n] = wav[:n]
    n_frames = 1 + (n - 400) // 160
    return x, n_frames

def decode_output(byte_matrix, byte_len):
    out = bytearray()
    for row, l in zip(byte_matrix, byte_len):
        l = int(l)
        if l <= 0:
            continue
        out += bytes(int(b) & 0xFF for b in row[:l])
    return out.decode("utf-8", errors="replace").strip()

def poll_job(job, label, every=15):
    print(f"[{label}] Polling job {job.job_id} ({job.name})...", flush=True)
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:120]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"):
            return st
        time.sleep(every)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    device = hub.Device(DEVICE_NAME)
    
    # 1. Poll compile job
    cjob = hub.get_job(COMPILE_JOB_ID)
    cst = poll_job(cjob, "COMPILE")
    if cst.code != "SUCCESS":
        print(f"Compile FAILED: {cst.message}", flush=True)
        # Save failure info
        res = {"device": DEVICE_NAME, "compile_job": COMPILE_JOB_ID, "compile_status": cst.code, "error": cst.message}
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        sys.exit(1)
        
    target_model = cjob.get_target_model()
    print(f"\nCompile SUCCESS! Target model ID: {target_model.model_id}\n", flush=True)

    # 2. Submit Profile Job
    print(f"--- Submitting Hardware Profile Job on {DEVICE_NAME} ---", flush=True)
    pjob = hub.submit_profile_job(
        model=target_model,
        device=device,
        name="Zipformer150M_Full_FP16_Profile"
    )
    print(f"Profile job: {pjob.job_id} ({pjob.url})", flush=True)

    # 3. Load 15 evaluation samples
    print("\n--- Loading Evaluation Samples ---", flush=True)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r["lang"] == "vi"]
        
    items = []
    for tag in ("clean", "snr5", "snr0"):
        for r in rows:
            fpath = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
            if os.path.exists(fpath):
                w, sr = sf.read(fpath, dtype="float32")
                items.append((tag, r["transcript"], w))
                
    print(f"Loaded {len(items)} audio samples for hardware inference.", flush=True)

    # 4. Submit Silicon Inference Job
    print(f"\n--- Submitting Silicon Inference Job ({len(items)} samples) ---", flush=True)
    infer_inputs = {
        "fb_raw_wave_flat": [x.astype(np.float32) for _, _, w in items for x, n in [prep_wave(w)]],
        "enc_x_lens": [np.array([n], np.int32) for _, _, w in items for x, n in [prep_wave(w)]]
    }
    
    ijob = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=infer_inputs,
        name="Zipformer150M_Full_FP16_Silicon_Inference"
    )
    print(f"Inference job: {ijob.job_id} ({ijob.url})", flush=True)

    # 5. Wait for Profile and Inference
    pst = poll_job(pjob, "PROFILE", every=15)
    ist = poll_job(ijob, "INFER", every=20)

    res = {
        "device": DEVICE_NAME,
        "compile_job": cjob.job_id,
        "compile_target_model": target_model.model_id,
        "profile_job": pjob.job_id,
        "profile_status": pst.code,
        "infer_job": ijob.job_id,
        "infer_status": ist.code,
    }

    if pst.code == "SUCCESS":
        profile_data = pjob.download_profile()
        es = profile_data.get("execution_summary", {})
        res["profile_metrics"] = {
            "estimated_inference_time_ms": es.get("estimated_inference_time", 0) / 1000.0,
            "first_load_time_sec": es.get("first_load_time", 0) / 1e6,
            "peak_memory_mb": es.get("estimated_inference_peak_memory", 0) / 1e6,
        }
        print(f"\nProfile metrics: {res['profile_metrics']}", flush=True)

    if ist.code == "SUCCESS":
        print("\nDownloading silicon output byte streams from Qualcomm NPU...", flush=True)
        out = ijob.download_output_data()
        
        # Identify output tensors
        bm_key = [k for k in out if np.asarray(out[k][0]).shape == (373, 12)][0]
        bl_key = [k for k in out if k != bm_key][0]
        
        predictions = []
        per_condition = {}
        
        print("\n================== SILICON RECOGNITION RESULTS ==================")
        for (tag, ref, w), bm, bl in zip(items, out[bm_key], out[bl_key]):
            hyp = decode_output(np.asarray(bm), np.asarray(bl))
            wer = jiwer.wer(norm(ref), norm(hyp))
            per_condition.setdefault(tag, []).append(wer)
            
            predictions.append({
                "condition": tag,
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(float(wer), 4)
            })
            print(f"[{tag:5s}] WER={wer*100:5.1f}% | REF: {ref}")
            print(f"        | HYP: {hyp}\n")

        res["by_condition"] = {t: {"wer": round(float(np.mean(v)), 4), "n": len(v)} for t, v in per_condition.items()}
        res["overall_wer"] = round(float(np.mean([p["wer"] for p in predictions])), 4)
        res["predictions"] = predictions
        print(f"Overall WER: {res['overall_wer']*100:.2f}% across {len(predictions)} samples.")
    else:
        print(f"Silicon inference FAILED: {ist.message}", flush=True)
        res["error"] = ist.message

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULT_PATH}")

if __name__ == "__main__":
    main()
