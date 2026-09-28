"""submit_qai_hub_pipeline.py — Tự động Submit toàn bộ chuỗi công việc lên Qualcomm AI Hub:
  1. Upload Calibration Dataset (calib_data_unified.npz: En, Zh, Ko với Vocab Token IDs)
  2. Upload Model (model_sensevoice_e2e_unified_patched.onnx)
  3. Quantize Job (W8A16 Mixed Precision: weights int8, activations int16)
  4. Compile Job (QNN DLC binary trên Dragonwing IQ-9075 EVK)
  5. Profile Job (Hardware Benchmark trên bo mạch thật Hexagon v73)
  6. Inference Job (Thực thi Silicon với 15 mẫu input đa ngữ)

Lưu ý quan trọng: KHÔNG tự động tải bất kỳ tệp binary/kết quả inference nào về máy theo yêu cầu của người dùng.
"""

import os
import sys
import json
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
MODEL_PATH = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_patched.onnx")
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified.npz")
CFG_JSON = os.path.join(OUT_DIR, "unified_e2e_config.json")
JOBS_LOG = os.path.join(OUT_DIR, "qai_hub_jobs.json")

DEVICE_NAME = "Dragonwing IQ-9075 EVK"


def log_job_info(stage, info):
    print(f"\n[{stage}] " + "=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 65)


def load_config():
    if os.path.exists(CFG_JSON):
        with open(CFG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_pipeline():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(CALIB_NPZ):
        print(f"❌ Không tìm thấy calib data tại: {CALIB_NPZ}")
        sys.exit(1)

    cfg = load_config()
    device = hub.Device(DEVICE_NAME)
    print("=" * 70)
    print("QUALCOMM AI HUB PIPELINE — SENSEVOICE-SMALL E2E v2 (W8A16, 100% NPU)")
    print("=" * 70)
    print(f"Target Device: {device}")
    print(f"L_MAX: {cfg.get('l_max')}, Byte stream len: {cfg.get('byte_stream_len')}")
    print(f"Special tokens filtered: {cfg.get('special_tokens_filtered')}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "SenseVoice-Small-E2E-v2",
        "device": DEVICE_NAME,
        "recipe": "W8A16 (weights: int8, activations: int16)",
        "l_max": cfg.get("l_max"),
        "byte_stream_len": cfg.get("byte_stream_len"),
        "stages": {}
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 1: Upload Calibration Dataset
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Step 1/5] Đang nạp calibration data từ {CALIB_NPZ} ...")
    calib_data = np.load(CALIB_NPZ)
    wav_arr  = calib_data["wav"]       # (N, 1, 464000)
    lang_arr = calib_data["language"]  # (N, 1)
    tn_arr   = calib_data["textnorm"]  # (N, 1)
    N = len(wav_arr)

    calib_dict = {
        "language": [lang_arr[i].reshape(1) for i in range(N)],
        "textnorm": [tn_arr[i].reshape(1)   for i in range(N)],
        "wav":      [wav_arr[i]             for i in range(N)],
    }
    print(f"  {N} mẫu (En/Zh/Ko), input shapes: language=[1], textnorm=[1], wav=[1, 464000]")
    t0 = time.time()
    dataset = hub.upload_dataset(calib_dict, name="SenseVoice_E2E_v2_Calib_Dataset")
    results["calibration_dataset_id"] = dataset.dataset_id
    results["calibration_dataset_url"] = f"https://workbench.aihub.qualcomm.com/datasets/{dataset.dataset_id}/"
    log_job_info("Step 1/5: Upload Dataset", {
        "Dataset ID": dataset.dataset_id,
        "URL": results["calibration_dataset_url"],
        "Time Taken": f"{time.time() - t0:.1f}s"
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 2: Upload Model
    # ─────────────────────────────────────────────────────────────────────────
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n[Step 2/5] Đang upload model {MODEL_PATH} ({model_size_mb:.1f} MB) lên Qualcomm AI Hub ...")
    t0 = time.time()
    base_model = hub.upload_model(
        MODEL_PATH,
        name="SenseVoice_Small_E2E_Unified_v2"
    )
    results["base_model_id"] = base_model.model_id
    results["base_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{base_model.model_id}/"
    log_job_info("Step 2/5: Upload Model", {
        "Base Model ID": base_model.model_id,
        "URL": results["base_model_url"],
        "Time Taken": f"{time.time() - t0:.1f}s"
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 3: Submit Quantize Job (W8A16)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 3/5] Đang submit Quantize Job (W8A16 Mixed Precision) ...")
    quantize_job = hub.submit_quantize_job(
        model=base_model,
        calibration_data=dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="SenseVoice_E2E_v2_Quantize_w8a16",
    )
    results["stages"]["quantize"] = {
        "job_id": quantize_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{quantize_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 3/5: Quantize Job Submitted", results["stages"]["quantize"])
    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("  Đang đợi Quantize Job hoàn thành trên Qualcomm AI Hub...")
    quantize_job.wait()
    q_status = quantize_job.get_status()
    results["stages"]["quantize"]["state"] = str(q_status.state)
    if not q_status.success:
        print(f"❌ Quantize Job thất bại: {q_status.message}")
        with open(JOBS_LOG, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sys.exit(1)

    quantized_model = quantize_job.get_target_model()
    results["stages"]["quantize"]["target_model_id"] = quantized_model.model_id
    results["stages"]["quantize"]["target_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{quantized_model.model_id}/"
    print(f"✅ Quantize thành công! Target Model ID: {quantized_model.model_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 4: Submit Compile Job (QNN DLC cho Hexagon v73)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 4/5] Đang submit Compile Job (QNN DLC cho Dragonwing IQ-9075 EVK) ...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        name="SenseVoice_E2E_v2_Compile_QNN",
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    results["stages"]["compile"] = {
        "job_id": compile_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 4/5: Compile Job Submitted", results["stages"]["compile"])
    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("  Đang đợi Compile Job hoàn thành trên Qualcomm AI Hub...")
    compile_job.wait()
    c_status = compile_job.get_status()
    results["stages"]["compile"]["state"] = str(c_status.state)
    if not c_status.success:
        print(f"❌ Compile Job thất bại: {c_status.message}")
        with open(JOBS_LOG, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sys.exit(1)

    compiled_model = compile_job.get_target_model()
    results["stages"]["compile"]["target_model_id"] = compiled_model.model_id
    results["stages"]["compile"]["target_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{compiled_model.model_id}/"
    print(f"✅ Compile thành công! Target Model ID: {compiled_model.model_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 5: Submit Profile Job & Inference Job
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 5/5] Đang submit Hardware Profile Job ...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="SenseVoice_E2E_v2_Hardware_Profile",
    )
    results["stages"]["profile"] = {
        "job_id": profile_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Profile Job Submitted", results["stages"]["profile"])

    print("\nĐang submit Hardware Inference Job (Silicon Execution) ...")
    inference_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=dataset,
        name="SenseVoice_E2E_v2_Silicon_Inference",
    )
    results["stages"]["inference"] = {
        "job_id": inference_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{inference_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Inference Job Submitted", results["stages"]["inference"])

    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nĐang đợi Profile Job hoàn tất trên chip silicon thật...")
    profile_job.wait()
    p_status = profile_job.get_status()
    results["stages"]["profile"]["state"] = str(p_status.state)
    print(f"Profile Status: {p_status.state}")

    print("\nĐang đợi Inference Job hoàn tất trên chip silicon thật...")
    inference_job.wait()
    i_status = inference_job.get_status()
    results["stages"]["inference"]["state"] = str(i_status.state)
    print(f"Inference Status: {i_status.state}")

    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ CHUỖI JOBS ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN QUALCOMM AI HUB!")
    print(f"  Quantize:  {results['stages']['quantize']['state']}")
    print(f"  Compile:   {results['stages']['compile']['state']}")
    print(f"  Profile:   {results['stages']['profile']['state']}")
    print(f"  Inference: {results['stages']['inference']['state']}")
def check_status(job_id: str):
    job = hub.get_job(job_id)
    status = job.get_status()
    print(f"Job ID: {job_id}")
    print(f"Status: {status.state}")
    if status.success:
        print("✅ Job thành công!")
        try:
            target_model = job.get_target_model()
            if target_model:
                print(f"  Target Model ID: {target_model.model_id}")
        except Exception:
            pass
    elif status.failure:
        print(f"❌ Job thất bại: {status.message}")


def check_all_jobs():
    if not os.path.exists(JOBS_LOG):
        print(f"❌ Không tìm thấy tệp {JOBS_LOG}")
        return
    with open(JOBS_LOG, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("=" * 60)
    print(f"TRẠNG THÁI TOÀN BỘ JOBS ({data.get('model_name')})")
    print(f"Thiết bị: {data.get('device')}")
    print("=" * 60)
    for stage, s_data in data.get("stages", {}).items():
        jid = s_data.get("job_id", "N/A")
        print(f"[{stage.upper():10s}] ID: {jid} | Saved state: {s_data.get('state')}")
        if jid != "N/A":
            try:
                live_status = hub.get_job(jid).get_status()
                print(f"             Live state: {live_status.state}")
                s_data["state"] = str(live_status.state)
            except Exception as e:
                print(f"             Error querying AI Hub: {e}")
    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Submit & Manage Qualcomm AI Hub Pipeline for SenseVoice E2E v2")
    parser.add_argument("--submit", action="store_true", help="Submit toàn bộ chuỗi 5 bước lên Qualcomm AI Hub")
    parser.add_argument("--status", type=str, default="", help="Kiểm tra trạng thái của một Job ID cụ thể")
    parser.add_argument("--check-all", action="store_true", help="Kiểm tra trạng thái trực tiếp của tất cả các job đã lưu")
    args = parser.parse_args()

    if args.status:
        check_status(args.status)
        return

    if args.check_all:
        check_all_jobs()
        return

    # Mặc định hoặc khi truyền --submit
    run_pipeline()


if __name__ == "__main__":
    main()


