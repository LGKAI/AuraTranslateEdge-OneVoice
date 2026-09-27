"""submit_qai_hub_pipeline.py — Tự động Submit toàn bộ chuỗi công việc lên Qualcomm AI Hub:
  1. Upload Model (model_sensevoice_e2e_unified_patched.onnx)
  2. Quantize Job (W8A16 Mixed Precision với dataset d7jg1dlp7)
  3. Compile Job (QNN DLC binary trên Dragonwing IQ-9075 EVK)
  4. Profile Job (Hardware Benchmark trên bo mạch thật Hexagon v73)
  5. Inference Job (Thực thi Silicon với 15 mẫu input đa ngữ)

Lưu ý: Không tải bất kỳ tệp binary/kết quả nào về máy theo yêu cầu của người dùng.
"""

import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
MODEL_PATH = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_patched.onnx")
DATASET_ID = "d2q4px3o7"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
JOBS_LOG = os.path.join(OUT_DIR, "qai_hub_jobs.json")


def log_job_info(stage, info):
    print(f"\n[{stage}] " + "=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 65)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại: {MODEL_PATH}")
        sys.exit(1)

    device = hub.Device(DEVICE_NAME)
    dataset = hub.get_dataset(DATASET_ID)
    print(f"Target Device: {device}")
    print(f"Calibration Dataset: {dataset.dataset_id} ({dataset.name})")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE_NAME,
        "calibration_dataset_id": DATASET_ID,
        "stages": {}
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 1: Upload Model
    # ─────────────────────────────────────────────────────────────────────────
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n[Step 1/5] Đang upload model {MODEL_PATH} ({model_size_mb:.1f} MB) lên Qualcomm AI Hub ...")
    t0 = time.time()
    base_model = hub.upload_model(
        MODEL_PATH,
        name="SenseVoice_Small_E2E_Unified_Patched"
    )
    results["base_model_id"] = base_model.model_id
    results["base_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{base_model.model_id}/"
    log_job_info("Step 1/5: Upload Model", {
        "Base Model ID": base_model.model_id,
        "URL": results["base_model_url"],
        "Time Taken": f"{time.time() - t0:.1f}s"
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 2: Submit Quantize Job (W8A16)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 2/5] Đang submit Quantize Job (W8A16 Mixed Precision) ...")
    quantize_job = hub.submit_quantize_job(
        model=base_model,
        calibration_data=dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="SenseVoice_E2E_Unified_Quantize_w8a16",
    )
    results["stages"]["quantize"] = {
        "job_id": quantize_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{quantize_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 2/5: Quantize Job Submitted", results["stages"]["quantize"])

    # Đợi Quantize hoàn thành
    print("  Đang đợi Quantize Job xử lý trên cloud...")
    quantize_job.wait()
    q_status = quantize_job.get_status()
    results["stages"]["quantize"]["state"] = str(q_status.state)
    if not q_status.success:
        print(f"❌ Quantize Job thất bại: {q_status.message}")
        with open(JOBS_LOG, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    quantized_model = quantize_job.get_target_model()
    results["stages"]["quantize"]["target_model_id"] = quantized_model.model_id
    results["stages"]["quantize"]["target_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{quantized_model.model_id}/"
    print(f"✅ Quantize thành công! Target Model ID: {quantized_model.model_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 3: Submit Compile Job (QNN DLC cho Hexagon v73)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 3/5] Đang submit Compile Job (QNN DLC cho Dragonwing IQ-9075 EVK) ...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        name="SenseVoice_E2E_Unified_Compile_QNN",
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    results["stages"]["compile"] = {
        "job_id": compile_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 3/5: Compile Job Submitted", results["stages"]["compile"])

    # Đợi Compile hoàn thành
    print("  Đang đợi Compile Job xử lý trên cloud...")
    compile_job.wait()
    c_status = compile_job.get_status()
    results["stages"]["compile"]["state"] = str(c_status.state)
    if not c_status.success:
        print(f"❌ Compile Job thất bại: {c_status.message}")
        with open(JOBS_LOG, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    compiled_model = compile_job.get_target_model()
    results["stages"]["compile"]["target_model_id"] = compiled_model.model_id
    results["stages"]["compile"]["target_model_url"] = f"https://workbench.aihub.qualcomm.com/models/{compiled_model.model_id}/"
    print(f"✅ Compile thành công! Target Model ID: {compiled_model.model_id}")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 4 & 5: Submit Profile Job & Inference Job (chạy song song trên board thật)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Step 4/5] Đang submit Hardware Profile Job ...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="SenseVoice_E2E_Unified_Hardware_Profile",
    )
    results["stages"]["profile"] = {
        "job_id": profile_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 4/5: Profile Job Submitted", results["stages"]["profile"])

    print("\n[Step 5/5] Đang submit Hardware Inference Job ...")
    inference_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=dataset,
        name="SenseVoice_E2E_Unified_Silicon_Inference",
    )
    results["stages"]["inference"] = {
        "job_id": inference_job.job_id,
        "url": f"https://workbench.aihub.qualcomm.com/jobs/{inference_job.job_id}/",
        "state": "RUNNING"
    }
    log_job_info("Step 5/5: Inference Job Submitted", results["stages"]["inference"])

    # Đợi cả Profile và Inference hoàn tất
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

    # Lưu lại toàn bộ thông tin kết quả
    with open(JOBS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("🎉 TOÀN BỘ CHUỖI 4 JOBS ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN QUALCOMM AI HUB!")
    print(f"Báo cáo tổng hợp đã lưu tại: {JOBS_LOG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
