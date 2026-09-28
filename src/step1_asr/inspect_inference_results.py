"""inspect_inference_results.py — So sánh chi tiết kết quả Inference:
  - Ground Truth từ data/asr/manifest.json
  - ORT FP32 Model tham chiếu (model_sensevoice_e2e_unified_patched.onnx)
  - NPU W8A16 Hardware thực thi trên Qualcomm Dragonwing IQ-9075 EVK (dataset-d70qx6e09.h5)
"""

import os
import sys
import glob
import json
import re
import argparse
import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
MANIFEST_PATH = os.path.join(ROOT, "data", "asr", "manifest.json")
MODEL_ONNX = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_patched.onnx")
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified.npz")
DEFAULT_H5 = os.path.join(OUT_DIR, "dataset-d70qx6e09.h5")
DEFAULT_JSON_OUT = os.path.join(OUT_DIR, "inference_results_v2.json")


def find_latest_h5():
    if os.path.exists(DEFAULT_H5):
        return DEFAULT_H5
    h5_files = glob.glob(os.path.join(OUT_DIR, "dataset-*.h5"))
    if h5_files:
        h5_files.sort(key=os.path.getmtime, reverse=True)
        return h5_files[0]
    return DEFAULT_H5


def main():
    parser = argparse.ArgumentParser(description="Inspect & compare SenseVoice NPU inference results.")
    parser.add_argument("--h5", type=str, default="", help="Đường dẫn file .h5 output từ Qualcomm AI Hub")
    parser.add_argument("--export-json", type=str, default=DEFAULT_JSON_OUT, help="Xuất kết quả giải mã ra file JSON")
    parser.add_argument("--skip-ort", action="store_true", help="Bỏ qua so sánh ORT FP32 (để chạy nhanh)")
    args = parser.parse_args()

    h5_path = args.h5 if args.h5 else find_latest_h5()
    if not os.path.exists(h5_path):
        print(f"❌ Không tìm thấy file kết quả H5: {h5_path}")
        print(f"Vui lòng tải file dataset từ Qualcomm AI Hub về đặt tại {OUT_DIR}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it.get("lang") in ["en", "zh", "ko"]]
    total_samples = len(sv_items)

    print("=" * 100)
    print("SO SÁNH KẾT QUẢ INFERENCE: GROUND TRUTH vs ORT FP32 vs NPU W8A16 SILICON")
    print(f"Hardware Target : Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)")
    print(f"Dataset File    : {h5_path}")
    print("=" * 100)

    f_h5 = h5py.File(h5_path, "r")
    data_group = f_h5["data"]["0"]

    sess = None
    calib = None
    if not args.skip_ort and os.path.exists(MODEL_ONNX) and os.path.exists(CALIB_NPZ):
        try:
            import onnxruntime as ort
            print("\nĐang khởi tạo ONNX Runtime FP32 để lấy kết quả tham chiếu ...")
            sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
            calib = np.load(CALIB_NPZ)
        except Exception as e:
            print(f"⚠️  Không thể nạp ORT FP32: {e}")
            sess = None

    results = []
    exact_matches = 0

    for i in range(total_samples):
        item = sv_items[i]
        ref_text = item.get("transcript", "").strip()
        lang = item.get("lang", "unknown")
        batch_key = f"batch_{i}"

        # 1. ORT FP32 (nếu có)
        ort_text = "N/A"
        if sess is not None and calib is not None:
            ort_feed = {
                "language": calib["language"][i].reshape(1),
                "textnorm": calib["textnorm"][i].reshape(1),
                "wav":      calib["wav"][i].reshape(1, -1),
            }
            ort_out = sess.run(None, ort_feed)[0][0]
            ort_bytes = [int(b) for b in ort_out if b > 0]
            ort_text = bytes(ort_bytes).decode("utf-8", errors="replace").strip()

        # 2. NPU W8A16
        raw_arr = data_group[batch_key][0]
        npu_bytes = [int(b) for b in raw_arr if b > 0]
        npu_text = bytes(npu_bytes).decode("utf-8", errors="replace").strip()

        # Check match
        ref_norm = re.sub(r"[^\w\s]", "", ref_text).replace(" ", "").lower()
        npu_norm = re.sub(r"[^\w\s]", "", npu_text).replace(" ", "").lower()
        is_exact = (ref_norm == npu_norm) and len(ref_norm) > 0
        if is_exact:
            exact_matches += 1

        if npu_text in [".", "。", ""]:
            err_type = "Blank Dominance / Silence collapse (bị câm hoàn toàn)"
        elif len(npu_text) < 5 and len(ref_text) > 10:
            err_type = "Severe Truncation (ngắt cụt nặng)"
        else:
            err_type = "Acoustic Degradation (lệch âm vị/từ vựng)"

        entry = {
            "index": i + 1,
            "batch_key": batch_key,
            "lang": lang,
            "audio_file": item["path"],
            "duration_s": item["duration_s"],
            "ground_truth": ref_text,
            "npu_decoded_text": npu_text,
            "raw_byte_length": len(npu_bytes),
            "match_status": "EXACT_MATCH" if is_exact else "MISMATCH",
            "error_type": err_type,
        }
        if sess is not None:
            entry["ort_fp32_text"] = ort_text
        results.append(entry)

        print(f"\n[Mẫu #{i+1:02d} | Ngôn ngữ: {lang.upper()}]")
        print(f"  📖 Ground Truth : {ref_text}")
        if sess is not None:
            print(f"  💻 ORT FP32     : {ort_text}")
        print(f"  ⚡ NPU W8A16    : {npu_text}")
        print(f"  Bytes: {len(npu_bytes)} | Đánh giá: {err_type}")

    f_h5.close()

    match_rate = round((exact_matches / total_samples) * 100, 2)
    print("\n" + "=" * 100)
    print(f"TỔNG KẾT: Khớp chính xác: {exact_matches}/{total_samples} ({match_rate}%)")
    print("=" * 100)

    if args.export_json:
        export_data = {
            "job_id": "jgjr6q6ep",
            "dataset_id": os.path.splitext(os.path.basename(h5_path))[0].replace("dataset-", ""),
            "target_device": "Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)",
            "precision": "W8A16 Mixed Precision",
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "exact_match_rate_percent": match_rate,
            "results": results
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Đã xuất kết quả JSON chi tiết: {args.export_json}")


if __name__ == "__main__":
    main()
