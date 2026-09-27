"""inspect_inference_results.py — So sánh chi tiết kết quả Inference:
  - Ground Truth từ data/asr/manifest.json
  - ORT FP32 Model tham chiếu (model_sensevoice_e2e_unified_patched.onnx)
  - NPU W8A16 Hardware thực thi trên Qualcomm Dragonwing IQ-9075 EVK (dataset-d74ny1er2.h5)
"""

import os
import sys
import json
import h5py
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5_PATH = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx", "dataset-d74ny1er2.h5")
MANIFEST_PATH = os.path.join(ROOT, "data", "asr", "manifest.json")
MODEL_ONNX = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx", "model_sensevoice_e2e_unified_patched.onnx")
CALIB_NPZ = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx", "calib_data_unified.npz")


def main():
    if not os.path.exists(H5_PATH):
        print(f"❌ Không tìm thấy file: {H5_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it.get("lang") in ["zh", "en", "ko"]]
    total_samples = len(sv_items)

    print("=" * 100)
    print("SO SÁNH KẾT QUẢ INFERENCE: GROUND TRUTH vs ORT FP32 vs NPU W8A16 SILICON")
    print(f"Hardware Target: Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)")
    print(f"Dataset Output:  {H5_PATH}")
    print("=" * 100)

    f_h5 = h5py.File(H5_PATH, "r")
    calib = np.load(CALIB_NPZ)
    
    print("\nĐang khởi tạo ONNX Runtime FP32 để lấy kết quả tham chiếu ...")
    sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])

    for i in range(total_samples):
        item = sv_items[i]
        ref_text = item.get("transcript", "").strip()
        lang = item.get("lang", "unknown")

        # 1. ORT FP32
        ort_feed = {
            "wav": calib["wav"][i],
            "language": calib["language"][i],
            "textnorm": calib["textnorm"][i]
        }
        ort_out = sess.run(None, ort_feed)[0][0]
        ort_bytes = [int(b) for b in ort_out if b > 0]
        ort_text = bytes(ort_bytes).decode("utf-8", errors="replace").strip()

        # 2. NPU W8A16
        batch_key = f"data/0/batch_{i}"
        raw_arr = f_h5[batch_key][0]
        npu_bytes = [int(b) for b in raw_arr if b > 0]
        npu_text = bytes(npu_bytes).decode("utf-8", errors="replace").strip()

        print(f"\n[Mẫu #{i+1:02d} | Ngôn ngữ: {lang.upper()}]")
        print(f"  📖 Ground Truth : {ref_text}")
        print(f"  💻 ORT FP32     : {ort_text}")
        print(f"  ⚡ NPU W8A16    : {npu_text}")

    f_h5.close()
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
