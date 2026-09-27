"""step4_s1_prepare_calib_unified.py — Chuẩn bị calibration data cho mô hình SenseVoice Unified E2E

Calibration dataset chứa các mẫu âm thanh thật (En, Zh, Ko) với kích thước tĩnh:
  - wav: [1, 464000] float32
  - language: [1] int32
  - textnorm: [1] int32

Dữ liệu được lưu dạng .npz và .h5 chuẩn theo yêu cầu của Qualcomm AI Hub API (submit_quantize_job).
"""

import os
import sys
import json
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "asr")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified.npz")

MAX_WAV_SAMPLES = 464000
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_ITN = 14


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it["lang"] in LID_DICT]

    wav_list = []
    lang_list = []
    tn_list = []

    print(f"[Calib] Đang xử lý {len(sv_items)} mẫu âm thanh (En/Zh/Ko) ...")
    for item in sv_items:
        lang = item["lang"]
        wav_path = os.path.join(ROOT, item["path"])
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        if len(wav) > MAX_WAV_SAMPLES:
            wav_pad = wav[:MAX_WAV_SAMPLES]
        else:
            wav_pad = np.pad(wav, (0, MAX_WAV_SAMPLES - len(wav)))
        wav_pad = wav_pad.reshape(1, MAX_WAV_SAMPLES)

        wav_list.append(wav_pad)
        lang_list.append(np.array([LID_DICT[lang]], dtype=np.int32))
        tn_list.append(np.array([TEXTNORM_ITN], dtype=np.int32))

        print(f"  + [{lang.upper()}] {os.path.basename(wav_path)} — {item['duration_s']}s")

    # Lưu thành định dạng npz
    # Mỗi key tương ứng với 1 input của đồ thị ONNX
    np.savez_compressed(
        CALIB_NPZ,
        wav=np.stack(wav_list, axis=0),         # [N, 1, 464000]
        language=np.stack(lang_list, axis=0),   # [N, 1]
        textnorm=np.stack(tn_list, axis=0)      # [N, 1]
    )

    print(f"\n✅ Đã lưu tập calibration data: {CALIB_NPZ}")
    data = np.load(CALIB_NPZ)
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")


if __name__ == "__main__":
    main()
