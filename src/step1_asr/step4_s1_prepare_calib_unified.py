"""step4_s1_prepare_calib_unified.py — Chuẩn bị calibration data cho SenseVoice Unified E2E v2

Cải tiến v2:
  1. Sử dụng Vocab Token IDs thực tế của SenseVoice:
     - zh: 24884 (<|zh|>)
     - en: 24885 (<|en|>)
     - ko: 24896 (<|ko|>)
     - withitn: 25016 (<|withitn|>)
  2. Thứ tự lưu các keys trong npz được chuẩn hóa theo bảng chữ cái:
     ["language", "textnorm", "wav"] khớp 100% với model DLC và pipeline Qualcomm AI Hub.
  3. Dùng bucket-based padding về MAX_WAV_SAMPLES (464000).
"""

import os
import sys
import json
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "asr")
OUT_DIR  = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified.npz")

FS = 16000
FRAME_SAMPLES = 400  # 25ms * 16kHz

# Vocab Token IDs thực tế trong vocab SenseVoice (tokens.json)
VOCAB_LID_DICT = {
    "zh": 24884,  # <|zh|>
    "en": 24885,  # <|en|>
    "ko": 24896,  # <|ko|>
}
VOCAB_TEXTNORM_ITN = 25016  # <|withitn|>

# Query IDs tương ứng (tầng embed [16, 560] nội bộ)
QUERY_LID_DICT = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "ko": 12,
}
QUERY_TEXTNORM_ITN = 14

LID_DICT = VOCAB_LID_DICT
TEXTNORM_ITN = VOCAB_TEXTNORM_ITN

# Buckets — đồng bộ với step4_s1_export_sensevoice_e2e_unified.py
WAV_BUCKETS = sorted([
    3 * FS,                   # 48000
    6 * FS,                   # 96000
    10 * FS,                  # 160000
    15 * FS,                  # 240000
    20 * FS,                  # 320000
    29 * FS + FRAME_SAMPLES,  # 464400 -> cắt thành 464000
])
MAX_WAV_SAMPLES = 464000  # hard limit


def best_bucket(n_samples: int) -> int:
    """Chọn bucket nhỏ nhất >= n_samples."""
    for b in WAV_BUCKETS:
        if n_samples <= b:
            return min(b, MAX_WAV_SAMPLES)
    return MAX_WAV_SAMPLES


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it["lang"] in LID_DICT]

    wav_list  = []
    lang_list = []
    tn_list   = []

    print(f"[Calib v2] Xử lý {len(sv_items)} mẫu (En/Zh/Ko) với Vocab Token IDs thực tế ...")
    for item in sv_items:
        lang     = item["lang"]
        wav_path = os.path.join(ROOT, item["path"])
        wav, sr  = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        n_samples = len(wav)
        bucket    = best_bucket(n_samples)

        # Pad về MAX_WAV_SAMPLES (shape cố định của model)
        # Phần từ 0..bucket là audio thực, bucket..MAX là silence zeros
        if n_samples > MAX_WAV_SAMPLES:
            wav_pad = wav[:MAX_WAV_SAMPLES]
        else:
            wav_pad = np.pad(wav, (0, MAX_WAV_SAMPLES - n_samples))

        wav_pad = wav_pad.reshape(1, MAX_WAV_SAMPLES)

        v_lid = LID_DICT[lang]
        q_lid = QUERY_LID_DICT[lang]

        wav_list.append(wav_pad)
        lang_list.append(np.array([v_lid], dtype=np.int32))
        tn_list.append(np.array([TEXTNORM_ITN], dtype=np.int32))

        dur = item.get("duration_s", round(n_samples / FS, 2))
        print(f"  [{lang.upper()}] {os.path.basename(wav_path)} {dur}s | Vocab LID={v_lid} (Query LID={q_lid}) -> shape=[1,{MAX_WAV_SAMPLES}]")

    # Lưu thành npz theo thứ tự bảng chữ cái: language, textnorm, wav
    np.savez_compressed(
        CALIB_NPZ,
        language=np.stack(lang_list, axis=0),    # [N, 1]
        textnorm=np.stack(tn_list, axis=0),      # [N, 1]
        wav=np.stack(wav_list, axis=0),          # [N, 1, MAX_WAV_SAMPLES]
    )

    print(f"\n[Calib v2] ✅ Saved: {CALIB_NPZ}")
    data = np.load(CALIB_NPZ)
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")


if __name__ == "__main__":
    main()