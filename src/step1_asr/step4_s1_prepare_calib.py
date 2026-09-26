"""Step 4 — Bước 3/6: Chuẩn bị calibration data cho quantization

Mục tiêu:
  - Extract fbank features từ 15 audio samples (5 En + 5 Zh + 5 Ko)
  - Lưu dưới dạng numpy arrays để dùng làm calibration_data trong submit_quantize_job
  - Calibration data dùng để ước lượng khoảng giá trị thực của mỗi activation tensor

Output:
  outputs/sensevoice_calib_data.npz  ← numpy arrays: speech, speech_lengths, language, textnorm

Cách chạy:
  python src/step1_asr/step4_s1_prepare_calib.py

Ghi chú từ meeting_prep_quantization.md:
  - 20-50 mẫu là đủ (bài học: tăng lên 600 không giúp thêm)
  - Quan trọng: phải cover đủ 3 ngôn ngữ để scale/zero-point đúng cho mọi lang token
  - Calibration data phải match đúng input format của ONNX model
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")
CALIB_PATH = os.path.join(ROOT, "outputs", "sensevoice_calib_data.npz")

# ONNX model uses fbank [B, T', 560] as speech input (NOT raw waveform)
# Separate language/textnorm embedding index inputs
# From export_meta.py dummy: language=[0], textnorm=[15]
LANG_EMBED_IDX = {"auto": 0, "en": 3, "zh": 4, "ko": 7}
TEXTNORM_ITN = 15   # use_itn=True
TEXTNORM_WOITN = 14


def extract_fbank_features(am, wav_path):
    """Extract fbank [1, T', 560] tu raw wav su dung frontend cua model."""
    import torch
    import soundfile as sf

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    wav_len = torch.tensor([wav.shape[0]], dtype=torch.int32)

    with torch.no_grad():
        frontend = am.kwargs.get('frontend')
        if frontend is not None:
            feats, feats_len = frontend(wav_t, wav_len)
        else:
            feats, feats_len = wav_t.unsqueeze(-1), wav_len
            
        if hasattr(am.model, 'normalize') and am.model.normalize is not None:
            feats, feats_len = am.model.normalize(feats, feats_len)
    return feats.numpy().astype(np.float32), feats_len.numpy().astype(np.int32), wav.shape[0] / sr


def extract_wav_inputs(am, wav_path, lang, onnx_input_names):
    """Chuan bi dict input cho 1 sample wav."""
    feats, feats_len, dur = extract_fbank_features(am, wav_path)
    lang_idx = LANG_EMBED_IDX.get(lang, 0)
    textnorm_idx = TEXTNORM_ITN

    # OVERRIDE: QAI Hub requires static shapes. We pad/truncate feats to exactly (1, 500, 560).
    MAX_LEN = 500
    current_len = feats.shape[1]
    padded_feats = np.zeros((1, MAX_LEN, 560), dtype=np.float32)
    
    if current_len > MAX_LEN:
        padded_feats[:, :MAX_LEN, :] = feats[:, :MAX_LEN, :]
        feats_len_val = np.array([MAX_LEN], dtype=np.int32)
    else:
        padded_feats[:, :current_len, :] = feats
        feats_len_val = np.array([current_len], dtype=np.int32)

    all_inputs = {
        "speech": padded_feats,
        "speech_lengths": feats_len_val,
        "language": np.array([lang_idx], dtype=np.int32),
        "textnorm": np.array([textnorm_idx], dtype=np.int32),
    }
    return {k: v for k, v in all_inputs.items() if k in onnx_input_names}, dur


def main():
    print("=" * 60)
    print("Step 4 — Bước 3: Prepare Calibration Data")
    print("=" * 60)

    onnx_path = os.path.join(OUT_DIR, "model_patched.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(OUT_DIR, "model.onnx")
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found. Chạy bước 1 và 2 trước.")
        sys.exit(1)

    print(f"[calib] Using ONNX: {onnx_path}")

    # Lấy input names từ ONNX
    try:
        import onnxruntime as ort
        sess_check = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_names = [inp.name for inp in sess_check.get_inputs()]
        print(f"[calib] ONNX input names: {onnx_input_names}")
    except Exception as e:
        print(f"[calib] ⚠️  Could not inspect ONNX inputs: {e}")
        onnx_input_names = ["speech", "speech_lengths", "language", "textnorm"]

    # Load FunASR model for frontend (fbank extraction)
    print("[calib] Loading SenseVoice model (needed for frontend/fbank extraction)...")
    from funasr import AutoModel
    try:
        am = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf",
                       device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    print("[calib] Model loaded.")

    # Load manifest
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    langs = ["en", "zh", "ko"]
    samples = [r for r in manifest if r["lang"] in langs]
    print(f"[calib] Found {len(samples)} En/Zh/Ko samples in manifest")

    # Thu thap calibration inputs
    calib_inputs_by_name = {name: [] for name in onnx_input_names}
    total_duration = 0.0
    processed = 0

    for row in samples:
        wav_path = os.path.join(ROOT, row["path"])
        lang = row["lang"]

        if not os.path.exists(wav_path):
            print(f"[calib] Missing: {wav_path}")
            continue

        try:
            inputs, dur = extract_wav_inputs(am, wav_path, lang, onnx_input_names)
            for name, arr in inputs.items():
                calib_inputs_by_name[name].append(arr)
            total_duration += dur
            processed += 1
            print(f"[calib] [{processed:2d}] {lang} {os.path.basename(wav_path)} "
                  f"dur={dur:.1f}s  shapes={[f'{k}:{v.shape}' for k,v in inputs.items()]}")
        except Exception as e:
            print(f"[calib] Failed {wav_path}: {e}")


    if processed == 0:
        print("[calib] ❌ No samples processed. Kiểm tra lại data/asr/ directory.")
        sys.exit(1)

    print(f"\n[calib] Processed {processed} samples, total audio: {total_duration:.1f}s")

    # Thống kê shape distribution (để debug nếu có vấn đề)
    print("\n[calib] Input shape statistics:")
    for name, arrs in calib_inputs_by_name.items():
        if arrs:
            shapes = [a.shape for a in arrs]
            print(f"  {name}: {len(arrs)} samples, shapes={set(str(s) for s in shapes)}")

    # Lưu calibration data
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)

    # Format cho qai_hub: list of dict, mỗi dict là 1 sample
    # Lưu dạng npz với prefix: input_{name}_{idx}
    save_dict = {}
    for name, arrs in calib_inputs_by_name.items():
        for idx, arr in enumerate(arrs):
            key = f"{name}_{idx:03d}"
            save_dict[key] = arr

    # Thêm metadata
    save_dict["__metadata_input_names"] = np.array(list(calib_inputs_by_name.keys()),
                                                     dtype=object)
    save_dict["__metadata_n_samples"] = np.array([processed])
    save_dict["__metadata_langs"] = np.array(
        [r["lang"] for r in samples if os.path.exists(os.path.join(ROOT, r["path"]))],
        dtype=object
    )

    np.savez(CALIB_PATH, **save_dict)
    file_size = os.path.getsize(CALIB_PATH) / 1e6
    print(f"\n[calib] ✅ Saved calibration data → {CALIB_PATH} ({file_size:.1f}MB)")
    print(f"  {processed} samples × {len(onnx_input_names)} inputs")

    # Verify có thể load lại
    loaded = np.load(CALIB_PATH, allow_pickle=True)
    print(f"[calib] Verify reload: {len(loaded.files)} arrays stored")

    print("\n" + "=" * 60)
    print("SUMMARY — Bước 3/6")
    print("=" * 60)
    print(f"  Samples processed:  {processed} (En: {sum(1 for r in samples if r['lang']=='en' and os.path.exists(os.path.join(ROOT,r['path'])))} / Zh: {sum(1 for r in samples if r['lang']=='zh' and os.path.exists(os.path.join(ROOT,r['path'])))} / Ko: {sum(1 for r in samples if r['lang']=='ko' and os.path.exists(os.path.join(ROOT,r['path'])))}")
    print(f"  Total audio:        {total_duration:.1f}s")
    print(f"  Output:             {CALIB_PATH}")
    print()
    print("Bước tiếp theo: chạy step4_s1_qai_hub_submit.py để submit jobs lên Qualcomm AI Hub")


if __name__ == "__main__":
    main()
