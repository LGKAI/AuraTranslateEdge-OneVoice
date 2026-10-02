# -*- coding: utf-8 -*-
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
DATA_DIR = os.path.join(ROOT, "data", "asr")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_WAV_SAMPLES = 464000
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14

manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
sv_items = [it for it in manifest if it["lang"] in LID_DICT]

wav_list, wav_len_list, lang_list, tn_list = [], [], [], []
for item in sv_items:
    lang = item["lang"]
    path = os.path.join(ROOT, item["path"])
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    real_len = min(len(wav), MAX_WAV_SAMPLES)
    if len(wav) > MAX_WAV_SAMPLES:
        wav_pad = wav[:MAX_WAV_SAMPLES]
    else:
        wav_pad = np.pad(wav, (0, MAX_WAV_SAMPLES - len(wav)))
    wav_list.append(wav_pad.reshape(1, MAX_WAV_SAMPLES).astype(np.float32))
    wav_len_list.append(np.array([real_len], dtype=np.int32))
    lang_list.append(np.array([LID_DICT[lang]], dtype=np.int32))
    tn_list.append(np.array([TEXTNORM_WITHITN], dtype=np.int32))
    print(f"  + [{lang.upper()}] {os.path.basename(path)} real_len={real_len}")

npz_path = os.path.join(OUT_DIR, "calib_data_unified_fixed.npz")
np.savez_compressed(
    npz_path,
    wav=np.stack(wav_list, axis=0),
    wav_len=np.stack(wav_len_list, axis=0),
    language=np.stack(lang_list, axis=0),
    textnorm=np.stack(tn_list, axis=0),
)
print(f"\nDa luu calibration: {npz_path}")
data = np.load(npz_path)
for k in data.files:
    print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
