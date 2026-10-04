# -*- coding: utf-8 -*-
"""Mo rong calibration de phu day du dai wav_len thuc te (gan toi max 464000),
tranh quantizer saturate tensor 'sum' noi bo cua acoustic model o gia tri thap (~180).
Dung audio that LAP LAI (tile) de tao cac do dai tong hop trai deu, khong dung silence
gia (tranh lam sai lech calibrate cac tensor khac)."""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
DATA_DIR = os.path.join(ROOT, "data", "asr")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")

MAX_WAV_SAMPLES = 464000
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14

manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
sv_items = [it for it in manifest if it["lang"] in LID_DICT]

def load_wav(item):
    wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav

wav_list, wav_len_list, lang_list, tn_list = [], [], [], []

# 1. 15 mau that goc (giu nguyen, dam bao khong mat coverage cu)
for item in sv_items:
    wav = load_wav(item)
    real_len = min(len(wav), MAX_WAV_SAMPLES)
    wav_pad = wav[:MAX_WAV_SAMPLES] if len(wav) > MAX_WAV_SAMPLES else np.pad(wav, (0, MAX_WAV_SAMPLES - len(wav)))
    wav_list.append(wav_pad.astype(np.float32))
    wav_len_list.append(real_len)
    lang_list.append(LID_DICT[item["lang"]])
    tn_list.append(TEXTNORM_WITHITN)
    print(f"  [goc] {item['lang']} {os.path.basename(item['path'])} len={real_len}")

# 2. Them cac muc do dai tong hop trai deu 50k -> 460k mau (~3s -> ~29s),
#    dung audio that LAP LAI (tile) de lap day, luan phien qua 3 ngon ngu.
target_lens = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 430000, 460000]
base_items = {"en": sv_items[0], "zh": sv_items[5], "ko": sv_items[10]}
langs_cycle = ["en", "zh", "ko"]

for k, target_len in enumerate(target_lens):
    lang = langs_cycle[k % 3]
    item = base_items[lang]
    wav = load_wav(item)
    n_tile = int(np.ceil(target_len / len(wav)))
    wav_tiled = np.tile(wav, n_tile)[:target_len]
    wav_pad = np.pad(wav_tiled, (0, MAX_WAV_SAMPLES - target_len)).astype(np.float32)
    wav_list.append(wav_pad)
    wav_len_list.append(target_len)
    lang_list.append(LID_DICT[lang])
    tn_list.append(TEXTNORM_WITHITN)
    print(f"  [tong hop] {lang} target_len={target_len} (tile audio that x{n_tile})")

npz_path = os.path.join(OUT_DIR, "calib_data_unified_v2_wide.npz")
np.savez_compressed(
    npz_path,
    wav=np.stack(wav_list, axis=0).reshape(len(wav_list), 1, MAX_WAV_SAMPLES),
    wav_len=np.array(wav_len_list, dtype=np.int32).reshape(-1, 1),
    language=np.array(lang_list, dtype=np.int32).reshape(-1, 1),
    textnorm=np.array(tn_list, dtype=np.int32).reshape(-1, 1),
)
print(f"\nDa luu: {npz_path}  (n={len(wav_list)} mau, wav_len range {min(wav_len_list)}-{max(wav_len_list)})")
