# -*- coding: utf-8 -*-
"""Tao calibration o domain FBANK (dau vao graph encoder+classifier) bang cach chay 25 mau
wav calibration da co qua frontend ONNX fp32 that."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
CALIB_WAV_NPZ = os.path.join(OUT_DIR, "calib_data_unified_v2_wide.npz")

sess_fe = ort.InferenceSession(os.path.join(OUT_DIR, "model_sv_frontend.onnx"), providers=["CPUExecutionProvider"])
calib = np.load(CALIB_WAV_NPZ)
n = len(calib["wav"])
print(f"n mau calibration: {n}")

fbank_list, sl_list, lang_list, tn_list = [], [], [], []
for i in range(n):
    wav = calib["wav"][i]; wav_len = calib["wav_len"][i]
    fbank, speech_lengths = sess_fe.run(None, {"wav": wav.astype(np.float32), "wav_len": wav_len.astype(np.int32)})
    fbank_list.append(fbank[0])
    sl_list.append(speech_lengths)
    lang_list.append(calib["language"][i])
    tn_list.append(calib["textnorm"][i])
    print(f"  {i}: wav_len={wav_len[0]} -> speech_lengths={speech_lengths[0]}")

out_path = os.path.join(OUT_DIR, "calib_data_fbank.npz")
np.savez_compressed(
    out_path,
    fbank=np.stack(fbank_list, axis=0),
    speech_lengths=np.stack(sl_list, axis=0),
    language=np.stack(lang_list, axis=0),
    textnorm=np.stack(tn_list, axis=0),
)
print(f"Da luu: {out_path}")
