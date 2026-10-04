# -*- coding: utf-8 -*-
"""Chay model GOP (combined, fp16-architecture nhung chay o CPU fp32 qua ONNXRuntime) tren
5 mau en, so sanh voi baseline goc (7.56%) va ket qua NPU that (12.78%) de xac dinh: gap
nam o ban than graph/encoder fp16, hay o nhieu phan cung NPU that con sot lai."""
import sys, os, json, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import jiwer

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")
ONNX_PATH = os.path.join(OUT_DIR, "model_sv_combined_single_inline.onnx")

LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14
MAX_WAV_SAMPLES = 464000
FS = 16000

manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
en_items = [it for it in manifest if it["lang"] == "en"]

def decode_bytes(arr):
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    raw = bytes(int(b) & 0xFF for b in arr)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()

def norm_wer(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()

print("Loading combined model qua ONNXRuntime CPU (fp32)...", flush=True)
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print("Input names:", [i.name for i in sess.get_inputs()], flush=True)

import soundfile as sf
results = []
for item in en_items:
    wav_path = os.path.join(ROOT, item["path"].replace("/", os.sep))
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == FS, f"sr={sr}"
    true_len = min(len(wav), MAX_WAV_SAMPLES)
    wav_padded = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)
    wav_padded[:true_len] = wav[:true_len]
    wav_len = np.array([true_len], dtype=np.int32)
    lang = np.array([LID_DICT["en"]], dtype=np.int32)
    tn = np.array([TEXTNORM_WITHITN], dtype=np.int32)

    out = sess.run(None, {
        "wav": wav_padded.reshape(1, MAX_WAV_SAMPLES),
        "wav_len": wav_len,
        "language": lang,
        "textnorm": tn,
    })
    hyp = decode_bytes(out[0])
    ref = item["transcript"]
    wer = jiwer.wer(norm_wer(ref), norm_wer(hyp)) if norm_wer(hyp) else 1.0
    results.append(wer)
    print(f"[{os.path.basename(item['path'])}] WER={wer*100:.1f}%  HYP={hyp[:80]!r}", flush=True)

print(f"\n=== EN avg WER (combined graph, CPU fp32): {np.mean(results)*100:.2f}% ===", flush=True)
print("So sanh: baseline goc fp32 = 7.56% | NPU that (sau fix) = 12.78%", flush=True)
