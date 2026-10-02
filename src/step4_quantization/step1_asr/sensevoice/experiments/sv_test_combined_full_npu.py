# -*- coding: utf-8 -*-
"""Test model GOP (combined single-graph, mnlz5pg3q) tren bo 15 mau test THAT tren NPU,
de biet chinh xac WER/CER cua chinh model dang dung trong demo (chua tung do truoc do -
truoc gio chi do model 2-graph tach roi)."""
import sys, os, json, re, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jiwer
import qai_hub as hub

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

COMBINED_TARGET_MODEL_ID = "mnlz5pg3q"
MAX_WAV_SAMPLES = 464000
FS = 16000
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14

device = hub.Device(DEVICE_NAME)

manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
sv_items = [it for it in manifest if it["lang"] in ("en", "zh", "ko")]

print("[1] Chuan bi dataset 15 mau that (wav goc, chua qua buoc trung gian nao)...", flush=True)
import soundfile as sf
wav_list, wav_len_list, lang_list, tn_list = [], [], [], []
for item in sv_items:
    wav_path = os.path.join(ROOT, item["path"].replace("/", os.sep))
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == FS
    true_len = min(len(wav), MAX_WAV_SAMPLES)
    wav_padded = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)
    wav_padded[:true_len] = wav[:true_len]
    wav_list.append(wav_padded.reshape(1, MAX_WAV_SAMPLES))
    wav_len_list.append(np.array([true_len], dtype=np.int32))
    lang_list.append(np.array([LID_DICT[item["lang"]]], dtype=np.int32))
    tn_list.append(np.array([TEXTNORM_WITHITN], dtype=np.int32))

print("[2] Upload dataset + submit inference tren NPU that (model GOP mnlz5pg3q)...", flush=True)
target = hub.get_model(COMBINED_TARGET_MODEL_ID)
dataset = hub.upload_dataset({
    "wav": wav_list, "wav_len": wav_len_list, "language": lang_list, "textnorm": tn_list,
})
print("  dataset_id=", dataset.dataset_id, flush=True)
infer_job = hub.submit_inference_job(model=target, device=device, inputs=dataset)
print("  infer_job_id=", infer_job.job_id, flush=True)
infer_job.wait()
print("  state:", infer_job.get_status().state, flush=True)

out = infer_job.download_output_data()
key = list(out.keys())[0]
outputs = out[key]

def decode_bytes(arr):
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    raw = bytes(int(b) & 0xFF for b in arr)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()

def norm_wer(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()

def cer(ref, hyp):
    ref_c = re.sub(r"\s+", "", ref)
    hyp_c = re.sub(r"\s+", "", hyp)
    return jiwer.cer(ref_c, hyp_c) if ref_c else 1.0

results = {"en": [], "zh": [], "ko": []}
for item, out_arr in zip(sv_items, outputs):
    hyp = decode_bytes(out_arr)
    ref = item["transcript"]
    lang = item["lang"]
    metric = cer(ref, hyp) if lang == "zh" else (jiwer.wer(norm_wer(ref), norm_wer(hyp)) if norm_wer(hyp) else 1.0)
    results[lang].append(metric)
    print(f"[{lang}] {os.path.basename(item['path'])} {'CER' if lang=='zh' else 'WER'}={metric*100:5.1f}%  HYP={hyp[:70]!r}", flush=True)

print("\n=== KET QUA MODEL GOP (single-graph, mnlz5pg3q) TREN NPU THAT ===", flush=True)
for lang, vals in results.items():
    print(f"  {lang}: n={len(vals)} {'CER' if lang=='zh' else 'WER'}={np.mean(vals)*100:.2f}%", flush=True)

with open(os.path.join(OUT_DIR, "combined_full_test_result.json"), "w", encoding="utf-8") as f:
    json.dump({lang: float(np.mean(vals)*100) for lang, vals in results.items()}, f, indent=2)
