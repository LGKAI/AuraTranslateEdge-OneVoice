# -*- coding: utf-8 -*-
"""usage: python eval_hw.py <key>   NPU audio vs CPU fused audio (cosine) + ASR round trip WER/CER + profile."""
import sys, os, re, json, collections
import numpy as np, onnxruntime as ort, jiwer, soundfile as sf, qai_hub as hub
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding="utf-8")
key = sys.argv[1]; HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); J = json.load(open(os.path.join(HERE, "jobs.json")))[key]
SENT = json.load(open(os.path.join(HERE, "..", "data", "sentences.json"), encoding="utf-8"))
d = np.load(os.path.join(HERE, f"hw_{key}.npz")); W = d["output_0"].reshape(len(SENT), -1); LAT = d["output_1"].reshape(-1)
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
LID = {"zh": 3, "en": 4, "ko": 12}
cos = lambda a, b: float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
nw = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()
nc = lambda s: re.sub(r"\s+", "", re.sub(r"[^\w\s]", "", s.lower()))


def asr_text(w44, lang):
    w = resample_poly(w44.astype(np.float32), 16000, 44100).astype(np.float32); n = min(len(w), 464000); b = np.zeros(464000, np.float32); b[:n] = w[:n]
    o = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([n], np.int32), "language": np.array([LID[lang]], np.int32), "textnorm": np.array([14], np.int32)})
    return bytes(int(x) & 0xFF for x in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


rows = []; os.makedirs(os.path.join(HERE, "wav_npu"), exist_ok=True)
for i, (lang, text) in enumerate(SENT):
    lat = int(round(float(LAT[i]))); w = W[i][:lat * 3072]
    ref, sr = sf.read(os.path.join(HERE, "wav_cpu", f"{i:02d}_{lang}_fusedcpu.wav"), dtype="float32")
    n = min(len(w), len(ref)); h = asr_text(w, lang)
    r = {"lang": lang, "text": text, "lat_npu": lat, "lat_cpu": len(ref) // 3072, "cos_vs_cpu": cos(w[:n], ref[:n]), "peak": float(np.abs(w).max()), "nan": bool(np.isnan(W[i]).any()),
         "tail_max": float(np.abs(W[i][lat * 3072:]).max()) if W[i].size > lat * 3072 else 0.0, "hyp": h,
         "wer": jiwer.wer(nw(text), nw(h)) if nw(h) else 1.0, "cer": jiwer.cer(nc(text), nc(h)) if nc(h) else 1.0}
    rows.append(r); sf.write(os.path.join(HERE, "wav_npu", f"{i:02d}_{lang}_{key}.wav"), w, 44100)
    print(f"[{i}] {lang} lat npu/cpu {r['lat_npu']}/{r['lat_cpu']} | cos vs CPU {r['cos_vs_cpu']:.4f} | peak {r['peak']:.2f} nan={r['nan']} | WER {r['wer']:.2f} CER {r['cer']:.2f}\n     hyp: {h}", flush=True)
for lang in ("en", "ko"):
    xs = [r for r in rows if r["lang"] == lang]
    print(f"NPU {key} {lang}: WER {np.mean([r['wer'] for r in xs])*100:.1f}%  CER {np.mean([r['cer'] for r in xs])*100:.1f}%  mean cos vs CPU {np.mean([r['cos_vs_cpu'] for r in xs]):.4f}  (n={len(xs)})")
json.dump(rows, open(os.path.join(HERE, f"eval_{key}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
p = hub.get_job(J["profile"]).download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]
u = collections.Counter(l.get("compute_unit", "?") for l in ex); tot = sum(l["execution_cycles"] for l in ex)
print(f"PROFILE {J['profile']}: {es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(u)} | peak mem {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB | first load {es.get('first_load_time',0)/1e6:.0f}s")
for l in sorted(ex, key=lambda l: -l["execution_cycles"])[:4]:
    print(f"   {l['execution_cycles']/tot*100:4.1f}% {l['name'][-70:]} ({l.get('type','')})")
