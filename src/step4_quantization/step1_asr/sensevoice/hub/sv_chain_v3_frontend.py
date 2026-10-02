# -*- coding: utf-8 -*-
"""Frontend v3 (chuan hoa theo frame): compile NPU -> inference 15 mau -> so fbank NPU vs fbank
fp32 GOC (muc tieu that) -> chain vao encoder quantized (jprlmj9kp) -> WER/CER."""
import sys, os, json, re, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, jiwer, onnxruntime as ort
import qai_hub as hub

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA = os.path.join(ROOT, "data", "asr")
device = hub.Device("Dragonwing IQ-9075 EVK")
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]
calib = np.load(os.path.join(OUT, "calib_data_unified_fixed.npz"))

print("[1] compile frontend v3", flush=True)
m = hub.upload_model(os.path.join(OUT, "model_sv_frontend_v3_inline.onnx"))
cj = hub.submit_compile_job(model=m, device=device, options="--target_runtime qnn_dlc --truncate_64bit_io")
print("  compile_job_id=", cj.job_id, flush=True)
cj.wait(); print("  state:", cj.get_status().state, flush=True)
fe_target = cj.get_target_model()
print("  fe_v3_target=", fe_target.model_id, flush=True)

print("[2] inference frontend v3 tren NPU", flush=True)
ds = hub.upload_dataset({"wav": [calib["wav"][i] for i in range(15)], "wav_len": [calib["wav_len"][i] for i in range(15)]})
ij = hub.submit_inference_job(model=fe_target, device=device, inputs=ds)
print("  fe_infer_job_id=", ij.job_id, flush=True)
ij.wait(); print("  state:", ij.get_status().state, flush=True)
fe_out = ij.download_output_data()
fk = [k for k in fe_out if np.asarray(fe_out[k][0]).reshape(-1).shape[0] == 500 * 560][0]
sk = [k for k in fe_out if k != fk][0]

print("[3] so fbank NPU v3 vs fbank fp32 GOC (muc tieu that)", flush=True)
ref = ort.InferenceSession(os.path.join(OUT, "model_sv_frontend_inline.onnx"), providers=["CPUExecutionProvider"])
print(f"{'mau':6s} {'cos_valid':>10s} {'relL2':>8s} {'maxabs':>8s}", flush=True)
coss = []
for i in range(15):
    rf, rs = ref.run(None, {"wav": calib["wav"][i].astype(np.float32), "wav_len": calib["wav_len"][i].reshape(1).astype(np.int32)})
    n = int(rs[0]); a = rf.reshape(500, 560)[:n].astype(np.float64); b = np.asarray(fe_out[fk][i]).reshape(500, 560)[:n].astype(np.float64)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)); coss.append(cos)
    print(f"{labels[i]:6s} {cos:10.6f} {np.linalg.norm(a-b)/np.linalg.norm(a):8.5f} {np.abs(a-b).max():8.4f}", flush=True)
print(f"  avg cos_valid = {np.mean(coss):.6f}   (v2 truoc do: ~0.80)", flush=True)

print("[4] chain -> encoder quantized jprlmj9kp", flush=True)
enc_target = hub.get_job("jprlmj9kp").get_target_model()
d2 = hub.upload_dataset({
    "fbank": [np.asarray(fe_out[fk][i]).reshape(1, 500, 560).astype(np.float32) for i in range(15)],
    "speech_lengths": [np.asarray(fe_out[sk][i]).reshape(1).astype(np.int32) for i in range(15)],
    "language": [calib["language"][i].astype(np.int32) for i in range(15)],
    "textnorm": [calib["textnorm"][i].astype(np.int32) for i in range(15)],
})
ej = hub.submit_inference_job(model=enc_target, device=device, inputs=d2)
print("  enc_infer_job_id=", ej.job_id, flush=True)
ej.wait(); print("  state:", ej.get_status().state, flush=True)
eo = ej.download_output_data(); outs = eo[list(eo.keys())[0]]

manifest = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))
items = [it for it in manifest if it["lang"] in ("en", "zh", "ko")]
def dec(a):
    a = np.asarray(a).astype(np.int64).reshape(-1)
    return bytes(int(b) & 0xFF for b in a).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
nw = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()
res = {"en": [], "zh": [], "ko": []}
for it, o in zip(items, outs):
    hyp = dec(o); ref_t = it["transcript"]; L = it["lang"]
    mt = jiwer.cer(re.sub(r"\s+", "", ref_t), re.sub(r"\s+", "", hyp)) if L == "zh" else (jiwer.wer(nw(ref_t), nw(hyp)) if nw(hyp) else 1.0)
    res[L].append(mt)
    print(f"[{L}] {os.path.basename(it['path'])} {mt*100:5.1f}%  {hyp[:70]!r}", flush=True)
print("\n=== KET QUA frontend v3 (per-frame norm) + encoder quantized, NPU THAT ===", flush=True)
for L, v in res.items():
    print(f"  {L}: {np.mean(v)*100:.2f}%", flush=True)
print("  (v2 truoc do: en 12.78 / zh 11.51 / ko 40.34 | fp32 muc tieu: en 6.23-7.56 / zh 11.20 / ko 38.27)", flush=True)
json.dump({"fe_v3_target": fe_target.model_id, **{L: float(np.mean(v) * 100) for L, v in res.items()}},
          open(os.path.join(OUT, "frontend_v3_result.json"), "w"), indent=2)
