# -*- coding: utf-8 -*-
"""True 2-graph chain: BERT (NPU job jp2oemxxg output) -> host: h^T @ E (3 M MAC) -> VITS g5 (NPU) -> wav ; ASR CER."""
import sys, os, re, json, time, numpy as np, onnxruntime as ort, jiwer, soundfile as sf, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
J = json.load(open("work/melo_jobs.json")); G = json.load(open("work/generic_jobs.json")); dev = hub.Device("Dragonwing IQ-9075 EVK")
SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
D = np.load("work/vits_hE_inputs.npz")                                              # h (CPU BERT), E, x, tone, lang, x_len
od = hub.get_job("jp2oemxxg").download_output_data(); hb = np.stack([np.asarray(a) for a in od[sorted(od.keys())[0]]]).reshape(len(SENT), 32, 1024)   # BERT output computed on the NPU
IN = {"x": [], "tone": [], "lang": [], "bert": [], "x_len": []}
for i in range(len(SENT)):
    IN["bert"].append((hb[i].T @ D["E"][i]).astype(np.float32)[None])               # [1,1024,96] expansion on the host
    for k in ("x", "tone", "lang", "x_len"): IN[k].append(D[k][i])
tm = hub.get_job("jp3o6emzp").get_target_model(); ij = hub.submit_inference_job(model=tm, device=dev, inputs=IN, name="melo_chain2_vits_infer")
print("chain2 VITS infer", ij.job_id, flush=True); G["chain2"] = {"infer": ij.job_id}; json.dump(G, open("work/generic_jobs.json", "w"), indent=1)
t0 = time.time(); ij.wait(); print("chain2 infer", ij.get_status().code, f"wall {time.time()-t0:.0f}s", flush=True)
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
o = ij.download_output_data(); ks = sorted(o.keys(), key=lambda s: int(s.split("_")[-1])); A = [np.asarray(a).reshape(-1) for a in o[ks[0]]]; YL = [int(np.asarray(a).reshape(-1)[0]) for a in o[ks[1]]]
nc = lambda s: re.sub(r"[^\w]", "", s); cs = []; os.makedirs("work/wav_chain2", exist_ok=True)
for i, t in enumerate(SENT):
    a = A[i][:YL[i] * 512].astype(np.float32); sf.write(f"work/wav_chain2/{i}.wav", a, 44100); x = resample_poly(a, 16000, 44100).astype(np.float32); m = min(len(x), 464000); b = np.zeros(464000, np.float32); b[:m] = x[:m]
    r = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([m], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(r[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip(); c = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cs.append(c)
    print(f"  [{i}] frames {YL[i]} | CER {c:.2f} | {h}", flush=True)
print("chain2 (BERT on NPU -> host expand -> VITS on NPU) mean CER %.3f" % np.mean(cs), flush=True)
