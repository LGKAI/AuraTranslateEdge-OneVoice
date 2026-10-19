# -*- coding: utf-8 -*-
"""Two NPU graphs chained: BERT(NPU) -> h -> VITS(NPU) -> wav. Evaluates (a) VITS with CPU BERT features, (b) VITS fed with the NPU BERT output (true chain)."""
import sys, os, re, json, time, collections, numpy as np, onnxruntime as ort, jiwer, soundfile as sf, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
J = json.load(open("work/generic_jobs.json")); dev = hub.Device("Dragonwing IQ-9075 EVK")
SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
nc = lambda s: re.sub(r"[^\w]", "", s)
def outs(job):
    od = job.download_output_data(); ks = sorted(od.keys(), key=lambda s: int(s.split("_")[-1])); return [np.stack([np.asarray(a) for a in od[k]]) for k in ks]
def cer_of(audio, yl, tag):
    cs = []; os.makedirs(f"work/wav_{tag}", exist_ok=True)
    for i, t in enumerate(SENT):
        n = int(np.asarray(yl[i]).reshape(-1)[0]) * 512; a = np.asarray(audio[i]).reshape(-1)[:n].astype(np.float32); sf.write(f"work/wav_{tag}/{i}.wav", a, 44100)
        x = resample_poly(a, 16000, 44100).astype(np.float32); m = min(len(x), 464000); b = np.zeros(464000, np.float32); b[:m] = x[:m]
        o = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([m], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
        h = bytes(int(v) & 0xFF for v in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
        c = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cs.append(c); print(f"  [{tag} {i}] {n/44100:.2f}s CER {c:.2f} | {h}", flush=True)
    print(f"{tag}: mean CER {np.mean(cs):.3f}", flush=True); return cs
def prof(jid, tag):
    p = hub.get_job(jid).download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]; u = collections.Counter(l.get("compute_unit", "?") for l in ex)
    print(f"{tag} PROFILE {jid}: {es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(u)} | peak mem {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB | first load {es.get('first_load_time',0)/1e6:.0f}s")
prof("jprxwy6vp", "BERT"); prof(J["vits_hE"]["profile"], "VITS")
a = outs(hub.get_job(J["vits_hE"]["infer"])); cer_of(a[0], a[1], "vits_cpuBERT")                    # VITS on NPU with CPU BERT features
hb = outs(hub.get_job("jp2oemxxg"))[0]                                                              # BERT output computed ON THE NPU: [n,1,32,1024]
D = np.load("work/vits_hE_inputs.npz"); hcpu = D["h"]; print("NPU BERT vs CPU BERT features cosine per sentence:", [round(float(np.dot(hb[i].ravel(), hcpu[i].ravel()) / np.linalg.norm(hb[i]) / np.linalg.norm(hcpu[i])), 5) for i in range(len(SENT))])
IN = {k: [D[k][i] for i in range(len(SENT))] for k in D.files}; IN["h"] = [hb[i].reshape(32, 1024).astype(np.float32) for i in range(len(SENT))]
tm = hub.get_job(J["vits_hE"]["compile"]).get_target_model(); ij = hub.submit_inference_job(model=tm, device=dev, inputs=IN, name="melo_chain_vits_infer"); print("chain VITS infer job", ij.job_id, flush=True)
J["chain"] = {"infer": ij.job_id}; json.dump(J, open("work/generic_jobs.json", "w"), indent=1); t0 = time.time(); ij.wait(); print("chain infer", ij.get_status().code, f"wall {time.time()-t0:.0f}s", flush=True)
if str(ij.get_status().code) == "SUCCESS": b = outs(ij); cer_of(b[0], b[1], "chain_npu_bert_vits")
