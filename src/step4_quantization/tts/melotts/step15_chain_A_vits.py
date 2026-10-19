# -*- coding: utf-8 -*-
"""Fully-NPU chain: graph A (static-table front + BERT + expansion; input = UTF-8 bytes) -> VITS g5 -> wav. ASR CER on 8 SENT + FLEURS clauses.
usage: python step15_chain_A_vits.py [n_clauses]"""
import sys, os, re, json, time, unicodedata, numpy as np, onnxruntime as ort, jiwer, soundfile as sf, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, "MeloTTS")
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
LB = 112; NC = int(sys.argv[1]) if len(sys.argv) > 1 else 32
dev = hub.Device("Dragonwing IQ-9075 EVK"); G = json.load(open("work/generic_jobs.json"))
S = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
for l in open(hf_hub_download("google/fleurs", "data/cmn_hans_cn/test.tsv", repo_type="dataset"), encoding="utf-8"):
    f = l.rstrip("\n").split("\t")
    if len(f) > 3:
        for c in re.split(r"[。！？；]", f[2]):
            c = c.strip()
            if 6 <= len(c) <= 20 and re.fullmatch(r"[一-鿿，、]+", c) and c not in S: S.append(c)
S = S[:len(json.load(open("work/cpu_baseline.json", encoding="utf-8"))) + NC]; print("texts", len(S), flush=True)
B = []
for t in S: b = np.zeros((1, LB), np.int32); by = unicodedata.normalize("NFC", t).encode(); b[0, :len(by)] = list(by); B.append(b)
tmA = hub.get_job(G[os.environ.get("GA","graphA")]["compile"]).get_target_model()
ja = hub.submit_inference_job(model=tmA, device=dev, inputs={"text_bytes": B}, name="melo_graphA_batch"); print("graphA infer", ja.job_id, flush=True); ja.wait(); print("graphA", ja.get_status().code, flush=True)
o = ja.download_output_data(); k = {int(n.split("_")[-1]): n for n in o}; X = o[k[0]]; Tn = o[k[1]]; XL = o[k[3]]; BE = o[k[4]]
json.dump({"graphA_batch": ja.job_id}, open("work/graphA_batch.json", "w"))
IN = {"x": [np.asarray(a) for a in X], "tone": [np.asarray(a) for a in Tn], "lang": [np.zeros((1, 96), np.int32) for _ in S], "bert": [np.asarray(a) for a in BE], "x_len": [np.asarray(a) for a in XL]}
tm = hub.get_job("jp3o6emzp").get_target_model(); ij = hub.submit_inference_job(model=tm, device=dev, inputs=IN, name="melo_graphA_vits"); print("VITS infer", ij.job_id, flush=True); ij.wait(); print("VITS", ij.get_status().code, flush=True)
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
od = ij.download_output_data(); ks = sorted(od.keys(), key=lambda s: int(s.split("_")[-1])); A = [np.asarray(a).reshape(-1) for a in od[ks[0]]]; YL = [int(np.asarray(a).reshape(-1)[0]) for a in od[ks[1]]]
nc = lambda s: re.sub(r"[^\w]", "", s); cs = []; os.makedirs("work/wav_A", exist_ok=True)
for i, t in enumerate(S):
    a = A[i][:YL[i] * 512].astype(np.float32); sf.write(f"work/wav_A/{i}.wav", a, 44100); x = resample_poly(a, 16000, 44100).astype(np.float32); m = min(len(x), 464000); b = np.zeros(464000, np.float32); b[:m] = x[:m]
    r = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([m], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(r[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip(); c = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cs.append(c)
    print(f"  [{i}] frames {YL[i]} | CER {c:.2f} | {t} | {h}", flush=True)
n8 = len(json.load(open("work/cpu_baseline.json", encoding="utf-8")))
print(f"FULL-NPU chain (text bytes -> graph A -> VITS) CER: first {n8} sentences {np.mean(cs[:n8]):.3f} | FLEURS clauses {np.mean(cs[n8:]):.3f} | all {np.mean(cs):.3f}", flush=True)
