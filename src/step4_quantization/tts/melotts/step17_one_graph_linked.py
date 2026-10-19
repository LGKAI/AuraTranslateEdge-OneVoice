# -*- coding: utf-8 -*-
"""Melo TTS as ONE NPU graph (UTF-8 bytes -> 44.1 kHz audio): link compiled DLC -> QNN context binary, profile (first-load time), inference on 8 SENT + 32 FLEURS clauses, ASR CER."""
import sys, os, re, json, collections, unicodedata, numpy as np, onnxruntime as ort, jiwer, soundfile as sf, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
dev = hub.Device("Dragonwing IQ-9075 EVK"); J = json.load(open("work/generic_jobs.json"))
S = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]; n8 = len(S)
for l in open(hf_hub_download("google/fleurs", "data/cmn_hans_cn/test.tsv", repo_type="dataset"), encoding="utf-8"):
    f = l.rstrip("\n").split("\t")
    if len(f) > 3:
        for c in re.split(r"[。！？；]", f[2]):
            c = c.strip()
            if 6 <= len(c) <= 20 and re.fullmatch(r"[一-鿿，、]+", c) and c not in S: S.append(c)
S = S[:n8 + 32]
B = []
for t in S: b = np.zeros((1, 112), np.int32); by = unicodedata.normalize("NFC", t).encode(); b[0, :len(by)] = list(by); B.append(b)
tm = hub.get_job(J["melo_one"]["compile"]).get_target_model()
lj = hub.submit_link_job(models=[tm], device=dev, name="melo_one_link"); print("link", lj.job_id, flush=True); lj.wait(); print("link", lj.get_status().code, flush=True)
cb = lj.get_target_model(); J["melo_one"]["link"] = lj.job_id
pj = hub.submit_profile_job(model=cb, device=dev, name="melo_one_link_profile"); ij = hub.submit_inference_job(model=cb, device=dev, inputs={"text_bytes": B}, name="melo_one_link_infer")
J["melo_one"].update(link_profile=pj.job_id, link_infer=ij.job_id); json.dump(J, open("work/generic_jobs.json", "w"), indent=1); print("profile", pj.job_id, "| infer", ij.job_id, flush=True)
ij.wait(); print("infer", ij.get_status().code, (ij.get_status().message or "")[:200], flush=True)
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
o = ij.download_output_data(); ks = sorted(o, key=lambda s: int(s.split("_")[-1])); nc = lambda s: re.sub(r"[^\w]", "", s); cs = []; os.makedirs("work/wav_one", exist_ok=True)
for i, t in enumerate(S):
    yl = int(np.asarray(o[ks[1]][i]).ravel()[0]); a = np.asarray(o[ks[0]][i]).ravel()[:yl * 512].astype(np.float32); sf.write(f"work/wav_one/{i}.wav", a, 44100)
    w = resample_poly(a, 16000, 44100).astype(np.float32); m = min(len(w), 464000); b = np.zeros(464000, np.float32); b[:m] = w[:m]
    r = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([m], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(r[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip(); c = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cs.append(c)
    print(f"  [{i}] frames {yl} | CER {c:.2f} | {t} | {h}", flush=True)
print(f"ONE-GRAPH Melo CER: first {n8} {np.mean(cs[:n8]):.3f} | FLEURS clauses {np.mean(cs[n8:]):.3f} | all {np.mean(cs):.3f}", flush=True)
pj.wait(); p = pj.download_profile(); es = p["execution_summary"]; ex = p["execution_detail"]
print("PROFILE", f"{es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(collections.Counter(l.get('compute_unit','?') for l in ex))} | first load {es.get('first_load_time',0)/1e6:.1f}s | peak {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB", flush=True)
