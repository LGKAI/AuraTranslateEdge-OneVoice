# -*- coding: utf-8 -*-
"""usage: python hub_run.py <key> <onnx> N L   -> compile (fp16, qnn_dlc) on Dragonwing IQ-9075 -> profile + ONE inference job with all 12 sentences.
Records job ids in jobs.json; outputs saved as hw_<key>.npz. Output names are output_N: sort by N (AI Hub dict order is NOT reliable)."""
import sys, os, json, time
import numpy as np, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS

key, onnx_path, N, L = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); JF = os.path.join(HERE, "jobs.json")
J = json.load(open(JF)) if os.path.exists(JF) else {}
dev = hub.Device("Dragonwing IQ-9075 EVK")
opts = os.environ.get("COMPILE_OPTS", "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io")
if key in J and J[key].get("compile"):
    cj = hub.get_job(J[key]["compile"])
else:
    cj = hub.submit_compile_job(model=onnx_path, device=dev, options=opts, name=f"supertonic_{key}")
    J[key] = {"compile": cj.job_id, "opts": opts}; json.dump(J, open(JF, "w"), indent=1)
print(key, "compile", cj.job_id, cj.url, flush=True)
cj.wait(); st = cj.get_status(); print(key, "compile", st.code, (st.message or "")[:400], flush=True)
if str(st.code) != "SUCCESS": sys.exit(1)
tm = cj.get_target_model()
tts = TTS(); m = tts.model; style = tts.get_voice_style(voice_name="M1")
SENT = json.load(open(os.path.join(HERE, "..", "data", "sentences.json"), encoding="utf-8"))
cps = []
for lang, text in SENT:
    v = [ord(c) for c in m.text_processor._preprocess_text(text, lang)]; a = np.zeros((1, N), np.int32); a[0, :len(v)] = v; cps.append(a)
pj = hub.submit_profile_job(model=tm, device=dev, name=f"supertonic_{key}_profile")
ij = hub.submit_inference_job(model=tm, device=dev, inputs={"cp": cps, "style_dp": [style.dp] * len(cps), "style_ttl": [style.ttl] * len(cps)}, name=f"supertonic_{key}_infer")
J[key].update(profile=pj.job_id, infer=ij.job_id); json.dump(J, open(JF, "w"), indent=1)
print(key, "profile", pj.job_id, "| infer", ij.job_id, flush=True)
t0 = time.time(); ij.wait(); st = ij.get_status(); print(key, "infer", st.code, f"wall {time.time()-t0:.0f}s", (st.message or "")[:300], flush=True)
if str(st.code) == "SUCCESS":
    od = ij.download_output_data(); ks = sorted(od.keys(), key=lambda k: int(k.split("_")[-1]))
    np.savez(os.path.join(HERE, f"hw_{key}.npz"), **{k: np.stack([np.asarray(a) for a in od[k]]) for k in ks}); print(key, "saved", ks, {k: np.asarray(od[k][0]).shape for k in ks}, flush=True)
pj.wait(); print(key, "profile", pj.get_status().code, flush=True)
