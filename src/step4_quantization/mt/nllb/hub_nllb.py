# -*- coding: utf-8 -*-
"""usage: python hub_nllb.py <key> <onnx> <T>  -> compile (fp16 qnn_dlc) on Dragonwing, profile + ONE inference job with the 8 reference sentences; compare ids with ORT CPU."""
import sys, os, json, time, numpy as np, onnxruntime as ort, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
key, path, T = sys.argv[1], sys.argv[2], int(sys.argv[3]); S = 32
ref = json.load(open("work/ref.json", encoding="utf-8")); from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"); V = tok.get_vocab()
A, L = [], []
for r in ref:
    a = np.ones((1, S), np.int32); a[0, :len(r["ids_in"])] = r["ids_in"]; A.append(a); L.append(np.array([V[r["tgt"]]], np.int32))
dev = hub.Device("Dragonwing IQ-9075 EVK"); JF = "work/nllb_jobs.json"; J = json.load(open(JF)) if os.path.exists(JF) else {}
if J.get(key, {}).get("compile") and os.environ.get("REUSE"): cj = hub.get_job(J[key]["compile"])      # reuse an already compiled model
else: cj = hub.submit_compile_job(model=path, device=dev, options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io", name=f"nllb_{key}")
J.setdefault(key, {})["compile"] = cj.job_id; json.dump(J, open(JF, "w"), indent=1); print(key, "compile", cj.job_id, flush=True)
cj.wait(); st = cj.get_status(); print(key, "compile", st.code, (st.message or "")[:300], flush=True)
if str(st.code) != "SUCCESS": sys.exit(1)
tm = cj.get_target_model()
pj = hub.submit_profile_job(model=tm, device=dev, name=f"nllb_{key}_profile")
ij = hub.submit_inference_job(model=tm, device=dev, inputs={"src_ids": A, "tgt_lang": L}, name=f"nllb_{key}_infer")
J[key].update(profile=pj.job_id, infer=ij.job_id); json.dump(J, open(JF, "w"), indent=1); print(key, "profile", pj.job_id, "| infer", ij.job_id, flush=True)
t0 = time.time(); ij.wait(); st = ij.get_status(); print(key, "infer", st.code, f"wall {time.time()-t0:.0f}s", (st.message or "")[:300], flush=True)
if str(st.code) == "SUCCESS":
    od = ij.download_output_data(); k = sorted(od.keys(), key=lambda s: int(s.split("_")[-1]))[0]; H = [np.asarray(x).reshape(-1) for x in od[k]]
    np.save(f"work/hw_{key}.npy", np.stack(H))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); same = 0
    for i, (a, l) in enumerate(zip(A, L)):
        w = s.run(None, {"src_ids": a, "tgt_lang": l})[0].reshape(-1); eq = np.array_equal(w, H[i]); same += eq
        print(f"  [{i}] {ref[i]['src'][:3]}->{ref[i]['tgt'][:3]} identical={eq} | hw {tok.decode([int(x) for x in H[i] if x > 3], skip_special_tokens=True)[:70]}", flush=True)
    print(key, f"IDENTICAL {same}/{len(A)}", flush=True)
try:
    pj.wait(); p = pj.download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]
    import collections; u = collections.Counter(l.get("compute_unit", "?") for l in ex)
    print(key, "PROFILE", f"{es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(u)} | peak mem {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB | first load {es.get('first_load_time',0)/1e6:.0f}s", flush=True)
except Exception as e: print(key, "profile failed:", str(e)[:200], flush=True)
