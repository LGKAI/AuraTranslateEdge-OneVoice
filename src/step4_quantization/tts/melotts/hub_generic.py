# -*- coding: utf-8 -*-
"""usage: python hub_generic.py <key> <onnx> <inputs.npz>  -> compile fp16 on Dragonwing, 1 inference job (all samples in npz), profile; cosine of each output vs ORT-CPU.
npz holds arrays named like the graph inputs, shape [n_samples, *input_shape]."""
import sys, os, json, time, numpy as np, onnxruntime as ort, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
key, path, npz = sys.argv[1:4]; d = np.load(npz); n = len(d[d.files[0]])
IN = {k: [d[k][i] for i in range(n)] for k in d.files}
dev = hub.Device("Dragonwing IQ-9075 EVK"); JF = "work/generic_jobs.json"; J = json.load(open(JF)) if os.path.exists(JF) else {}
cj = hub.get_job(J[key]["compile"]) if (os.environ.get("REUSE") and J.get(key, {}).get("compile")) else hub.submit_compile_job(model=path, device=dev, options=os.environ.get("COMPILE_OPTS", "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io"), name=f"gen_{key}")
J.setdefault(key, {})["compile"] = cj.job_id; json.dump(J, open(JF, "w"), indent=1); print(key, "compile", cj.job_id, flush=True)
cj.wait(); st = cj.get_status(); print(key, "compile", st.code, (st.message or "")[:300], flush=True)
if str(st.code) != "SUCCESS": sys.exit(1)
tm = cj.get_target_model(); pj = hub.submit_profile_job(model=tm, device=dev, name=f"gen_{key}_profile"); ij = hub.submit_inference_job(model=tm, device=dev, inputs=IN, name=f"gen_{key}_infer")
J[key].update(profile=pj.job_id, infer=ij.job_id); json.dump(J, open(JF, "w"), indent=1); print(key, "profile", pj.job_id, "| infer", ij.job_id, flush=True)
t0 = time.time(); ij.wait(); st = ij.get_status(); print(key, "infer", st.code, f"wall {time.time()-t0:.0f}s", (st.message or "")[:300], flush=True)
if str(st.code) == "SUCCESS":
    od = ij.download_output_data(); ks = sorted(od.keys(), key=lambda s: int(s.split("_")[-1])); s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    cs = [[] for _ in ks]
    for i in range(n):
        ref = s.run(None, {k: v[i] for k, v in IN.items()})
        for j, k in enumerate(ks):
            r = np.asarray(ref[j]).astype(np.float64).ravel(); a = np.asarray(od[k][i]).astype(np.float64).ravel(); cs[j].append(float(r @ a / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)))
    np.savez(f"work/hw_{key}.npz", **{k: np.stack([np.asarray(od[k][i]) for i in range(n)]) for k in ks})
    for j, k in enumerate(ks): print(key, f"OUTPUT {k}: cosine vs ORT-CPU per sample", [round(x, 5) for x in cs[j]], flush=True)
try:
    pj.wait(); p = pj.download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]; import collections; u = collections.Counter(l.get("compute_unit", "?") for l in ex)
    print(key, "PROFILE", f"{es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(u)} | peak mem {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB | first load {es.get('first_load_time',0)/1e6:.0f}s", flush=True)
except Exception as e: print(key, "profile failed:", str(e)[:200], flush=True)
