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
    os.makedirs("results", exist_ok=True)
    os.makedirs("work", exist_ok=True)
    np.save(f"work/hw_{key}.npy", np.stack(H))
    np.save(f"results/hw_{key}.npy", np.stack(H))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); same = 0
    predictions = []
    for i, (a, l) in enumerate(zip(A, L)):
        w = s.run(None, {"src_ids": a, "tgt_lang": l})[0].reshape(-1); eq = bool(np.array_equal(w, H[i])); same += int(eq)
        hyp_text = tok.decode([int(x) for x in H[i] if x > 3], skip_special_tokens=True)
        predictions.append({
            "index": i,
            "src_lang": ref[i]["src"],
            "tgt_lang": ref[i]["tgt"],
            "source_text": ref[i]["text"],
            "reference_text": ref[i]["hyp"],
            "silicon_hypothesis": hyp_text,
            "identical_to_cpu": eq,
            "raw_hw_ids": [int(x) for x in H[i]]
        })
        print(f"  [{i}] {ref[i]['src'][:3]}->{ref[i]['tgt'][:3]} identical={eq} | hw {hyp_text[:70]}", flush=True)
    print(key, f"IDENTICAL {same}/{len(A)}", flush=True)
profile_summary = {}
try:
    pj.wait(); p = pj.download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]
    import collections; u = collections.Counter(l.get("compute_unit", "?") for l in ex)
    profile_summary = {
        "estimated_inference_time_ms": round(es.get("estimated_inference_time", 0) / 1000.0, 2),
        "total_layers": len(ex),
        "compute_units": dict(u),
        "peak_memory_mb": round(es.get("estimated_inference_peak_memory", 0) / 1e6, 2),
        "first_load_time_sec": round(es.get("first_load_time", 0) / 1e6, 2)
    }
    print(key, "PROFILE", f"{profile_summary['estimated_inference_time_ms']} ms | layers {len(ex)} {dict(u)} | peak mem {profile_summary['peak_memory_mb']} MB | first load {profile_summary['first_load_time_sec']}s", flush=True)
except Exception as e:
    print(key, "profile failed:", str(e)[:200], flush=True)

# Save comprehensive result summary
hub_results = {
    "key": key,
    "model_path": path,
    "unrolled_steps_T": T,
    "device": "Dragonwing IQ-9075 EVK",
    "jobs": {
        "compile_job_id": cj.job_id,
        "compile_url": f"https://workbench.aihub.qualcomm.com/jobs/{cj.job_id}/",
        "profile_job_id": pj.job_id,
        "profile_url": f"https://workbench.aihub.qualcomm.com/jobs/{pj.job_id}/",
        "infer_job_id": ij.job_id,
        "infer_url": f"https://workbench.aihub.qualcomm.com/jobs/{ij.job_id}/"
    },
    "accuracy": {
        "identical_to_cpu": f"{same}/{len(A)}",
        "accuracy_percent": round(100.0 * same / len(A), 2)
    },
    "hardware_profile": profile_summary,
    "predictions": predictions
}
json.dump(hub_results, open(f"results/hub_silicon_test_results_{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(hub_results, open("results/hub_silicon_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Results successfully saved to results/hub_silicon_test_results_{key}.json and results/hw_{key}.npy", flush=True)
