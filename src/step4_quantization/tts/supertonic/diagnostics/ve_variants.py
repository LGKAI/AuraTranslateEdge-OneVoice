# -*- coding: utf-8 -*-
"""VE-only NPU experiments. usage:  python ve_variants.py prep            -> writes ve_erftanh.onnx + ve_samples.npz (calib/eval from CPU fp32 chain)
                                    python ve_variants.py run <variant>   -> variant in {fp16, erftanh, w8a16, w16a16}
Per-step error is measured on real fp32 inputs (x_t from the fp32 chain), so it isolates ONE VE call on the chip."""
import sys, os, json, time
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto as TP
sys.stdout.reconfigure(encoding="utf-8")
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); N = 96; L = int(os.environ.get('VAR_L', '150')); D = os.path.join(HERE, f"static_N{N}_L{L}")
VE = os.path.join(D, "vector_estimator_static.onnx"); cpu = ["CPUExecutionProvider"]
CAL, EVAL = [1, 3, 4, 7, 8, 10], [0, 2, 6, 9]
IN = ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask", "current_step", "total_step"]


def prep():
    from supertonic import TTS
    tts = TTS(); m = tts.model; st = tts.get_voice_style(voice_name="M1")
    S = {k: ort.InferenceSession(os.path.join(D, f"{k}_static.onnx"), providers=cpu) for k in ["duration_predictor", "text_encoder", "vector_estimator"]}
    SENT = json.load(open(os.path.join(HERE, "..", "data", "sentences.json"), encoding="utf-8")); out = {k: [] for k in IN + ["ref", "sent", "step"]}
    for i in CAL + EVAL:
        lang, text = SENT[i]; ids, _ = m.text_processor([text], lang); n = ids.shape[1]
        ip = np.zeros((1, N), np.int64); ip[:, :n] = ids; mp = np.zeros((1, 1, N), np.float32); mp[:, :, :n] = 1
        dur = S["duration_predictor"].run(None, {"text_ids": ip, "style_dp": st.dp, "text_mask": mp})[0] / 1.05
        emb = S["text_encoder"].run(None, {"text_ids": ip, "style_ttl": st.ttl, "text_mask": mp})[0]
        lat = int(np.ceil(int(dur[0] * 44100) / 3072)); lm = np.zeros((1, 1, L), np.float32); lm[:, :, :lat] = 1
        x = np.random.RandomState(0).randn(1, 144, L).astype(np.float32) * lm
        for s in range(5):
            feed = {"noisy_latent": x, "text_emb": emb, "style_ttl": st.ttl, "latent_mask": lm, "text_mask": mp, "current_step": np.array([s], np.float32), "total_step": np.array([5], np.float32)}
            y = S["vector_estimator"].run(None, feed)[0]
            for k in IN: out[k].append(feed[k])
            out["ref"].append(y); out["sent"].append(i); out["step"].append(s); x = y
    np.savez(os.path.join(HERE, f"ve_samples_L{L}.npz" if L != 150 else "ve_samples.npz"), **{k: np.stack(v) if k in IN + ["ref"] else np.array(v) for k, v in out.items()})
    # Erf -> tanh approx: erf(z) ~= tanh(1.1283792*z + 0.1009112*z^3)
    mdl = onnx.load(VE); nodes = []; cnt = 0
    mdl.graph.initializer.extend([numpy_helper.from_array(np.array(1.1283792, np.float32), "erf_a"), numpy_helper.from_array(np.array(0.1009112, np.float32), "erf_b")])
    for n in mdl.graph.node:
        if n.op_type == "Erf":
            z, o = n.input[0], n.output[0]; p = f"erf{cnt}_"; cnt += 1
            nodes += [helper.make_node("Mul", [z, z], [p + "z2"]), helper.make_node("Mul", [p + "z2", z], [p + "z3"]), helper.make_node("Mul", [z, "erf_a"], [p + "t1"]),
                      helper.make_node("Mul", [p + "z3", "erf_b"], [p + "t2"]), helper.make_node("Add", [p + "t1", p + "t2"], [p + "s"]), helper.make_node("Tanh", [p + "s"], [o])]
        else:
            nodes.append(n)
    del mdl.graph.node[:]; mdl.graph.node.extend(nodes); onnx.save(mdl, os.path.join(HERE, "ve_erftanh.onnx"))
    s = ort.InferenceSession(os.path.join(HERE, "ve_erftanh.onnx"), providers=cpu); d = np.load(os.path.join(HERE, f"ve_samples_L{L}.npz" if L != 150 else "ve_samples.npz"))
    cs = []
    for j in range(len(d["ref"])):
        y = s.run(None, {k: d[k][j] for k in IN})[0]; r = d["ref"][j]; cs.append(float(np.dot(y.ravel(), r.ravel()) / (np.linalg.norm(y) * np.linalg.norm(r))))
    print(f"prep done: {len(d['ref'])} samples; replaced {cnt} Erf; CPU fp32 erf->tanh approx cos vs exact: min {min(cs):.6f} mean {np.mean(cs):.6f}")


def run(var):
    import qai_hub as hub
    d = np.load(os.path.join(HERE, f"ve_samples_L{L}.npz" if L != 150 else "ve_samples.npz")); dev = hub.Device("Dragonwing IQ-9075 EVK")
    JF = os.path.join(HERE, f"ve_jobs_{var}_L{L}.json"); J = {}
    src = os.path.join(HERE, "ve_finite.onnx") if var == "finite" else os.path.join(HERE, "ve_erftanh.onnx") if var == "erftanh" else (os.path.join(HERE, "ve_zb.onnx") if var.endswith("zb") else VE)
    opts = "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io"
    if var in ("w8a16", "w16a16", "w8a16zb", "w16a16zb"):
        cal = {k: [d[k][j] for j, sidx in enumerate(d["sent"]) if int(sidx) in CAL] for k in IN}
        wd = hub.QuantizeDtype.INT8 if var.startswith("w8") else hub.QuantizeDtype.INT16
        qj = hub.submit_quantize_job(model=src, calibration_data=cal, weights_dtype=wd, activations_dtype=hub.QuantizeDtype.INT16, name=f"ve_{var}_quant")
        print(var, "quantize", qj.job_id, flush=True); qj.wait(); print(var, "quantize", qj.get_status().code, flush=True)
        if str(qj.get_status().code) != "SUCCESS": return
        model = qj.get_target_model(); opts = "--target_runtime qnn_dlc --quantize_io --truncate_64bit_io"; J.setdefault(var, {})["quant"] = qj.job_id
    else:
        model = src; J.setdefault(var, {})
    cj = hub.submit_compile_job(model=model, device=dev, options=opts, name=f"ve_{var}_compile"); J[var]["compile"] = cj.job_id; json.dump(J, open(JF, "w"), indent=1)
    print(var, "compile", cj.job_id, flush=True); cj.wait(); st = cj.get_status(); print(var, "compile", st.code, (st.message or "")[:300], flush=True)
    if str(st.code) != "SUCCESS": return
    idx = [j for j, sidx in enumerate(d["sent"]) if int(sidx) in EVAL]
    ij = hub.submit_inference_job(model=cj.get_target_model(), device=dev, inputs={k: [d[k][j] for j in idx] for k in IN}, name=f"ve_{var}_infer")
    J[var]["infer"] = ij.job_id; json.dump(J, open(JF, "w"), indent=1); print(var, "infer", ij.job_id, flush=True)
    t0 = time.time(); ij.wait(); st = ij.get_status(); print(var, "infer", st.code, f"{time.time()-t0:.0f}s", (st.message or "")[:200], flush=True)
    if str(st.code) != "SUCCESS": return
    od = ij.download_output_data(); k = sorted(od.keys(), key=lambda s: int(s.split("_")[-1]))[0]
    cs = {s: [] for s in range(5)}
    for jj, j in enumerate(idx):
        y = np.asarray(od[k][jj]).reshape(d["ref"][j].shape); r = d["ref"][j]
        cs[int(d["step"][j])].append(float(np.dot(y.ravel(), r.ravel()) / (np.linalg.norm(y) * np.linalg.norm(r))))
    print(var, "ONE-STEP cos vs CPU fp32 per ODE step:", {s: round(float(np.mean(v)), 5) for s, v in cs.items()}, flush=True)


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else run(sys.argv[2])
