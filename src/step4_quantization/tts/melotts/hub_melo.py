# -*- coding: utf-8 -*-
"""usage: python hub_melo.py <key> <onnx> <N> <Y>  -> compile (fp16 qnn_dlc) on Dragonwing, ONE inference job (8 sentences) + profile; NPU audio -> SenseVoice CER; cosine vs ORT-CPU."""
import sys, os, json, re, time, numpy as np, onnxruntime as ort, qai_hub as hub, torch, jiwer, soundfile as sf
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from melo.api import TTS
from melo import utils
key, path, N, Y = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tts = TTS(language="ZH", device="cpu"); SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
IN = {k: [] for k in ["x", "tone", "lang", "bert", "x_len"]}
for t in SENT:
    bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(t, "ZH", tts.hps, "cpu", tts.symbol_to_id); n = len(ph)
    pad = lambda v: np.concatenate([v.numpy(), np.zeros(N - n)]).astype(np.int32)[None]
    bp = np.zeros((1, 1024, N), np.float32); bp[0, :, :n] = bert.numpy()
    for k, v in zip(IN, [pad(ph), pad(tone), pad(lang), bp, np.array([n], np.int32)]): IN[k].append(v)
QUANT = os.environ.get("QUANT")          # w8a16 | w16a16 | None
dev = hub.Device("Dragonwing IQ-9075 EVK"); JF = "work/melo_jobs.json"; J = json.load(open(JF)) if os.path.exists(JF) else {}
OPTS = os.environ.get("COMPILE_OPTS", "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io")
model_in = path
if QUANT:
    import onnx
    from onnx import numpy_helper
    zb = path.replace(".onnx", "_zb.onnx")
    if not os.path.exists(zb):
        mm = onnx.load(path); W = {i.name: i for i in mm.graph.initializer}; k = 0
        for nd in mm.graph.node:
            if nd.op_type == "Conv" and len(nd.input) == 2:
                mm.graph.initializer.append(numpy_helper.from_array(np.zeros(W[nd.input[1]].dims[0], np.float32), f"zb_{k}")); nd.input.append(f"zb_{k}"); k += 1
        onnx.save(mm, zb); print("zero bias added to", k, "convs", flush=True)
    from datasets_zh import calib_inputs
    cal = calib_inputs(tts, N)
    wd = hub.QuantizeDtype.INT8 if QUANT.startswith("w8") else hub.QuantizeDtype.INT16
    qj = hub.submit_quantize_job(model=zb, calibration_data=cal, weights_dtype=wd, activations_dtype=hub.QuantizeDtype.INT16, name=f"melo_{key}_quant")
    J.setdefault(key, {})["quant"] = qj.job_id; json.dump(J, open(JF, "w"), indent=1); print(key, "quantize", qj.job_id, flush=True); qj.wait(); st_ = qj.get_status(); print(key, "quantize", st_.code, (st_.message or "")[:300], flush=True)
    if str(st_.code) != "SUCCESS": sys.exit(1)
    model_in = qj.get_target_model(); OPTS = "--target_runtime qnn_dlc --quantize_io --truncate_64bit_io"
if J.get(key, {}).get("compile") and os.environ.get("REUSE"): cj = hub.get_job(J[key]["compile"])
else: cj = hub.submit_compile_job(model=model_in, device=dev, options=OPTS, name=f"melo_{key}")
J.setdefault(key, {})["compile"] = cj.job_id; json.dump(J, open(JF, "w"), indent=1); print(key, "compile", cj.job_id, flush=True)
cj.wait(); st = cj.get_status(); print(key, "compile", st.code, (st.message or "")[:300], flush=True)
if str(st.code) != "SUCCESS": sys.exit(1)
tm = cj.get_target_model()
pj = hub.submit_profile_job(model=tm, device=dev, name=f"melo_{key}_profile"); ij = hub.submit_inference_job(model=tm, device=dev, inputs=IN, name=f"melo_{key}_infer")
J[key].update(profile=pj.job_id, infer=ij.job_id); json.dump(J, open(JF, "w"), indent=1); print(key, "profile", pj.job_id, "| infer", ij.job_id, flush=True)
t0 = time.time(); ij.wait(); st = ij.get_status(); print(key, "infer", st.code, f"wall {time.time()-t0:.0f}s", (st.message or "")[:300], flush=True)
if str(st.code) == "SUCCESS":
    od = ij.download_output_data(); ks = sorted(od.keys(), key=lambda s: int(s.split("_")[-1]))     # output_0 = audio, output_1 = y_len
    A = [np.asarray(a).reshape(-1) for a in od[ks[0]]]; YL = [int(np.asarray(a).reshape(-1)[0]) for a in od[ks[1]]]; np.savez(f"work/hw_{key}.npz", audio=np.stack(A), y_len=np.array(YL))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
    nc = lambda x: re.sub(r"[^\w]", "", x); cers = []; os.makedirs(f"work/wav_{key}", exist_ok=True)
    for i, t in enumerate(SENT):
        ref = s.run(None, {k: v[i] for k, v in IN.items()}); L = int(ref[1][0]) * 512; r = ref[0][0, :L].astype(np.float64); a = A[i][:L].astype(np.float64)
        c = float(r @ a / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)); sf.write(f"work/wav_{key}/{i}.wav", A[i][:YL[i] * 512], 44100)
        x = resample_poly(A[i][:YL[i] * 512].astype(np.float32), 16000, 44100).astype(np.float32); m_ = min(len(x), 464000); buf = np.zeros(464000, np.float32); buf[:m_] = x[:m_]
        o = asr.run(None, {"wav": buf.reshape(1, -1), "wav_len": np.array([m_], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
        h = bytes(int(v) & 0xFF for v in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
        cer = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cers.append(cer)
        print(f"  [{i}] frames npu/cpu {YL[i]}/{int(ref[1][0])} | cos {c:.4f} | peak {np.abs(A[i][:L]).max():.2f} | CER {cer:.2f} | {h}", flush=True)
    print(key, "NPU mean CER %.3f" % np.mean(cers), flush=True)
try:
    pj.wait(); p = pj.download_profile(); ex = p["execution_detail"]; es = p["execution_summary"]; import collections; u = collections.Counter(l.get("compute_unit", "?") for l in ex)
    print(key, "PROFILE", f"{es['estimated_inference_time']/1000:.1f} ms | layers {len(ex)} {dict(u)} | peak mem {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB | first load {es.get('first_load_time',0)/1e6:.0f}s", flush=True)
except Exception as e: print(key, "profile failed:", str(e)[:200], flush=True)
