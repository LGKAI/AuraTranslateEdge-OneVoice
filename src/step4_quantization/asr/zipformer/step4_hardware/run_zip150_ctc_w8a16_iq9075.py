# -*- coding: utf-8 -*-
"""ZipFormer-150M CTC (khong train, head co san) -> w8a16 -> compile qnn_dlc -> profile + inference tren IQ-9075.
ONNX tinh da dung boi build_zip150_ctc_static.py (outputs/zip150_ctc/zip150_ctc_static_1500.onnx, 597MB,
WER CTC local: clean 4.17%, snr5 8.72%, snr0 5.42% -- tot hon RNN-T 30M dang dung 6.22%).
Chay trong tools/venv_nllb_export."""
import json, os, re, sys, time
import numpy as np, onnxruntime as ort, soundfile as sf, jiwer
import sentencepiece as spm, kaldi_native_fbank as knf
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
ONNX = os.path.join(ROOT, "outputs", "zip150_ctc", "zip150_ctc_static_1500.onnx")
RESULT = os.path.join(ROOT, "outputs", "zip150_ctc", "result_iq9075.json")
DEVICE = "Dragonwing IQ-9075 EVK"
T = 1500
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()
sp = spm.SentencePieceProcessor(); sp.load("third_party_zipformer_150m_crctc/bpe.model")


def fbank(wav):
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = -400.0
    f = knf.OnlineFbank(o); f.accept_waveform(16000, wav.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def pad(feats):
    n = min(feats.shape[0], T); x = np.zeros((T, 80), np.float32); x[:n] = feats[:n]
    return x, n


def collapse(ids):
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev: out.append(int(t))
        prev = t
    return sp.decode(out)


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f):
            w, _ = sf.read(f, dtype="float32"); items.append((tag, r["transcript"], *pad(fbank(w))))
print(f"eval={len(items)} mau", flush=True)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
ref_logits, ref_lens = [], []
t0 = time.time()
for _, _, x, n in items:
    lg, el = sess.run(None, {"x": x[None], "x_lens": np.array([n], np.int64)})
    ref_logits.append(lg); ref_lens.append(int(el[0]))
print(f"local fp32 xong {time.time()-t0:.0f}s", flush=True)

calib_x = [x[None].astype(np.float32) for _, _, x, n in items[:5]]                 # 5 cau sach lam calibration
calib_lens = [np.array([n], np.int64) for _, _, x, n in items[:5]]


def poll(job, label, every=25):
    while True:
        st = job.get_status(); print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:150]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"): return st
        time.sleep(every)


device = hub.Device(DEVICE)
if os.environ.get("QUANT_JOB"):
    qjob = hub.get_job(os.environ["QUANT_JOB"]); print("REUSE QUANT", qjob.job_id, flush=True)
else:
    calib = hub.upload_dataset({"x": calib_x, "x_lens": calib_lens}, name="zip150_ctc_calib")
    qjob = hub.submit_quantize_job(model=ONNX, calibration_data=calib, weights_dtype=hub.QuantizeDtype.INT8,
                                   activations_dtype=hub.QuantizeDtype.INT16, name="zip150-ctc-w8a16")
    print("QUANT", qjob.job_id, qjob.url, flush=True)
if poll(qjob, "quant").code != "SUCCESS": sys.exit("quantize FAILED")

if os.environ.get("COMPILE_JOB"):
    cjob = hub.get_job(os.environ["COMPILE_JOB"]); print("REUSE COMPILE", cjob.job_id, flush=True)
else:
    cjob = hub.submit_compile_job(model=qjob.get_target_model(), device=device,
                                  input_specs={"x": ((1, T, 80), "float32"), "x_lens": ((1,), "int64")},
                                  options="--target_runtime qnn_dlc --quantize_io --truncate_64bit_io", name="zip150-ctc-compile")
    print("COMPILE", cjob.job_id, cjob.url, flush=True)
cst = poll(cjob, "compile")
res = {"quant_job": qjob.job_id, "compile_job": cjob.job_id, "compile_status": cst.code, "compile_msg": (cst.message or "")[:300]}
json.dump(res, open(RESULT, "w"), indent=1)
if cst.code != "SUCCESS": sys.exit("compile FAILED: " + (cst.message or ""))
tm = cjob.get_target_model()
pjob = hub.submit_profile_job(model=tm, device=device, name="zip150-ctc-profile")
ijob = hub.submit_inference_job(model=tm, device=device,
                                inputs={"x": [x[None].astype(np.float32) for _, _, x, n in items],
                                        "x_lens": [np.array([n], np.int32) for _, _, x, n in items]},  # HTP can int32, khong phai int64
                                name="zip150-ctc-infer-v2")
print("PROFILE", pjob.job_id, pjob.url, "\nINFER", ijob.job_id, ijob.url, flush=True)
ist = poll(ijob, "infer"); pst = poll(pjob, "profile")
res.update(profile_job=pjob.job_id, infer_job=ijob.job_id, infer_status=ist.code, profile_status=pst.code)
if pst.code == "SUCCESS":
    p = pjob.download_profile(); es = p["execution_summary"]
    res["profile_ms"] = es["estimated_inference_time"] / 1000
    res["profile_load_s"] = es["first_load_time"] / 1e6
    ex = p["execution_detail"]; tot = sum(l["execution_cycles"] for l in ex)
    top = sorted(ex, key=lambda l: -l["execution_cycles"])[:3]
    res["top_layers"] = [{"name": l["name"][-70:], "pct": round(l["execution_cycles"] / tot * 100, 1)} for l in top]
if ist.code == "SUCCESS":
    out = ijob.download_output_data(); key = [k for k in out if "logit" in k.lower()][0]
    lkey = [k for k in out if k != key][0]
    per = {}
    for (tag, ref, x, n), hl, hel, rl, rn in zip(items, out[key], out[lkey], ref_logits, ref_lens):
        hl = np.asarray(hl).reshape(1, T, 2000); real = int(np.asarray(hel).reshape(-1)[0])
        a, b = rl[0, :min(rn, real)].reshape(-1), hl[0, :min(rn, real)].reshape(-1)
        d = per.setdefault(tag, {"cos": [], "wer_fp": [], "wer_hw": []})
        d["cos"].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        d["wer_fp"].append(jiwer.wer(norm(ref), norm(collapse(rl[0, :rn].argmax(-1)))))
        d["wer_hw"].append(jiwer.wer(norm(ref), norm(collapse(hl[0, :real].argmax(-1)))))
    res["by_condition"] = {t: {k: round(float(np.mean(v)), 4) for k, v in d.items()} | {"n": len(d["cos"])} for t, d in per.items()}
json.dump(res, open(RESULT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
