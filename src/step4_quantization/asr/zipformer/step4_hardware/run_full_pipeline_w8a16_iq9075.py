# -*- coding: utf-8 -*-
"""Graph GOP DUY NHAT: raw audio -> fbank(matmul) -> encoder+CTC(150M) -> argmax -> collapse -> detokenize
-> byte_matrix/byte_len. Quantize w8a16 -> compile qnn_dlc -> profile + inference THAT tren Dragonwing IQ-9075.
Da verify local (ONNX Runtime CPU) khop CHINH XAC voi pipeline cu tach rieng: clean 6.22%, snr5 8.72%, snr0 4.17%.
Chay trong tools/venv_nllb_export."""
import json, os, re, sys, time
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
ONNX = "outputs/zip150_full_npu/zip150_full_pipeline.onnx"
RESULT = "outputs/zip150_full_npu/result_iq9075.json"
DEVICE = "Dragonwing IQ-9075 EVK"
N_SAMPLES = 240240
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()


def prep_wave(wav):
    n = min(wav.shape[0], N_SAMPLES)
    x = np.zeros((N_SAMPLES,), dtype=np.float32)
    x[:n] = wav[:n]                                   # KHONG scale 32768 -- da xac nhan bug that, xem fbank_matmul_verify.py
    n_frames = 1 + (n - 400) // 160
    return x, n_frames


def decode_output(byte_matrix, byte_len):
    out = bytearray()
    for row, l in zip(byte_matrix, byte_len):
        l = int(l)
        if l <= 0:
            continue
        out += bytes(int(b) & 0xFF for b in row[:l])
    return out.decode("utf-8", errors="replace")


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f):
            w, _ = sf.read(f, dtype="float32"); items.append((tag, r["transcript"], w))
print(f"eval={len(items)} mau", flush=True)

calib_x, calib_lens = [], []
for _, _, w in items[:5]:
    x, n = prep_wave(w); calib_x.append(x); calib_lens.append(np.array([n], np.int64))   # fb_raw_wave_flat la 1-D [N_SAMPLES], KHONG them chieu batch


def poll(job, label, every=30):
    while True:
        st = job.get_status(); print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:150]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"): return st
        time.sleep(every)


device = hub.Device(DEVICE)
if os.environ.get("QUANT_JOB"):
    qjob = hub.get_job(os.environ["QUANT_JOB"]); print("REUSE QUANT", qjob.job_id, flush=True)
else:
    calib = hub.upload_dataset({"fb_raw_wave_flat": calib_x, "enc_x_lens": calib_lens}, name="zip150_full_calib")
    print("calib uploaded", calib, flush=True)
    qjob = hub.submit_quantize_job(model=ONNX, calibration_data=calib, weights_dtype=hub.QuantizeDtype.INT8,
                                   activations_dtype=hub.QuantizeDtype.INT16, name="zip150-full-w8a16")
    print("QUANT", qjob.job_id, qjob.url, flush=True)
if poll(qjob, "quant").code != "SUCCESS": sys.exit("quantize FAILED")

if os.environ.get("COMPILE_JOB"):
    cjob = hub.get_job(os.environ["COMPILE_JOB"]); print("REUSE COMPILE", cjob.job_id, flush=True)
else:
    cjob = hub.submit_compile_job(
        model=qjob.get_target_model(), device=device,
        input_specs={"fb_raw_wave_flat": ((N_SAMPLES,), "float32"), "enc_x_lens": ((1,), "int64")},
        options="--target_runtime qnn_dlc --quantize_io --truncate_64bit_io", name="zip150-full-compile")
    print("COMPILE", cjob.job_id, cjob.url, flush=True)
cst = poll(cjob, "compile")
res = {"quant_job": qjob.job_id, "compile_job": cjob.job_id, "compile_status": cst.code, "compile_msg": (cst.message or "")[:500]}
json.dump(res, open(RESULT, "w"), indent=1)
if cst.code != "SUCCESS":
    print("COMPILE FAILED:", cst.message, flush=True); sys.exit(1)

tm = cjob.get_target_model()
pjob = hub.submit_profile_job(model=tm, device=device, name="zip150-full-profile")
ijob = hub.submit_inference_job(
    model=tm, device=device,
    inputs={"fb_raw_wave_flat": [x.astype(np.float32) for _, _, w in items for x, n in [prep_wave(w)]],
            "enc_x_lens": [np.array([n], np.int32) for _, _, w in items for x, n in [prep_wave(w)]]},   # HTP can int32
    name="zip150-full-infer")
print("PROFILE", pjob.job_id, pjob.url, "\nINFER", ijob.job_id, ijob.url, flush=True)
ist = poll(ijob, "infer"); pst = poll(pjob, "profile")
res.update(profile_job=pjob.job_id, infer_job=ijob.job_id, infer_status=ist.code, profile_status=pst.code)
if pst.code == "SUCCESS":
    p = pjob.download_profile(); es = p["execution_summary"]
    res["profile_ms"] = es["estimated_inference_time"] / 1000
    res["profile_load_s"] = es["first_load_time"] / 1e6
    res["peak_mem_mb"] = es["estimated_inference_peak_memory"] / 1e6
if ist.code == "SUCCESS":
    out = ijob.download_output_data()
    print("output keys:", list(out.keys()), flush=True)
    bm_key = [k for k in out if np.asarray(out[k][0]).shape == (373, 12)][0]
    bl_key = [k for k in out if k != bm_key][0]
    per = {}
    for (tag, ref, w), bm, bl in zip(items, out[bm_key], out[bl_key]):
        hyp = decode_output(np.asarray(bm), np.asarray(bl))
        wr = jiwer.wer(norm(ref), norm(hyp))
        per.setdefault(tag, []).append(wr)
    res["by_condition"] = {t: {"wer": round(float(np.mean(v)), 4), "n": len(v)} for t, v in per.items()}
    res["examples"] = []
    for (tag, ref, w), bm, bl in list(zip(items, out[bm_key], out[bl_key]))[:6]:
        hyp = decode_output(np.asarray(bm), np.asarray(bl))
        res["examples"].append({"tag": tag, "ref": ref, "hyp": hyp})
json.dump(res, open(RESULT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
