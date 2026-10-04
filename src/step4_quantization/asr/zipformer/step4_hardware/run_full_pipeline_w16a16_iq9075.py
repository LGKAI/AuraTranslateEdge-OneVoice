# -*- coding: utf-8 -*-
"""Ban sua loi: nang tu w8a16 len w16a16 (ca weight cung INT16, khong chi INT8) cho graph GOP DUY NHAT.
Ly do: da xac nhan bang thuc nghiem (diag_first_word_drop_fetch.py, diag_fbank_fakequant.py) -- encoder+CTC
w8a16 rieng KHONG loi (WER 7.02%, khong mat tu dau cau); fake-quant int8 tren TRONG SO fbank cung KHONG
tai hien duoc loi -- nen nghi van con lai la ACTIVATION quantize noi bo cua graph gop (khac boundary I/O
cua ban encoder-only truoc day). w16a16 la cong thuc chuan da dung khi PTQ lam hong do chinh xac trong
du an nay (vd NLLB, Zipformer 30M truoc day). Cung mo rong calibration: 5 sach + 5 nhieu, khong chi 5 sach.
Chay trong tools/venv_nllb_export."""
import json, os, re, sys, time
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
ONNX = "outputs/zip150_full_npu/zip150_full_pipeline.onnx"
RESULT = "outputs/zip150_full_npu/result_iq9075_w16a16.json"
DEVICE = "Dragonwing IQ-9075 EVK"
N_SAMPLES = 240240
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()


def prep_wave(wav):
    n = min(wav.shape[0], N_SAMPLES)
    x = np.zeros((N_SAMPLES,), dtype=np.float32)
    x[:n] = wav[:n]
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
for _, _, w in items[:5] + items[5:8]:          # 5 sach + 3 nhieu5dB -- da dang hon ban truoc (chi 5 sach)
    x, n = prep_wave(w); calib_x.append(x); calib_lens.append(np.array([n], np.int64))


def poll(job, label, every=30):
    while True:
        st = job.get_status(); print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:150]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"): return st
        time.sleep(every)


device = hub.Device(DEVICE)
if os.environ.get("QUANT_JOB"):
    qjob = hub.get_job(os.environ["QUANT_JOB"]); print("REUSE QUANT", qjob.job_id, flush=True)
else:
    calib = hub.upload_dataset({"fb_raw_wave_flat": calib_x, "enc_x_lens": calib_lens}, name="zip150_full_calib_w16a16")
    print("calib uploaded", calib, flush=True)
    qjob = hub.submit_quantize_job(model=ONNX, calibration_data=calib, weights_dtype=hub.QuantizeDtype.INT16,
                                   activations_dtype=hub.QuantizeDtype.INT16, name="zip150-full-w16a16")
    print("QUANT", qjob.job_id, qjob.url, flush=True)
if poll(qjob, "quant").code != "SUCCESS": sys.exit("quantize FAILED")

if os.environ.get("COMPILE_JOB"):
    cjob = hub.get_job(os.environ["COMPILE_JOB"]); print("REUSE COMPILE", cjob.job_id, flush=True)
else:
    cjob = hub.submit_compile_job(
        model=qjob.get_target_model(), device=device,
        input_specs={"fb_raw_wave_flat": ((N_SAMPLES,), "float32"), "enc_x_lens": ((1,), "int64")},
        options="--target_runtime qnn_dlc --quantize_io --truncate_64bit_io", name="zip150-full-compile-w16a16")
    print("COMPILE", cjob.job_id, cjob.url, flush=True)
cst = poll(cjob, "compile")
res = {"quant_job": qjob.job_id, "compile_job": cjob.job_id, "compile_status": cst.code, "compile_msg": (cst.message or "")[:500]}
with open(RESULT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
if cst.code != "SUCCESS":
    print("COMPILE FAILED:", cst.message, flush=True); sys.exit(1)

tm = cjob.get_target_model()
pjob = hub.submit_profile_job(model=tm, device=device, name="zip150-full-profile-w16a16")
ijob = hub.submit_inference_job(
    model=tm, device=device,
    inputs={"fb_raw_wave_flat": [x.astype(np.float32) for _, _, w in items for x, n in [prep_wave(w)]],
            "enc_x_lens": [np.array([n], np.int32) for _, _, w in items for x, n in [prep_wave(w)]]},
    name="zip150-full-infer-w16a16")
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
    for (tag, ref, w), bm, bl in list(zip(items, out[bm_key], out[bl_key]))[:10]:
        hyp = decode_output(np.asarray(bm), np.asarray(bl))
        res["examples"].append({"tag": tag, "ref": ref, "hyp": hyp})
with open(RESULT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
