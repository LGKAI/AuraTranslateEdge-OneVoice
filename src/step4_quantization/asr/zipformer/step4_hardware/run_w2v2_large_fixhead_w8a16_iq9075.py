# -*- coding: utf-8 -*-
"""wav2vec2-large-vi-vlsp2020 (CTC, khong vong lap) -> w8a16 -> compile -> profile + inference tren Dragonwing IQ-9075.

Khac ban try_no_loop_wav2vec2_large_iq9075.py (14/8, bo cuoc vi WER local = 1.0):
  1. Dung lai head `feature_transform.*` (Linear->BN->ReLU x3) ma Wav2Vec2ForCTC chuan cua HF am tham bo qua
     -> day la nguyen nhan WER 1.0, khong phai model hong.
  2. Chuan hoa dau vao dung nhu processor (zero-mean/unit-var TREN DO DAI THAT, roi moi dem 0) cho ca
     calibration lan inference (ban cu dua song tho, khong chuan hoa).
  3. Bucket tinh 15 s (240000 mau) = khop bucket 1500 frame cua Zipformer, phu du ca 5 cau test.
  4. Do them tren audio nhieu (data/asr_mixed) de so voi Zipformer (5dB 6.22% / 0dB 4.10%).
ONNX da export boi outputs/w2v2_ctc_onnx/w2v2_large_static_15s.onnx (cosine ORT vs torch = 0.9999997).
Chay trong tools/venv_nllb_export (torch 2.5.1 + transformers 4.46.3 + qai-hub)."""
import glob, json, os, re, sys, time
import numpy as np
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
import jiwer
import qai_hub as hub
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
ONNX = os.path.join(ROOT, "outputs", "w2v2_ctc_onnx", "w2v2_large_static_15s.onnx")
RESULT = os.path.join(ROOT, "outputs", "w2v2_ctc_onnx", "result_large_iq9075.json")
DEVICE = "Dragonwing IQ-9075 EVK"
N = 240000
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()

path = glob.glob("third_party_wav2vec2_large_vi/**/snapshots/*/", recursive=True)[0]
proc = Wav2Vec2Processor.from_pretrained(path)
model = Wav2Vec2ForCTC.from_pretrained(path).eval()
sd = torch.load(path + "pytorch_model.bin", map_location="cpu", weights_only=False)


def head(h):
    for i in (1, 2, 3):
        y = F.linear(h, sd[f"feature_transform.linear{i}.weight"], sd[f"feature_transform.linear{i}.bias"])
        m, v, g, b = (sd[f"feature_transform.bn{i}.{n}"] for n in ("running_mean", "running_var", "weight", "bias"))
        h = F.relu((y - m) / torch.sqrt(v + 1e-5) * g + b)
    return h


def prep(wav):
    x = proc(wav[:N], sampling_rate=16000, return_tensors="pt").input_values      # normalize tren do dai that
    return F.pad(x, (0, N - x.shape[1])).numpy().astype(np.float32)               # (1, N)


def local_logits(x):
    with torch.no_grad():
        return model.lm_head(head(model.wav2vec2(torch.from_numpy(x)).last_hidden_state)).numpy()


def decode(logits):
    return proc.batch_decode(torch.from_numpy(logits).argmax(-1))[0]


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []   # (nhan, ref, wav)
for r in rows:
    w, _ = sf.read(r["path"], dtype="float32"); items.append(("clean", r["transcript"], w))
for snr in (5, 0):
    for r in rows:
        f = r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_snr{snr}.wav")
        if os.path.exists(f):
            w, _ = sf.read(f, dtype="float32"); items.append((f"snr{snr}", r["transcript"], w))
calib_items = [it for it in items if it[0] == "clean"]        # + snr10/snr15 (khong dung de danh gia)
for r in rows[:5]:
    for snr in (10, 15):
        f = r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_snr{snr}.wav")
        if os.path.exists(f):
            w, _ = sf.read(f, dtype="float32"); calib_items.append((f"snr{snr}", "", w))
X = [prep(w) for _, _, w in items]
XC = [prep(w) for _, _, w in calib_items]
print(f"eval={len(X)} mau ({sorted(set(t for t,_,_ in items))}) | calibration={len(XC)} mau", flush=True)

t0 = time.time(); ref_logits = [local_logits(x) for x in X]; print(f"local fp32 xong {time.time()-t0:.0f}s", flush=True)


def poll(job, label, every=30):
    while True:
        st = job.get_status(); print(f"[{label}] {job.job_id}: {st.code} {(st.message or '')[:150]}", flush=True)
        if st.code in ("SUCCESS", "FAILED", "CANCELLED"): return st
        time.sleep(every)


device = hub.Device(DEVICE)
if os.environ.get("QUANT_JOB"):        # tai dung job quantize da SUCCESS (tranh upload lai 1.27GB)
    qjob = hub.get_job(os.environ["QUANT_JOB"]); print("REUSE QUANT", qjob.job_id, flush=True)
else:
    calib = hub.upload_dataset({"input_values": XC}, name="w2v2_large_vi_calib_norm")
    qjob = hub.submit_quantize_job(model=ONNX, calibration_data=calib, weights_dtype=hub.QuantizeDtype.INT8,
                                   activations_dtype=hub.QuantizeDtype.INT16, name="w2v2-large-vi-w8a16")
    print("QUANT", qjob.job_id, qjob.url, flush=True)
if poll(qjob, "quant").code != "SUCCESS": sys.exit("quantize FAILED")
cjob = hub.submit_compile_job(model=qjob.get_target_model(), device=device,
                              input_specs={"input_values": ((1, N), "float32")},
                              options=f"--target_runtime {os.environ.get('COMPILE_RUNTIME', 'qnn_dlc')} --quantize_io --truncate_64bit_io", name="w2v2-large-vi-compile")
print("COMPILE", cjob.job_id, cjob.url, flush=True)
cst = poll(cjob, "compile")
res = {"quant_job": qjob.job_id, "compile_job": cjob.job_id, "compile_status": cst.code, "compile_msg": (cst.message or "")[:300]}
json.dump(res, open(RESULT, "w"), indent=1)
if cst.code != "SUCCESS": sys.exit("compile FAILED: " + (cst.message or ""))
tm = cjob.get_target_model()
pjob = hub.submit_profile_job(model=tm, device=device, name="w2v2-large-vi-profile")
ijob = hub.submit_inference_job(model=tm, device=device, inputs={"input_values": X}, name="w2v2-large-vi-infer")
print("PROFILE", pjob.job_id, pjob.url, "\nINFER", ijob.job_id, ijob.url, flush=True)
ist = poll(ijob, "infer"); pst = poll(pjob, "profile")
res.update(profile_job=pjob.job_id, infer_job=ijob.job_id, infer_status=ist.code, profile_status=pst.code)
if pst.code == "SUCCESS":
    prof = pjob.download_profile()
    res["profile_summary"] = {k: (v if isinstance(v, (int, float, str)) else str(v)[:200])
                              for k, v in prof.get("execution_summary", {}).items()}
if ist.code == "SUCCESS":
    out = ijob.download_output_data(); key = list(out.keys())[0]
    hw = [np.asarray(a).reshape(ref_logits[0].shape) for a in out[key]]
    per = {}
    for (tag, ref, _), rl, hl in zip(items, ref_logits, hw):
        a, b = rl.reshape(-1), hl.reshape(-1)
        d = per.setdefault(tag, {"cos": [], "argmax_match": [], "wer_fp": [], "wer_hw": []})
        d["cos"].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        d["argmax_match"].append(float((rl.argmax(-1) == hl.argmax(-1)).mean()))
        d["wer_fp"].append(jiwer.wer(norm(ref), norm(decode(rl))))
        d["wer_hw"].append(jiwer.wer(norm(ref), norm(decode(hl))))
    res["by_condition"] = {t: {k: round(float(np.mean(v)), 4) for k, v in d.items()} | {"n": len(d["cos"])}
                           for t, d in per.items()}
    res["examples"] = [{"tag": t, "ref": r, "fp": decode(rl), "hw": decode(hl)}
                       for (t, r, _), rl, hl in list(zip(items, ref_logits, hw))[:2]]
json.dump(res, open(RESULT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
