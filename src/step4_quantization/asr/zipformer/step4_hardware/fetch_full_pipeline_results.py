# -*- coding: utf-8 -*-
"""Lay lai ket qua tu 2 job da SUCCESS (jgjrw6o1p infer, jpvl9yez5 profile) -- khong chay lai tu dau."""
import json, os, re
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
RESULT = "outputs/zip150_full_npu/result_iq9075.json"
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

ijob = hub.get_job("jgjrw6o1p")
pjob = hub.get_job("jpvl9yez5")
res = {"quant_job": "j5qlm1y7p", "compile_job": "jgnz7q8kg", "compile_status": "SUCCESS",
      "profile_job": "jpvl9yez5", "infer_job": "jgjrw6o1p", "infer_status": "SUCCESS", "profile_status": "SUCCESS"}

p = pjob.download_profile(); es = p["execution_summary"]
res["profile_ms"] = es["estimated_inference_time"] / 1000
res["profile_load_s"] = es["first_load_time"] / 1e6
res["peak_mem_mb"] = es["estimated_inference_peak_memory"] / 1e6
print("profile:", res["profile_ms"], "ms |", res["profile_load_s"], "s load |", res["peak_mem_mb"], "MB peak")

out = ijob.download_output_data()
print("output keys:", list(out.keys()))
bm_key = [k for k in out if np.asarray(out[k][0]).shape == (373, 12)][0]
bl_key = [k for k in out if k != bm_key][0]

per = {}
examples = []
for (tag, ref, w), bm, bl in zip(items, out[bm_key], out[bl_key]):
    hyp = decode_output(np.asarray(bm), np.asarray(bl))
    wr = jiwer.wer(norm(ref), norm(hyp))
    per.setdefault(tag, []).append(wr)
    if len(examples) < 10:
        examples.append({"tag": tag, "ref": ref, "hyp": hyp, "wer": round(wr, 4)})

res["by_condition"] = {t: {"wer": round(float(np.mean(v)), 4), "n": len(v)} for t, v in per.items()}
res["examples"] = examples
with open(RESULT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)

import sys
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
