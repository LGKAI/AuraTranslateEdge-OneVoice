# -*- coding: utf-8 -*-
"""Danh gia 90 cau (30 en/zh/ko) tren NPU THAT, pipeline moi:
  Stage 1: frontend v3 (chuan hoa theo frame)  -> fbank
  Stage 2: encoder quantized W8A16             -> byte_stream
Moi stage = 1 batch job (tra 'thue tai model' 1 lan/stage)."""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import qai_hub as hub
from sv_eval30_common import *

OUT = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
FE_V3_TARGET = "mq26z3d0n"        # frontend v3 da compile (job j57ez99vp)
ENC_COMPILE_JOB = "jprlmj9kp"     # encoder quantized W8A16
device = hub.Device("Dragonwing IQ-9075 EVK")

items = load_items()
N = len(items)
print(f"[0] {N} cau: " + ", ".join(f"{L}={sum(i['lang']==L for i in items)}" for L in ('en', 'zh', 'ko')), flush=True)

wavs, lens = zip(*[load_wav_padded(it) for it in items])
t0 = time.time()

print("[1] Stage 1 frontend v3 (1 batch job)", flush=True)
ds1 = hub.upload_dataset({"wav": list(wavs), "wav_len": list(lens)})
j1 = hub.submit_inference_job(model=hub.get_model(FE_V3_TARGET), device=device, inputs=ds1)
print("  fe_job_id=", j1.job_id, flush=True)
j1.wait(); print("  state:", j1.get_status().state, f"({time.time()-t0:.0f}s)", flush=True)
o1 = j1.download_output_data()
fk = [k for k in o1 if np.asarray(o1[k][0]).reshape(-1).shape[0] == 500 * 560][0]
sk = [k for k in o1 if k != fk][0]
fbanks = [np.asarray(o1[fk][i]).reshape(1, 500, 560).astype(np.float32) for i in range(N)]
sls = [np.asarray(o1[sk][i]).reshape(1).astype(np.int32) for i in range(N)]

print("[2] Stage 2 encoder quantized (1 batch job)", flush=True)
enc = hub.get_job(ENC_COMPILE_JOB).get_target_model()
ds2 = hub.upload_dataset({
    "fbank": fbanks, "speech_lengths": sls,
    "language": [np.array([LID[it["lang"]]], dtype=np.int32) for it in items],
    "textnorm": [np.array([TEXTNORM_WITHITN], dtype=np.int32)] * N,
})
j2 = hub.submit_inference_job(model=enc, device=device, inputs=ds2)
print("  enc_job_id=", j2.job_id, flush=True)
j2.wait(); print("  state:", j2.get_status().state, f"({time.time()-t0:.0f}s)", flush=True)
o2 = j2.download_output_data()
outs = o2[list(o2.keys())[0]]
hyps = [decode_bytes(o) for o in outs]

per, summ = summarize(items, hyps)
print_summary("EVAL30 TREN NPU THAT: frontend v3 + encoder quantized", summ)
res = {"pipeline": "frontend_v3 (mq26z3d0n) + encoder_quantized (jprlmj9kp)", "fe_job": j1.job_id,
       "enc_job": j2.job_id, "summary": summ, "per_sample": per}
json.dump(res, open(os.path.join(OUT, "eval30_npu_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved eval30_npu_results.json", flush=True)
