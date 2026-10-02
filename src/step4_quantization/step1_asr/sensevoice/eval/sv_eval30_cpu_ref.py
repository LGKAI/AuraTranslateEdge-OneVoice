# -*- coding: utf-8 -*-
"""Tham chieu fp32 tren CPU (ONNXRuntime) cho cung 90 cau: tran chat luong ly tuong
cua graph (frontend fp32 chinh xac + encoder fp32). Dung model GOP model_sv_combined_single_inline.onnx."""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, onnxruntime as ort
from sv_eval30_common import *

OUT = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
items = load_items()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(os.path.join(OUT, "model_sv_combined_single_inline.onnx"), so, providers=["CPUExecutionProvider"])
hyps, t0 = [], time.time()
for i, it in enumerate(items):
    wav, ln = load_wav_padded(it)
    o = sess.run(None, {"wav": wav, "wav_len": ln,
                        "language": np.array([LID[it["lang"]]], dtype=np.int32),
                        "textnorm": np.array([TEXTNORM_WITHITN], dtype=np.int32)})
    hyps.append(decode_bytes(o[0]))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(items)} xong ({time.time()-t0:.0f}s)", flush=True)
per, summ = summarize(items, hyps)
print_summary("EVAL30 THAM CHIEU fp32 CPU (graph gop)", summ)
json.dump({"pipeline": "combined graph fp32 CPU (reference)", "summary": summ, "per_sample": per},
          open(os.path.join(OUT, "eval30_cpu_ref_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved eval30_cpu_ref_results.json", flush=True)
