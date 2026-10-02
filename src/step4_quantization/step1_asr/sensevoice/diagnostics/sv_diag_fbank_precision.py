# -*- coding: utf-8 -*-
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import qai_hub as hub

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified_fixed.npz")
FE_ONNX = os.path.join(OUT_DIR, "model_sv_frontend_inline.onnx")

calib = np.load(CALIB_NPZ)
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]
# result quality tu chain that (de doi chieu)
wer_cer = {"en_0":94.7,"en_1":95.2,"en_2":42.4,"en_3":53.3,"en_4":0.0,
           "zh_0":100.0,"zh_1":95.2,"zh_2":100.0,"zh_3":100.0,"zh_4":100.0,
           "ko_0":100.0,"ko_1":100.0,"ko_2":100.0,"ko_3":100.0,"ko_4":100.0}

print("[1] Tai fbank that tu NPU (job jgk21e6yg)...")
fe_job = hub.get_job("jgk21e6yg")
fe_out = fe_job.download_output_data()
fbank_key = [k for k in fe_out if np.asarray(fe_out[k][0]).reshape(-1).shape[0] == 500*560][0]
fbank_npu_list = fe_out[fbank_key]

print("[2] Chay frontend LOCAL fp32 (CPU, onnxruntime) tren cung wav...")
sess = ort.InferenceSession(FE_ONNX, providers=["CPUExecutionProvider"])
in_names = [i.name for i in sess.get_inputs()]
print("  frontend onnx inputs:", in_names)

print("\n[3] So sanh fbank NPU that vs fbank CPU fp32 (cosine sim, max_abs_diff, rel L2)...")
for i in range(len(fbank_npu_list)):
    wav = calib["wav"][i].astype(np.float32)
    wav_len = calib["wav_len"][i].reshape(1).astype(np.int32)
    out = sess.run(None, {"wav": wav, "wav_len": wav_len})
    fbank_cpu = out[0].reshape(-1)
    fbank_npu = np.asarray(fbank_npu_list[i]).reshape(-1)
    a, b = fbank_cpu.astype(np.float64), fbank_npu.astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    lbl = labels[i] if i < len(labels) else f"idx{i}"
    q = wer_cer.get(lbl, -1)
    print(f"  [{lbl}] cos_sim={cos:.6f}  max_abs_diff={max_abs:.5f}  rel_L2={rel_l2:.5f}  (chain_result={q}%)")
