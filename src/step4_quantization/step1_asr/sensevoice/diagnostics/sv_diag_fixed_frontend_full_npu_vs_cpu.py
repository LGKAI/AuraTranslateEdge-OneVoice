# -*- coding: utf-8 -*-
"""So sanh fbank DAY DU (sau mel/log/LFR/CMVN) cua frontend DA FIX: NPU that (job jp4yzyjvp)
vs CPU fp32. Truoc gio chi moi verify Stage A (power spectrum) tren NPU -- chua verify phan
log/LFR/CMVN sau khi fix. Gia thuyet: bo scale x32768 lam gia tri mel qua nho -> tran so
DUOI (underflow) fp16 o vung im lang / bin tan so cao."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import qai_hub as hub

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
FE_FIXED = os.path.join(OUT_DIR, "model_sv_frontend_fixed_inline.onnx")
calib = np.load(os.path.join(OUT_DIR, "calib_data_unified_fixed.npz"))
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]

fe_out = hub.get_job("jp4yzyjvp").download_output_data()
fbank_key = [k for k in fe_out if np.asarray(fe_out[k][0]).reshape(-1).shape[0] == 500 * 560][0]
npu_list = fe_out[fbank_key]

sess = ort.InferenceSession(FE_FIXED, providers=["CPUExecutionProvider"])

print(f"{'mau':6s} {'cos_valid':>10s} {'relL2_valid':>12s} {'maxabs':>9s} {'frames_xau(>0.5)':>17s} {'min_npu':>9s} {'min_cpu':>9s}")
for i in range(len(npu_list)):
    wav = calib["wav"][i].astype(np.float32)
    wl = calib["wav_len"][i].reshape(1).astype(np.int32)
    cpu_fb, cpu_sl = sess.run(None, {"wav": wav, "wav_len": wl})
    cpu_fb = cpu_fb.reshape(500, 560).astype(np.float64)
    npu_fb = np.asarray(npu_list[i]).reshape(500, 560).astype(np.float64)
    n_valid = int(np.asarray(cpu_sl).reshape(-1)[0])
    a, b = cpu_fb[:n_valid], npu_fb[:n_valid]
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    maxabs = float(np.abs(a - b).max())
    bad_frames = int((np.abs(a - b).max(axis=1) > 0.5).sum())
    print(f"{labels[i]:6s} {cos:10.6f} {rel:12.5f} {maxabs:9.4f} {bad_frames:6d}/{n_valid:<10d} {b.min():9.3f} {a.min():9.3f}")
