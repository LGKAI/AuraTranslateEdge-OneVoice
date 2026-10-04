# -*- coding: utf-8 -*-
"""Compile Stage A FIXED (khong scale x32768 truoc FFT) len NPU that, chay inference,
so sanh voi tham chieu CPU fp32 (cung khong scale) de kiem tra fix co giai quyet duoc
van de tran so fp16 hay khong."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import qai_hub as hub

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified_fixed.npz")
ONNX_PATH = os.path.join(OUT_DIR, "model_sv_frontend_stageA_power_fixed_inline.onnx")

device = hub.Device(DEVICE_NAME)
calib = np.load(CALIB_NPZ)
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]

print("[1] Upload + compile Stage A FIXED len NPU that (KHONG quantize)...", flush=True)
model = hub.upload_model(ONNX_PATH)
print("  model_id=", model.model_id, flush=True)
cjob = hub.submit_compile_job(model=model, device=device, options="--target_runtime qnn_dlc --truncate_64bit_io")
print("  compile_job_id=", cjob.job_id, flush=True)
cjob.wait()
print("  compile state:", cjob.get_status().state, flush=True)
target = cjob.get_target_model()

print("\n[2] Chay inference tren NPU that (15-sample test set, wav KHONG scale)...", flush=True)
infer_dict = {"wav": [calib["wav"][i] for i in range(len(calib["wav"]))]}
dataset = hub.upload_dataset(infer_dict)
infer_job = hub.submit_inference_job(model=target, device=device, inputs=dataset)
print("  infer_job_id=", infer_job.job_id, flush=True)
infer_job.wait()
print("  infer state:", infer_job.get_status().state, flush=True)
out = infer_job.download_output_data()
key = list(out.keys())[0]
npu_outputs = out[key]

print("\n[3] Tham chieu CPU fp32 (ONNXRuntime, cung KHONG scale) + so sanh...", flush=True)
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
results = []
for i in range(len(npu_outputs)):
    wav_np = calib["wav"][i].astype(np.float32)
    cpu_out = sess.run(None, {"wav": wav_np})[0].reshape(-1)
    npu_out = np.asarray(npu_outputs[i]).reshape(-1)
    a, b = cpu_out.astype(np.float64), npu_out.astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    lbl = labels[i] if i < len(labels) else f"idx{i}"
    results.append((lbl, cos, max_abs, rel_l2))
    print(f"  [{lbl}] cos_sim={cos:.6f}  max_abs_diff={max_abs:.6f}  rel_L2={rel_l2:.5f}", flush=True)

avg_cos = float(np.mean([r[1] for r in results]))
avg_rel_l2 = float(np.mean([r[3] for r in results]))
print(f"\n>>> Stage A FIXED (khong scale x32768): avg_cos_sim={avg_cos:.6f}  avg_rel_L2={avg_rel_l2:.5f}", flush=True)
print(f">>> Stage A GOC (co scale x32768, da biet): avg_cos_sim=0.240837  avg_rel_L2=0.97893", flush=True)

with open(os.path.join(OUT_DIR, "stageA_fixed_diag.json"), "w", encoding="utf-8") as f:
    json.dump({"avg_cos_sim": avg_cos, "avg_rel_l2": avg_rel_l2}, f, indent=2)
print("Saved.", flush=True)
