# -*- coding: utf-8 -*-
"""Compile 2 checkpoint (stageA_power, stageB_mellog) len NPU that, chay inference tren
15-sample test set, so sanh voi tham chieu CPU fp32 (ONNXRuntime) de tim buoc gay sai lech."""
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import qai_hub as hub

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
CALIB_NPZ = os.path.join(OUT_DIR, "calib_data_unified_fixed.npz")

STAGES = [
    ("stageA_power", "model_sv_frontend_stageA_power_inline.onnx"),
    ("stageB_mellog", "model_sv_frontend_stageB_mellog_inline.onnx"),
]

device = hub.Device(DEVICE_NAME)
calib = np.load(CALIB_NPZ)
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]

results_summary = {}

for stage_name, onnx_file in STAGES:
    print(f"\n{'='*70}\n[STAGE {stage_name}]\n{'='*70}", flush=True)
    onnx_path = os.path.join(OUT_DIR, onnx_file)

    print("  [1] Upload + compile len NPU that (KHONG quantize)...", flush=True)
    model = hub.upload_model(onnx_path)
    print("    model_id=", model.model_id, flush=True)
    cjob = hub.submit_compile_job(model=model, device=device, options="--target_runtime qnn_dlc --truncate_64bit_io")
    print("    compile_job_id=", cjob.job_id, flush=True)
    cjob.wait()
    print("    compile state:", cjob.get_status().state, flush=True)
    if str(cjob.get_status().state) != "State.SUCCESS":
        print(f"    !!! COMPILE THAT BAI cho {stage_name}, bo qua stage nay.", flush=True)
        continue
    target = cjob.get_target_model()

    print("  [2] Chay inference tren NPU that (15-sample test set)...", flush=True)
    infer_dict = {"wav": [calib["wav"][i] for i in range(len(calib["wav"]))]}
    dataset = hub.upload_dataset(infer_dict)
    infer_job = hub.submit_inference_job(model=target, device=device, inputs=dataset)
    print("    infer_job_id=", infer_job.job_id, flush=True)
    infer_job.wait()
    print("    infer state:", infer_job.get_status().state, flush=True)
    out = infer_job.download_output_data()
    key = list(out.keys())[0]
    npu_outputs = out[key]

    print("  [3] Tham chieu CPU fp32 (ONNXRuntime) + so sanh...", flush=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    stage_results = []
    for i in range(len(npu_outputs)):
        wav_np = calib["wav"][i].astype(np.float32)
        cpu_out = sess.run(None, {"wav": wav_np})[0].reshape(-1)
        npu_out = np.asarray(npu_outputs[i]).reshape(-1)
        a, b = cpu_out.astype(np.float64), npu_out.astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        max_abs = float(np.max(np.abs(a - b)))
        rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
        lbl = labels[i] if i < len(labels) else f"idx{i}"
        stage_results.append((lbl, cos, max_abs, rel_l2))
        print(f"    [{lbl}] cos_sim={cos:.6f}  max_abs_diff={max_abs:.5f}  rel_L2={rel_l2:.5f}", flush=True)

    avg_cos = float(np.mean([r[1] for r in stage_results]))
    avg_rel_l2 = float(np.mean([r[3] for r in stage_results]))
    results_summary[stage_name] = {"avg_cos_sim": avg_cos, "avg_rel_l2": avg_rel_l2}
    print(f"  >>> {stage_name}: avg_cos_sim={avg_cos:.6f}  avg_rel_L2={avg_rel_l2:.5f}", flush=True)

print(f"\n{'='*70}\nTOM TAT SO SANH THEO TUNG STAGE\n{'='*70}", flush=True)
for stage_name, r in results_summary.items():
    print(f"  {stage_name}: avg_cos_sim={r['avg_cos_sim']:.6f}  avg_rel_L2={r['avg_rel_l2']:.5f}", flush=True)
print("  stageC_full_fbank (da biet truoc): avg_cos_sim~0.90-0.97  avg_rel_L2~0.42-0.59", flush=True)

with open(os.path.join(OUT_DIR, "stage_diag_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print("\nSaved:", os.path.join(OUT_DIR, "stage_diag_summary.json"), flush=True)
