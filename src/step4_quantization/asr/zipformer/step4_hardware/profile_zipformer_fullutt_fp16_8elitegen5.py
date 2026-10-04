"""Step 4 -- pivot experiment: does moving off QCS6490 (Hexagon v68, HTP opset
68, floating tensors rejected -- see step4.md SS4c-4) to Snapdragon 8 Elite
Gen 5 QRD (Hexagon v81, device attribute htp-supports-fp16:true) let the
Zipformer full-utterance encoder compile at fp16, preserving the signal that
int8 destroyed (cosine similarity 0.21 vs fp32 on QCS6490)? Same graph,
same weights, same 1500-frame fixed budget as profile_zipformer_fullutt_qcs6490.py
-- only the target device changes.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Snapdragon 8 Elite Gen 5 QRD"
FIXED_FRAMES = 1500


def find_encoder():
    return os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")


def main():
    device = hub.Device(DEVICE_NAME)
    model_path = find_encoder()
    print(f"[8elite_fp16] device: {device.name}  attrs: {device.attributes}")
    print(f"[8elite_fp16] model: {model_path}")

    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type float16 --quantize_io"
    print(f"[8elite_fp16] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs, options=options,
    )
    print(f"[8elite_fp16] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[8elite_fp16] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[8elite_fp16] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[8elite_fp16] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[8elite_fp16] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[8elite_fp16] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[8elite_fp16] === REAL Snapdragon 8 Elite Gen 5 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[8elite_fp16] full report: {profile_job.url}")
    print(f"[8elite_fp16] target_model_id: {target_model.model_id}")
    print(f"[8elite_fp16] compile_job_id: {compile_job.job_id}")


if __name__ == "__main__":
    main()
