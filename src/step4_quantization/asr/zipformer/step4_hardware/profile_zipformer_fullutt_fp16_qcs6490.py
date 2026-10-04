"""Step 4 -- experiment: does dropping int8 for float16 fix the encoder
quantization-corruption finding in diag_zipformer_quant_gap.py (real
hardware int8 encoder_out has mean cosine similarity 0.21 vs the fp32
reference -- the encoder's representation is being destroyed by int8 PTQ,
not just imprecise)? Same graph, same weights, same fixed 1500-frame
budget as profile_zipformer_fullutt_qcs6490.py -- only the quantization
option changes (int8 -> float16, and no calibration data needed since
float16 doesn't require activation range calibration).
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
FIXED_FRAMES = 1500


def find_encoder():
    return os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")


def main():
    device = hub.Device(DEVICE_NAME)
    model_path = find_encoder()
    print(f"[fullutt_fp16] device: {device.name}")
    print(f"[fullutt_fp16] model: {model_path}")

    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int16 --quantize_io"
    print(f"[fullutt_fp16] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs, options=options,
    )
    print(f"[fullutt_fp16] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[fullutt_fp16] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[fullutt_fp16] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[fullutt_fp16] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[fullutt_fp16] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[fullutt_fp16] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[fullutt_fp16] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[fullutt_fp16] full report: {profile_job.url}")
    print(f"[fullutt_fp16] target_model_id: {target_model.model_id}")
    print(f"[fullutt_fp16] compile_job_id: {compile_job.job_id}")


if __name__ == "__main__":
    main()
