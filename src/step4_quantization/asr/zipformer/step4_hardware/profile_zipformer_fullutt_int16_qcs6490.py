"""Step 4 -- experiment #2: does int16 fixed-point (instead of int8) preserve
the encoder's representation, where int8 destroyed it (diag_zipformer_quant_gap.py:
mean cosine similarity 0.21 vs fp32 reference) and float16 wasn't accepted at
all by this device's HTP backend ("Tensor 'x' has a floating-point type which
is not supported by the targeted device", both with and without --quantize_io,
see profile_zipformer_fullutt_fp16_qcs6490.py). int16 is still a fixed-point
quantization scheme (needs calibration data, like int8), just with 65536
levels instead of 256 -- same graph/weights/frame budget as the int8 and
fp16 attempts, only the quantization option changes.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
FIXED_FRAMES = 1500
CALIB_DATASET = "d7gwn3z62"


def find_encoder():
    return os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")


def main():
    device = hub.Device(DEVICE_NAME)
    model_path = find_encoder()
    print(f"[fullutt_int16] device: {device.name}")
    print(f"[fullutt_int16] model: {model_path}")

    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64"),
    }
    calib = hub.get_dataset(CALIB_DATASET)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int16 --quantize_io"
    print(f"[fullutt_int16] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[fullutt_int16] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[fullutt_int16] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[fullutt_int16] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[fullutt_int16] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[fullutt_int16] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[fullutt_int16] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[fullutt_int16] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[fullutt_int16] full report: {profile_job.url}")
    print(f"[fullutt_int16] target_model_id: {target_model.model_id}")
    print(f"[fullutt_int16] compile_job_id: {compile_job.job_id}")


if __name__ == "__main__":
    main()
