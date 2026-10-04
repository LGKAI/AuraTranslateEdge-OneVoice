"""Step 4 -- quick experiment: is Piper Stage 1's OOMKilled compile failure
(step4.md SS4h: 3x OOMKilled, confirmed node-count-driven not tensor-size-driven
via a phoneme_len 40->16 diagnostic) specific to the int8 quantization pass,
or does it OOM even at plain float precision (no quantization at all, no
calibration data needed)? Same NonZero-fixed graph as profile_piper_stage1_qcs6490.py,
only the compile options change (drop --quantize_full_type/--quantize_io).
Cheap, fast to test either way it lands.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
MODEL_PATH = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")
FIXED_PHONEME_LEN = 40


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[stage1_float] device: {device.name}")
    print(f"[stage1_float] model: {MODEL_PATH}")

    input_specs = {
        "input": ((1, FIXED_PHONEME_LEN), "int64"),
        "input_lengths": ((1,), "int64"),
        "scales": ((3,), "float32"),
        "/dp/RandomNormalLike_output_0": ((1, 2, FIXED_PHONEME_LEN), "float32"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io"
    print(f"[stage1_float] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=MODEL_PATH, device=device, input_specs=input_specs, options=options,
    )
    print(f"[stage1_float] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[stage1_float] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[stage1_float] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[stage1_float] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[stage1_float] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[stage1_float] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[stage1_float] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[stage1_float] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
