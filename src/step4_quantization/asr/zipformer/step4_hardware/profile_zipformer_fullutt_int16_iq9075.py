"""Step 4 -- test Zipformer full-utterance encoder at int16 on Dragonwing
IQ-9075 EVK (Hexagon v73 -- exactly meets the ">=73" HTP op version
requirement that blocked int16 on QCS6490's Hexagon v68, see step4.md
SS4c-4). Also a further data point on the SS4c-5 graph-correctness-bug
hypothesis: fp16 on Snapdragon 8 Elite Gen 5 (Hexagon v81) gave the SAME
~0.21 cosine similarity corruption as int8 on QCS6490, suggesting the bug
is device/precision-independent -- testing a genuinely different
quantization path (int16, not yet tried anywhere successfully) on a third,
free AI Hub device adds real signal either way.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
FIXED_FRAMES = 1500
CALIB_DATASET = "d7gwn3z62"


def find_encoder():
    return os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")


def main():
    device = hub.Device(DEVICE_NAME)
    model_path = find_encoder()
    print(f"[iq9075_int16] device: {device.name}  attrs: {device.attributes}")
    print(f"[iq9075_int16] model: {model_path}")

    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64"),
    }
    calib = hub.get_dataset(CALIB_DATASET)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int16 --quantize_io"
    print(f"[iq9075_int16] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[iq9075_int16] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[iq9075_int16] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[iq9075_int16] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[iq9075_int16] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[iq9075_int16] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[iq9075_int16] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[iq9075_int16] === REAL IQ-9075 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[iq9075_int16] full report: {profile_job.url}")
    print(f"[iq9075_int16] target_model_id: {target_model.model_id}")
    print(f"[iq9075_int16] compile_job_id: {compile_job.job_id}")


if __name__ == "__main__":
    main()
