"""Step 4 -- compile + profile Zipformer's encoder with a LARGE fixed
frame budget (1500 frames ~ 15s) so a whole real utterance fits in ONE
call, no chunking needed. Fixes the quality-degradation finding in
step4.md SS4c-3 (the encoder is an offline/full-context model with no
cache-state export, so independent-chunk processing loses context) --
same graph surgery (Cast bool->int32) as the original 103-frame encoder
in prepare_zipformer_for_qnn.py, just recompiled at a bigger fixed shape.
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
    print(f"[fullutt_encoder] device: {device.name}")
    print(f"[fullutt_encoder] model: {model_path}")

    input_specs = {
        "x": ((1, FIXED_FRAMES, 80), "float32"),
        "x_lens": ((1,), "int64"),
    }
    calib = hub.get_dataset(CALIB_DATASET)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[fullutt_encoder] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[fullutt_encoder] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[fullutt_encoder] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[fullutt_encoder] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[fullutt_encoder] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[fullutt_encoder] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[fullutt_encoder] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[fullutt_encoder] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[fullutt_encoder] full report: {profile_job.url}")
    print(f"[fullutt_encoder] target_model_id: {target_model.model_id}")


if __name__ == "__main__":
    main()
