"""Step 4 -- compile + profile Piper Stage 2 (encoder+flow+vocoder,
max-length-padded to MAX_FRAMES) on real QCS6490 hardware via AI Hub.

Prototype-scale budget for this first real-hardware attempt: phoneme
budget 40, frame budget 400 (~4.6s audio at Piper's 86fps). NOTE: real
Vietnamese sentences in this project's own MT test corpus run 170-409
phonemes (see build_piper_stage2_calibration.py output) -- these budgets
are enough to test whether QNN can compile the max-length-padding
technique AT ALL, not necessarily enough for every real sentence. The
right production MAX_FRAMES/FIXED_PHONEME_LEN needs the device's actual
typical-utterance-length requirement, not decided here.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
STAGE2_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage2_synth.onnx")
CALIB_DATASET = "d7m86gxv2"

FIXED_PHONEME_LEN = 40
MAX_FRAMES = 400


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[stage2] device: {device.name}")
    print(f"[stage2] model: {STAGE2_MODEL}")

    # key order matches model.graph.input's actual order (input, input_lengths,
    # scales, /dp/Split_output_0, /Cast_2_output_0, /RandomNormalLike_output_0)
    # -- AI Hub's shape validation appears to zip provided vs inferred shapes
    # positionally, not by name, so a mismatched dict order here causes a
    # spurious "does not match shapes inferred from the model" failure even
    # when every individual shape is actually correct and compatible.
    input_specs = {
        "input": ((1, FIXED_PHONEME_LEN), "int64"),
        "input_lengths": ((1,), "int64"),
        "scales": ((3,), "float32"),
        "/dp/Split_output_0": ((1, 1, FIXED_PHONEME_LEN), "float32"),
        "/Cast_2_output_0": ((1, 1, MAX_FRAMES), "float32"),
        "/RandomNormalLike_output_0": ((1, 192, MAX_FRAMES), "float32"),
    }

    calib = hub.get_dataset(CALIB_DATASET)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[stage2] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=STAGE2_MODEL, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[stage2] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[stage2] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[stage2] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[stage2] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[stage2] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[stage2] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[stage2] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[stage2] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
