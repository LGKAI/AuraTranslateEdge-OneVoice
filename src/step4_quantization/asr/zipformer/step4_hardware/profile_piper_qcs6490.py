"""Step 4 -- submit Piper (Vietnamese TTS) to Qualcomm AI Hub for a real
compile + profile job on QCS6490 (via the "Dragonwing RB3 Gen 2 Vision
Kit" device -- no "Rubik Pi 3" entry exists on AI Hub by name, but this
kit uses the identical QCS6490 chipset, so its numbers are directly
representative of the project's chosen demo device).

Piper's ONNX graph has dynamic input shapes (phoneme sequence length is
not fixed) -- AI Hub compilation needs concrete shapes, so this pins
a representative phoneme length via input_specs.
"""
import os

import qai_hub as hub

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "step3_tts", "vi_VN-vais1000-medium.onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"  # real QCS6490 hardware, same chip as Rubik Pi 3
PHONEME_LEN = 50  # representative length for a medium-length sentence


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[profile_piper] device: {device.name} ({device.os})")

    input_specs = {
        "input": ((1, PHONEME_LEN), "int64"),
        "input_lengths": ((1,), "int64"),
        "scales": ((3,), "float32"),
    }

    print(f"[profile_piper] submitting compile job for {MODEL_PATH} ...")
    compile_job = hub.submit_compile_job(
        model=MODEL_PATH,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
    )
    print(f"[profile_piper] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[profile_piper] compile status: {status}")
    if status.code != "COMPLETED":
        print(f"[profile_piper] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()

    print("[profile_piper] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[profile_piper] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[profile_piper] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[profile_piper] === REAL QCS6490 RESULTS ===")
    for k in ("estimated_inference_time", "estimated_inference_peak_memory",
              "compute_unit_breakdown"):
        if k in exec_summary:
            print(f"  {k}: {exec_summary[k]}")
    print(f"[profile_piper] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
