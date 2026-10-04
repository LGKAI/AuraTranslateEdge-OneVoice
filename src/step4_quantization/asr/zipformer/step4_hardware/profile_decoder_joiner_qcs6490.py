"""Step 4 -- compile + profile Zipformer's decoder and joiner (the two
remaining RNN-T components) on real QCS6490 hardware via Qualcomm AI Hub.

Both are tiny, structurally clean graphs (no RandomNormalLike / NonZero /
DynamicQuantizeLinear -- verified via node-type audit before writing this
script), unlike the encoder which needed 11 attempts and real graph
surgery. This script compiles them as-is, using the real calibration data
built by build_decoder_joiner_calibration.py.
"""
import os
import sys

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = os.path.join(ROOT, "third_party_zipformer",
                     "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

DECODER_CALIB = "d7m86g0m2"
JOINER_CALIB = "d9kvgey82"


def snap_dir():
    return os.path.join(SNAP, os.listdir(SNAP)[0])


def compile_and_profile(name, model_path, input_specs, calib_dataset_id, extra_opts=""):
    device = hub.Device(DEVICE_NAME)
    print(f"[{name}] device: {device.name}")
    print(f"[{name}] model: {model_path}")

    calib = hub.get_dataset(calib_dataset_id)
    options = f"--target_runtime qnn_context_binary --quantize_full_type int8 --quantize_io {extra_opts}".strip()
    print(f"[{name}] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=model_path, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[{name}] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[{name}] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[{name}] compile FAILED: {status.message}. See {compile_job.url}")
        return None

    target_model = compile_job.get_target_model()
    print(f"[{name}] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[{name}] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[{name}] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print(f"[{name}] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[{name}] full report: {profile_job.url}")
    return exec_summary


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    snap = snap_dir()

    if target in ("decoder", "both"):
        compile_and_profile(
            "decoder",
            os.path.join(snap, "decoder-epoch-20-avg-10.onnx"),
            {"y": ((1, 2), "int64")},
            DECODER_CALIB,
            extra_opts="--truncate_64bit_io",
        )

    if target in ("joiner", "both"):
        compile_and_profile(
            "joiner",
            os.path.join(snap, "joiner-epoch-20-avg-10.onnx"),
            {"encoder_out": ((1, 512), "float32"), "decoder_out": ((1, 512), "float32")},
            JOINER_CALIB,
        )


if __name__ == "__main__":
    main()
