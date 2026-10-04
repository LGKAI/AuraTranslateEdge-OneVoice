"""Step 4 -- compile + profile Supertonic's 4 ONNX submodels on real
QCS6490 hardware via Qualcomm AI Hub. Unlike Piper, none of these needed
any graph surgery (confirmed clean of RandomNormalLike/NonZero/
DynamicQuantizeLinear in their pristine fp32 form) -- this compiles the
originals directly against real calibration data.
"""
import json
import os
import sys

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.expanduser("~/.cache/supertonic3/onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

FIXED_TEXT_LEN = 80
FIXED_LATENT_LEN = 150

with open(os.path.join(ROOT, "outputs", "supertonic_calibration_ids.json")) as f:
    CALIB = json.load(f)


def compile_and_profile(name, model_path, input_specs, calib_dataset_id):
    device = hub.Device(DEVICE_NAME)
    print(f"[{name}] device: {device.name}")
    print(f"[{name}] model: {model_path}")

    calib = hub.get_dataset(calib_dataset_id)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
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
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("duration_predictor", "all"):
        compile_and_profile(
            "duration_predictor",
            os.path.join(SRC_DIR, "duration_predictor.onnx"),
            {"text_ids": ((1, FIXED_TEXT_LEN), "int64"),
             "style_dp": ((1, 8, 16), "float32"),
             "text_mask": ((1, 1, FIXED_TEXT_LEN), "float32")},
            CALIB["duration_predictor"],
        )

    if target in ("text_encoder", "all"):
        compile_and_profile(
            "text_encoder",
            os.path.join(SRC_DIR, "text_encoder.onnx"),
            {"text_ids": ((1, FIXED_TEXT_LEN), "int64"),
             "style_ttl": ((1, 50, 256), "float32"),
             "text_mask": ((1, 1, FIXED_TEXT_LEN), "float32")},
            CALIB["text_encoder"],
        )

    if target in ("vector_estimator", "all"):
        # key order matches model.graph.input's actual order (noisy_latent,
        # text_emb, style_ttl, latent_mask, text_mask, current_step,
        # total_step) -- AI Hub's shape validation zips provided vs inferred
        # shapes positionally, not by name (confirmed via Piper Stage 2's
        # identical failure mode: "shape inference" error that turned out to
        # be a dict-order mismatch, not a real shape incompatibility). The
        # first attempt here had text_mask before latent_mask, opposite of
        # the graph's real order.
        compile_and_profile(
            "vector_estimator",
            os.path.join(SRC_DIR, "vector_estimator.onnx"),
            {"noisy_latent": ((1, 144, FIXED_LATENT_LEN), "float32"),
             "text_emb": ((1, 256, FIXED_TEXT_LEN), "float32"),
             "style_ttl": ((1, 50, 256), "float32"),
             "latent_mask": ((1, 1, FIXED_LATENT_LEN), "float32"),
             "text_mask": ((1, 1, FIXED_TEXT_LEN), "float32"),
             "current_step": ((1,), "float32"),
             "total_step": ((1,), "float32")},
            CALIB["vector_estimator"],
        )

    if target in ("vocoder", "all"):
        compile_and_profile(
            "vocoder",
            os.path.join(SRC_DIR, "vocoder.onnx"),
            {"latent": ((1, 144, FIXED_LATENT_LEN), "float32")},
            CALIB["vocoder"],
        )


if __name__ == "__main__":
    main()
