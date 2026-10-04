"""Step 4 -- submit Zipformer's encoder (Vietnamese ASR, int8 ONNX) to
Qualcomm AI Hub for a real compile + profile job on QCS6490.

Unlike Piper/Supertonic (both have an internal RandomNormalLike op for
stochastic sampling -- QNN has no translation for that op, confirmed via
outputs/profile_piper_qcs6490.log), Zipformer's RNN-T encoder is fully
deterministic (no internal RNG), so it should not hit the same blocker.
"""
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZIPFORMER_SNAP = os.path.join(ROOT, "third_party_zipformer",
                               "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"  # real QCS6490 hardware, same chip as Rubik Pi 3


def find_encoder():
    # Surgically-patched graph: the 3 boolean Slice nodes (the actual QNN
    # blocker, confirmed via outputs/profile_zipformer_realcalib2_qcs6490.log
    # -- "Failed to validate op /encoder/Slice_1", no BOOL datatype on HTP)
    # are wrapped in Cast(bool->int8->bool), verified numerically EXACT
    # (max_abs_diff=0.0) against the original in prepare_zipformer_for_qnn.py.
    # fp32 still (not the author's int8 build): that build uses
    # DynamicQuantizeLinear, which QNN's converter has no translation for --
    # int8 quantization here comes from AI Hub's own --quantize_full_type
    # instead, applied at compile time below.
    return os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")


def main():
    model_path = find_encoder()
    device = hub.Device(DEVICE_NAME)
    print(f"[profile_zipformer] device: {device.name}")
    print(f"[profile_zipformer] model: {model_path}")

    import onnx
    m = onnx.load(model_path)
    for inp in m.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"[profile_zipformer] input: {inp.name} {dims} elem_type={inp.type.tensor_type.elem_type}")

    input_specs = {
        "x": ((1, 103, 80), "float32"),   # 103 frames ~= 1s of 80-dim fbank at 10ms hop
        "x_lens": ((1,), "int64"),        # matches the real ONNX graph; --truncate_64bit_io handles device-side cast
    }

    print("[profile_zipformer] submitting compile job (AI Hub-native static int8 quantization, real calibration data) ...")
    calib_dataset = hub.get_dataset("d7x8l3y59")  # 5 real Vietnamese audio samples, windowed to 103 frames
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io",
        calibration_data=calib_dataset,
    )
    print(f"[profile_zipformer] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[profile_zipformer] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[profile_zipformer] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[profile_zipformer] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[profile_zipformer] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[profile_zipformer] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[profile_zipformer] === REAL QCS6490 RESULTS ===")
    for k, v in exec_summary.items():
        print(f"  {k}: {v}")
    print(f"[profile_zipformer] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
