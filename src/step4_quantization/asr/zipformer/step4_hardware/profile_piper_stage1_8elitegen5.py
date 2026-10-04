"""Step 4 -- pivot experiment: does Piper Stage 1's OOMKilled compile
(confirmed node-count-driven, independent of precision -- 4 failures on
QCS6490, see step4.md SS4h) also happen on Snapdragon 8 Elite Gen 5 QRD?
The AI Hub compiler backend/build-machine memory limit is likely shared
across target devices (it's a compile-time resource, not a device-side
one), so this is a real, cheap thing to verify rather than assume.
Same NonZero-fixed graph, same phoneme_len=40 as profile_piper_stage1_qcs6490.py.
"""
import os

import numpy as np
import qai_hub as hub
from piper.voice import PiperVoice

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Snapdragon 8 Elite Gen 5 QRD"
MODEL_PATH = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")
VOICE_ONNX = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")

FIXED_PHONEME_LEN = 40


def build_calibration():
    import json
    voice = PiperVoice.load(VOICE_ONNX)
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_sentences = [r["vi"] for r in manifest][:6]

    rng = np.random.default_rng(0)
    scales_default = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    inputs, input_lens, scales_list, noises = [], [], [], []

    for text in vi_sentences:
        phonemes = voice.phonemize(text)
        ids_nested = voice.phonemes_to_ids(phonemes[0]) if phonemes else None
        if not ids_nested:
            continue
        real_ids = np.array(ids_nested, dtype=np.int64).reshape(1, -1)
        real_len = real_ids.shape[1]
        if real_len < FIXED_PHONEME_LEN:
            pad = np.zeros((1, FIXED_PHONEME_LEN - real_len), dtype=np.int64)
            ids = np.concatenate([real_ids, pad], axis=1)
            input_len = real_len
        else:
            ids = real_ids[:, :FIXED_PHONEME_LEN]
            input_len = FIXED_PHONEME_LEN

        inputs.append(ids)
        input_lens.append(np.array([input_len], dtype=np.int64))
        scales_list.append(scales_default.copy())
        noises.append(rng.standard_normal((1, 2, FIXED_PHONEME_LEN)).astype(np.float32))
        print(f"[stage1_8elite_calib] '{text[:30]}...' -> {real_len} real phonemes (windowed to {FIXED_PHONEME_LEN})")

    dataset = hub.upload_dataset(
        {"input": inputs, "input_lengths": input_lens, "scales": scales_list,
         "/dp/RandomNormalLike_output_0": noises},
        name="piper_stage1_nononzero_calibration_8elitegen5",
    )
    print(f"[stage1_8elite_calib] dataset id: {dataset.dataset_id}")
    return dataset.dataset_id


def main():
    device = hub.Device(DEVICE_NAME)
    print(f"[stage1_8elite] device: {device.name}  attrs: {device.attributes}")
    print(f"[stage1_8elite] model: {MODEL_PATH}")

    calib_id = os.environ.get("STAGE1_CALIB_ID") or build_calibration()
    calib = hub.get_dataset(calib_id)

    input_specs = {
        "input": ((1, FIXED_PHONEME_LEN), "int64"),
        "input_lengths": ((1,), "int64"),
        "scales": ((3,), "float32"),
        "/dp/RandomNormalLike_output_0": ((1, 2, FIXED_PHONEME_LEN), "float32"),
    }
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[stage1_8elite] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=MODEL_PATH, device=device, input_specs=input_specs,
        options=options, calibration_data=calib,
    )
    print(f"[stage1_8elite] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[stage1_8elite] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[stage1_8elite] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print("[stage1_8elite] submitting profile job ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"[stage1_8elite] profile job: {profile_job.job_id}  {profile_job.url}")
    profile_job.wait()
    print(f"[stage1_8elite] profile status: {profile_job.get_status()}")

    results = profile_job.download_profile()
    exec_summary = results.get("execution_summary", {})
    print("[stage1_8elite] === REAL Snapdragon 8 Elite Gen 5 RESULTS ===")
    for k, v in exec_summary.items():
        if k == "all_inference_times":
            continue
        print(f"  {k}: {v}")
    print(f"[stage1_8elite] full report: {profile_job.url}")


if __name__ == "__main__":
    main()
