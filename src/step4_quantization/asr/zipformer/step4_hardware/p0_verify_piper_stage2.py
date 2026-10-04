"""P0 -- numeric ground-truth check for Piper Stage 2 (job j5m89yo7p,
step4.md SS4e). Until now only compile+profile (timing) was ever measured,
never real output correctness. Builds a REAL input chain: real phonemes
(PiperVoice) -> real /dp/Split_output_0 (from the bit-exact-verified Stage 1
NonZero-fixed model) -> host-computed frame-validity mask + real z_p noise,
then compares local fp32 (stage2_synth.onnx via onnxruntime) vs the real
compiled hardware target.
"""
import os
import json

import numpy as np
import onnxruntime as ort
import qai_hub as hub
from piper.voice import PiperVoice

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE1_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")
STAGE2_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage2_synth.onnx")
VOICE_ONNX = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
COMPILE_JOB = "j5m89yo7p"

FIXED_PHONEME_LEN = 40
MAX_FRAMES = 400


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    voice = PiperVoice.load(VOICE_ONNX)
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    text = manifest[0]["vi"]
    print(f"[p0_piper_stage2] real sentence: '{text}'")

    phonemes = voice.phonemize(text)
    ids_nested = voice.phonemes_to_ids(phonemes[0])
    real_ids = np.array(ids_nested, dtype=np.int64).reshape(1, -1)
    real_len = real_ids.shape[1]
    if real_len < FIXED_PHONEME_LEN:
        pad = np.zeros((1, FIXED_PHONEME_LEN - real_len), dtype=np.int64)
        phoneme_ids = np.concatenate([real_ids, pad], axis=1)
        input_len = real_len
    else:
        phoneme_ids = real_ids[:, :FIXED_PHONEME_LEN]
        input_len = FIXED_PHONEME_LEN
    input_lengths = np.array([input_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    print(f"[p0_piper_stage2] {real_len} real phonemes (windowed to {FIXED_PHONEME_LEN})")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    rng = np.random.default_rng(0)
    dp_noise = rng.standard_normal((1, 2, FIXED_PHONEME_LEN)).astype(np.float32)

    stage1_sess = ort.InferenceSession(STAGE1_MODEL, so, providers=["CPUExecutionProvider"])
    dp_split = stage1_sess.run(["/dp/Split_output_0"], {
        "input": phoneme_ids, "input_lengths": input_lengths, "scales": scales,
        "/dp/RandomNormalLike_output_0": dp_noise,
    })[0]
    print(f"[p0_piper_stage2] real /dp/Split_output_0 shape={dp_split.shape}")

    # host-side duration -> real frame count -> validity mask (same formula
    # as prepare_piper_stage2_for_qnn.py's removed branch: exp(logdur) ->
    # length_scale -> ceil -> sum, clipped to MAX_FRAMES)
    length_scale = 1.0 / scales[1]
    durations = np.ceil(np.exp(dp_split) * scales[1]).astype(np.int64)  # scales[1]=length_scale per Piper convention
    real_frames = int(min(MAX_FRAMES, durations.sum()))
    print(f"[p0_piper_stage2] real_frames={real_frames} (budget {MAX_FRAMES})")

    mask = (np.arange(MAX_FRAMES)[None, None, :] < real_frames).astype(np.float32)
    z_noise = rng.standard_normal((1, 192, MAX_FRAMES)).astype(np.float32)

    stage2_inputs = {
        "input": phoneme_ids, "input_lengths": input_lengths, "scales": scales,
        "/dp/Split_output_0": dp_split, "/Cast_2_output_0": mask,
        "/RandomNormalLike_output_0": z_noise,
    }

    print(f"[p0_piper_stage2] local fp32 run: {STAGE2_MODEL}")
    stage2_sess = ort.InferenceSession(STAGE2_MODEL, so, providers=["CPUExecutionProvider"])
    local_out = stage2_sess.run(None, stage2_inputs)[0]
    print(f"[p0_piper_stage2] local output shape={local_out.shape}")

    device = hub.Device(DEVICE_NAME)
    target_model = hub.get_job(COMPILE_JOB).get_target_model()
    print(f"[p0_piper_stage2] submitting REAL inference job on {device.name} ...")
    # compiled with --truncate_64bit_io: real hardware needs int32 for the
    # declared int64 inputs (input, input_lengths) -- confirmed via a real
    # "Cannot assign data from unexpected type. Expected int32, got int64" error
    hw_inputs = {k: [v] for k, v in stage2_inputs.items()}
    hw_inputs["input"] = [phoneme_ids.astype(np.int32)]
    hw_inputs["input_lengths"] = [input_lengths.astype(np.int32)]
    job = hub.submit_inference_job(model=target_model, device=device, inputs=hw_inputs, name="p0-piper-stage2")
    print(f"[p0_piper_stage2] job: {job.job_id}  {job.url}")
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        print(f"[p0_piper_stage2] inference FAILED: {status.message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    print(f"[p0_piper_stage2] hardware output shape={hw_out.shape}")

    T = min(local_out.shape[-1], hw_out.shape[-1])
    a, b = local_out.reshape(-1)[:T], hw_out.reshape(-1)[:T]
    print(f"\n[p0_piper_stage2] === RESULT ===")
    print(f"[p0_piper_stage2] cos_sim={cos_sim(a, b):.4f}  max_abs_diff={max_abs_diff(a, b):.4e}")
    print(f"[p0_piper_stage2] local: mean={a.mean():.4f} std={a.std():.4f}")
    print(f"[p0_piper_stage2] hw:    mean={b.mean():.4f} std={b.std():.4f}")


if __name__ == "__main__":
    main()
