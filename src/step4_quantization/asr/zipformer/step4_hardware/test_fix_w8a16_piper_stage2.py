"""w8a16 generalization test #2: Piper Stage 2 (flow-based TTS synthesizer).
Same recipe that fixed NLLB (cos_sim 0.18->0.9998, step4.md SS4p) and
partially fixed Zipformer (0.21->0.80, SS4q): submit_quantize_job with
weights=INT8, activations=INT16, then compile on IQ-9075 EVK (Hexagon v73,
required -- QCS6490/v68 does not support w8a16), then real inference vs
local fp32 reference. Baseline: Piper Stage2 int8-only was corrupted
(SS4j "compile success != correct output").

Reuses the exact real-input construction from p0_verify_piper_stage2.py.
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
DEVICE_FALLBACK = "Dragonwing IQ-9075 EVK"

FIXED_PHONEME_LEN = 40
MAX_FRAMES = 400


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_inputs(voice, text, rng):
    phonemes = voice.phonemize(text)
    ids_nested = voice.phonemes_to_ids(phonemes[0])
    real_ids = np.array(ids_nested, dtype=np.int64).reshape(1, -1)
    real_len = real_ids.shape[1]
    if real_len < FIXED_PHONEME_LEN:
        padv = np.zeros((1, FIXED_PHONEME_LEN - real_len), dtype=np.int64)
        phoneme_ids = np.concatenate([real_ids, padv], axis=1)
        input_len = real_len
    else:
        phoneme_ids = real_ids[:, :FIXED_PHONEME_LEN]
        input_len = FIXED_PHONEME_LEN
    input_lengths = np.array([input_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    so = ort.SessionOptions(); so.log_severity_level = 3
    dp_noise = rng.standard_normal((1, 2, FIXED_PHONEME_LEN)).astype(np.float32)
    stage1_sess = ort.InferenceSession(STAGE1_MODEL, so, providers=["CPUExecutionProvider"])
    dp_split = stage1_sess.run(["/dp/Split_output_0"], {
        "input": phoneme_ids, "input_lengths": input_lengths, "scales": scales,
        "/dp/RandomNormalLike_output_0": dp_noise,
    })[0]

    length_scale = 1.0 / scales[1]
    durations = np.ceil(np.exp(dp_split) * scales[1]).astype(np.int64)
    real_frames = int(min(MAX_FRAMES, durations.sum()))
    mask = (np.arange(MAX_FRAMES)[None, None, :] < real_frames).astype(np.float32)
    z_noise = rng.standard_normal((1, 192, MAX_FRAMES)).astype(np.float32)

    return {
        "input": phoneme_ids, "input_lengths": input_lengths, "scales": scales,
        "/dp/Split_output_0": dp_split, "/Cast_2_output_0": mask,
        "/RandomNormalLike_output_0": z_noise,
    }, real_frames


def main():
    voice = PiperVoice.load(VOICE_ONNX)
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    rng = np.random.default_rng(0)

    eval_text = manifest[0]["vi"]
    eval_inputs, real_frames = build_inputs(voice, eval_text, rng)
    print(f"[p2-w8a16] eval text: '{eval_text}'  real_frames={real_frames}", flush=True)

    # calibration: 4 more real sentences
    calib_inputs = []
    for row in manifest[1:5]:
        inp, _ = build_inputs(voice, row["vi"], rng)
        calib_inputs.append(inp)
    print(f"[p2-w8a16] {len(calib_inputs)} calibration sentences", flush=True)

    so = ort.SessionOptions(); so.log_severity_level = 3
    stage2_sess = ort.InferenceSession(STAGE2_MODEL, so, providers=["CPUExecutionProvider"])
    out_fp = stage2_sess.run(None, eval_inputs)[0]
    print(f"[p2-w8a16] local fp32 out shape={out_fp.shape} std={out_fp.std():.4f}", flush=True)
    np.save(os.path.join(ROOT, "outputs", "piper_stage2_w8a16_fp32ref.npy"), out_fp)

    RECIPE = os.environ.get("RECIPE", "w8a16")
    WDT = hub.QuantizeDtype.INT16 if RECIPE.startswith("w16") else hub.QuantizeDtype.INT8
    ADT = hub.QuantizeDtype.INT16 if RECIPE.endswith("a16") else hub.QuantizeDtype.INT8
    print(f"[p2-w8a16] RECIPE={RECIPE} weights={WDT} activations={ADT}", flush=True)

    calib_ds_entries = {k: [c[k] for c in calib_inputs] for k in calib_inputs[0]}
    calib_ds = hub.upload_dataset(calib_ds_entries, name=f"piper-stage2-calib-{RECIPE}")
    print("[p2-w8a16] submit_quantize_job...", flush=True)
    qjob = hub.submit_quantize_job(
        model=STAGE2_MODEL, calibration_data=calib_ds,
        weights_dtype=WDT, activations_dtype=ADT, name=f"piper-stage2-{RECIPE}-quant")
    print(f"[p2-w8a16] quant job: {qjob.job_id}  {qjob.url}", flush=True)


if __name__ == "__main__":
    main()
