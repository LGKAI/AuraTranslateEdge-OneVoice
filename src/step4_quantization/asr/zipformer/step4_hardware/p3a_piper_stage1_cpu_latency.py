"""P3(a) -- measure REAL Piper Stage 1 (duration predictor) latency on CPU
(onnxruntime), to check the strategic-plan claim that it's cheap enough to
just run off-NPU while Stage 1's OOMKilled compile blocker (step4.md SS4h)
remains unresolved. Uses the NonZero-fixed, bit-exact-verified graph, real
Vietnamese phonemes, 50 real timed runs (after 5 warmup runs) for a stable
median/mean.
"""
import os
import json
import time

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE1_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")
VOICE_ONNX = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
FIXED_PHONEME_LEN = 40
N_WARMUP = 5
N_TIMED = 50


def main():
    voice = PiperVoice.load(VOICE_ONNX)
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    text = manifest[0]["vi"]
    phonemes = voice.phonemize(text)
    ids_nested = voice.phonemes_to_ids(phonemes[0])
    real_ids = np.array(ids_nested, dtype=np.int64).reshape(1, -1)
    real_len = real_ids.shape[1]
    if real_len < FIXED_PHONEME_LEN:
        pad = np.zeros((1, FIXED_PHONEME_LEN - real_len), dtype=np.int64)
        ids = np.concatenate([real_ids, pad], axis=1)
        input_len = real_len
    else:
        ids = real_ids[:, :FIXED_PHONEME_LEN]
        input_len = FIXED_PHONEME_LEN
    input_lengths = np.array([input_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    print(f"[p3a] real sentence: '{text}' -> {real_len} phonemes (windowed to {FIXED_PHONEME_LEN})")

    rng = np.random.default_rng(0)
    dp_noise = rng.standard_normal((1, 2, FIXED_PHONEME_LEN)).astype(np.float32)
    inputs = {"input": ids, "input_lengths": input_lengths, "scales": scales,
              "/dp/RandomNormalLike_output_0": dp_noise}

    n_threads = os.cpu_count()
    print(f"[p3a] host CPU threads available: {n_threads}")

    for label, intra_threads in [("single-thread (worst case, matches a small edge core)", 1),
                                  (f"multi-thread ({n_threads} threads, matches a full edge cluster)", n_threads)]:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = intra_threads
        sess = ort.InferenceSession(STAGE1_MODEL, so, providers=["CPUExecutionProvider"])

        for _ in range(N_WARMUP):
            sess.run(None, inputs)

        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            sess.run(None, inputs)
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"[p3a] {label}:")
        print(f"  mean={times.mean():.3f}ms  median={np.median(times):.3f}ms  "
              f"p95={np.percentile(times, 95):.3f}ms  min={times.min():.3f}ms  max={times.max():.3f}ms")


if __name__ == "__main__":
    main()
