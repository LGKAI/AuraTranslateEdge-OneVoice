"""Step 4 -- build a REAL calibration dataset for Piper Stage 2 (encoder +
flow + vocoder), from real Vietnamese sentences phonemized by Piper's own
espeak-ng frontend (same voice as production: vi_VN-vais1000-medium),
run through the REAL Stage 1 (duration predictor) to get a genuine
log-duration for each, matching the same "real activations over synthetic"
rule used for the encoder/joiner calibration.
"""
import os
import json

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
STAGE1_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration.onnx")
VOICE_ONNX = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
MAX_FRAMES = 400
N_SAMPLES = 6
FIXED_PHONEME_LEN = 40  # must match prepare_piper_stage1_for_qnn.py's --phoneme-len (the noise input shape baked into stage1_duration.onnx)


def main():
    voice = PiperVoice.load(VOICE_ONNX)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_sentences = [r["vi"] for r in manifest][:N_SAMPLES]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s1_sess = ort.InferenceSession(STAGE1_MODEL, so, providers=["CPUExecutionProvider"])

    inputs, input_lens, scales_list, logdurs, znoises, masks = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    scales_default = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    for text in vi_sentences:
        phonemes = voice.phonemize(text)
        ids_nested = voice.phonemes_to_ids(phonemes[0]) if phonemes else None
        if not ids_nested:
            continue
        real_ids = np.array(ids_nested, dtype=np.int64).reshape(1, -1)
        real_len = real_ids.shape[1]

        # pad/truncate to the FIXED compile-time phoneme budget (same
        # windowing approach as Zipformer's FIXED_FRAMES calibration) --
        # padding id 0 is Piper's own "_" pad/silence phoneme (id map
        # confirms "_": [0] in vi_VN-vais1000-medium.onnx.json)
        if real_len < FIXED_PHONEME_LEN:
            pad = np.zeros((1, FIXED_PHONEME_LEN - real_len), dtype=np.int64)
            ids = np.concatenate([real_ids, pad], axis=1)
            input_len = real_len
        else:
            ids = real_ids[:, :FIXED_PHONEME_LEN]
            input_len = FIXED_PHONEME_LEN

        dp_noise = rng.standard_normal((1, 2, FIXED_PHONEME_LEN)).astype(np.float32)
        logdur = s1_sess.run(
            ["/dp/Split_output_0"],
            {"input": ids, "input_lengths": np.array([input_len], dtype=np.int64),
             "scales": scales_default, "/dp/RandomNormalLike_output_0": dp_noise},
        )[0]

        z_noise = rng.standard_normal((1, 192, MAX_FRAMES)).astype(np.float32)

        # real per-utterance total frame count, same formula as the graph's
        # own (now-removed) Exp->Mul->Ceil->ReduceSum chain -- length_scale
        # is scales[1] (order: noise_scale, length_scale, noise_w, matching
        # vi_VN-vais1000-medium.onnx.json's inference block)
        per_phoneme_dur = np.ceil(np.exp(logdur.reshape(-1)) * float(scales_default[1]))
        real_total = min(MAX_FRAMES, max(1, int(per_phoneme_dur[:input_len].sum())))
        mask = (np.arange(MAX_FRAMES) < real_total).astype(np.float32).reshape(1, 1, MAX_FRAMES)

        inputs.append(ids)
        input_lens.append(np.array([input_len], dtype=np.int64))
        scales_list.append(scales_default.copy())
        logdurs.append(logdur.astype(np.float32))
        znoises.append(z_noise)
        masks.append(mask)
        print(f"[build_stage2_calib] '{text[:40]}...' -> {real_len} real phonemes "
              f"(windowed to {FIXED_PHONEME_LEN}), real_total_frames={real_total}")

    # key order matches model.graph.input's actual order -- AI Hub matches
    # calibration data to graph inputs positionally, not by name (confirmed:
    # a prior attempt with /RandomNormalLike_output_0 before /Cast_2_output_0
    # failed with "Calibration data set has input '/RandomNormalLike_output_0'
    # but expected '/Cast_2_output_0'").
    import qai_hub as hub
    dataset = hub.upload_dataset(
        {"input": inputs, "input_lengths": input_lens, "scales": scales_list,
         "/dp/Split_output_0": logdurs, "/Cast_2_output_0": masks,
         "/RandomNormalLike_output_0": znoises},
        name="piper_stage2_calibration_v3",
    )
    print(f"[build_stage2_calib] dataset id: {dataset.dataset_id}")


if __name__ == "__main__":
    main()
