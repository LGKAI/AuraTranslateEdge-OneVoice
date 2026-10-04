"""Step 4 -- build a REAL calibration dataset for Zipformer's QNN
quantization, from real Vietnamese audio (data/asr/vi/*.wav) run through
an 80-dim kaldi-style fbank frontend (kaldi_native_fbank, already a
dependency of funasr_onnx). AI Hub's own docs recommend real, in-domain
calibration data over calibration_data=None for accuracy -- and
calibration_data=None didn't get compile past the packaging stage anyway
(outputs/profile_zipformer_quantio_qcs6490.log).
"""
import os
import sys
import json

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "asr", "manifest.json")
N_SAMPLES = 8
FIXED_FRAMES = 103  # must match input_specs["x"] in profile_zipformer_qcs6490.py


def compute_fbank(wav, sr):
    assert sr == 16000, f"expected 16kHz, got {sr}"
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    feats = np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)
    return feats  # (T, 80)


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"][:N_SAMPLES]

    xs, x_lens = [], []
    for item in vi_items:
        path = os.path.join(ROOT, item["path"])
        wav, sr = sf.read(path, dtype="float32")
        feats = compute_fbank(wav, sr)
        # AI Hub's QNN compile needs a FIXED input shape (input_specs pins x to
        # (1, FIXED_FRAMES, 80)) -- window each real utterance's real features
        # to that length instead of using the full variable length.
        if feats.shape[0] < FIXED_FRAMES:
            pad = np.zeros((FIXED_FRAMES - feats.shape[0], 80), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:FIXED_FRAMES]
        xs.append(feats[None, :, :])  # (1, FIXED_FRAMES, 80)
        x_lens.append(np.array([FIXED_FRAMES], dtype=np.int64))
        print(f"[build_calibration] {item['path']}: windowed to {feats.shape[0]} frames")

    print(f"[build_calibration] uploading {len(xs)} calibration samples to AI Hub ...")
    dataset = hub.upload_dataset(dict(x=xs, x_lens=x_lens), name="zipformer_vi_calibration")
    print(f"[build_calibration] dataset id: {dataset.dataset_id}")
    print(f"[build_calibration] dataset url: {dataset.url}")


if __name__ == "__main__":
    main()
