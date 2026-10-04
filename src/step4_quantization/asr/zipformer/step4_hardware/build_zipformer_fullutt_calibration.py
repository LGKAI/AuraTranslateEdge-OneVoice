"""Step 4 -- build REAL calibration data for a Zipformer encoder compiled
with a LARGE fixed frame budget (single-shot whole-utterance encoding,
not the 103-frame chunk budget used in SS4c). This is the fix for the
finding in step4.md SS4c-3: the "hynt/Zipformer-30M-RNNT-6000h" checkpoint
is an OFFLINE model (its own README describes it via `OfflineRecognizer`,
"12s audio in 0.3s" batch-style framing) with no cache-state ONNX export
-- naive independent-chunk processing loses cross-chunk context and
genuinely degrades quality (confirmed locally: full-utterance fp32
transcribes correctly, chunked fp32 degrades, chunked+int8 real-hardware
degrades to zero). Rather than needing a different, streaming-trained
checkpoint, the fix stays within the SAME model+weights: compile the
encoder with a big enough fixed frame budget to cover a whole real
utterance in ONE call (no chunking needed at all), matching how the model
was actually trained/intended to run.
"""
import os
import json

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "asr", "manifest.json")
FIXED_FRAMES = 1500  # covers all 5 real Vietnamese samples (max real: 1489 frames / 14.88s)


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
    return feats


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]

    xs, x_lens = [], []
    for item in vi_items:
        path = os.path.join(ROOT, item["path"])
        wav, sr = sf.read(path, dtype="float32")
        feats = compute_fbank(wav, sr)
        real_len = feats.shape[0]
        if real_len < FIXED_FRAMES:
            pad = np.zeros((FIXED_FRAMES - real_len, 80), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:FIXED_FRAMES]
        xs.append(feats[None, :, :])
        x_lens.append(np.array([min(real_len, FIXED_FRAMES)], dtype=np.int64))
        print(f"[build_fullutt_calib] {item['path']}: real {real_len} frames "
              f"(windowed to {FIXED_FRAMES})")

    print(f"[build_fullutt_calib] uploading {len(xs)} calibration samples ...")
    dataset = hub.upload_dataset(dict(x=xs, x_lens=x_lens), name="zipformer_fullutt_calibration")
    print(f"[build_fullutt_calib] dataset id: {dataset.dataset_id}")


if __name__ == "__main__":
    main()
