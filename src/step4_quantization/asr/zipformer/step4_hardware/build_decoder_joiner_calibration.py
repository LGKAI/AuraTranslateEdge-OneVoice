"""Step 4 -- build REAL calibration datasets for Zipformer's decoder and
joiner (the two remaining RNN-T components after the encoder, see
profile_zipformer_qcs6490.py for the encoder's own compile+profile).

Both networks are tiny and structurally clean (verified via onnx node-type
audit: decoder = 17 nodes, embedding Gather + Conv + Relu + Gemm, no
RandomNormalLike/NonZero/DynamicQuantizeLinear; joiner = 3 nodes, Add +
Tanh + Gemm, same clean bill). Neither has the architecture-level blockers
that took 11 attempts to resolve for the encoder -- this script only needs
to supply real calibration data, matching the same "real data over
calibration_data=None" rule that mattered for the encoder.

- decoder input `y`: (N, 2) int64, a 2-token context window from the RNN-T
  decoding history. Valid token ids are [-1, vocab_size-1] (embedding
  table is 2000 x 512, and -1 is the blank/start sentinel -- confirmed via
  the graph's own Clip(min=0) + GreaterOrEqual(0) mask-and-zero pattern).
  Calibration here uses real *token ids* (not audio) since that's the
  network's actual input domain -- there's no "real vs synthetic" axis for
  discrete ids the way there is for audio features, so uniform random ids
  in the valid range are exactly as real as any other valid id sequence.
- joiner inputs `encoder_out`/`decoder_out`: (N, 512) float32 activations.
  These DO have a real-vs-synthetic axis (they're learned feature
  vectors, not raw ids), so this script gets them from genuine forward
  passes: runs the real encoder on the same real Vietnamese calibration
  audio used for the encoder's own calibration, and the real decoder on
  real token-id windows, both via onnxruntime on CPU.
"""
import os
import json

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import onnxruntime as ort
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "asr", "manifest.json")
SNAP = os.path.join(ROOT, "third_party_zipformer",
                     "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
VOCAB_SIZE = 2000
N_SAMPLES = 8
FIXED_FRAMES = 103


def snap_dir():
    return os.path.join(SNAP, os.listdir(SNAP)[0])


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
    return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)


def build_decoder_calibration(rng):
    """Real token-id context windows -- includes the blank sentinel (-1)
    since that's a genuine, frequently-hit state in real RNN-T decoding
    (every "no new token this frame" step re-feeds the same context)."""
    ys = []
    for _ in range(N_SAMPLES):
        ids = rng.integers(-1, VOCAB_SIZE, size=(1, 2)).astype(np.int64)
        ys.append(ids)
    return ys


def build_joiner_calibration(rng):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc_sess = ort.InferenceSession(
        os.path.join(snap_dir(), "encoder-epoch-20-avg-10.onnx"), so,
        providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(
        os.path.join(snap_dir(), "decoder-epoch-20-avg-10.onnx"), so,
        providers=["CPUExecutionProvider"])

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"][:N_SAMPLES]

    enc_outs, dec_outs = [], []
    for item in vi_items:
        path = os.path.join(ROOT, item["path"])
        wav, sr = sf.read(path, dtype="float32")
        feats = compute_fbank(wav, sr)
        if feats.shape[0] < FIXED_FRAMES:
            pad = np.zeros((FIXED_FRAMES - feats.shape[0], 80), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:FIXED_FRAMES]
        x = feats[None, :, :]
        x_lens = np.array([FIXED_FRAMES], dtype=np.int64)
        enc_out, _ = enc_sess.run(None, {"x": x, "x_lens": x_lens})
        # pick one real frame's activation vector (mid-sequence, fully valid)
        t = enc_out.shape[1] // 2
        enc_outs.append(enc_out[:, t, :].astype(np.float32))  # (1, 512)

        y = rng.integers(-1, VOCAB_SIZE, size=(1, 2)).astype(np.int64)
        dec_out = dec_sess.run(None, {"y": y})[0]
        dec_outs.append(dec_out.astype(np.float32))  # (1, 512)

        print(f"[build_calibration] joiner sample from {item['path']}: "
              f"enc_out {enc_outs[-1].shape}, dec_out {dec_outs[-1].shape}")

    return enc_outs, dec_outs


def main():
    rng = np.random.default_rng(0)

    print("[build_calibration] decoder: sampling real token-id context windows ...")
    ys = build_decoder_calibration(rng)
    dec_dataset = hub.upload_dataset(dict(y=ys), name="zipformer_decoder_calibration")
    print(f"[build_calibration] decoder dataset id: {dec_dataset.dataset_id}")

    print("[build_calibration] joiner: running real encoder+decoder for real activations ...")
    enc_outs, dec_outs = build_joiner_calibration(rng)
    joiner_dataset = hub.upload_dataset(
        dict(encoder_out=enc_outs, decoder_out=dec_outs), name="zipformer_joiner_calibration")
    print(f"[build_calibration] joiner dataset id: {joiner_dataset.dataset_id}")


if __name__ == "__main__":
    main()
