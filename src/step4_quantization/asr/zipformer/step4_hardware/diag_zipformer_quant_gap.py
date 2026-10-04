"""Step 4 -- diagnose WHY the full-utterance (no-chunking) int8 real-hardware
Zipformer decode still produced zero tokens (step4.md SS4c-3 follow-up).
Reuses the ALREADY-COMPLETED real encoder inference job (jpv7m6kjp, no new
hardware cost) and compares its output against a LOCAL fp32 onnxruntime run
of the same (pre-quantization) encoder graph on the identical input, to
isolate whether int8 quantization is corrupting the ENCODER itself (vs. a
downstream decoder/joiner issue).
"""
import os
import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")
ENCODER_INFERENCE_JOB = "jpv7m6kjp"
FIXED_FRAMES = 1500


def compute_fbank(wav, sr):
    assert sr == 16000
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)


def main():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav_path = os.path.join(ROOT, item["path"])

    wav, sr = sf.read(wav_path, dtype="float32")
    feats = compute_fbank(wav, sr)
    real_frames = feats.shape[0]
    feats_padded = np.concatenate(
        [feats, np.zeros((FIXED_FRAMES - real_frames, 80), dtype=np.float32)], axis=0)
    x = feats_padded[None, :, :].astype(np.float32)
    x_lens = np.array([real_frames], dtype=np.int64)

    print(f"[diag] local fp32 encoder run: {ENCODER_ONNX}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[diag] local encoder outputs: {out_names}")
    local_out = sess.run(out_names, {"x": x, "x_lens": x_lens})
    local_encoder_out = local_out[0]
    local_T = int(np.array(local_out[1]).reshape(-1)[0])
    print(f"[diag] local fp32 encoder_out shape={local_encoder_out.shape}, valid T={local_T}")
    local_encoder_out = local_encoder_out[:, :local_T, :]

    print(f"[diag] fetching REAL hardware encoder job {ENCODER_INFERENCE_JOB} output (no re-submit) ...")
    job = hub.get_job(ENCODER_INFERENCE_JOB)
    hw_out = job.download_output_data()
    hw_encoder_out = np.array(hw_out["output_0"][0])
    hw_T = int(np.array(hw_out["output_1"][0]).reshape(-1)[0])
    print(f"[diag] hardware encoder_out shape={hw_encoder_out.shape}, valid T={hw_T}")
    hw_encoder_out = hw_encoder_out[:, :hw_T, :]

    T = min(local_T, hw_T)
    a = local_encoder_out[:, :T, :].astype(np.float64)
    b = hw_encoder_out[:, :T, :].astype(np.float64)

    print(f"[diag] === comparing first {T} frames ===")
    print(f"[diag] local fp32  encoder_out: mean={a.mean():.4f} std={a.std():.4f} "
          f"min={a.min():.4f} max={a.max():.4f}")
    print(f"[diag] hardware int8 encoder_out: mean={b.mean():.4f} std={b.std():.4f} "
          f"min={b.min():.4f} max={b.max():.4f}")

    # per-frame cosine similarity, a cheap signal-preservation metric that's
    # robust to any overall scale shift the int8 quantizer might introduce
    a_flat = a.reshape(T, -1)
    b_flat = b.reshape(T, -1)
    num = (a_flat * b_flat).sum(axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-9
    cos_sim = num / den
    print(f"[diag] per-frame cosine similarity: mean={cos_sim.mean():.4f} "
          f"min={cos_sim.min():.4f} max={cos_sim.max():.4f}")
    print(f"[diag] frames with cos_sim < 0.5: {int((cos_sim < 0.5).sum())} / {T}")
    print(f"[diag] frames with cos_sim < 0.9: {int((cos_sim < 0.9).sum())} / {T}")

    if b.std() < 1e-3:
        print("[diag] CONCLUSION: hardware encoder_out is essentially CONSTANT "
              "(near-zero variance) -- the int8-quantized encoder itself is "
              "producing degenerate/collapsed output, not just noisy output.")
    elif cos_sim.mean() < 0.5:
        print("[diag] CONCLUSION: hardware encoder_out diverges substantially "
              "from the fp32 reference -- int8 quantization is corrupting the "
              "ENCODER's representation.")
    else:
        print("[diag] CONCLUSION: hardware encoder_out is reasonably aligned "
              "with fp32 reference -- the zero-token issue likely lies "
              "downstream (decoder/joiner quantization), not the encoder.")


if __name__ == "__main__":
    main()
