"""P1 -- 10-minute test: is the ~0.21 cosine-similarity corruption caused by
the padding mask logic misbehaving under QNN? Re-run the SAME already-compiled
int8 encoder (QCS6490, job jgd241rz5) but declare x_lens = FIXED_FRAMES (1500,
i.e. "no padding, everything is valid") instead of the real 724. If the mask
logic is the culprit, telling the graph there's no padding at all should
route around whatever is broken and cosine similarity should jump back to
~1.0. Compare against a local fp32 reference run with the SAME (padding-free)
x_lens, on the SAME real audio (zero-padded, since our clip is shorter than
1500 frames either way -- only the DECLARED x_lens changes, the tensor
content is identical).
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
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
ENCODER_COMPILE_JOB = "jgd241rz5"  # int8, QCS6490, FIXED_FRAMES=1500
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


def cosine(a, b):
    a_flat = a.reshape(a.shape[1], -1)
    b_flat = b.reshape(b.shape[1], -1)
    num = (a_flat * b_flat).sum(axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-9
    return num / den


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

    # KEY CHANGE: declare x_lens = FIXED_FRAMES (no padding claimed), not real_frames
    x_lens_nopad_i64 = np.array([FIXED_FRAMES], dtype=np.int64)
    x_lens_nopad_i32 = np.array([FIXED_FRAMES], dtype=np.int32)

    print(f"[p1] real_frames={real_frames}, declaring x_lens={FIXED_FRAMES} (claims no padding)")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    local_out = sess.run(out_names, {"x": x, "x_lens": x_lens_nopad_i64})
    local_encoder_out = local_out[0]
    local_T = int(np.array(local_out[1]).reshape(-1)[0])
    print(f"[p1] local fp32 (no-pad-declared) encoder_out shape={local_encoder_out.shape}, T={local_T}")

    device = hub.Device(DEVICE_NAME)
    target_model = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()
    print(f"[p1] submitting REAL inference job on {device.name} (x_lens={FIXED_FRAMES}) ...")
    job = hub.submit_inference_job(
        model=target_model, device=device, inputs={"x": [x], "x_lens": [x_lens_nopad_i32]},
        name="p1-nopad-encoder")
    print(f"[p1] job: {job.job_id}  {job.url}")
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        print(f"[p1] inference FAILED: {status.message}")
        return
    hw_out = job.download_output_data()
    hw_encoder_out = np.array(hw_out["output_0"][0])
    hw_T = int(np.array(hw_out["output_1"][0]).reshape(-1)[0])
    print(f"[p1] hardware (no-pad-declared) encoder_out shape={hw_encoder_out.shape}, T={hw_T}")

    T = min(local_T, hw_T, local_encoder_out.shape[1], hw_encoder_out.shape[1])
    a = local_encoder_out[:, :T, :].astype(np.float64)
    b = hw_encoder_out[:, :T, :].astype(np.float64)
    cos_sim = cosine(a, b)
    print(f"[p1] === RESULT: comparing first {T} frames ===")
    print(f"[p1] per-frame cosine similarity: mean={cos_sim.mean():.4f} min={cos_sim.min():.4f} max={cos_sim.max():.4f}")
    print(f"[p1] frames with cos_sim > 0.99: {int((cos_sim > 0.99).sum())} / {T}")

    if cos_sim.mean() > 0.9:
        print("[p1] CONCLUSION: PADDING/MASK IS THE ROOT CAUSE -- declaring no padding "
              "restores the signal. Fix: feed x_lens=FIXED_FRAMES in production (let the "
              "model process the zero-padding as if real, decoder will emit blank there).")
    else:
        print("[p1] CONCLUSION: padding/mask is NOT the (sole) root cause -- corruption "
              "persists even with no padding declared. Need P2 (host-computed mask) or "
              "deeper graph-level investigation (QAIRT local debugging).")


if __name__ == "__main__":
    main()
