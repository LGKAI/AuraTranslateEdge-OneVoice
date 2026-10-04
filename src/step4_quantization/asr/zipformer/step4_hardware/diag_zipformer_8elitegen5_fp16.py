"""Step 4 -- verify Zipformer fp16 encoder on Snapdragon 8 Elite Gen 5 QRD
actually preserves the signal (not just "compiles") -- same real-inference +
cosine-similarity method as diag_zipformer_quant_gap.py, which showed int8
on QCS6490 gave mean cosine similarity 0.21 (representation destroyed).
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
DEVICE_NAME = "Snapdragon 8 Elite Gen 5 QRD"
ENCODER_COMPILE_JOB = "jgjqyvo15"
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
    x_lens_i32 = np.array([real_frames], dtype=np.int32)
    x_lens_i64 = np.array([real_frames], dtype=np.int64)

    print(f"[diag8elite] local fp32 encoder run: {ENCODER_ONNX}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    local_out = sess.run(out_names, {"x": x, "x_lens": x_lens_i64})
    local_encoder_out = local_out[0]
    local_T = int(np.array(local_out[1]).reshape(-1)[0])
    print(f"[diag8elite] local fp32 encoder_out shape={local_encoder_out.shape}, valid T={local_T}")
    local_encoder_out = local_encoder_out[:, :local_T, :]

    device = hub.Device(DEVICE_NAME)
    target_model = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()
    print(f"[diag8elite] submitting REAL inference job on {device.name} ...")
    job = hub.submit_inference_job(
        model=target_model, device=device, inputs={"x": [x], "x_lens": [x_lens_i32]},
        name="encoder-8elite-fp16")
    print(f"[diag8elite] job: {job.job_id}  {job.url}")
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        print(f"[diag8elite] inference FAILED: {status.message}")
        return
    hw_out = job.download_output_data()
    hw_encoder_out = np.array(hw_out["output_0"][0])
    hw_T = int(np.array(hw_out["output_1"][0]).reshape(-1)[0])
    print(f"[diag8elite] hardware fp16 encoder_out shape={hw_encoder_out.shape}, valid T={hw_T}")
    hw_encoder_out = hw_encoder_out[:, :hw_T, :]

    T = min(local_T, hw_T)
    a = local_encoder_out[:, :T, :].astype(np.float64)
    b = hw_encoder_out[:, :T, :].astype(np.float64)

    print(f"[diag8elite] === comparing first {T} frames ===")
    print(f"[diag8elite] local fp32  encoder_out: mean={a.mean():.4f} std={a.std():.4f} "
          f"min={a.min():.4f} max={a.max():.4f}")
    print(f"[diag8elite] hardware fp16 encoder_out: mean={b.mean():.4f} std={b.std():.4f} "
          f"min={b.min():.4f} max={b.max():.4f}")

    a_flat = a.reshape(T, -1)
    b_flat = b.reshape(T, -1)
    num = (a_flat * b_flat).sum(axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-9
    cos_sim = num / den
    print(f"[diag8elite] per-frame cosine similarity: mean={cos_sim.mean():.4f} "
          f"min={cos_sim.min():.4f} max={cos_sim.max():.4f}")
    print(f"[diag8elite] frames with cos_sim < 0.9: {int((cos_sim < 0.9).sum())} / {T}")
    print(f"[diag8elite] frames with cos_sim > 0.99: {int((cos_sim > 0.99).sum())} / {T}")

    if cos_sim.mean() > 0.99:
        print("[diag8elite] CONCLUSION: fp16 on Snapdragon 8 Elite Gen 5 preserves the "
              "encoder representation essentially perfectly -- the int8 corruption seen "
              "on QCS6490 is fully avoided.")
    elif cos_sim.mean() > 0.9:
        print("[diag8elite] CONCLUSION: fp16 preserves the representation reasonably well "
              "(minor precision loss, not corruption).")
    else:
        print("[diag8elite] CONCLUSION: fp16 STILL diverges substantially from fp32 -- "
              "unexpected, needs further investigation.")


if __name__ == "__main__":
    main()
