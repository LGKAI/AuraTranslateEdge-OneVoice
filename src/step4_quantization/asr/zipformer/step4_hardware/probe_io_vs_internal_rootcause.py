"""Step 4 -- root-cause probe: is the ~0.21 cosine similarity on HTP caused by
OUTPUT quantization (boundary encoding) or INTERNAL int8 activation error?

Free experiment -- no new AI Hub job. Reuses:
- LOCAL fp32 Zipformer encoder (outputs/zipformer-qnn/encoder_no_bool_slice.onnx)
- ALREADY-RUN hardware inference job jpv7m6kjp (re-downloaded, no re-submit)

Compares 4 signals on the same real Vietnamese audio:
  local_fp32      : onnxruntime CPU fp32 (ground truth)
  hw_int8         : the real HTP output (already dequantized to float by AI Hub)
  sim_io_int8     : local_fp32 with per-tensor symmetric int8 quantize+dequant
                    applied ONLY to the output vector  (boundary quantization)
  sim_io_wide     : same, but with the int8 scale widened 5x -- simulates a
                    calibration outlier inflating the output encoding range

If sim_io_int8 ~= hw_int8 (cos_sim ~0.2): the *output* quantization alone
destroys the representation -> root cause is I/O encoding (fixable with
per-channel / better-encodings / quantization_overrides).
If sim_io_int8 >> hw_int8 (cos_sim ~0.95) but hw stays at 0.2: the boundary
isn't the cause -> corruption is INTERNAL (int8 activations accumulating
through depth) -> root cause is compute, fixable only with mixed precision.
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
HW_JOB = "jpv7m6kjp"
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


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def per_frame_cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[1], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[1], -1)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return num / den


def symmetric_int8_qdq(x):
    """Per-tensor symmetric int8 quantize + dequantize (mimics a single
    output encoding: one scale for the whole tensor)."""
    mx = float(np.abs(x).max())
    scale = mx / 127.0
    if scale == 0:
        return x.copy()
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).astype(np.float32)


def symmetric_int8_qdq_wide(x, widen):
    mx = float(np.abs(x).max()) * widen
    scale = mx / 127.0
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale).astype(np.float32)


def asymmetric_int8_qdq(x):
    mn, mx = float(x.min()), float(x.max())
    scale = (mx - mn) / 255.0
    if scale == 0:
        return x.copy()
    zp = int(round(-mn / scale))
    zp = max(0, min(255, zp))
    q = np.round(x / scale + zp).clip(0, 255).astype(np.uint8)
    return ((q.astype(np.float32) - zp) * scale).astype(np.float32)


def stats(x, name):
    x = np.asarray(x, dtype=np.float64)
    print(f"  {name:18s} mean={x.mean():+.4f} std={x.std():.4f} "
          f"min={x.min():+.4f} max={x.max():+.4f}")


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

    print(f"[probe] encoder ONNX : {ENCODER_ONNX}")
    print(f"[probe] audio        : {item['path']}  ({real_frames} frames, ~{real_frames/100:.1f}s)")

    # --- LOCAL fp32 ---
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    local_out = sess.run(out_names, {"x": x, "x_lens": x_lens})
    local_enc = local_out[0]
    local_T = int(np.array(local_out[1]).reshape(-1)[0])
    local_enc = local_enc[:, :local_T, :]  # (1, T, 512)
    print(f"[probe] local fp32  encoder_out shape={local_enc.shape} valid_T={local_T}")

    # --- HARDWARE (re-download existing job, no re-submit) ---
    job = hub.get_job(HW_JOB)
    hw = job.download_output_data()
    hw_enc = np.array(hw["output_0"][0])
    hw_T = int(np.array(hw["output_1"][0]).reshape(-1)[0])
    hw_enc = hw_enc[:, :hw_T, :]
    print(f"[probe] hardware    encoder_out shape={hw_enc.shape} valid_T={hw_T}")

    T = min(local_T, hw_T)
    a = local_enc[:, :T, :].astype(np.float64)
    b = hw_enc[:, :T, :].astype(np.float64)

    print(f"\n[probe] === FULL-TENSOR STATS (first {T} frames) ===")
    stats(a, "local_fp32")
    stats(b, "hw_int8")

    print(f"\n[probe] === FULL COS_SIM ===")
    print(f"  local_fp32 vs hw_int8: cos_sim = {cos_sim(a, b):.4f}  (known: ~0.21)")
    pfc = per_frame_cos(a, b)
    print(f"  per-frame: mean={pfc.mean():.4f} min={pfc.min():.4f} max={pfc.max():.4f}")

    # --- SIM 1: boundary int8 quantize of the LOCAL fp32 output ---
    print(f"\n[probe] === LOCAL SIM: per-tensor int8 quantize of OUTPUT only ===")
    sim_sym = symmetric_int8_qdq(a.astype(np.float32)).astype(np.float64)
    sim_asym = asymmetric_int8_qdq(a.astype(np.float32)).astype(np.float64)
    print(f"  symmetric int8  vs local_fp32 : cos_sim = {cos_sim(a, sim_sym):.4f}")
    print(f"  asymmetric int8 vs local_fp32 : cos_sim = {cos_sim(a, sim_asym):.4f}")
    sim_sym_w5 = symmetric_int8_qdq_wide(a.astype(np.float32), 5.0).astype(np.float64)
    print(f"  symmetric int8 (scale x5)     : cos_sim = {cos_sim(a, sim_sym_w5):.4f}")
    for w in (10.0, 50.0, 200.0):
        sim_w = symmetric_int8_qdq_wide(a.astype(np.float32), w).astype(np.float64)
        print(f"  symmetric int8 (scale x{int(w)})     : cos_sim = {cos_sim(a, sim_w):.4f}")

    # --- SIM 2: simulate per-tensor int8 on each frame (per-frame encoding) ---
    print(f"\n[probe] === LOCAL SIM: per-FRAME int8 quantize of output ===")
    per_frame_sim = np.zeros_like(a)
    for t in range(T):
        per_frame_sim[0, t] = symmetric_int8_qdq(a[0, t].astype(np.float32)).astype(np.float64)
    print(f"  per-frame symmetric int8      : cos_sim = {cos_sim(a, per_frame_sim):.4f}")

    # --- IMPACT: what widen factor would make local-sim drop to 0.21? ---
    print(f"\n[probe] === searching: what int8 scale-widen gives cos_sim ~0.21 (matching hw)? ===")
    lo, hi = 1.0, 2000.0
    for _ in range(40):
        w = (lo + hi) / 2
        sim_w = symmetric_int8_qdq_wide(a.astype(np.float32), w).astype(np.float64)
        c = cos_sim(a, sim_w)
        if c > 0.21:
            lo = w
        else:
            hi = w
    print(f"  -> boundary-only int8 needs scale widened ~{hi:.0f}x actual range to drop to cos_sim 0.21")
    print(f"     (i.e. one outlier ~{hi:.0f}x the real max would be needed to explain hw via I/O quant alone)")

    print("\n[probe] === INTERPRETATION ===")
    cs_io = cos_sim(a, sim_sym)
    if cs_io > 0.9:
        print("  Boundary int8 quantization preserves output (cos_sim>0.9). "
              "Corruption is INTERNAL (int8 activations through depth).")
    elif cs_io < 0.4:
        print("  Boundary int8 quantization alone destroys output. "
              "Root cause is the I/O output encoding -- fixable via per-channel / overrides.")
    else:
        print(f"  Boundary gives cos_sim {cs_io:.2f} -- partial. Internal + boundary both contribute.")


if __name__ == "__main__":
    main()