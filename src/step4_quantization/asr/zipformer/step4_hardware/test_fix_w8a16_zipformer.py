"""Step 4 fix-test (Zipformer encoder, w8a16): same recipe that fixed NLLB
(int8 weights + int16 activations via submit_quantize_job). Zipformer has NO
-30000 mask bias like NLLB, so only the w8a16 swap is needed -- if cos_sim
recovers >0.95 it proves the recipe generalizes beyond NLLB.

Pipeline: w8a16 quantize -> compile on IQ-9075 (Hexagon v73) -> inference ->
cos_sim vs fp32 local. Baseline to beat: Zipformer int8-only = 0.21.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_static_1500.onnx")
DEVICE_PRIMARY = "Dragonwing RB3 Gen 2 Vision Kit"   # QCS6490, v68
DEVICE_FALLBACK = "Dragonwing IQ-9075 EVK"            # v73
FIXED_FRAMES = int(os.environ.get("FIXED_FRAMES", "1500"))


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


def pad(feats, T=FIXED_FRAMES):
    real = feats.shape[0]
    if real >= T:
        return feats[:T][None, :, :], np.array([T], dtype=np.int64), real
    z = np.zeros((T - real, 80), dtype=np.float32)
    return np.concatenate([feats, z], axis=0)[None, :, :], np.array([real], dtype=np.int64), real


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def per_frame_cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[1], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[1], -1)
    T = min(a.shape[0], b.shape[0])
    num = (a[:T] * b[:T]).sum(axis=1)
    den = np.linalg.norm(a[:T], axis=1) * np.linalg.norm(b[:T], axis=1) + 1e-9
    out = num / den
    return out.mean(), out.min(), out.max()


def main():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi = [r for r in manifest if r["lang"] == "vi"]
    # calibrate on first 4, eval on smallest (matches diag_zipformer_quant_gap)
    calib_items = vi[:4]
    eval_item = min(vi, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    calib_x, calib_lens = [], []
    for it in calib_items:
        wav, sr = sf.read(os.path.join(ROOT, it["path"]), dtype="float32")
        x, x_lens, _ = pad(compute_fbank(wav, sr))
        calib_x.append(x[0][None, :, :])  # keep batch dim (1,1500,80) per entry
        calib_lens.append(np.array([int(x_lens[0])], dtype=np.int64))
    print(f"[z-w8a16] {len(calib_x)} calibration audios, FIXED_FRAMES={FIXED_FRAMES}", flush=True)

    # eval entry
    wav, sr = sf.read(os.path.join(ROOT, eval_item["path"]), dtype="float32")
    eval_x, eval_lens, eval_real = pad(compute_fbank(wav, sr))
    print(f"[z-w8a16] eval audio: {eval_item['path']}  ({eval_real} frames ~{eval_real/100:.1f}s)", flush=True)

    # ---- local fp32 reference ----
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_fp = sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})[0]
    T_fp = int(sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})[1][0])
    out_fp = out_fp[:, :T_fp, :]
    print(f"[z-w8a16] local fp32 out shape={out_fp.shape} std={out_fp.std():.4f}", flush=True)

    # ---- Step 1: w8a16 quantize ----
    RECIPE = os.environ.get("RECIPE", "w8a16")
    WDT = hub.QuantizeDtype.INT16 if RECIPE.startswith("w16") else hub.QuantizeDtype.INT8
    ADT = hub.QuantizeDtype.INT16 if RECIPE.endswith("a16") else hub.QuantizeDtype.INT8
    QMODEL_REUSE = os.environ.get("QMODEL_REUSE", "")
    print(f"[z-w8a16] RECIPE={RECIPE} weights={WDT} activations={ADT}", flush=True)
    if QMODEL_REUSE:
        qmodel = hub.get_model(QMODEL_REUSE)
        print(f"[z-w8a16] REUSING quantized model {QMODEL_REUSE}", flush=True)
    else:
        calib_ds = hub.upload_dataset({"x": calib_x, "x_lens": calib_lens}, name=f"zipf-calib-{RECIPE}")
        print(f"[z-w8a16] submit_quantize_job...", flush=True)
        qjob = hub.submit_quantize_job(
            model=ENCODER_ONNX, calibration_data=calib_ds,
            weights_dtype=WDT, activations_dtype=ADT, name=f"zipf-{RECIPE}-quant")
        print(f"[z-w8a16] quant job: {qjob.job_id}  {qjob.url}", flush=True)
        qjob.wait()
        if qjob.get_status().code != "SUCCESS":
            print(f"[z-w8a16] quantize FAIL: {qjob.get_status().message[:600]}"); return
        qmodel = qjob.get_target_model()
    print(f"[z-w8a16] quantized model: {qmodel.model_id}", flush=True)

    # ---- Step 2: compile ----
    compile_dev = None; cmodel = None
    for dev_name in [DEVICE_PRIMARY, DEVICE_FALLBACK]:
        device = hub.Device(dev_name)
        print(f"[z-w8a16] submit_compile_job on {dev_name}...", flush=True)
        cjob = hub.submit_compile_job(
            model=qmodel, device=device,
            input_specs={"x": ((1, FIXED_FRAMES, 80), "float32"),
                         "x_lens": ((1,), "int64")},
            options="--target_runtime qnn_context_binary --quantize_io --truncate_64bit_io",
            name=f"zipf-w8a16-compile-{dev_name.split()[1].lower()}")
        print(f"[z-w8a16] compile job: {cjob.job_id}  {cjob.url}", flush=True)
        cjob.wait()
        st = cjob.get_status()
        if st.code == "SUCCESS":
            print(f"[z-w8a16] compile SUCCESS on {dev_name}", flush=True)
            cmodel = cjob.get_target_model(); compile_dev = dev_name; break
        else:
            print(f"[z-w8a16] compile FAIL on {dev_name}: {st.message[:300]}", flush=True)
    if cmodel is None:
        print("[z-w8a16] compile failed on both devices."); return

    # ---- Step 3: inference ----
    device = hub.Device(compile_dev)
    print(f"[z-w8a16] submit_inference_job on {device.name}...", flush=True)
    ijob = hub.submit_inference_job(
        model=cmodel, device=device,
        inputs={"x": [eval_x[0].astype(np.float32)], "x_lens": [int(eval_lens[0])]},
        name="zipf-w8a16-infer")
    print(f"[z-w8a16] inference job: {ijob.job_id}", flush=True)
    ijob.wait()
    if ijob.get_status().code != "SUCCESS":
        print(f"[z-w8a16] inference FAIL: {ijob.get_status().message[:400]}"); return
    hw_data = ijob.download_output_data()
    hw = np.array(hw_data["output_0"][0])
    hw_T = int(np.array(hw_data["output_1"][0]).reshape(-1)[0])
    hw = hw[:, :hw_T, :]
    T = min(T_fp, hw_T)
    print(f"[z-w8a16] hw out shape={hw.shape} valid_T={hw_T} std={hw.std():.4f}", flush=True)

    cs = cos_sim(out_fp[:, :T, :], hw[:, :T, :])
    pf_mean, pf_min, pf_max = per_frame_cos(out_fp[:, :T, :], hw[:, :T, :])
    print(f"\n[z-w8a16] === VERDICT ({compile_dev}) ===")
    print(f"[z-w8a16] cos_sim(fp32, {RECIPE}_hw) = {cs:.4f}   (int8-only baseline = 0.21)")
    print(f"[z-w8a16] per-frame: mean={pf_mean:.4f} min={pf_min:.4f} max={pf_max:.4f}")
    print(f"[z-w8a16] fp32 std={out_fp.std():.4f}  hw std={hw.std():.4f}")
    if cs > 0.95:
        print(f"[z-w8a16] FIX CONFIRMED on Zipformer too -- w8a16 generalizes.")
    elif cs > 0.6:
        print(f"[z-w8a16] PARTIAL improvement -- need per-tensor overrides or other outlier.")
    else:
        print(f"[z-w8a16] STILL BROKEN -- Zipformer has a different/additional root cause.")


if __name__ == "__main__":
    main()