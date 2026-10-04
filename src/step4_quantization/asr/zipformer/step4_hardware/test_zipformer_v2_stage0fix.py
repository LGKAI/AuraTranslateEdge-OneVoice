"""Step 4 -- compile the v2 encoder (stage-0 bool-mask fix applied, see
fix_zipformer_stage0_mask.py) on QCS6490 int8, then immediately run the same
cosine-similarity diagnostic used throughout SS4c-4/4c-5 to check whether
this fixes the ~0.21 corruption.
"""
import os
import json

import numpy as np
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice_v2.onnx")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
FIXED_FRAMES = 1500
CALIB_DATASET = "d7gwn3z62"


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
    device = hub.Device(DEVICE_NAME)
    print(f"[v2test] device: {device.name}")
    print(f"[v2test] model: {ENCODER_ONNX}")

    input_specs = {"x": ((1, FIXED_FRAMES, 80), "float32"), "x_lens": ((1,), "int64")}
    calib = hub.get_dataset(CALIB_DATASET)
    options = "--target_runtime qnn_context_binary --truncate_64bit_io --quantize_full_type int8 --quantize_io"
    print(f"[v2test] compile options: {options}")

    compile_job = hub.submit_compile_job(
        model=ENCODER_ONNX, device=device, input_specs=input_specs,
        options=options, calibration_data=calib)
    print(f"[v2test] compile job: {compile_job.job_id}  {compile_job.url}")
    compile_job.wait()
    status = compile_job.get_status()
    print(f"[v2test] compile status: {status}")
    if status.code != "SUCCESS":
        print(f"[v2test] compile FAILED: {status.message}. See {compile_job.url}")
        return

    target_model = compile_job.get_target_model()
    print(f"[v2test] compile SUCCESS, target_model_id={target_model.model_id}")

    # --- real audio diagnostic ---
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
    feats = compute_fbank(wav, sr)
    real_frames = feats.shape[0]
    feats_padded = np.concatenate([feats, np.zeros((FIXED_FRAMES - real_frames, 80), dtype=np.float32)], axis=0)
    x = feats_padded[None, :, :].astype(np.float32)
    x_lens_i64 = np.array([real_frames], dtype=np.int64)
    x_lens_i32 = np.array([real_frames], dtype=np.int32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    local_out = sess.run(out_names, {"x": x, "x_lens": x_lens_i64})
    local_encoder_out = local_out[0]
    local_T = int(np.array(local_out[1]).reshape(-1)[0])
    local_encoder_out = local_encoder_out[:, :local_T, :]
    print(f"[v2test] local fp32 encoder_out shape={local_encoder_out.shape}, T={local_T}")

    print(f"[v2test] submitting REAL inference job on {device.name} ...")
    job = hub.submit_inference_job(
        model=target_model, device=device, inputs={"x": [x], "x_lens": [x_lens_i32]}, name="v2test-encoder")
    print(f"[v2test] job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[v2test] inference FAILED: {job.get_status().message}")
        return
    hw_out = job.download_output_data()
    hw_encoder_out = np.array(hw_out["output_0"][0])
    hw_T = int(np.array(hw_out["output_1"][0]).reshape(-1)[0])
    hw_encoder_out = hw_encoder_out[:, :hw_T, :]
    print(f"[v2test] hardware encoder_out shape={hw_encoder_out.shape}, T={hw_T}")

    T = min(local_T, hw_T)
    a = local_encoder_out[:, :T, :].astype(np.float64)
    b = hw_encoder_out[:, :T, :].astype(np.float64)
    a_flat, b_flat = a.reshape(T, -1), b.reshape(T, -1)
    num = (a_flat * b_flat).sum(axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-9
    cos_sim = num / den
    print(f"\n[v2test] === RESULT: cosine similarity mean={cos_sim.mean():.4f} "
          f"min={cos_sim.min():.4f} max={cos_sim.max():.4f} ===")
    print(f"[v2test] frames with cos_sim > 0.99: {int((cos_sim > 0.99).sum())} / {T}")
    if cos_sim.mean() > 0.9:
        print("[v2test] *** FIX CONFIRMED: stage-0 bool-mask bug was the (or a major) root cause ***")
    else:
        print("[v2test] fix did not resolve it -- more investigation needed")


if __name__ == "__main__":
    main()
