"""P0 -- numeric ground-truth check for Zipformer decoder (jpey63rv5) and
joiner (jp16oxdn5). These were only ever compile+profile'd (timing), and
used qualitatively inside the E2E decode loop (which produced 0 tokens
because of the encoder bug) -- never directly verified against a local
fp32 reference. Uses real token-context values (not random) and a real
encoder_out vector captured from a local fp32 run.
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
SNAP_ROOT = os.path.join(ROOT, "third_party_zipformer",
                          "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"
DECODER_COMPILE_JOB = "jpey63rv5"
JOINER_COMPILE_JOB = "jp16oxdn5"
CONTEXT_SIZE = 2


def snap_dir():
    return os.path.join(SNAP_ROOT, os.listdir(SNAP_ROOT)[0])


def decoder_onnx():
    return os.path.join(snap_dir(), "decoder-epoch-20-avg-10.onnx")


def joiner_onnx():
    return os.path.join(snap_dir(), "joiner-epoch-20-avg-10.onnx")


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
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def max_abs_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    device = hub.Device(DEVICE_NAME)

    # --- get a REAL encoder_out vector from a local fp32 encoder run (not
    # random noise) so decoder/joiner are tested on realistic activations ---
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
    feats = compute_fbank(wav, sr)
    x = feats[None, :103, :].astype(np.float32)  # first 103 real frames, well-formed input
    x_lens = np.array([103], dtype=np.int64)
    enc_sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    enc_out = enc_sess.run(None, {"x": x, "x_lens": x_lens})[0]
    real_encoder_frame = enc_out[:, 0, :].astype(np.float32)  # (1, 512), first valid frame
    print(f"[p0_dec_joiner] real encoder_out frame shape={real_encoder_frame.shape}")

    # --- decoder: real "start" context (matches the RNN-T greedy loop's
    # actual initial hypothesis, -1 sentinel per BLANK/start convention) ---
    hyp = [-1] * CONTEXT_SIZE
    y = np.array(hyp, dtype=np.int64).reshape(1, CONTEXT_SIZE)
    dec_sess = ort.InferenceSession(decoder_onnx(), so, providers=["CPUExecutionProvider"])
    decoder_out_local = dec_sess.run(None, {"y": y})[0]
    print(f"[p0_dec_joiner] local decoder_out shape={decoder_out_local.shape}")

    decoder_model = hub.get_job(DECODER_COMPILE_JOB).get_target_model()
    y_i32 = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
    print(f"[p0_dec_joiner] submitting REAL decoder inference job ...")
    job = hub.submit_inference_job(model=decoder_model, device=device, inputs={"y": [y_i32]}, name="p0-decoder")
    print(f"[p0_dec_joiner] job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[p0_dec_joiner] decoder inference FAILED: {job.get_status().message}")
        return
    decoder_out_hw = np.array(job.download_output_data()["output_0"][0])
    print(f"[p0_dec_joiner] hardware decoder_out shape={decoder_out_hw.shape}")
    print(f"[p0_dec_joiner] DECODER: cos_sim={cos_sim(decoder_out_local, decoder_out_hw):.4f} "
          f"max_abs_diff={max_abs_diff(decoder_out_local, decoder_out_hw):.4e}")

    # --- joiner: real encoder_out frame + real decoder_out (both fp32 local values) ---
    join_sess = ort.InferenceSession(joiner_onnx(), so, providers=["CPUExecutionProvider"])
    joiner_out_local = join_sess.run(None, {
        "encoder_out": real_encoder_frame.reshape(1, 512),
        "decoder_out": decoder_out_local.reshape(1, 512),
    })[0]
    print(f"[p0_dec_joiner] local joiner_out shape={joiner_out_local.shape}")

    joiner_model = hub.get_job(JOINER_COMPILE_JOB).get_target_model()
    print(f"[p0_dec_joiner] submitting REAL joiner inference job ...")
    job = hub.submit_inference_job(
        model=joiner_model, device=device,
        inputs={"encoder_out": [real_encoder_frame.reshape(1, 512).astype(np.float32)],
                "decoder_out": [decoder_out_local.reshape(1, 512).astype(np.float32)]},
        name="p0-joiner")
    print(f"[p0_dec_joiner] job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[p0_dec_joiner] joiner inference FAILED: {job.get_status().message}")
        return
    joiner_out_hw = np.array(job.download_output_data()["output_0"][0])
    print(f"[p0_dec_joiner] hardware joiner_out shape={joiner_out_hw.shape}")
    print(f"[p0_dec_joiner] JOINER: cos_sim={cos_sim(joiner_out_local, joiner_out_hw):.4f} "
          f"max_abs_diff={max_abs_diff(joiner_out_local, joiner_out_hw):.4e}")

    print(f"\n[p0_dec_joiner] argmax local={int(np.argmax(joiner_out_local))} "
          f"hw={int(np.argmax(joiner_out_hw))}")


if __name__ == "__main__":
    main()
