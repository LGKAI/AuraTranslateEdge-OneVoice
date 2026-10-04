"""Dataset-level Zipformer evaluation with hardware w8a16 encoder.

This script does the part that actually matters for the current deployment
question:

1. run the full-utterance encoder on Qualcomm AI Hub / IQ-9075 for every
   Vietnamese sample in data/asr/manifest.json
2. decode the resulting encoder activations locally with the already-verified
   decoder/joiner ONNX models
3. report per-sample and aggregate cosine/WER so the hardware path is judged
   on a real dataset, not a single cherry-picked utterance

The hardware bottleneck here is the encoder. The decoder/joiner are tiny and
have already been numerically verified against local ONNXRuntime, so using the
local pair for dataset scoring keeps the experiment fast enough to finish while
still reflecting the deployed encoder quality.
"""
import csv
import json
import os
import sys
import time

import jiwer
import numpy as np
import onnxruntime as ort
import qai_hub as hub
import soundfile as sf
import kaldi_native_fbank as knf
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import normalize_text


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
ENCODER_COMPILE_JOB = os.environ.get("ENCODER_COMPILE_JOB", "j5qvdwxeg")
FIXED_FRAMES = int(os.environ.get("FIXED_FRAMES", "1500"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "5"))
RESULT_CSV = os.path.join(ROOT, "outputs", "zipformer_w8a16_dataset_eval.csv")
RESULT_JSON = os.path.join(ROOT, "outputs", "zipformer_w8a16_dataset_eval.json")

SNAP_ROOT = os.path.join(
    ROOT, "third_party_zipformer", "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots"
)
REAL_ARTIFACT_DIR = os.path.join(ROOT, "third_party_zipformer_real")
LOCAL_ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_static_1500.onnx")


def snap_dir():
    return os.path.join(SNAP_ROOT, sorted(os.listdir(SNAP_ROOT))[0])


def artifact_path(filename):
    real = os.path.join(REAL_ARTIFACT_DIR, filename)
    if os.path.exists(real) and os.path.getsize(real) > 0:
        return real
    return os.path.join(snap_dir(), filename)


LOCAL_DECODER_ONNX = artifact_path("decoder-epoch-20-avg-10.onnx")
LOCAL_JOINER_ONNX = artifact_path("joiner-epoch-20-avg-10.onnx")
LOCAL_BPE = artifact_path("bpe.model")

BLANK_ID = 0
CONTEXT_SIZE = 2


def compute_fbank(wav, sr):
    assert sr == 16000, f"expected 16kHz, got {sr}"
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav.tolist())
    fbank.input_finished()
    return np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)


def pad(feats, frames=FIXED_FRAMES):
    real = feats.shape[0]
    if real >= frames:
        return feats[:frames][None, :, :], np.array([frames], dtype=np.int32), real
    z = np.zeros((frames - real, 80), dtype=np.float32)
    return np.concatenate([feats, z], axis=0)[None, :, :], np.array([real], dtype=np.int32), real


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {st.message[:160] if st.message else ''}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def local_decode(encoder_out, decoder_sess, joiner_sess, max_tokens=64, max_sym_per_frame=3):
    """Greedy RNNT decode using local verified decoder/joiner."""
    T = encoder_out.shape[1]
    hyp = [-1] * CONTEXT_SIZE
    y = np.array([hyp], dtype=np.int64)
    decoder_out = decoder_sess.run(None, {"y": y})[0]

    tokens = []
    t = 0
    last_emit_frame = None
    sym_this_frame = 0
    while t < T and len(tokens) < max_tokens:
        emit_idx = None
        emit_token = None
        for i in range(t, T):
            enc = encoder_out[0, i, :].reshape(1, 512).astype(np.float32)
            dec = decoder_out.reshape(1, 512).astype(np.float32)
            logits = joiner_sess.run(None, {"encoder_out": enc, "decoder_out": dec})[0]
            token = int(np.argmax(np.asarray(logits).reshape(-1)))
            if token != BLANK_ID:
                emit_idx = i
                emit_token = token
                break
        if emit_idx is None:
            break
        tokens.append(emit_token)
        hyp = hyp[1:] + [emit_token]
        y = np.array([hyp], dtype=np.int64)
        decoder_out = decoder_sess.run(None, {"y": y})[0]
        if emit_idx == last_emit_frame:
            sym_this_frame += 1
        else:
            last_emit_frame = emit_idx
            sym_this_frame = 1
        t = emit_idx + 1 if sym_this_frame >= max_sym_per_frame else emit_idx

    sp = spm.SentencePieceProcessor()
    sp.load(LOCAL_BPE)
    return tokens, sp.decode(tokens)


def main():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    vi_items = sorted(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    if MAX_SAMPLES > 0:
        vi_items = vi_items[:MAX_SAMPLES]

    device = hub.Device(DEVICE_NAME)
    encoder_model = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc_sess = ort.InferenceSession(LOCAL_ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(LOCAL_DECODER_ONNX, so, providers=["CPUExecutionProvider"])
    join_sess = ort.InferenceSession(LOCAL_JOINER_ONNX, so, providers=["CPUExecutionProvider"])

    prepared = []
    for idx, item in enumerate(vi_items, 1):
        wav_path = os.path.join(ROOT, item["path"])
        wav, sr = sf.read(wav_path, dtype="float32")
        feats = compute_fbank(wav, sr)
        x, x_lens, real_frames = pad(feats)
        print(f"[dataset] ({idx}/{len(vi_items)}) {os.path.basename(wav_path)} frames={real_frames}", flush=True)

        # local fp32 encoder reference
        fp_out, fp_len = enc_sess.run(None, {"x": x.astype(np.float32), "x_lens": x_lens.astype(np.int64)})
        fp_T = int(np.asarray(fp_len).reshape(-1)[0])
        fp_out = np.asarray(fp_out)[:, :fp_T, :]
        prepared.append({
            "item": item,
            "x": x.astype(np.float32),
            "x_lens": x_lens.astype(np.int32),
            "real_frames": real_frames,
            "fp_out": fp_out,
        })

    t0 = time.perf_counter()
    hw_job = hub.submit_inference_job(
        model=encoder_model,
        device=device,
        inputs={
            "x": [p["x"] for p in prepared],
            "x_lens": [p["x_lens"] for p in prepared],
        },
        name=f"zipf-w8a16-dataset-batch{len(prepared)}",
    )
    print(f"[dataset] hw batch job {hw_job.job_id} {hw_job.url}", flush=True)
    st = poll(hw_job, "encoder-batch")
    if st.code != "SUCCESS":
        raise RuntimeError(f"encoder batch failed: {st.message}")
    hw_data = hw_job.download_output_data()
    hw_wall = time.perf_counter() - t0

    rows = []
    for idx, p in enumerate(prepared, 1):
        item = p["item"]
        hw_out = np.asarray(hw_data["output_0"][idx - 1])
        hw_T = int(np.asarray(hw_data["output_1"][idx - 1]).reshape(-1)[0])
        hw_out = hw_out[:, :hw_T, :]

        T = min(p["fp_out"].shape[1], hw_out.shape[1])
        fp_cut = p["fp_out"][:, :T, :]
        hw_cut = hw_out[:, :T, :]
        enc_cos = cos_sim(fp_cut, hw_cut)

        hyp_fp_tokens, hyp_fp = local_decode(fp_cut, dec_sess, join_sess)
        hyp_hw_tokens, hyp_hw = local_decode(hw_cut, dec_sess, join_sess)
        ref = item.get("transcript", "")

        row = {
            "file": os.path.basename(item["path"]),
            "real_frames": int(p["real_frames"]),
            "valid_T": int(T),
            "encoder_cos": round(enc_cos, 6),
            "ref": ref,
            "fp_hyp": hyp_fp,
            "hw_hyp": hyp_hw,
            "fp_wer": round(jiwer.wer(normalize_text(ref), normalize_text(hyp_fp)), 4),
            "hw_wer": round(jiwer.wer(normalize_text(ref), normalize_text(hyp_hw)), 4),
            "hw_wall_sec": round(hw_wall, 2),
            "hw_job": hw_job.job_id,
            "hw_url": hw_job.url,
            "fp_tokens": " ".join(map(str, hyp_fp_tokens)),
            "hw_tokens": " ".join(map(str, hyp_hw_tokens)),
        }
        rows.append(row)
        print(
            f"[dataset] {row['file']} cos={row['encoder_cos']:.4f} "
            f"fpWER={row['fp_wer']:.3f} hwWER={row['hw_wer']:.3f} hw='{hyp_hw[:60]}'",
            flush=True,
        )

    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "device": DEVICE_NAME,
        "encoder_compile_job": ENCODER_COMPILE_JOB,
        "num_samples": len(rows),
        "mean_encoder_cos": float(np.mean([r["encoder_cos"] for r in rows])),
        "mean_fp_wer": float(np.mean([r["fp_wer"] for r in rows])),
        "mean_hw_wer": float(np.mean([r["hw_wer"] for r in rows])),
        "median_hw_wall_sec": float(np.median([r["hw_wall_sec"] for r in rows])),
    }
    payload = {"summary": summary, "rows": rows}
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(RESULT_CSV)
    print(RESULT_JSON)


if __name__ == "__main__":
    main()
