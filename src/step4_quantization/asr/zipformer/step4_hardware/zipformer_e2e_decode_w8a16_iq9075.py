"""End-to-end Zipformer RNN-T decode with w8a16/int16 encoder on AI Hub.

Encoder:
  - w8a16/int16 activation compile job j5qvdwxeg on Dragonwing IQ-9075 EVK

Decoder/joiner:
  - reuse the best known compiled models by default. They are tiny and were
    previously verified numerically; this script runs real inference jobs for
    them as part of greedy decode.

The goal is not another cosine probe. This runs the actual RNN-T greedy loop
and saves the emitted token ids + decoded text.
"""
import json
import os
import time

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub
import sentencepiece as spm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

ENCODER_COMPILE_JOB = os.environ.get("ENCODER_COMPILE_JOB", "j5qvdwxeg")
DECODER_COMPILE_JOB = os.environ.get("DECODER_COMPILE_JOB", "jpey63rv5")
JOINER_COMPILE_JOB = os.environ.get("JOINER_COMPILE_JOB", "jp16oxdn5")
PERMISSIVE_QNN_OPTIONS = "--qnn_options context_binary_compatibility=permissive"

BLANK_ID = 0
CONTEXT_SIZE = 2
FIXED_FRAMES = 1500
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))
MAX_SYM_PER_FRAME = int(os.environ.get("MAX_SYM_PER_FRAME", "3"))
RESULT_JSON = os.path.join(ROOT, "outputs", "zipformer_e2e_w8a16_iq9075.json")

SNAP_ROOT = os.path.join(
    ROOT, "third_party_zipformer", "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots"
)


def snap_dir():
    return os.path.join(SNAP_ROOT, os.listdir(SNAP_ROOT)[0])


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


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {st.message[:180] if st.message else ''}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def run_inference(model, inputs, device, name, result, options=""):
    job = hub.submit_inference_job(model=model, device=device, inputs=inputs, name=name, options=options)
    print(f"[{name}] job {job.job_id} {job.url}", flush=True)
    st = poll(job, name)
    result.setdefault("jobs", []).append({
        "name": name,
        "job_id": job.job_id,
        "url": job.url,
        "status": st.code,
        "message": st.message,
    })
    if st.code != "SUCCESS":
        raise RuntimeError(f"{name} failed: {st.message}")
    return job.download_output_data()


def main():
    device = hub.Device(DEVICE_NAME)
    result = {
        "device": DEVICE_NAME,
        "encoder_compile_job": ENCODER_COMPILE_JOB,
        "decoder_compile_job": DECODER_COMPILE_JOB,
        "joiner_compile_job": JOINER_COMPILE_JOB,
        "jobs": [],
    }

    encoder_model = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()
    decoder_model = hub.get_job(DECODER_COMPILE_JOB).get_target_model()
    joiner_model = hub.get_job(JOINER_COMPILE_JOB).get_target_model()
    result["encoder_model_id"] = encoder_model.model_id
    result["decoder_model_id"] = decoder_model.model_id
    result["joiner_model_id"] = joiner_model.model_id

    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav_path = os.path.join(ROOT, item["path"])
    wav, sr = sf.read(wav_path, dtype="float32")
    feats = compute_fbank(wav, sr)
    real_frames = feats.shape[0]
    if real_frames > FIXED_FRAMES:
        raise RuntimeError(f"{real_frames} frames exceeds {FIXED_FRAMES}")

    feats_padded = np.concatenate(
        [feats, np.zeros((FIXED_FRAMES - real_frames, 80), dtype=np.float32)], axis=0
    )
    x = feats_padded[None, :, :].astype(np.float32)
    x_lens = np.array([real_frames], dtype=np.int32)
    result["eval_path"] = item["path"]
    result["reference"] = item.get("transcript", "")
    result["real_frames"] = int(real_frames)

    enc_out = run_inference(
        encoder_model, {"x": [x], "x_lens": [x_lens]}, device, "encoder-w8a16-fullutt", result
    )
    encoder_out = np.array(enc_out["output_0"][0])
    T = int(np.array(enc_out["output_1"][0]).reshape(-1)[0])
    encoder_out = encoder_out[:, :T, :]
    result["encoder_valid_T"] = int(T)
    result["encoder_output_shape"] = list(encoder_out.shape)

    hyp = [-1] * CONTEXT_SIZE
    y_in = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
    dec_out = run_inference(
        decoder_model, {"y": [y_in]}, device, "decoder-init", result,
        options=PERMISSIVE_QNN_OPTIONS,
    )
    decoder_out = np.array(dec_out["output_0"][0])

    tokens = []
    t = 0
    sym_this_frame = 0
    last_emit_frame = None
    while t < T and len(tokens) < MAX_TOKENS:
        remaining = T - t
        enc_batch = [encoder_out[0, tt, :].reshape(1, 512).astype(np.float32) for tt in range(t, T)]
        dec_batch = [decoder_out.reshape(1, 512).astype(np.float32)] * remaining
        joiner_out = run_inference(
            joiner_model,
            {"encoder_out": enc_batch, "decoder_out": dec_batch},
            device,
            f"joiner-scan-t{t}",
            result,
            options=PERMISSIVE_QNN_OPTIONS,
        )

        emit_idx = None
        emit_token = None
        for i, logit_arr in enumerate(joiner_out["output_0"]):
            logit = np.array(logit_arr).reshape(-1)
            token = int(np.argmax(logit))
            if token != BLANK_ID:
                emit_idx = t + i
                emit_token = token
                break

        if emit_idx is None:
            result["decode_stop"] = f"all blank from frame {t}"
            break

        tokens.append(emit_token)
        hyp = hyp[1:] + [emit_token]
        print(f"[decode] frame {emit_idx}: token {emit_token}; tokens={tokens}", flush=True)
        result["tokens"] = tokens
        result["partial_num_tokens"] = len(tokens)
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        y_in = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
        dec_out = run_inference(
            decoder_model, {"y": [y_in]}, device, f"decoder-t{emit_idx}", result,
            options=PERMISSIVE_QNN_OPTIONS,
        )
        decoder_out = np.array(dec_out["output_0"][0])
        if emit_idx == last_emit_frame:
            sym_this_frame += 1
        else:
            last_emit_frame = emit_idx
            sym_this_frame = 1
        t = emit_idx + 1 if sym_this_frame >= MAX_SYM_PER_FRAME else emit_idx

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(snap_dir(), "bpe.model"))
    hyp_text = sp.decode(tokens)
    result["tokens"] = tokens
    result["hypothesis"] = hyp_text
    result["num_tokens"] = len(tokens)

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] wrote {RESULT_JSON}", flush=True)


if __name__ == "__main__":
    main()
