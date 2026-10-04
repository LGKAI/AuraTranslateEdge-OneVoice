"""Step 4 -- genuine end-to-end Zipformer RNN-T greedy decode using REAL
AI Hub inference jobs, this time against the FULL-UTTERANCE encoder
(FIXED_FRAMES=1500, compile job jgd241rz5 / target_model_id mq2y0227m,
see step4.md SS4c-3) instead of the original 103-frame chunked encoder.

This is the direct fix for the chunking-induced context-loss finding:
the "hynt/Zipformer-30M-RNNT-6000h" checkpoint is an OFFLINE (full-context)
model with no cache-state ONNX export, so independent 103-frame chunks
each lost context and the chunked+int8 real-hardware decode produced ZERO
tokens (see zipformer_e2e_decode.py's chunking loop and step4.md SS4c-3).
Recompiling the SAME encoder/weights at a 1500-frame (~15s) fixed budget
lets one real utterance fit in ONE non-chunked encoder call, matching how
the model was actually trained to run -- no architecture change, no model
swap, just a bigger fixed compile-time shape. Decoder/joiner reuse the
already-profiled SS4c-2 models unchanged (their I/O doesn't depend on the
encoder's frame budget).
"""
import os
import json

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

ENCODER_COMPILE_JOB = "jgd241rz5"  # full-utterance, FIXED_FRAMES=1500
DECODER_COMPILE_JOB = "jpey63rv5"
JOINER_COMPILE_JOB = "jp16oxdn5"

BLANK_ID = 0
CONTEXT_SIZE = 2
FIXED_FRAMES = 1500
MAX_TOKENS = 50

SNAP_ROOT = os.path.join(ROOT, "third_party_zipformer",
                          "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")


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
    n = fbank.num_frames_ready
    return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)


def run_inference(model_id, inputs, device, name):
    print(f"  [{name}] submitting real inference job on {device.name} ...", flush=True)
    job = hub.submit_inference_job(model=model_id, device=device, inputs=inputs, name=name)
    print(f"  [{name}] job: {job.job_id}  {job.url}", flush=True)
    job.wait()
    status = job.get_status()
    if status.code != "SUCCESS":
        raise RuntimeError(f"{name} inference FAILED: {status.message}")
    return job.download_output_data()


def main():
    device = hub.Device(DEVICE_NAME)
    ENCODER_MODEL = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()
    DECODER_MODEL = hub.get_job(DECODER_COMPILE_JOB).get_target_model()
    JOINER_MODEL = hub.get_job(JOINER_COMPILE_JOB).get_target_model()

    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    # same utterance as the earlier chunked attempt, for a direct before/after comparison
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav_path = os.path.join(ROOT, item["path"])
    print(f"[zipformer_fullutt] utterance: {wav_path}")
    print(f"[zipformer_fullutt] reference: {item['transcript']}")

    wav, sr = sf.read(wav_path, dtype="float32")
    feats = compute_fbank(wav, sr)
    real_frames = feats.shape[0]
    assert real_frames <= FIXED_FRAMES, f"{real_frames} frames exceeds {FIXED_FRAMES} budget"
    print(f"[zipformer_fullutt] real audio frames={real_frames} (single-shot, budget={FIXED_FRAMES})")

    feats_padded = np.concatenate(
        [feats, np.zeros((FIXED_FRAMES - real_frames, 80), dtype=np.float32)], axis=0)
    x = feats_padded[None, :, :].astype(np.float32)
    x_lens = np.array([real_frames], dtype=np.int32)

    enc_out = run_inference(ENCODER_MODEL, {"x": [x], "x_lens": [x_lens]}, device, "encoder-fullutt")
    encoder_out = np.array(enc_out["output_0"][0])
    T = int(np.array(enc_out["output_1"][0]).reshape(-1)[0])
    encoder_out = encoder_out[:, :T, :]
    print(f"[zipformer_fullutt] encoder_out shape={encoder_out.shape}, valid frames T={T}")

    # --- greedy RNN-T decode, real decoder/joiner calls, batched joiner scan ---
    hyp = [-1] * CONTEXT_SIZE
    y_in = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
    dec_out = run_inference(DECODER_MODEL, {"y": [y_in]}, device, "decoder-init")
    decoder_out = np.array(dec_out["output_0"][0])

    tokens = []
    t = 0
    calls = 2  # encoder + initial decoder
    while t < T and len(tokens) < MAX_TOKENS:
        remaining = T - t
        enc_batch = [encoder_out[0, tt, :].reshape(1, 512).astype(np.float32) for tt in range(t, T)]
        dec_batch = [decoder_out.reshape(1, 512).astype(np.float32)] * remaining
        joiner_out = run_inference(
            JOINER_MODEL, {"encoder_out": enc_batch, "decoder_out": dec_batch}, device,
            f"joiner-scan-t{t}")
        calls += 1
        logits_list = joiner_out["output_0"]

        emit_idx = None
        emit_token = None
        for i in range(remaining):
            logit = np.array(logits_list[i]).reshape(-1)
            y = int(np.argmax(logit))
            if y != BLANK_ID:
                emit_idx = t + i
                emit_token = y
                break

        if emit_idx is None:
            print(f"[zipformer_fullutt] frames {t}..{T-1} all blank -- decode finished")
            break

        tokens.append(emit_token)
        hyp = hyp[1:] + [emit_token]
        print(f"[zipformer_fullutt] frame {emit_idx}: emit token_id={emit_token} "
              f"(hyp so far: {tokens})")

        y_in = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
        dec_out = run_inference(DECODER_MODEL, {"y": [y_in]}, device, f"decoder-t{emit_idx}")
        decoder_out = np.array(dec_out["output_0"][0])
        calls += 1
        t = emit_idx  # re-scan same frame in case of multiple symbols

    print(f"\n[zipformer_fullutt] === DONE: {calls} real AI Hub inference calls total ===")

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(snap_dir(), "bpe.model"))
    hyp_text = sp.decode(tokens)
    print(f"[zipformer_fullutt] token ids: {tokens}")
    print(f"[zipformer_fullutt] HYPOTHESIS (real NPU end-to-end, single-shot): {hyp_text}")
    print(f"[zipformer_fullutt] REFERENCE:                                    {item['transcript']}")


if __name__ == "__main__":
    main()
