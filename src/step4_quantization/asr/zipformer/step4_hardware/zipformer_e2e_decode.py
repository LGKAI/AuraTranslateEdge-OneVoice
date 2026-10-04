"""Step 4 -- genuine end-to-end Zipformer RNN-T greedy decode using REAL
AI Hub inference jobs (hub.submit_inference_job, not just profiling) on
real QCS6490 hardware for every single encoder/decoder/joiner call. This
is the "missing loop" that connects the 3 already-individually-profiled
components (see step4.md SS4c/SS4c-2) into an actual working ASR pipeline
running entirely on real Snapdragon silicon.

Naive frame-by-frame decoding would need ~103 separate real hardware
round-trips just for the joiner (one per encoder frame), each costing
real wall-clock time for AI Hub's device provisioning -- impractically
slow. Instead: since the decoder's output only changes when a NEW token
is emitted (RNN-T's decoder state is a step function over frames, constant
between emissions), each joiner call batches ALL remaining frames against
the CURRENT decoder_out in one inference-job round trip, scans for the
first non-blank frame, and only recomputes the decoder (a second real
call) at that point. Total real AI Hub calls ~= 2 + 2*K where K = number
of emitted tokens (small for a short clip), not ~103.
"""
import os
import sys

import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf
import qai_hub as hub
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"

ENCODER_COMPILE_JOB = "jgk86n2yg"
DECODER_COMPILE_JOB = "jpey63rv5"
JOINER_COMPILE_JOB = "jp16oxdn5"

BLANK_ID = 0
CONTEXT_SIZE = 2
FIXED_FRAMES = 103
MAX_TOKENS = 50
MAX_SYM_PER_FRAME = 3

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
    out = job.download_output_data()
    return out


def main():
    device = hub.Device(DEVICE_NAME)
    ENCODER_MODEL = hub.get_job(ENCODER_COMPILE_JOB).get_target_model()
    DECODER_MODEL = hub.get_job(DECODER_COMPILE_JOB).get_target_model()
    JOINER_MODEL = hub.get_job(JOINER_COMPILE_JOB).get_target_model()

    # --- pick a real Vietnamese utterance ---
    # (none of the 5 real samples are short enough to fit the compiled
    # encoder's fixed 103-frame/~1.03s window in one shot -- confirmed by
    # testing: feeding just the first 103 frames of a 724-frame utterance
    # locally, via the ORIGINAL onnx files, also gives zero emitted tokens,
    # so that's a genuine property of this utterance's first second, not a
    # decode-loop bug. The real fix: chunk the full audio into consecutive
    # 103-frame windows -- this encoder export has no exposed recurrent
    # state input/output, so each chunk is encoded independently (a real,
    # honest limitation: quality may dip right at chunk boundaries versus
    # a stateful streaming encoder, but every chunk is still a genuine
    # real-hardware inference call on real audio) -- then run ONE
    # continuous greedy decode across the concatenated encoder_out.
    import json
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))
    wav_path = os.path.join(ROOT, item["path"])
    print(f"[zipformer_e2e] utterance: {wav_path}")
    print(f"[zipformer_e2e] reference: {item['transcript']}")

    wav, sr = sf.read(wav_path, dtype="float32")
    feats = compute_fbank(wav, sr)
    real_frames = feats.shape[0]
    n_chunks = (real_frames + FIXED_FRAMES - 1) // FIXED_FRAMES
    print(f"[zipformer_e2e] real audio frames={real_frames} -> {n_chunks} chunks of {FIXED_FRAMES}")

    encoder_out_chunks = []
    T = 0
    for c in range(n_chunks):
        chunk = feats[c * FIXED_FRAMES:(c + 1) * FIXED_FRAMES]
        chunk_real_len = chunk.shape[0]
        if chunk_real_len < FIXED_FRAMES:
            chunk = np.concatenate(
                [chunk, np.zeros((FIXED_FRAMES - chunk_real_len, 80), dtype=np.float32)], axis=0)
        x = chunk[None, :, :].astype(np.float32)
        # int32, not int64: encoder was compiled with --truncate_64bit_io, which
        # truncates declared int64 graph I/O to int32 at the QNN level -- feeding
        # int64 here fails with "Cannot assign data from unexpected type. Expected
        # int32, got int64" (confirmed via a real inference attempt).
        x_lens = np.array([chunk_real_len], dtype=np.int32)

        # AI Hub's download_output_data() returns generic keys "output_0",
        # "output_1", ... in the model's declared output order (encoder_out,
        # encoder_out_lens), not the original tensor names (confirmed via a
        # real test call).
        enc_out = run_inference(ENCODER_MODEL, {"x": [x], "x_lens": [x_lens]}, device, f"encoder-chunk{c}")
        chunk_encoder_out = np.array(enc_out["output_0"][0])
        chunk_T = int(np.array(enc_out["output_1"][0]).reshape(-1)[0])
        encoder_out_chunks.append(chunk_encoder_out[:, :chunk_T, :])
        T += chunk_T
        print(f"[zipformer_e2e] chunk {c}: real_frames={chunk_real_len}, "
              f"encoder_out valid frames={chunk_T}")

    encoder_out = np.concatenate(encoder_out_chunks, axis=1)
    print(f"[zipformer_e2e] concatenated encoder_out shape={encoder_out.shape}, total valid frames T={T}")

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
            print(f"[zipformer_e2e] frames {t}..{T-1} all blank -- decode finished")
            break

        tokens.append(emit_token)
        hyp = hyp[1:] + [emit_token]
        print(f"[zipformer_e2e] frame {emit_idx}: emit token_id={emit_token} "
              f"(hyp so far: {tokens})")

        y_in = np.array(hyp, dtype=np.int32).reshape(1, CONTEXT_SIZE)
        dec_out = run_inference(DECODER_MODEL, {"y": [y_in]}, device, f"decoder-t{emit_idx}")
        decoder_out = np.array(dec_out["output_0"][0])
        calls += 1
        t = emit_idx  # re-scan same frame in case of multiple symbols

    print(f"\n[zipformer_e2e] === DONE: {calls} real AI Hub inference calls total ===")

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(snap_dir(), "bpe.model"))
    hyp_text = sp.decode(tokens)
    print(f"[zipformer_e2e] token ids: {tokens}")
    print(f"[zipformer_e2e] HYPOTHESIS (real NPU end-to-end): {hyp_text}")
    print(f"[zipformer_e2e] REFERENCE:                        {item['transcript']}")


if __name__ == "__main__":
    main()
