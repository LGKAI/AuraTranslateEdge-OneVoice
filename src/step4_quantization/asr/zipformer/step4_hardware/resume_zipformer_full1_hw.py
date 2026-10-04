"""Resume the long Zipformer full-hardware greedy decode from saved jobs."""
import json
import os
import time

import numpy as np
import qai_hub as hub
import sentencepiece as spm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
ENCODER_JOB = os.environ.get("ENCODER_JOB", "jp81n0d85")
DECODER_MODEL_ID = os.environ.get("DECODER_MODEL_ID", "mno9oo49n")
JOINER_MODEL_ID = os.environ.get("JOINER_MODEL_ID", "mq2y00p0m")
START_T = int(os.environ.get("START_T", "68"))
TOKENS = [int(x) for x in os.environ.get("TOKENS", "166,26,104,51,379,279,18,1050").split(",") if x]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "30"))
MAX_SYM_PER_FRAME = int(os.environ.get("MAX_SYM_PER_FRAME", "3"))
RESULT_JSON = os.path.join(ROOT, "outputs", "zipformer_full1_hw_resume.json")
PERMISSIVE_QNN_OPTIONS = "--qnn_options context_binary_compatibility=permissive"
BLANK_ID = 0


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {st.message[:160] if st.message else ''}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def run(model, inputs, device, name, jobs):
    job = hub.submit_inference_job(
        model=model, device=device, inputs=inputs, name=name, options=PERMISSIVE_QNN_OPTIONS
    )
    print(f"[{name}] job {job.job_id} {job.url}", flush=True)
    st = poll(job, name)
    jobs.append({"name": name, "job_id": job.job_id, "url": job.url, "status": st.code, "message": st.message})
    if st.code != "SUCCESS":
        raise RuntimeError(f"{name} failed: {st.message}")
    return job.download_output_data()


def write(result):
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    device = hub.Device(DEVICE_NAME)
    decoder_model = hub.get_model(DECODER_MODEL_ID)
    joiner_model = hub.get_model(JOINER_MODEL_ID)

    enc_data = hub.get_job(ENCODER_JOB).download_output_data()
    encoder_out = np.asarray(enc_data["output_0"][0])
    T = int(np.asarray(enc_data["output_1"][0]).reshape(-1)[0])
    encoder_out = encoder_out[:, :T, :]

    jobs = []
    tokens = list(TOKENS)
    t = START_T
    last_emit_frame = START_T
    sym_this_frame = 1

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(ROOT, "third_party_zipformer_real", "bpe.model"))
    result = {
        "device": DEVICE_NAME,
        "encoder_job": ENCODER_JOB,
        "decoder_model_id": DECODER_MODEL_ID,
        "joiner_model_id": JOINER_MODEL_ID,
        "start_t": START_T,
        "encoder_valid_T": T,
        "tokens": tokens,
        "hypothesis": sp.decode(tokens),
        "jobs": jobs,
    }
    write(result)

    while t < T and len(tokens) < MAX_TOKENS:
        hyp = tokens[-2:] if len(tokens) >= 2 else [-1] * (2 - len(tokens)) + tokens
        y = np.asarray(hyp, dtype=np.int32).reshape(1, 2)
        dec = run(decoder_model, {"y": [y]}, device, f"resume-decoder-t{t}", jobs)
        decoder_out = np.asarray(dec["output_0"][0])

        remaining = T - t
        enc_batch = [encoder_out[0, tt, :].reshape(1, 512).astype(np.float32) for tt in range(t, T)]
        dec_batch = [decoder_out.reshape(1, 512).astype(np.float32)] * remaining
        join = run(
            joiner_model,
            {"encoder_out": enc_batch, "decoder_out": dec_batch},
            device,
            f"resume-joiner-t{t}",
            jobs,
        )

        emit = None
        for i, arr in enumerate(join["output_0"]):
            tok = int(np.argmax(np.asarray(arr).reshape(-1)))
            if tok != BLANK_ID:
                emit = (t + i, tok)
                break
        if emit is None:
            result["decode_stop"] = f"all blank from frame {t}"
            break

        emit_frame, tok = emit
        tokens.append(tok)
        print(f"[decode] frame {emit_frame}: token {tok}; text={sp.decode(tokens)}", flush=True)
        if emit_frame == last_emit_frame:
            sym_this_frame += 1
        else:
            last_emit_frame = emit_frame
            sym_this_frame = 1
        t = emit_frame + 1 if sym_this_frame >= MAX_SYM_PER_FRAME else emit_frame
        result.update({"tokens": tokens, "hypothesis": sp.decode(tokens), "current_t": t, "partial_num_tokens": len(tokens)})
        write(result)

    result.update({"tokens": tokens, "hypothesis": sp.decode(tokens), "num_tokens": len(tokens), "final_t": t})
    write(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
