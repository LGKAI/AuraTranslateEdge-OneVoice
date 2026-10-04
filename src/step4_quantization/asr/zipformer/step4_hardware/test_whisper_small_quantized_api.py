"""Run Qualcomm's fetched Whisper-Small-Quantized QNN bundle via AI Hub API.

This avoids importing qai_hub_models.models.whisper_small_quantized, which is
blocked on Windows by AIMET-ONNX. The fetched bundle already contains QNN
context binaries and metadata with quantized I/O parameters, so we can submit
those binaries directly to AI Hub and drive the Whisper decode loop ourselves.

Default is intentionally tiny (one Vietnamese sample, six generated tokens) so
we can prove the API path before spending many remote decoder round trips.
"""
import json
import os
import sys
import time

import jiwer
import numpy as np
import qai_hub as hub
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import normalize_text


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE = os.path.join(ROOT, "whisper_small_quantized-qnn_context_binary-w8a16-qualcomm_qcs9075")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
MODEL_ID = "openai/whisper-small"
SAMPLE_RATE = 16000
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "6"))
RESULT_JSON = os.path.join(ROOT, "outputs", "whisper_small_quantized_api_vi1.json")
CACHE_JSON = os.path.join(ROOT, "outputs", "whisper_small_quantized_hub_model_ids.json")


def qparams(meta, file_name, io_kind, name):
    spec = meta["model_files"][file_name][io_kind][name]
    qp = spec.get("quantization_parameters")
    return spec["shape"], spec["dtype"], qp


def quantize(x, dtype, qp):
    if qp is None:
        return x.astype(dtype)
    q = np.round(x / qp["scale"] + qp["zero_point"])
    if dtype == "uint8":
        return np.clip(q, 0, 255).astype(np.uint8)
    if dtype == "uint16":
        return np.clip(q, 0, 65535).astype(np.uint16)
    raise ValueError(dtype)


def dequantize(x, qp):
    if qp is None:
        return x
    return (x.astype(np.float32) - qp["zero_point"]) * qp["scale"]


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        print(f"[{label}] {job.job_id}: {st.code} {st.message[:160] if st.message else ''}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def run_job(model_path, inputs, device, name, result):
    job = hub.submit_inference_job(model=model_path, device=device, inputs=inputs, name=name)
    print(f"[{name}] {job.job_id} {job.url}", flush=True)
    st = poll(job, name)
    result.setdefault("jobs", []).append({
        "name": name,
        "job_id": job.job_id,
        "url": job.url,
        "status": st.code,
        "message": st.message,
    })
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if st.code != "SUCCESS":
        raise RuntimeError(f"{name} failed: {st.message}")
    return job.download_output_data()


def get_cached_models():
    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON, encoding="utf-8") as f:
            cached = json.load(f)
        encoder = hub.get_model(cached["encoder_model_id"])
        decoder = hub.get_model(cached["decoder_model_id"])
        return encoder, decoder, cached

    encoder = hub.upload_model(os.path.join(BUNDLE, "encoder.bin"))
    decoder = hub.upload_model(os.path.join(BUNDLE, "decoder.bin"))
    cached = {
        "encoder_model_id": encoder.model_id,
        "decoder_model_id": decoder.model_id,
    }
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)
    return encoder, decoder, cached


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return wav


def first_vi_sample():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi = [r for r in manifest if r["lang"] == "vi"]
    return min(vi, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))


def main():
    with open(os.path.join(BUNDLE, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    item = first_vi_sample()
    wav_path = os.path.join(ROOT, item["path"])
    ref = item["transcript"]
    result = {
        "model": "whisper_small_quantized",
        "device": DEVICE_NAME,
        "bundle": BUNDLE,
        "audio": item["path"],
        "reference": ref,
        "max_tokens": MAX_TOKENS,
        "jobs": [],
    }

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="vi", task="transcribe")
    features = feature_extractor(load_audio(wav_path), sampling_rate=SAMPLE_RATE, return_tensors="np")[
        "input_features"
    ].astype(np.float32)

    enc_shape, enc_dtype, enc_qp = qparams(meta, "encoder.bin", "inputs", "input_features")
    assert list(features.shape) == enc_shape, (features.shape, enc_shape)
    enc_in = quantize(features, enc_dtype, enc_qp)

    device = hub.Device(DEVICE_NAME)
    encoder_model, decoder_model, cached = get_cached_models()
    result["cached_model_ids"] = cached
    enc_out = run_job(encoder_model, {"input_features": [enc_in]}, device, "wsq-encoder-vi1", result)

    dec_meta = meta["model_files"]["decoder.bin"]
    inputs_meta = dec_meta["inputs"]
    outputs_meta = dec_meta["outputs"]
    num_layers = len([k for k in inputs_meta if k.startswith("k_cache_self_") and k.endswith("_in")])

    # Force Vietnamese transcription. Without these prompt tokens, Whisper Tiny
    # locally drifted into English on the same files.
    prompt_ids = tokenizer.get_decoder_prompt_ids(language="vi", task="transcribe")
    forced = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    forced.extend(tok for _, tok in prompt_ids)
    generated = []

    kv_self = {}
    for i in range(num_layers):
        for prefix in ("k", "v"):
            name = f"{prefix}_cache_self_{i}_in"
            shape, dtype, qp = qparams(meta, "decoder.bin", "inputs", name)
            kv_self[name] = quantize(np.zeros(shape, dtype=np.float32), dtype, qp)

    attention = np.full(inputs_meta["attention_mask"]["shape"], -100.0, dtype=np.float32)
    position = 0
    last_token = forced[0]

    for step in range(MAX_TOKENS):
        if step + 1 < len(forced):
            last_token = forced[step]
            next_forced = forced[step + 1]
        else:
            next_forced = None

        attention[:, :, :, 200 - position - 1 :] = 0.0
        _, attn_dtype, attn_qp = qparams(meta, "decoder.bin", "inputs", "attention_mask")
        dec_inputs = {
            "input_ids": [np.array([[last_token]], dtype=np.int32)],
            "position_ids": [np.array([position], dtype=np.int32)],
        }
        # Preserve metadata order after the first two scalar inputs.
        for name in inputs_meta:
            if name in dec_inputs:
                continue
            if name == "attention_mask":
                dec_inputs[name] = [quantize(attention, attn_dtype, attn_qp)]
            elif name.startswith(("k_cache_self_", "v_cache_self_")):
                dec_inputs[name] = [kv_self[name]]
            else:
                dec_inputs[name] = [np.asarray(enc_out[name][0])]

        dec_out = run_job(decoder_model, dec_inputs, device, f"wsq-decoder-step{step}", result)

        logits_name = "logits"
        _, _, logits_qp = qparams(meta, "decoder.bin", "outputs", logits_name)
        logits = dequantize(np.asarray(dec_out[logits_name][0]), logits_qp).reshape(-1)
        token = int(np.argmax(logits))
        if next_forced is not None:
            token = next_forced
        else:
            generated.append(token)
        result["forced_prompt_tokens"] = forced
        result["generated_tokens"] = generated
        result["partial_text"] = tokenizer.decode(generated, skip_special_tokens=True).strip()

        for i in range(num_layers):
            for prefix in ("k", "v"):
                out_name = f"{prefix}_cache_self_{i}_out"
                in_name = f"{prefix}_cache_self_{i}_in"
                kv_self[in_name] = np.asarray(dec_out[out_name][0])

        last_token = token
        position += 1
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if token == tokenizer.eos_token_id:
            break

    hyp = tokenizer.decode(generated, skip_special_tokens=True).strip()
    result["hypothesis"] = hyp
    result["wer"] = jiwer.wer(normalize_text(ref), normalize_text(hyp)) if hyp else 1.0
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
