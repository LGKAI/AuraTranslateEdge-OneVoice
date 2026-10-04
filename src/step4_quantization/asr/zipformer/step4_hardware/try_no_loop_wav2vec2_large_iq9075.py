"""Try the best-quality no-loop Vietnamese ASR candidate on Qualcomm AI Hub.

Target model:
  nguyenvulebinh/wav2vec2-large-vi-vlsp2020

Why this one first:
  - CTC / no decoder loop
  - published WER is the best among the current no-loop Vietnamese candidates
  - standard Hugging Face architecture, so ONNX export is much more likely to
    compile on QNN than a custom RNNT/streaming graph

This script:
  1. downloads the checkpoint
  2. runs a local CPU decode on the 7.2s Vietnamese sample
  3. exports a fixed-shape ONNX graph
  4. builds a small calibration dataset from local Vietnamese audio
  5. submits AI Hub quantize -> compile -> inference on Dragonwing IQ-9075 EVK

If it succeeds, the full pipeline is:
  waveform -> model -> logits -> greedy CTC decode
"""
import json
import os
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import qai_hub as hub
from huggingface_hub import snapshot_download
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_wav, normalize_text


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "wav2vec2_large_vi"
CACHE_DIR = ROOT / "third_party_wav2vec2_large_vi"
MODEL_ID = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
DEVICE_PRIMARY = "Dragonwing IQ-9075 EVK"
SAMPLE_PATH = ROOT / "data" / "asr" / "vi_1.wav"
MAX_SECONDS = 7.2
SR = 16000
FIXED_SAMPLES = int(MAX_SECONDS * SR)
ONNX_PATH = OUT_DIR / "wav2vec2_large_vi.onnx"
RESULT_JSON = OUT_DIR / "wav2vec2_large_vi_iq9075.json"


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        msg = st.message or ""
        print(f"[{label}] {job.job_id}: {st.code} {msg[:180]}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def load_audio_fixed(path):
    wav = load_wav(str(path), sr=SR)
    if len(wav) >= FIXED_SAMPLES:
        return wav[:FIXED_SAMPLES].astype(np.float32)
    out = np.zeros((FIXED_SAMPLES,), dtype=np.float32)
    out[: len(wav)] = wav
    return out


def local_decode(model, feature_extractor, tokenizer, wav):
    inputs = feature_extractor(wav, sampling_rate=SR, return_tensors="pt")
    input_values = inputs.input_values
    with torch.no_grad():
        logits = model(input_values=input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    hyp = tokenizer.batch_decode(pred_ids)[0].strip()
    return hyp, logits


def export_onnx(model, onnx_path):
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_values):
            return self.inner(input_values=input_values).logits

    wrapper = Wrapper(model).eval()
    dummy = torch.zeros((1, FIXED_SAMPLES), dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=False,
        dynamic_axes=None,
        external_data=True,
    )


def build_calibration_dataset(sample_paths):
    xs = []
    for p in sample_paths:
        wav = load_audio_fixed(p)
        xs.append(wav[None, :].astype(np.float32))
    return hub.upload_dataset({"input_values": xs}, name="wav2vec2_large_vi_calib")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[w2v2] downloading {MODEL_ID} ...", flush=True)
    model_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(CACHE_DIR),
    )

    print(f"[w2v2] loading processor/model from {model_dir}", flush=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Transformers >= 4.55 blocks `torch.load` on torch < 2.6 by default.
    # This repo snapshot is trusted and only ships `pytorch_model.bin`, so we
    # intentionally bypass that guard for this one local benchmark run.
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils
    import transformers.models.wav2vec2.modeling_wav2vec2 as wav2vec2_mod

    def _allow_torch_load():
        return None

    modeling_utils.check_torch_load_is_safe = _allow_torch_load
    import_utils.check_torch_load_is_safe = _allow_torch_load
    wav2vec2_mod.create_bidirectional_mask = lambda **kwargs: None
    model = AutoModelForCTC.from_pretrained(model_dir, attn_implementation="eager").eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    with open(ROOT / "data" / "asr" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    vi_items = sorted(vi_items, key=lambda r: os.path.getsize(ROOT / r["path"]))
    sample_item = next(r for r in manifest if r["path"] == "data/asr/vi/vi_1.wav")
    sample_wav = load_audio_fixed(ROOT / sample_item["path"])

    t0 = time.perf_counter()
    local_hyp, _ = local_decode(model, feature_extractor, tokenizer, sample_wav)
    local_elapsed = time.perf_counter() - t0
    ref = sample_item["transcript"]
    local_wer = jiwer.wer(normalize_text(ref), normalize_text(local_hyp))
    print(f"[w2v2] local CPU hyp: {local_hyp}")
    print(f"[w2v2] local CPU WER={local_wer:.4f}  latency={local_elapsed:.3f}s", flush=True)

    print(f"[w2v2] exporting ONNX -> {ONNX_PATH}", flush=True)
    export_onnx(model, ONNX_PATH)

    import onnx
    onnx.checker.check_model(str(ONNX_PATH))
    print("[w2v2] ONNX checker passed", flush=True)

    device = hub.Device(DEVICE_PRIMARY)
    calib_paths = [ROOT / r["path"] for r in vi_items[:4]]
    calib_ds = build_calibration_dataset(calib_paths)
    print(f"[w2v2] calibration dataset uploaded with {len(calib_paths)} files", flush=True)

    input_specs = {"input_values": ((1, FIXED_SAMPLES), "float32")}
    print("[w2v2] submit quantize job ...", flush=True)
    qjob = hub.submit_quantize_job(
        model=str(ONNX_PATH),
        calibration_data=calib_ds,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="wav2vec2-large-vi-w8a16",
    )
    print(f"[w2v2] quant job: {qjob.job_id} {qjob.url}", flush=True)
    qst = poll(qjob, "w2v2-quant")
    if qst.code != "SUCCESS":
        print(f"[w2v2] quantization failed: {qst.message}", flush=True)
        return
    qmodel = qjob.get_target_model()
    print(f"[w2v2] quantized model: {qmodel.model_id}", flush=True)

    print("[w2v2] submit compile job ...", flush=True)
    cjob = hub.submit_compile_job(
        model=qmodel,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_context_binary --truncate_64bit_io --quantize_io",
        name="wav2vec2-large-vi-compile",
    )
    print(f"[w2v2] compile job: {cjob.job_id} {cjob.url}", flush=True)
    cst = poll(cjob, "w2v2-compile")
    if cst.code != "SUCCESS":
        print(f"[w2v2] compile failed: {cst.message}", flush=True)
        return
    target_model = cjob.get_target_model()
    print(f"[w2v2] compiled target model: {target_model.model_id}", flush=True)

    print("[w2v2] submit inference job ...", flush=True)
    ijob = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"input_values": [sample_wav[None, :].astype(np.float32)]},
        name="wav2vec2-large-vi-infer",
    )
    print(f"[w2v2] inference job: {ijob.job_id} {ijob.url}", flush=True)
    ist = poll(ijob, "w2v2-infer")
    if ist.code != "SUCCESS":
        print(f"[w2v2] inference failed: {ist.message}", flush=True)
        return

    out = ijob.download_output_data()
    logits = np.asarray(out["logits"][0])
    pred_ids = np.argmax(logits, axis=-1)
    hyp = tokenizer.batch_decode(pred_ids)[0].strip()
    wer = jiwer.wer(normalize_text(ref), normalize_text(hyp))

    result = {
        "model": MODEL_ID,
        "device": DEVICE_PRIMARY,
        "reference": ref,
        "local_hypothesis": local_hyp,
        "local_wer": float(local_wer),
        "local_latency_s": float(local_elapsed),
        "hardware_hypothesis": hyp,
        "hardware_wer": float(wer),
        "onnx_path": str(ONNX_PATH),
        "quant_job_id": qjob.job_id,
        "compile_job_id": cjob.job_id,
        "inference_job_id": ijob.job_id,
    }
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
