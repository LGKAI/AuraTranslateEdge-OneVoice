"""Try the smaller Vietnamese CTC model that is already cached locally.

This is the practical fallback after the larger no-loop candidate blew up in
ONNX export. It is still a no-loop pipeline:

  waveform -> Wav2Vec2 CTC -> greedy decode

If it compiles on IQ-9075, we stop here.
"""
import json
import os
import time
from pathlib import Path

import jiwer
import numpy as np
import qai_hub as hub
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_wav, normalize_text


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "third_party_viet_ctc"
OUT_DIR = ROOT / "outputs" / "viet_ctc_base"
ONNX_PATH = OUT_DIR / "viet_ctc_base.onnx"
RESULT_JSON = OUT_DIR / "viet_ctc_base_iq9075.json"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
SAMPLE_PATH = ROOT / "data" / "asr" / "vi_1.wav"
SR = 16000
FIXED_SAMPLES = 115200


def poll(job, label, sleep_s=20):
    while True:
        st = job.get_status()
        msg = st.message or ""
        print(f"[{label}] {job.job_id}: {st.code} {msg[:180]}", flush=True)
        if st.code in {"SUCCESS", "FAILED", "CANCELLED"}:
            return st
        time.sleep(sleep_s)


def pad_audio(wav):
    if len(wav) >= FIXED_SAMPLES:
        return wav[:FIXED_SAMPLES].astype(np.float32)
    out = np.zeros((FIXED_SAMPLES,), dtype=np.float32)
    out[: len(wav)] = wav
    return out


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


def local_decode(model, feature_extractor, tokenizer, wav):
    inputs = feature_extractor(wav, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(pred_ids)[0].strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[viet-ctc] loading from {MODEL_DIR}", flush=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils
    import transformers.models.wav2vec2.modeling_wav2vec2 as wav2vec2_mod

    def _allow_torch_load():
        return None

    modeling_utils.check_torch_load_is_safe = _allow_torch_load
    import_utils.check_torch_load_is_safe = _allow_torch_load
    wav2vec2_mod.create_bidirectional_mask = lambda **kwargs: None

    model = AutoModelForCTC.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    for layer in model.wav2vec2.feature_extractor.conv_layers:
        conv = layer.conv
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(torch.zeros(conv.out_channels, dtype=conv.weight.dtype))
    model.config.conv_bias = True

    with open(ROOT / "data" / "asr" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    sample_item = next(r for r in manifest if r["path"] == "data/asr/vi/vi_1.wav")
    wav = pad_audio(load_wav(str(ROOT / sample_item["path"]), sr=SR))

    t0 = time.perf_counter()
    hyp = local_decode(model, feature_extractor, tokenizer, wav)
    local_elapsed = time.perf_counter() - t0
    ref = sample_item["transcript"]
    local_wer = jiwer.wer(normalize_text(ref), normalize_text(hyp))
    print(f"[viet-ctc] local hyp: {hyp}")
    print(f"[viet-ctc] local WER={local_wer:.4f} latency={local_elapsed:.3f}s", flush=True)

    print(f"[viet-ctc] exporting ONNX -> {ONNX_PATH}", flush=True)
    export_onnx(model, ONNX_PATH)
    import onnx
    onnx.checker.check_model(str(ONNX_PATH))
    print("[viet-ctc] ONNX checker passed", flush=True)

    device = hub.Device(DEVICE_NAME)
    calib_items = [r for r in manifest if r["lang"] == "vi"][:4]
    calib_x = [pad_audio(load_wav(str(ROOT / r["path"]), sr=SR))[None, :].astype(np.float32) for r in calib_items]
    calib_ds = hub.upload_dataset({"input_values": calib_x}, name="viet_ctc_base_calib")

    print("[viet-ctc] submit quantize job ...", flush=True)
    qjob = hub.submit_quantize_job(
        model=str(ONNX_PATH),
        calibration_data=calib_ds,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="viet-ctc-base-w8a16",
    )
    print(f"[viet-ctc] quant job: {qjob.job_id} {qjob.url}", flush=True)
    qst = poll(qjob, "viet-quant")
    if qst.code != "SUCCESS":
        print(f"[viet-ctc] quantization failed: {qst.message}", flush=True)
        return
    qmodel = qjob.get_target_model()

    print("[viet-ctc] submit compile job ...", flush=True)
    cjob = hub.submit_compile_job(
        model=qmodel,
        device=device,
        input_specs={"input_values": ((1, FIXED_SAMPLES), "float32")},
        options="--target_runtime qnn_context_binary --truncate_64bit_io --quantize_io",
        name="viet-ctc-base-compile",
    )
    print(f"[viet-ctc] compile job: {cjob.job_id} {cjob.url}", flush=True)
    cst = poll(cjob, "viet-compile")
    if cst.code != "SUCCESS":
        print(f"[viet-ctc] compile failed: {cst.message}", flush=True)
        return
    target_model = cjob.get_target_model()

    print("[viet-ctc] submit inference job ...", flush=True)
    ijob = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"input_values": [wav[None, :].astype(np.float32)]},
        name="viet-ctc-base-infer",
    )
    print(f"[viet-ctc] inference job: {ijob.job_id} {ijob.url}", flush=True)
    ist = poll(ijob, "viet-infer")
    if ist.code != "SUCCESS":
        print(f"[viet-ctc] inference failed: {ist.message}", flush=True)
        return

    out = ijob.download_output_data()
    logits = np.asarray(out["logits"][0])
    pred_ids = np.argmax(logits, axis=-1)
    hw_hyp = tokenizer.batch_decode(pred_ids)[0].strip()
    hw_wer = jiwer.wer(normalize_text(ref), normalize_text(hw_hyp))

    result = {
        "model": str(MODEL_DIR),
        "device": DEVICE_NAME,
        "reference": ref,
        "local_hypothesis": hyp,
        "local_wer": float(local_wer),
        "local_latency_s": float(local_elapsed),
        "hardware_hypothesis": hw_hyp,
        "hardware_wer": float(hw_wer),
        "quant_job_id": qjob.job_id,
        "compile_job_id": cjob.job_id,
        "inference_job_id": ijob.job_id,
    }
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
