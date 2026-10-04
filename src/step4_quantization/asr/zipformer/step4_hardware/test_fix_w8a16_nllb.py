"""Step 4 fix-test (v2): w8a16 via submit_quantize_job with activations=INT16.

submit_compile_job's restricted `options` rejects the local QAIRT converter
flags like --act_bitwidth. But submit_quantize_job exposes EXPLICIT
weights_dtype / activations_dtype via hub.QuantizeDtype. This is the AI Hub
way to set 16-bit activations -- the equivalent of Qualcomm's own
whisper_small_quantized w8a16 recipe.

Pipeline:
  1. submit_quantize_job(int8 weights, INT16 activations, real Vi calib)
  2. submit_compile_job on the quantized model
  3. submit_inference_job -> download output -> cos_sim vs fp32

Expected: cos_sim > 0.95 (int16 activations ~lossless for transformer).
Baseline to beat: hardware int8-only cos_sim = 0.18.
"""
import os
import json
import numpy as np
import onnxruntime as ort
import qai_hub as hub
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask_static_optimized.onnx")
DEVICE_PRIMARY = "Dragonwing RB3 Gen 2 Vision Kit"
DEVICE_FALLBACK = "Dragonwing IQ-9075 EVK"
B = "/encoder/Where_1_output_0"
FIXED = 64
MODEL_ID = "facebook/nllb-200-distilled-600M"


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def host_mask(mask, val=None):
    if val is None:
        val = float(os.environ.get("MASK_VAL", "-30000"))
    inv = (1 - mask).astype(np.float32)
    return np.broadcast_to(inv * val, (1, 1, FIXED, FIXED)).copy()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="vie_Latn")
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        rows = json.load(f)
    # multi-language calibration from all 4 src langs the encoder may see.
    SRC_FLORES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
    calib_texts = []
    for lang in ("vi", "en", "zh", "ko"):
        for r in rows:
            calib_texts.append((r[lang], SRC_FLORES[lang]))
    eval_text = rows[0]["vi"]

    # build calibration entries (several real Vi sentences)
    ids_c, bias_c = [], []
    for text, src in calib_texts:
        tok.src_lang = src
        enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED)
        ids = enc["input_ids"].astype(np.int32)         # int32 for the compiled-model inference path
        mask = enc["attention_mask"].astype(np.int64)
        # quantize_job reads dtype straight off the model (int64) -- feed int64 there.
        ids_c.append(ids.astype(np.int64)); bias_c.append(host_mask(mask))
    print(f"[w8a16] {len(ids_c)} calibration sentences", flush=True)

    # eval single entry
    tok.src_lang = "vie_Latn"
    enc = tok(eval_text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED)
    eval_ids = enc["input_ids"].astype(np.int32)
    eval_mask = enc["attention_mask"].astype(np.int64)
    eval_bias = host_mask(eval_mask)

    # ---- local fp32 reference ----
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    out_fp = sess.run(None, {"input_ids": eval_ids.astype(np.int64), B: eval_bias})[0]
    print(f"[w8a16] local fp32 out shape={out_fp.shape} std={out_fp.std():.4f}", flush=True)

    # ---- Step 1: submit_quantize_job (w8a16) ----
    QMODEL_ID = os.environ.get("QMODEL_REUSE", "")   # set to reuse an already-quantized model
    RECIPE = os.environ.get("RECIPE", "w8a16")       # w8a16 | w16a16 | w8a8
    WDT = hub.QuantizeDtype.INT16 if RECIPE.startswith("w16") else hub.QuantizeDtype.INT8
    ADT = hub.QuantizeDtype.INT16 if "a16" in RECIPE else hub.QuantizeDtype.INT8
    print(f"[w8a16] RECIPE={RECIPE}  weights={WDT} activations={ADT}", flush=True)
    if QMODEL_ID:
        qmodel = hub.get_model(QMODEL_ID)
        print(f"[w8a16] REUSING quantized model {QMODEL_ID}", flush=True)
    else:
        calib_ds = hub.upload_dataset({"input_ids": ids_c, B: bias_c}, name=f"nllb-calib-{RECIPE}")
        print(f"[w8a16] submit_quantize_job weights={WDT} activations={ADT} ...", flush=True)
        qjob = hub.submit_quantize_job(
            model=MODEL_PATH, calibration_data=calib_ds,
            weights_dtype=WDT, activations_dtype=ADT,
            name=f"nllb-{RECIPE}-quant")
        print(f"[w8a16] quant job: {qjob.job_id}  {qjob.url}", flush=True)
        qjob.wait()
        if qjob.get_status().code != "SUCCESS":
            print(f"[w8a16] quantize FAIL: {qjob.get_status().message[:600]}"); return
        qmodel = qjob.get_target_model()
    print(f"[w8a16] quantized model: {qmodel.model_id}", flush=True)

    # ---- Step 2: compile ----
    compile_dev_name = None
    cmodel = None
    for dev_name in [DEVICE_PRIMARY, DEVICE_FALLBACK]:
        device = hub.Device(dev_name)
        print(f"[w8a16] submit_compile_job on {dev_name}...", flush=True)
        cjob = hub.submit_compile_job(
            model=qmodel, device=device,
            input_specs={"input_ids": ((1, FIXED), "int64"), B: ((1, 1, FIXED, FIXED), "float32")},
            options="--target_runtime qnn_context_binary --quantize_io --truncate_64bit_io",
            name=f"nllb-w8a16-compile-{dev_name.split()[1].lower()}")
        print(f"[w8a16] compile job: {cjob.job_id}  {cjob.url}", flush=True)
        cjob.wait()
        st = cjob.get_status()
        if st.code == "SUCCESS":
            print(f"[w8a16] compile SUCCESS on {dev_name}", flush=True)
            cmodel = cjob.get_target_model()
            compile_dev_name = dev_name
            break
        else:
            print(f"[w8a16] compile FAIL on {dev_name}: {st.message[:400]}", flush=True)
    if cmodel is None:
        print("[w8a16] compile failed on both devices."); return

    # ---- Step 3: inference + cos_sim (use the SAME device that compiled = SUCCESS) ----
    device = hub.Device(compile_dev_name)
    print(f"[w8a16] submit_inference_job on {device.name}...", flush=True)
    ijob = hub.submit_inference_job(
        model=cmodel, device=device,
        inputs={"input_ids": [eval_ids], B: [eval_bias]}, name="nllb-w8a16-infer")
    print(f"[w8a16] inference job: {ijob.job_id}", flush=True)
    ijob.wait()
    if ijob.get_status().code != "SUCCESS":
        print(f"[w8a16] inference FAIL: {ijob.get_status().message[:400]}"); return
    hw = np.array(ijob.download_output_data()["output_0"][0])
    print(f"[w8a16] hw out shape={hw.shape} std={hw.std():.4f}", flush=True)
    print(f"[w8a16] (compiled+inferred on {compile_dev_name})", flush=True)

    cs = cos_sim(out_fp, hw)
    print(f"\n[w8a16] === VERDICT ===")
    print(f"[w8a16] cos_sim(fp32, w8a16_hw) = {cs:.4f}   (int8-only baseline = 0.18)")
    if cs > 0.95:
        print("[w8a16] FIX FOUND: w8a16 (int8 w + int16 act) recovers correctness.")
    elif cs > 0.6:
        print("[w8a16] PARTIAL improvement -> need finer mixed precision / per-tensor overrides.")
    else:
        print("[w8a16] NO IMPROVEMENT via w8a16 alone.")


if __name__ == "__main__":
    main()