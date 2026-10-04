"""Decisive cheap test: is the ~0.2 cosine-similarity "corruption" actually
a TENSOR LAYOUT mismatch (QNN/HTP internally uses NHWC-style ordering; the
converter might return a permuted-axis array that download_output_data()
just reshapes flat into our assumed NCHW/row-major order) rather than real
numerical corruption? Brute-forces every axis permutation of each hardware
output against the local fp32 reference and reports the best cosine
similarity found. Zero new AI Hub cost -- reuses already-compiled target
models with a single real inference call each (no compile needed).
"""
import os
import json
import itertools

import numpy as np
import onnxruntime as ort
import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_NAME = "Dragonwing RB3 Gen 2 Vision Kit"


def cos_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_permutation(local_arr, hw_arr):
    """local_arr, hw_arr: same shape. Tests the FLAT-BUFFER reinterpretation
    hypothesis: take hw_arr's raw flat data (in its current memory order),
    reshape it into every dimension-permutation of the declared shape, then
    transpose each candidate back to the declared shape and compare. This
    catches the case where the SDK mislabeled which axis is which when it
    reshaped raw device memory into the declared output shape -- NOT just
    permuting an already-correctly-shaped array (which is a no-op for most
    of these tests since a real reshape-order bug wouldn't preserve shape
    under a simple axis transpose for non-square dims)."""
    shape = local_arr.shape
    ndim = len(shape)
    flat = hw_arr.reshape(-1)
    results = []
    for dim_perm in itertools.permutations(range(ndim)):
        candidate_shape = tuple(shape[i] for i in dim_perm)
        reshaped = flat.reshape(candidate_shape)
        # transpose back so axis dim_perm[k] (which now sits at position k)
        # goes back to position dim_perm[k] in the final array
        inverse_perm = np.argsort(dim_perm)
        back = np.transpose(reshaped, inverse_perm)
        cs = cos_sim(local_arr, back)
        results.append((cs, dim_perm))
    results.sort(key=lambda x: -x[0])
    return results


def run_test(name, local_arr, hw_arr):
    print(f"\n=== {name} ===")
    print(f"  shape: {local_arr.shape}")
    identity_cs = cos_sim(local_arr, hw_arr)
    print(f"  identity (no permutation) cos_sim: {identity_cs:.4f}")
    results = best_permutation(local_arr, hw_arr)
    print(f"  tried {len(results)} shape-preserving permutations")
    for cs, perm in results[:5]:
        print(f"    perm={perm} cos_sim={cs:.4f}")
    best_cs, best_perm = results[0]
    if best_cs > 0.95:
        print(f"  *** LAYOUT BUG CONFIRMED: perm={best_perm} gives cos_sim={best_cs:.4f} ***")
    else:
        print(f"  no permutation fixes it (best={best_cs:.4f}) -- not a layout issue for this tensor")
    return best_cs, best_perm


def test_nllb():
    from transformers import AutoTokenizer
    MODEL_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask_static.onnx")
    COMPILE_JOB = "jpv7mzjzp"  # the -100 mask int8 compile (or reuse jpeyxrv15/jg9d23kw5 AIMET one)
    FIXED_SEQ_LEN = 64

    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="vie_Latn")
    with open(os.path.join(ROOT, "data", "mt", "manifest.json"), encoding="utf-8") as f:
        row = json.load(f)[0]
    text = row["vi"]
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True, max_length=FIXED_SEQ_LEN)
    ids = enc["input_ids"].astype(np.int64)
    mask = enc["attention_mask"].astype(np.int64)
    inv = (1 - mask).astype(np.float32)
    bias = np.broadcast_to((inv * -100.0)[:, None, None, :], (1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)).copy()

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    local_out = sess.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]

    device = hub.Device(DEVICE_NAME)
    target_model = hub.get_job(COMPILE_JOB).get_target_model()
    ids_i32 = ids.astype(np.int32)
    print("[nllb] submitting fresh real inference (reusing compiled model, no new compile cost) ...")
    job = hub.submit_inference_job(
        model=target_model, device=device,
        inputs={"input_ids": [ids_i32], "/Where_1_output_0": [bias]}, name="permtest-nllb")
    print(f"[nllb] job: {job.job_id}  {job.url}")
    job.wait()
    if job.get_status().code != "SUCCESS":
        print(f"[nllb] inference FAILED: {job.get_status().message}")
        return
    hw_out = np.array(job.download_output_data()["output_0"][0])
    run_test("NLLB encoder (1,64,1024)", local_out[0], hw_out[0])


if __name__ == "__main__":
    test_nllb()
