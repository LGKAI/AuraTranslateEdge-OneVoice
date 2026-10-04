"""Step 4 -- submit_quantize_job (the proper AIMET path, unlike submit_compile_job
which accepts input_specs to pin shapes at compile time) requires the ONNX
model itself to already have STATIC input shapes -- confirmed via a real
error: "Model input 'input_ids' has dynamic shapes. Please use a static shape."
Pins input_ids to (1, FIXED_SEQ_LEN), matching what was already being done
via input_specs for the old submit_compile_job path.
"""
import os

import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask.onnx")
OUT_PATH = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask_static.onnx")
FIXED_SEQ_LEN = 64


def main():
    print(f"[fix_static] loading {IN_PATH}")
    model = onnx.load(IN_PATH)

    new_inputs = []
    for i in model.graph.input:
        if i.name == "input_ids":
            new_inp = helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, [1, FIXED_SEQ_LEN])
            new_inputs.append(new_inp)
            print(f"[fix_static] pinned input_ids to [1, {FIXED_SEQ_LEN}]")
        else:
            new_inputs.append(i)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    # stale value_info from the old dynamic-shape graph can conflict with
    # the new static shapes during re-inference -- clear and let onnx
    # re-derive it (same pattern as the Piper Stage2 value_info fix)
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model)

    onnx.checker.check_model(model)
    print("[fix_static] onnx.checker passed")

    onnx.save(model, OUT_PATH)
    print(f"[fix_static] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")

    print("[fix_static] verifying numerical equivalence vs the dynamic-shape original ...")
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 25000, size=(1, FIXED_SEQ_LEN)).astype(np.int64)
    bias = np.zeros((1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(IN_PATH, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]

    new = ort.InferenceSession(OUT_PATH, so, providers=["CPUExecutionProvider"])
    new_out = new.run(None, {"input_ids": ids, "/Where_1_output_0": bias})[0]

    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64) - np.asarray(new_out, dtype=np.float64)).max())
    print(f"[fix_static] max_abs_diff={diff:.3e}")
    if diff == 0.0:
        print("[fix_static] OK -- bit-exact, safe to use for submit_quantize_job")
    else:
        print("[fix_static] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
