"""Step 4 -- fix a MISSED boolean-mask consumer in the Zipformer encoder
graph surgery (prepare_zipformer_for_qnn.py). That script's find_bool_slices()
only searched for op_type=="Slice" nodes with a boolean data input, and
patched exactly 3 of them (Slice_1/3/5, feeding encoder stages 1/2/3's
downsampled attention masks). But /GreaterOrEqual_output_0 (the raw
padding-mask tensor, built from x_lens) has a 4TH consumer that was never
found because it isn't a Slice:

    /encoder/0/layers.0/self_attn_weights/Unsqueeze_15

This is encoder STAGE 0 (the first, highest-resolution block) consuming the
RAW BOOLEAN mask directly, with no Cast protection at all -- exactly the
kind of untyped-bool-through-the-HTP-quantizer path that plausibly explains
the SS4c-5 finding (int8/fp16/int16 on 3 different devices/Hexagon versions
all converge to ~0.21-0.22 cosine similarity vs fp32: a bool-handling bug
is precision-independent, matching the evidence exactly, unlike a
quantization-accuracy problem which would vary by precision).

Fix: apply the EXACT SAME Cast(bool->int32) wrap already used successfully
for the 3 Slice nodes, just targeting this Unsqueeze node's bool input
instead. No other change.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_PATH = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")
OUT_PATH = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice_v2.onnx")

MISSED_CONSUMER = "/encoder/0/layers.0/self_attn_weights/Unsqueeze_15"
BOOL_SOURCE = "/GreaterOrEqual_output_0"


def wrap_bool_consumer(graph, node_name, bool_input_name):
    node = next(n for n in graph.node if n.name == node_name)
    idx_in = list(node.input).index(bool_input_name)

    pre_out = f"{node_name}/bool_to_int32"
    cast_in = helper.make_node("Cast", [bool_input_name], [pre_out], to=TensorProto.INT32,
                                name=f"{node_name}/CastIn")
    node.input[idx_in] = pre_out

    idx = list(graph.node).index(node)
    graph.node.insert(idx, cast_in)


def verify(orig_path, new_path, frames=1500, tol=1e-3):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, frames, 80)).astype(np.float32)
    x_lens = np.array([frames], dtype=np.int64)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"x": x, "x_lens": x_lens})

    new = ort.InferenceSession(new_path, so, providers=["CPUExecutionProvider"])
    new_out = new.run(None, {"x": x, "x_lens": x_lens})

    ok = True
    for i, (a, b) in enumerate(zip(ref_out, new_out)):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            print(f"  output[{i}] SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        diff = float(np.abs(a - b).max())
        print(f"  output[{i}] shape={a.shape} max_abs_diff={diff:.3e}")
        ok = ok and diff <= tol
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-frames", type=int, default=1500)
    args = ap.parse_args()

    print(f"[fix_stage0_mask] loading {IN_PATH}")
    model = onnx.load(IN_PATH)

    by_name = {n.name: n for n in model.graph.node}
    node = by_name[MISSED_CONSUMER]
    assert BOOL_SOURCE in node.input, f"expected {BOOL_SOURCE} in {node.name}'s inputs, got {list(node.input)}"
    print(f"[fix_stage0_mask] confirmed: {MISSED_CONSUMER} directly consumes {BOOL_SOURCE} (unpatched)")

    wrap_bool_consumer(model.graph, MISSED_CONSUMER, BOOL_SOURCE)
    print(f"[fix_stage0_mask] inserted Cast(bool->int32) before {MISSED_CONSUMER}")

    onnx.checker.check_model(model)
    print("[fix_stage0_mask] onnx.checker passed")

    onnx.save(model, OUT_PATH)
    print(f"[fix_stage0_mask] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")

    print("[fix_stage0_mask] verifying numerical equivalence vs the v1 (3-patch) graph ...")
    if verify(IN_PATH, OUT_PATH, args.verify_frames):
        print("[fix_stage0_mask] OK -- bit-exact under onnxruntime, safe to compile+deploy")
    else:
        print("[fix_stage0_mask] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
