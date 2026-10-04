"""Step 4 -- ONNX graph surgery to make Zipformer's encoder QNN/NPU-compilable,
WITHOUT retraining and WITHOUT touching any weight.

The blocker (confirmed on real hardware, outputs/profile_zipformer_realcalib2_qcs6490.log):

    Failed to validate op /encoder/Slice_1 with error 0xc26
    Supported I/O datatype sets: BF16 / FP16 / INT16 / INT8 / INT32

Three Slice nodes in the encoder operate on a BOOLEAN tensor (the padding
mask, downsampled at each encoder stage). QNN's HTP backend has no boolean
I/O datatype at all, so these can never validate as-is -- everything else
in the graph compiled fine in earlier attempts (the compiler got well into
OPTIMIZING_MODEL before failing on this specific op).

A first attempt ran the whole graph through onnxsim (constant-fold with a
fixed input shape, in the hope of resolving the boolean Slice away
entirely) -- that DID remove the boolean Slice, but onnxsim has a real bug
on this graph's `/encoder/2/downsample/Reshape_1` node (produces an
inconsistent hard-coded reshape target regardless of frame count -- tried
both 103 and 128 frames, identical failure both times, ruling out an
odd/even parity issue).

This script instead does the MINIMAL surgical fix: wrap each boolean
Slice with Cast(bool->int8) / Cast(int8->bool) so the Slice itself
operates on a QNN-supported dtype. Nothing else in the graph is touched --
x stays dynamic-shaped in the exported graph (AI Hub's own input_specs
mechanism pins it at compile time, exactly as it already did successfully
for every other op in earlier attempts).
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP_ROOT = os.path.join(ROOT, "third_party_zipformer",
                          "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
OUT_DIR = os.path.join(ROOT, "outputs", "zipformer-qnn")
BOOL_SLICE_NODES = ("/encoder/Slice_1", "/encoder/1/encoder/0/self_attn_weights/Slice_1"
                     if False else None)  # discovered dynamically below instead


def source_encoder():
    snap = os.path.join(SNAP_ROOT, os.listdir(SNAP_ROOT)[0])
    return os.path.join(snap, "encoder-epoch-20-avg-10.onnx")


def find_bool_slices(model):
    from onnx import shape_inference
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    types = {vi.name: vi.type.tensor_type.elem_type
             for vi in list(inferred.graph.value_info) + list(inferred.graph.input)}
    return [n.name for n in inferred.graph.node
            if n.op_type == "Slice" and types.get(n.input[0]) == TensorProto.BOOL]


def wrap_bool_slice(graph, node_name):
    """Insert Cast(bool->int32) before the Slice's data input only. The
    Slice's output then flows as int32 (not bool) into its consumers:
    Unsqueeze (dtype-agnostic, shape-only) then an EXISTING Cast node
    already present in the original graph (bool -> float, feeding a
    Where for masking). That existing Cast now sees int32 instead of
    bool as its source -- numerically identical, since 0/1 round-trips
    exactly through int32 the same as through bool.

    int32, not int8: QNN's HTP quantizer tries to assign a quantization
    encoding (scale/zero-point) to every int8 tensor, and fails to do so
    sensibly for a {0,1}-valued mask that was never meant to be quantized
    as if it were a real activation ("OpConfig validation failed for
    Reshape" on the Unsqueeze right after the Slice -- confirmed via
    outputs/profile_zipformer_surgical2_qcs6490.log). QNN's own
    "Supported I/O datatype sets" error listed INT_32 under a separate
    "OTHERS" category alongside the quantized BF16/FP16/INT16/INT8
    buckets -- int32 tensors are treated as plain integers, not put
    through the quantizer at all, which is exactly what a 0/1 mask needs.

    No cast-back-to-bool needed (QNN's quantizer also rejects a Cast
    whose OUTPUT is bool -- confirmed via
    outputs/profile_zipformer_surgical_qcs6490.log)."""
    node = next(n for n in graph.node if n.name == node_name)
    data_in = node.input[0]

    pre_out = f"{node_name}/bool_to_int32"
    cast_in = helper.make_node("Cast", [data_in], [pre_out], to=TensorProto.INT32,
                                name=f"{node_name}/CastIn")
    node.input[0] = pre_out

    idx = list(graph.node).index(node)
    graph.node.insert(idx, cast_in)


def verify(orig_path, new_path, frames, tol=1e-3):
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
    ap.add_argument("--verify-frames", type=int, default=103,
                     help="frame count used only for the numerical sanity check")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    src = source_encoder()
    model = onnx.load(src)
    print(f"[prepare] source: {src}  ({len(model.graph.node)} nodes)")

    targets = find_bool_slices(model)
    print(f"[prepare] boolean Slice nodes found: {targets}")
    for name in targets:
        wrap_bool_slice(model.graph, name)

    remaining = find_bool_slices(model)
    print(f"[prepare] boolean Slice nodes remaining after patch: {remaining}")

    onnx.checker.check_model(model)
    print("[prepare] onnx.checker passed")

    out_path = os.path.join(OUT_DIR, "encoder_no_bool_slice.onnx")
    onnx.save(model, out_path)
    print(f"[prepare] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB), "
          f"{len(model.graph.node)} nodes")

    print("[prepare] verifying numerical equivalence vs the original graph ...")
    if verify(src, out_path, args.verify_frames):
        print("[prepare] outputs match the original within tolerance")
    else:
        print("[prepare] outputs DIVERGE -- do not deploy this graph")


if __name__ == "__main__":
    main()
