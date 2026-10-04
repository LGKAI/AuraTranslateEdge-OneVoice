"""Step 4 -- NLLB encoder QNN fix, take 2. The first attempt (shrinking
the -3.4e38 mask-fill constant to -30000, see fix_nllb_mask_constant.py)
did NOT fix the real problem -- a second real compile attempt failed on
the SAME tensor name, "/Cast_1_output_0_pre_quant", which turned out to
be upstream of that constant entirely: it's just `Cast(attention_mask,
to=float)`, the very first step of building the extended attention bias.
QNN's quantizer rejects it regardless of what value it eventually feeds.

Fix (same successful pattern as Piper Stage 2, step4.md SS4e): don't
patch the internal float-cast machinery at all -- delete the whole
mask-construction branch and replace its final output with a
host-computed graph input. Confirmed via a real forward pass: that final
tensor (/Where_1_output_0) is a standard (1,1,seq_len,seq_len) additive
attention bias with exactly two values, {0.0, -3.4e38} (attend / block),
constant across the query dimension (plain self-attention key-padding
mask, not causal) -- trivial to compute on the host directly from the
raw attention_mask input: `(1 - mask)[:,None,None,:] * NEG_BIAS`,
broadcast over the query axis. Uses the same clamped -30000 bias (verified
bit-exact vs -3.4e38 for real padding patterns) so the value itself
was already right; the fix is HOW it reaches the graph, not WHAT it is.

`attention_mask` remains a real graph input -- it still feeds /Shape_1 and
/Shape_2 (shape-only queries, harmless, unrelated to this fix) elsewhere
in the graph, confirmed via consumer audit before deleting anything.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_MODEL = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model.onnx")
OUT_MODEL = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_hostmask.onnx")

MASK_OUTPUT = "/Where_1_output_0"
NEG_BIAS = -30000.0


def find_safe_to_delete_ancestors(model, target_output, stop_at=("attention_mask", "input_ids")):
    nodes = list(model.graph.node)
    by_output = {}
    for n in nodes:
        for o in n.output:
            by_output[o] = n

    # all (output_tensor -> node) pairs reachable backward from target
    ancestors = set()
    frontier = [target_output]
    seen_tensors = set()
    while frontier:
        t = frontier.pop()
        if t in seen_tensors or t in stop_at:
            continue
        seen_tensors.add(t)
        n = by_output.get(t)
        if n is None:
            continue
        ancestors.add(n.name)
        frontier.extend(n.input)

    # a node is safe to delete only if EVERY output it produces is consumed
    # exclusively by other nodes inside the ancestor set (never by anything
    # outside it, and never a graph output itself) -- EXCEPT the node that
    # directly produces target_output itself: it's always deleted (its
    # "outside consumers" are exactly what will bind to the new input we're
    # about to add with the same tensor name, so having outside consumers
    # is expected and fine specifically for this one node).
    target_producer = by_output.get(target_output)
    graph_outputs = {o.name for o in model.graph.output}
    safe = set()
    for name in ancestors:
        node = next(n for n in nodes if n.name == name)
        if node is target_producer:
            safe.add(name)
            continue
        if any(o in graph_outputs for o in node.output):
            continue
        outside_consumer = False
        for o in node.output:
            for other in nodes:
                if other.name in ancestors:
                    continue
                if o in other.input:
                    outside_consumer = True
                    break
            if outside_consumer:
                break
        if not outside_consumer:
            safe.add(name)
    return safe


def apply_fix(model, seq_len):
    safe = find_safe_to_delete_ancestors(model, MASK_OUTPUT)
    print(f"[fix_nllb_hostmask] {len(safe)} nodes safe to delete (exclusive ancestors of {MASK_OUTPUT})")

    kept = [n for n in model.graph.node if n.name not in safe]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    inp = helper.make_tensor_value_info(MASK_OUTPUT, TensorProto.FLOAT, [1, 1, seq_len, seq_len])
    model.graph.input.append(inp)

    del model.graph.value_info[:]

    # attention_mask is now an orphaned input -- its only consumers
    # (/Shape_1, /Shape_2, /Unsqueeze_1) were all exclusively inside the
    # deleted mask-construction branch (confirmed via consumer audit: zero
    # remaining consumers). A real compile attempt caught this the hard
    # way: "[QNN_CPU] Expected number of inputs for Graph is 2 instead 3
    # provided" -- the converter already drops unused inputs, but
    # input_specs/calibration still declared 3, causing a mismatch. Remove
    # the now-dead declared input to match.
    remaining_consumers = any("attention_mask" in n.input for n in model.graph.node)
    if not remaining_consumers:
        keep_inputs = [i for i in model.graph.input if i.name != "attention_mask"]
        del model.graph.input[:]
        model.graph.input.extend(keep_inputs)
        print("[fix_nllb_hostmask] removed now-orphaned 'attention_mask' input (zero remaining consumers)")

    return len(safe)


def host_mask(attention_mask, neg_bias=NEG_BIAS):
    inv = (1 - attention_mask).astype(np.float32)  # 1 where padding
    bias = inv * neg_bias                          # 0 where valid, -30000 where padding
    return np.broadcast_to(bias[:, None, None, :], (bias.shape[0], 1, bias.shape[1], bias.shape[1])).copy()


def verify(orig_path, new_path, seq_len, tol=1e-2):
    rng = np.random.default_rng(0)
    input_ids = rng.integers(3, 1000, size=(1, seq_len)).astype(np.int64)
    attention_mask = np.zeros((1, seq_len), dtype=np.int64)
    attention_mask[:, :30] = 1

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    mask_bias = host_mask(attention_mask)
    new = ort.InferenceSession(new_path, so, providers=["CPUExecutionProvider"])
    # attention_mask itself is NOT fed here -- it's a dropped/orphaned
    # input in the fixed graph (see apply_fix), only its derived bias is
    new_out = new.run(None, {"input_ids": input_ids, MASK_OUTPUT: mask_bias})[0]

    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64)
                         - np.asarray(new_out, dtype=np.float64)).max())
    print(f"  max_abs_diff={diff:.3e}  (output range ~[{ref_out.min():.2f}, {ref_out.max():.2f}])")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    print(f"[fix_nllb_hostmask] loading {SRC_MODEL} ...")
    model = onnx.load(SRC_MODEL, load_external_data=True)
    print(f"[fix_nllb_hostmask] {len(model.graph.node)} nodes before fix")

    n = apply_fix(model, args.seq_len)
    print(f"[fix_nllb_hostmask] {len(model.graph.node)} nodes after fix (removed {n})")

    onnx.checker.check_model(model, full_check=False)
    print("[fix_nllb_hostmask] onnx.checker passed")

    onnx.save(model, OUT_MODEL)
    print(f"[fix_nllb_hostmask] wrote {OUT_MODEL} ({os.path.getsize(OUT_MODEL)/1e6:.1f} MB)")

    print("[fix_nllb_hostmask] verifying vs original (real padding pattern) ...")
    if verify(SRC_MODEL, OUT_MODEL, args.seq_len):
        print("[fix_nllb_hostmask] OK -- safe to deploy")
    else:
        print("[fix_nllb_hostmask] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
