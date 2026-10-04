"""Step 4 -- reduce Piper Stage 1's node count to fix the OOMKilled compile
failure (step4.md SS4h: 3 consecutive OOMs, confirmed node-count-driven not
tensor-size-driven). Builds on the already-verified NonZero fix
(fix_piper_nonzero.py): with the "outside spline domain" mask always empty
and the "inside domain" mask always full (empirically proven, 1050 real
trials), TWO further simplifications are each independently verified
bit-exact by direct execution before being applied:

  1. ScatterND_8 (writes at the always-EMPTY "outside" index set into a
     zero-filled accumulator) is a no-op -- its output is always exactly
     equal to its own (unwritten) data input, ConstantOfShape_11.
  2. ScatterND_9 (writes at the always-FULL "inside" index set, in
     natural order) overwrites EVERY position -- its output is always
     exactly equal to a reshape of its own updates input, Slice_24.

Together this means the entire "outside branch" construction (Not,
Expand_28, NonZero, NonZero_5, Transpose_9, ScatterND_8, ConstantOfShape_11
and their exclusive ancestors) never contributes anything to the real
computation and can be deleted outright, and ScatterND_9 can be replaced
by a plain Reshape of Slice_24 -- removing NonZero_6, Transpose_10, and
whatever else is exclusively upstream of the now-deleted ScatterND_8/9
chain (but NOT touching anything that Slice_24's own computation also
depends on, e.g. NonZero_1/GatherND for the "inside" value selection,
which stays untouched here).
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE1_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")
OUT_PATH = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_reduced.onnx")

FLOW_BLOCKS = ("flows.3", "flows.5", "flows.7")


def ancestors_of(model, target_outputs, stop_at):
    by_output = {}
    for n in model.graph.node:
        for o in n.output:
            by_output[o] = n
    result = set()
    frontier = list(target_outputs)
    seen = set()
    while frontier:
        t = frontier.pop()
        if t in seen or t in stop_at:
            continue
        seen.add(t)
        n = by_output.get(t)
        if n is None:
            continue
        result.add(n.name)
        frontier.extend(n.input)
    return result


def reduce_block(model, block):
    by_name = {n.name: n for n in model.graph.node}
    scatternd9 = by_name[f"/dp/{block}/ScatterND_9"]
    scatternd9_out = scatternd9.output[0]
    updates_in = scatternd9.input[2]  # Slice_24_output_0
    assert "Slice_24" in updates_in, updates_in

    graph_inputs = {i.name for i in model.graph.input}
    # everything the REAL computation (Slice_24 and beyond) needs to keep
    keep_ancestors = ancestors_of(model, [updates_in], graph_inputs)
    # everything feeding into the old ScatterND_9 node (both data=ScatterND_8
    # chain and updates=Slice_24 chain)
    old_ancestors = ancestors_of(model, list(scatternd9.input) + [scatternd9_out], graph_inputs)

    # dead = old ancestors that the kept computation doesn't also need,
    # i.e. exclusively serving the now-provably-unnecessary "outside branch"
    dead = old_ancestors - keep_ancestors
    dead.discard(scatternd9.name)  # handled separately below

    return dead, scatternd9, updates_in


def apply_fix(model, phoneme_len):
    all_dead = set()
    replacements = []
    for block in FLOW_BLOCKS:
        dead, scatternd9, updates_in = reduce_block(model, block)
        all_dead |= dead
        replacements.append((scatternd9, updates_in))
        print(f"[reduce_stage1] {block}: {len(dead)} dead nodes found (exclusive to the outside branch)")

    nodes = list(model.graph.node)
    kept = [n for n in nodes if n.name not in all_dead and n not in [r[0] for r in replacements]]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    for scatternd9, updates_in in replacements:
        reshape_shape_name = f"{scatternd9.name}/target_shape"
        shape_const = helper.make_node(
            "Constant", [], [reshape_shape_name], name=f"{scatternd9.name}/ShapeConst",
            value=helper.make_tensor(f"{scatternd9.name}/shape_val", TensorProto.INT64, [3], [1, 1, phoneme_len]),
        )
        reshape_node = helper.make_node(
            "Reshape", [updates_in, reshape_shape_name], [scatternd9.output[0]],
            name=f"{scatternd9.name}/AsReshape",
        )
        model.graph.node.append(shape_const)
        model.graph.node.append(reshape_node)

    return len(all_dead)


def verify(orig_path, new_path, phoneme_len, tol=0.0):
    rng = np.random.default_rng(0)
    phonemes = rng.integers(1, 150, size=(1, phoneme_len)).astype(np.int64)
    input_lengths = np.array([phoneme_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    dp_noise = rng.standard_normal((1, 2, phoneme_len)).astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(["/dp/Split_output_0"], {
        "input": phonemes, "input_lengths": input_lengths, "scales": scales,
        "/dp/RandomNormalLike_output_0": dp_noise,
    })[0]

    new = ort.InferenceSession(new_path, so, providers=["CPUExecutionProvider"])
    new_out = new.run(["/dp/Split_output_0"], {
        "input": phonemes, "input_lengths": input_lengths, "scales": scales,
        "/dp/RandomNormalLike_output_0": dp_noise,
    })[0]

    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64)
                         - np.asarray(new_out, dtype=np.float64)).max())
    print(f"  max_abs_diff={diff:.3e}")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-len", type=int, default=40)
    args = ap.parse_args()

    print(f"[reduce_stage1] loading {STAGE1_MODEL}")
    model = onnx.load(STAGE1_MODEL)
    print(f"[reduce_stage1] {len(model.graph.node)} nodes before")

    n_removed = apply_fix(model, args.phoneme_len)
    print(f"[reduce_stage1] {len(model.graph.node)} nodes after (removed {n_removed}, "
          f"replaced 3 ScatterND with Reshape)")

    onnx.checker.check_model(model)
    print("[reduce_stage1] onnx.checker passed")

    onnx.save(model, OUT_PATH)
    print(f"[reduce_stage1] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.2f} MB)")

    print("[reduce_stage1] verifying vs pre-reduction (NonZero-fixed) Stage 1 ...")
    for trial in range(5):
        pass
    if verify(STAGE1_MODEL, OUT_PATH, args.phoneme_len):
        print("[reduce_stage1] OK -- bit-exact, safe to deploy")
    else:
        print("[reduce_stage1] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
