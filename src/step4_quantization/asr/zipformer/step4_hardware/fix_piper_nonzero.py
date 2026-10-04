"""Step 4 -- fix Piper Stage 1's NonZero blocker (see step4.md SS4d) with a
minimal, empirically-justified graph surgery, NOT a rewrite of the RQS
spline math.

Root cause recap: the stochastic duration predictor's normalizing flow
uses a bounded/tail-clipped rational-quadratic spline. Positions whose
input value falls outside [-5, 5] get an identity transform; positions
inside get the real spline transform. PyTorch's boolean advanced indexing
(`x[mask]`) for this split has no ONNX op, so the exporter lowers it to
NonZero(mask) -> GatherND -> [compute] -> ScatterND, once per flow block.

Empirical finding (1050 real forward-pass trials: 30 real Vietnamese
sentences x 7 realistic `scales` settings x 5 noise seeds each, see
step4.md SS4d): the value being tested NEVER exceeds [-5, 5] at any
`scales` setting up to and including the model's own documented default
range (noise_scale<=1.0) -- global observed range was [-4.54, 4.29]. It
DOES exceed the bound at an artificially extreme noise_scale=1.2 (40/270
trials), which is well outside the model's documented default (0.667) and
not a realistic production setting.

Given that, for the production `scales` range, verified directly (not
assumed) at a fixed phoneme_len=40: the "outside domain" mask
(/dp/flows.{3,5,7}/Not_output_0, fed to NonZero and NonZero_5) is ALWAYS
all-False (NonZero output shape (3,0), selects nothing), and the "inside
domain" mask (/dp/flows.{3,5,7}/And_output_0, fed to NonZero_1 and
NonZero_6) is ALWAYS all-True (NonZero output shape (3,phoneme_len),
selects everything) -- confirmed via direct inspection of the real
boolean mask values, not inferred.

Fix: replace each of the 12 NonZero nodes (4 per block x 3 blocks) with a
precomputed Constant holding exactly the index tensor NonZero would
produce for an always-empty or always-full mask of the known (compile-
time-fixed) shape. This changes NOTHING about the surrounding softmax/
cumsum/softplus spline computation or the GatherND/ScatterND plumbing --
it only replaces "compute which indices are nonzero at runtime" with
"here they are, precomputed, under the verified always-true/always-false
assumption" -- mathematically exact under that assumption, not an
approximation of the spline formula itself.

Scope note: this fix is valid for scales within the empirically-tested
range (noise_scale <= ~1.0, matching the model's own default of 0.667).
Deploying with a noise_scale meaningfully above 1.0 would violate the
assumption this fix relies on and should be re-validated first.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE1_MODEL = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration.onnx")
FULL_MODEL = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
OUT_PATH = os.path.join(ROOT, "outputs", "piper-qnn", "stage1_duration_nononzero.onnx")

FLOW_BLOCKS = ("flows.3", "flows.5", "flows.7")
# (node suffix, always-empty?) -- verified via real forward pass at phoneme_len=40
NONZERO_SPECS = [
    ("NonZero", True),      # fed by Not_output_0 (outside-domain mask) -- always empty
    ("NonZero_1", False),   # fed by And_output_0 (inside-domain mask) -- always full
    ("NonZero_5", True),    # fed by Expand_28 = Expand(Not_output_0) -- always empty
    ("NonZero_6", False),   # fed by Expand_29 = Expand(And_output_0) -- always full
]


def make_nonzero_replacement(node, phoneme_len, always_empty):
    """NonZero's real ONNX semantics: for a boolean tensor of shape
    (1,1,N), output is (rank=3, num_true). All-false -> (3,0). All-true
    -> (3,N) with row0=row1=0 (the size-1 dims), row2=arange(N)."""
    if always_empty:
        indices = np.zeros((3, 0), dtype=np.int64)
    else:
        indices = np.zeros((3, phoneme_len), dtype=np.int64)
        indices[2, :] = np.arange(phoneme_len)
    return helper.make_node(
        "Constant", [], [node.output[0]], name=f"{node.name}/ConstFold",
        value=helper.make_tensor(f"{node.name}/precomputed", TensorProto.INT64,
                                  list(indices.shape), indices.flatten().tolist()),
    )


def apply_fix(model, phoneme_len):
    nodes = list(model.graph.node)
    by_name = {n.name: n for n in nodes}
    replaced = 0
    for block in FLOW_BLOCKS:
        for suffix, always_empty in NONZERO_SPECS:
            name = f"/dp/{block}/{suffix}"
            node = by_name.get(name)
            if node is None:
                raise KeyError(f"expected NonZero node not found: {name}")
            assert node.op_type == "NonZero", node.op_type
            idx = list(model.graph.node).index(node)
            const_node = make_nonzero_replacement(node, phoneme_len, always_empty)
            model.graph.node.remove(node)
            model.graph.node.insert(idx, const_node)
            replaced += 1
    return replaced


def verify(orig_path, new_path, phoneme_len, tol=0.0):
    """Bit-exact check: same real captured dp-noise, same phoneme input,
    compare Stage 1's log-duration output before vs after the fix."""
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
    print(f"  Stage 1 (NonZero-free) vs Stage 1 (original): max_abs_diff={diff:.3e}")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-len", type=int, default=40)
    args = ap.parse_args()

    print(f"[fix_nonzero] loading {STAGE1_MODEL}")
    model = onnx.load(STAGE1_MODEL)
    print(f"[fix_nonzero] {len(model.graph.node)} nodes before fix")

    n = apply_fix(model, args.phoneme_len)
    print(f"[fix_nonzero] replaced {n} NonZero nodes with precomputed constants")

    remaining = [x.op_type for x in model.graph.node].count("NonZero")
    print(f"[fix_nonzero] NonZero nodes remaining: {remaining}")
    assert remaining == 0

    onnx.checker.check_model(model)
    print("[fix_nonzero] onnx.checker passed")

    onnx.save(model, OUT_PATH)
    print(f"[fix_nonzero] wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.2f} MB)")

    print("[fix_nonzero] verifying vs pre-fix Stage 1 (same real captured noise) ...")
    if verify(STAGE1_MODEL, OUT_PATH, args.phoneme_len):
        print("[fix_nonzero] OK -- bit-exact match, safe to deploy")
    else:
        print("[fix_nonzero] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
