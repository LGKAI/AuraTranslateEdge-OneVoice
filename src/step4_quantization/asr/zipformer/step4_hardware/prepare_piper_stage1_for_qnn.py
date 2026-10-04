"""Step 4 -- 2-stage compile split for Piper (VITS), Stage 1 of 2:
extract the deterministic-shape duration predictor into its own
QNN-compilable ONNX graph, WITHOUT retraining and WITHOUT touching weights.

Why a split at all: Piper's full graph has two internal RandomNormalLike
nodes, and QNN has no translation for that op (confirmed via
outputs/profile_piper_qcs6490.log). Naively deleting both nodes and
replacing them with graph inputs of a fixed shape is only valid if that
shape is actually knowable at compile time. Traced both by hand
(node ancestry, not onnx.shape_inference -- that tool reports "unknown"
even for shapes that ARE static, since it doesn't const-propagate through
Shape/Gather/Concat chains) and confirmed empirically by running the full
graph across different phoneme content and different `scales` values:

  - /dp/RandomNormalLike (noise for the stochastic duration predictor):
    shape is always (1, 2, phoneme_len) -- a function of the INPUT phoneme
    count only. Confirmed identical across 3 random seeds x 2 `scales`
    settings x 2 phoneme lengths. Safe to hoist to a host-generated input.
  - /RandomNormalLike (z_p noise before the normalizing flow): shape's
    last dim varies with phoneme content AND with `scales` (e.g. (1,192,22)
    vs (1,192,26) for the SAME phoneme_len=20) -- traced its ancestry
    through /Sub <- /Reshape_1, /Slice_1, which sit downstream of the
    duration-driven attention/length-regulator expansion. This one is
    genuinely data-dependent on the predicted (stochastic) total frame
    count, confirming step4.md Part B §4b's conclusion with a harder
    empirical check instead of just trusting shape_inference's "unknown".

So only the FIRST RandomNormalLike can be hoisted cleanly with a static
shape. That's exactly the duration predictor's own noise input, and the
duration predictor (`dp` submodule) has exactly one leaf output tensor
that leaves the `/dp/` node prefix: /dp/Split_output_0 (consumed by the
top-level /Exp node that turns log-duration into duration). That makes
`dp` a clean, self-contained Stage 1: (phonemes, input_lengths, scales) in,
log-duration out, no data-dependent shape anywhere inside it.

Stage 2 (text encoder + flow + vocoder, everything after /Exp) still needs
the second RandomNormalLike AND has a genuinely data-dependent output
length -- that one is NOT solved by this script. See
prepare_piper_stage2_for_qnn.py (max-length padding design, not yet
implemented) and step4.md Part B for the full picture.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_MODEL = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
OUT_DIR = os.path.join(ROOT, "outputs", "piper-qnn")

STAGE1_OUTPUT = "/dp/Split_output_0"
DP_NOISE_TENSOR = "/dp/RandomNormalLike_output_0"


def extract_stage1(src_path, out_path):
    onnx.utils.extract_model(
        src_path, out_path,
        input_names=["input", "input_lengths", "scales"],
        output_names=[STAGE1_OUTPUT],
    )


def hoist_noise_to_input(model, phoneme_len):
    """Delete /dp/RandomNormalLike, add a graph input with the same output
    tensor name so every existing consumer is rewired for free."""
    nodes = list(model.graph.node)
    target = next(n for n in nodes if DP_NOISE_TENSOR in n.output)
    assert target.op_type == "RandomNormalLike", target.op_type
    model.graph.node.remove(target)

    inp = helper.make_tensor_value_info(
        DP_NOISE_TENSOR, TensorProto.FLOAT, [1, 2, phoneme_len])
    model.graph.input.append(inp)


def dedupe_value_info(model):
    """ONNX IR forbids a tensor name in both value_info and model IO --
    extract_model carries over value_info entries that now collide with
    our new input/output boundary (AI Hub's compiler rejects this at
    compile time, not via onnx.checker)."""
    io_names = {t.name for t in model.graph.input} | {t.name for t in model.graph.output}
    keep = [vi for vi in model.graph.value_info if vi.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)


def verify(full_model_path, stage1_path, phoneme_len, tol=1e-4):
    """Feed Stage 1 the EXACT noise the full model's RandomNormalLike drew
    (captured by exposing it as an extra output of the full graph), and
    confirm Stage 1 alone reproduces the same log-duration output."""
    rng = np.random.default_rng(0)
    phonemes = rng.integers(1, 150, size=(1, phoneme_len)).astype(np.int64)
    input_lengths = np.array([phoneme_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    full = onnx.load(full_model_path)
    full.graph.output.append(onnx.ValueInfoProto(name=STAGE1_OUTPUT))
    full.graph.output.append(onnx.ValueInfoProto(name=DP_NOISE_TENSOR))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    full_sess = ort.InferenceSession(full.SerializeToString(), so, providers=["CPUExecutionProvider"])
    full_out = full_sess.run(
        [STAGE1_OUTPUT, DP_NOISE_TENSOR],
        {"input": phonemes, "input_lengths": input_lengths, "scales": scales},
    )
    ref_logdur, captured_noise = full_out

    s1_sess = ort.InferenceSession(stage1_path, so, providers=["CPUExecutionProvider"])
    s1_out = s1_sess.run(
        [STAGE1_OUTPUT],
        {
            "input": phonemes,
            "input_lengths": input_lengths,
            "scales": scales,
            DP_NOISE_TENSOR: captured_noise,
        },
    )[0]

    diff = float(np.abs(np.asarray(ref_logdur, dtype=np.float64)
                         - np.asarray(s1_out, dtype=np.float64)).max())
    print(f"  Stage 1 vs full-model log-duration: max_abs_diff={diff:.3e}")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-len", type=int, default=40,
                     help="fixed phoneme count for the AI Hub input_specs / verification")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "stage1_duration_raw.onnx")
    print(f"[stage1] extracting duration-predictor subgraph -> {raw_path}")
    extract_stage1(SRC_MODEL, raw_path)

    model = onnx.load(raw_path)
    print(f"[stage1] extracted {len(model.graph.node)} nodes")
    hoist_noise_to_input(model, args.phoneme_len)
    dedupe_value_info(model)
    onnx.checker.check_model(model)
    print("[stage1] onnx.checker passed after noise hoist")

    out_path = os.path.join(OUT_DIR, "stage1_duration.onnx")
    onnx.save(model, out_path)
    print(f"[stage1] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")

    print("[stage1] verifying vs full model (exact captured noise fed back in) ...")
    if verify(SRC_MODEL, out_path, args.phoneme_len):
        print("[stage1] OK -- Stage 1 output matches the full model within tolerance")
    else:
        print("[stage1] DIVERGE -- do not use this graph")


if __name__ == "__main__":
    main()
