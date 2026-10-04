"""Step 4 -- 2-stage compile split for Piper (VITS), Stage 2 of 2:
text encoder + main flow + HiFi-GAN vocoder, made QNN-compilable via
max-length padding, WITHOUT retraining and WITHOUT touching weights.

Recap of the two blockers found for this half of the graph (step4.md
Part B SS4d): (1) `/RandomNormalLike` (z_p noise before the main flow) --
QNN has no translation for that op; (2) the final waveform length is
data-dependent (driven by the stochastic duration predictor's output),
which conflicts with QNN's static-shape compilation.

Both turned out to trace back to the SAME root tensor. Node-ancestry trace
(not onnx.shape_inference, which reports "unknown" even where the value is
fully determined by the graph structure):

    /ReduceMax (keepdims=0) over /Cast_output_0
      <- /Clip_output_0 <- /ReduceSum_output_0 <- /Ceil_output_0 <- ...
         <- /dp/Split_output_0 (Stage 1's duration-predictor output)

/ReduceMax has exactly ONE consumer: /Range (build 0..total_frames-1).
That Range feeds a /Less comparison against /Unsqueeze_1_output_0 -- which
traces back through /Cast_output_0 too, but as a SEPARATE branch (not
through /ReduceMax). In other words: /ReduceMax only controls the
ALLOCATED SIZE of the frame-validity mask; the mask's CONTENT (which
positions are real vs. padding) is computed independently from the same
real duration value. That mask (via /Cast_2_output_0) is then
multiplied elementwise into every /flow/flows.*/ layer and the vocoder --
never used to dynamically reshape anything downstream. And the z_p noise
shape (/Transpose_3_output_0, source of /RandomNormalLike) is itself
downstream of the SAME Range/mask construction (the attention-expansion
MatMul that turns per-phoneme encoder stats into per-frame stats).

Net effect: pinning /ReduceMax's output to a fixed MAX_FRAMES constant
cascades through the graph's own existing shape-propagation and fixes
every downstream SHAPE at once, including the RandomNormalLike shape.
That part worked. But 3 real compile attempts after that (v2, v3, v4 --
see outputs/profile_piper_stage2_qcs6490_v{2,3,4}.log) kept failing one
hop further upstream each time: "/Clip_output_0 ... floating-point type",
then "/ReduceSum_output_0 ... floating-point type", even though each
target tensor was immediately Cast to int32 right after being produced.
Root cause, confirmed by checking real consumer counts: QNN's quantizer
(under --quantize_full_type int8) inspects and tries to quantize the
output of EVERY node in the graph, independent of what a later Cast does
to it -- a node that is inherently float32 at its point of creation
(/Ceil, /ReduceSum, /Clip -- all float because /Ceil requires float I/O
per the ONNX spec) gets flagged regardless of how quickly it's cast away.
Patching one tensor at a time just relocated the same failure one hop
upstream each round.

Fix: stop patching hop-by-hop and remove the whole branch instead. Traced
the FULL node chain from /Cast_2_output_0 (the actual float mask consumed
by 30 Mul ops throughout /flow/flows.*/) back to /ReduceSum -- every node
in that chain (/Cast_2, /Unsqueeze_2, /Less, /Unsqueeze, /Unsqueeze_1,
/Range, /Clip, /ReduceSum) has EXACTLY ONE consumer, i.e. it's a clean,
self-contained side-branch off /Ceil (whose only OTHER consumer, /CumSum,
is the real content-alignment computation and is left completely
untouched). Since the host already computes the real total-duration value
from Stage 1's output anyway (to truncate the waveform afterward), it can
just as easily compute this validity MASK too -- so the whole branch is
deleted and /Cast_2_output_0 becomes a new graph input, computed by the
host as `(arange(MAX_FRAMES) < real_total_frames)`. This also sidesteps
the quantizer issue entirely: there's no leftover float32 node output in
the graph for it to choke on.
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

STAGE1_OUTPUT = "/dp/Split_output_0"   # Stage 1's log-duration output == Stage 2's extra input
Z_NOISE_TENSOR = "/RandomNormalLike_output_0"
HOP_LENGTH = 256   # VITS/Piper default; sample_rate=22050 per vi_VN-vais1000-medium.onnx.json


def extract_stage2(src_path, out_path):
    onnx.utils.extract_model(
        src_path, out_path,
        input_names=["input", "input_lengths", "scales", STAGE1_OUTPUT],
        output_names=["output"],
    )


MASK_TENSOR = "/Cast_2_output_0"
# /Cast_output_0 (from /Cast) has TWO consumers: /ReduceMax (buffer-size
# branch) and /Unsqueeze_1 (the actual per-position comparison) -- both
# must go, not just the one that happens to feed /Range.
MASK_BRANCH_NODES = ("/Cast_2", "/Unsqueeze_2", "/Less", "/Unsqueeze",
                      "/Unsqueeze_1", "/Range", "/ReduceMax",
                      "/Clip", "/Cast", "/ReduceSum")


def replace_mask_branch_with_host_input(model, max_frames):
    """Delete the entire /ReduceSum -> ... -> /Cast_2 frame-validity-mask
    branch (8 nodes, each verified to have exactly 1 consumer -- see the
    module docstring) and replace its output tensor with a graph INPUT of
    the same name, so all 30 existing Mul consumers throughout
    /flow/flows.*/ rewire for free. /Ceil itself (and its OTHER consumer,
    /CumSum -- the real per-position content-alignment computation) is
    left completely untouched.

    The host computes this mask the same way sample_noisy_latent-style
    code already needs to for truncation: real_total_frames from Stage 1's
    duration output, then `arange(max_frames) < real_total_frames`."""
    nodes = list(model.graph.node)
    by_name = {n.name: n for n in nodes}

    for name in MASK_BRANCH_NODES:
        node = by_name[name]
        model.graph.node.remove(node)

    inp = helper.make_tensor_value_info(MASK_TENSOR, TensorProto.FLOAT, ["batch_size", 1, max_frames])
    model.graph.input.append(inp)


def hoist_z_noise(model, max_frames):
    nodes = list(model.graph.node)
    target = next(n for n in nodes if Z_NOISE_TENSOR in n.output)
    assert target.op_type == "RandomNormalLike", target.op_type
    model.graph.node.remove(target)

    inp = helper.make_tensor_value_info(Z_NOISE_TENSOR, TensorProto.FLOAT, ["batch_size", 192, max_frames])
    model.graph.input.append(inp)


def dedupe_value_info(model):
    """ONNX IR forbids a tensor name appearing in both value_info and
    model IO (input/output) -- extract_model carries over value_info
    entries from the original graph that now collide with our new
    input/output boundary (AI Hub's compiler rejects this at the
    ONNXRUNTIME/QNN converter stage, not with a generic onnx.checker
    error, so this only surfaces on a real compile attempt).

    Also drops EVERY OTHER value_info entry, not just colliding ones:
    retype_mask_branch_to_int32() changes /Cast's output dtype in place,
    but the carried-over value_info for /Cast_output_0 (and its
    downstream Unsqueeze/Range/Less) still declares the OLD int64 type --
    onnxruntime's loader hard-fails on that mismatch ("Type Error: Type
    (tensor(int64)) ... does not match expected type (tensor(int32))").
    value_info is purely advisory (shape/type inference re-derives it at
    load time), so clearing it entirely is safe post-surgery."""
    del model.graph.value_info[:]


def verify(full_model_path, stage2_path, phoneme_len, max_frames, tol=2e-3):
    """Run the FULL original model once to get a real, in-budget sample
    (real total_frames <= max_frames so no truncation loss occurs), capture
    its randomness sources (dp's noise + z_p's noise) and the real total
    frame count, and confirm Stage 2 alone -- fed the REAL captured
    z_p noise padded out to max_frames, the real log-duration from Stage 1,
    and a host-computed validity mask -- reproduces the same waveform in
    the valid region."""
    rng = np.random.default_rng(0)
    phonemes = rng.integers(1, 150, size=(1, phoneme_len)).astype(np.int64)
    input_lengths = np.array([phoneme_len], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    full = onnx.load(full_model_path)
    extra = [STAGE1_OUTPUT, Z_NOISE_TENSOR, "/Cast_output_0"]
    for name in extra:
        full.graph.output.append(onnx.ValueInfoProto(name=name))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    full_sess = ort.InferenceSession(full.SerializeToString(), so, providers=["CPUExecutionProvider"])
    out_names = ["output"] + extra
    full_out = full_sess.run(out_names, {"input": phonemes, "input_lengths": input_lengths, "scales": scales})
    ref_wav, logdur, z_noise, real_total = full_out
    real_total = int(np.asarray(real_total).reshape(()))
    print(f"  real total_frames={real_total} (max_frames budget={max_frames})")
    if real_total > max_frames:
        print(f"  SKIP: this sample needs {real_total} frames > budget {max_frames}, pick a smaller phoneme_len")
        return False

    pad_amt = max_frames - z_noise.shape[2]
    if pad_amt > 0:
        pad_noise = rng.standard_normal((1, 192, pad_amt)).astype(np.float32)
        z_noise_padded = np.concatenate([z_noise, pad_noise], axis=2)
    else:
        z_noise_padded = z_noise[:, :, :max_frames]

    mask = (np.arange(max_frames) < real_total).astype(np.float32).reshape(1, 1, max_frames)

    s2_sess = ort.InferenceSession(stage2_path, so, providers=["CPUExecutionProvider"])
    s2_out = s2_sess.run(["output"], {
        "input": phonemes, "input_lengths": input_lengths, "scales": scales,
        STAGE1_OUTPUT: logdur, Z_NOISE_TENSOR: z_noise_padded, MASK_TENSOR: mask,
    })[0]

    real_samples = real_total * HOP_LENGTH
    ref_trim = np.asarray(ref_wav, dtype=np.float64).reshape(-1)[:real_samples]
    s2_trim = np.asarray(s2_out, dtype=np.float64).reshape(-1)[:real_samples]
    n = min(len(ref_trim), len(s2_trim))
    diff = float(np.abs(ref_trim[:n] - s2_trim[:n]).max())
    print(f"  Stage 2 (truncated to real length, {n} samples) vs full-model waveform: max_abs_diff={diff:.3e}")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoneme-len", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=400,
                     help="fixed compile-time frame budget; real utterances are truncated to their real length after inference")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "stage2_synth_raw.onnx")
    print(f"[stage2] extracting encoder+flow+vocoder subgraph -> {raw_path}")
    extract_stage2(SRC_MODEL, raw_path)

    model = onnx.load(raw_path)
    print(f"[stage2] extracted {len(model.graph.node)} nodes")

    replace_mask_branch_with_host_input(model, args.max_frames)
    print(f"[stage2] removed the ReduceSum..Cast_2 mask branch (8 nodes), "
          f"replaced with host-computed input shape (1,1,{args.max_frames})")
    hoist_z_noise(model, args.max_frames)
    print(f"[stage2] hoisted /RandomNormalLike -> input shape (1,192,{args.max_frames})")

    dedupe_value_info(model)
    print("[stage2] deduped value_info vs model IO (ONNX IR hygiene, AI Hub compile requirement)")

    onnx.checker.check_model(model)
    print("[stage2] onnx.checker passed")

    out_path = os.path.join(OUT_DIR, "stage2_synth.onnx")
    onnx.save(model, out_path)
    print(f"[stage2] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")

    print("[stage2] verifying vs full model (real captured z-noise, padded; truncated compare) ...")
    if verify(SRC_MODEL, out_path, args.phoneme_len, args.max_frames):
        print("[stage2] OK -- Stage 2 output matches the full model within tolerance in the valid region")
    else:
        print("[stage2] DIVERGE or SKIPPED -- see message above")


if __name__ == "__main__":
    main()
