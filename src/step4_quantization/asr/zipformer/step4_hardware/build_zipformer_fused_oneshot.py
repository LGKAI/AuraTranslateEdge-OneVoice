"""Build a bounded, one-shot Zipformer RNN-T graph.

The graph contains the existing encoder followed by an unrolled greedy
decoder.  It is intentionally bounded: K output symbols and S encoder frames
are fixed at compile time.  This is the deployable shape for a runtime that
cannot execute an ONNX Loop.  The graph emits token ids; SentencePiece stays
outside the neural graph.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENC = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")
DEC = os.path.join(ROOT, "third_party_zipformer_real", "decoder-epoch-20-avg-10.onnx")
JOIN = os.path.join(ROOT, "third_party_zipformer_real", "joiner-epoch-20-avg-10.onnx")
OUT = os.path.join(ROOT, "outputs", "zipformer-qnn", "zipformer_fused_oneshot.onnx")


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def copy_subgraph(src, prefix, input_map, output_map, nodes, initializers):
    """Copy a small ONNX graph while wiring its public ports to our graph."""
    names = set()
    for n in src.node:
        names.add(n.name)
        names.update(n.input)
        names.update(n.output)
    mapping = {n: prefix + n for n in names if n}
    mapping.update(input_map)
    mapping.update(output_map)

    for init in src.initializer:
        a = numpy_helper.to_array(init)
        new_name = mapping.get(init.name, prefix + init.name)
        initializers.append(numpy_helper.from_array(a, new_name))

    for node in src.node:
        nodes.append(helper.make_node(
            node.op_type,
            [mapping.get(x, x) for x in node.input],
            [mapping.get(x, x) for x in node.output],
            name=prefix + (node.name or node.op_type),
            **{a.name: onnx.helper.get_attribute_value(a) for a in node.attribute},
        ))


def build(k, scan_frames, output_path):
    enc = onnx.load(ENC)
    dec = onnx.load(DEC)
    join = onnx.load(JOIN)

    nodes = list(enc.graph.node)
    initializers = list(enc.graph.initializer)
    inputs = list(enc.graph.input)

    # Use the encoder's fixed compile shape for the scan.  At inference the
    # encoder length output masks padded frames, so no dynamic slicing is used.
    init = lambda name, arr: initializers.append(
        numpy_helper.from_array(np.asarray(arr), name)
    )
    init("fused_scan_frames", np.arange(scan_frames, dtype=np.int32))
    init("fused_frame_weights", (scan_frames - np.arange(scan_frames)).astype(np.float32))
    init("fused_shape_t512", np.array([scan_frames, 512], dtype=np.int64))
    init("fused_one_i32", np.array(1, dtype=np.int32))
    init("fused_one_bool", np.array(True, dtype=np.bool_))
    init("fused_zero_i32", np.array(0, dtype=np.int32))
    init("fused_blank_i64", np.array(0, dtype=np.int64))
    init("fused_initial_context", np.array([[-1, -1]], dtype=np.int64))
    init("fused_axes_01", np.array([0, 1], dtype=np.int64))
    init("fused_axis_0", np.array([0], dtype=np.int64))
    init("fused_ctx_slice_start", np.array([1], dtype=np.int64))
    init("fused_ctx_slice_end", np.array([3], dtype=np.int64))
    init("fused_ctx_slice_axis", np.array([1], dtype=np.int64))

    # [1,S,512] -> [S,512]. The final frames are masked by encoder_out_lens.
    nodes += [
        helper.make_node("Reshape", ["encoder_out", "fused_shape_t512"], ["fused_enc2d"], name="fused_reshape_enc"),
        helper.make_node("Constant", [], ["fused_t_limit"], name="fused_t_limit", value=numpy_helper.from_array(np.array(scan_frames, dtype=np.int32))),
        helper.make_node("Cast", ["encoder_out_lens"], ["fused_valid_len"], name="fused_cast_len", to=TensorProto.INT32),
    ]

    ctx = "fused_initial_context"
    frame = "fused_zero_i32"
    active = "fused_one_bool"
    token_outputs = []

    for i in range(k):
        p = f"fused_d{i}_"
        dec_out = f"fused_dec_out_{i}"
        copy_subgraph(dec.graph, p, {"y": ctx}, {"decoder_out": dec_out}, nodes, initializers)

        # Broadcast decoder state over all encoder frames and run joiner once.
        dec_expand = f"fused_dec_expand_{i}"
        logits = f"fused_logits_{i}"
        raw_tokens = f"fused_raw_tokens_{i}"
        valid_mask = f"fused_valid_mask_{i}"
        score = f"fused_score_{i}"
        idx64 = f"fused_idx64_{i}"
        idx = f"fused_idx_{i}"
        tok64 = f"fused_tok64_{i}"
        tok = f"fused_tok_{i}"
        tok_col = f"fused_tok_col_{i}"
        found = f"fused_found_{i}"
        chosen = f"fused_chosen_{i}"
        chosen_out = f"fused_chosen_out_{i}"
        chosen_col = f"fused_chosen_col_{i}"
        next_frame = f"fused_next_frame_{i}"
        next_active = f"fused_next_active_{i}"
        next_ctx = f"fused_ctx_{i}"
        nodes += [
            helper.make_node("Expand", [dec_out, "fused_shape_t512"], [dec_expand], name=f"fused_expand_{i}"),
        ]
        copy_subgraph(join.graph, f"fused_j{i}_", {"encoder_out": "fused_enc2d", "decoder_out": dec_expand}, {"logit": logits}, nodes, initializers)
        nodes += [
            helper.make_node("ArgMax", [logits], [raw_tokens], name=f"fused_argmax_{i}", axis=1, keepdims=0),
            helper.make_node("GreaterOrEqual", ["fused_scan_frames", frame], [f"fused_after_{i}" ]),
            helper.make_node("Less", ["fused_scan_frames", "fused_valid_len"], [f"fused_before_{i}" ]),
            helper.make_node("And", [f"fused_after_{i}", f"fused_before_{i}"], [valid_mask]),
            helper.make_node("Equal", [raw_tokens, "fused_blank_i64"], [f"fused_blank_cmp_{i}" ]),
            helper.make_node("Not", [f"fused_blank_cmp_{i}"], [f"fused_not_blank_{i}" ]),
            helper.make_node("And", [valid_mask, f"fused_not_blank_{i}"], [f"fused_candidate_{i}" ]),
            helper.make_node("Cast", [f"fused_candidate_{i}"], [f"fused_candidate_f_{i}"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [f"fused_candidate_f_{i}", "fused_frame_weights"], [score], name=f"fused_score_mul_{i}"),
            helper.make_node("ArgMax", [score], [idx64], name=f"fused_pick_frame_{i}", axis=0, keepdims=0),
            helper.make_node("Cast", [idx64], [idx], to=TensorProto.INT32),
            helper.make_node("Gather", [raw_tokens, idx64], [tok64], axis=0),
            helper.make_node("Cast", [tok64], [tok], to=TensorProto.INT32),
            helper.make_node("Gather", [f"fused_candidate_{i}", idx64], [found], axis=0),
            helper.make_node("Where", [found, tok64, "fused_blank_i64"], [chosen]),
            helper.make_node("Unsqueeze", [chosen, "fused_axes_01"], [tok_col]),
            helper.make_node("Cast", [chosen], [chosen_out], to=TensorProto.INT32),
            helper.make_node("Unsqueeze", [chosen_out, "fused_axis_0"], [chosen_col]),
            helper.make_node("Add", [idx, "fused_one_i32"], [f"fused_idx_plus_{i}" ]),
            helper.make_node("Where", [found, f"fused_idx_plus_{i}", "fused_t_limit"], [next_frame]),
            helper.make_node("And", [active, found], [next_active]),
            helper.make_node("Concat", [ctx, tok_col], [f"fused_ctx_cat_{i}"], axis=1),
            helper.make_node("Slice", [f"fused_ctx_cat_{i}", "fused_ctx_slice_start", "fused_ctx_slice_end", "fused_ctx_slice_axis"], [next_ctx]),
        ]
        token_outputs.append(chosen_col)
        ctx, frame, active = next_ctx, next_frame, next_active

    # Each token is scalar int32. Stack into [K] for a stable output.
    nodes.append(helper.make_node("Concat", token_outputs, ["token_ids"], name="fused_token_stack", axis=0))
    graph = helper.make_graph(
        nodes,
        "zipformer_fused_oneshot",
        inputs,
        [vi("token_ids", TensorProto.INT32, [k])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 9)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"wrote {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB), K={k}, S={scan_frames}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--scan-frames", type=int, default=375)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    build(args.max_tokens, args.scan_frames, args.output)
