# -*- coding: utf-8 -*-
"""Unified build script for Zipformer-150M CTC Single Static Graph.
1. Load encoder-epoch-11-avg-2.onnx
2. Apply surgical Cast(bool->int32) to boolean Slice ops (QNN HTP requirement)
3. Attach pretrained CTC head (ctc_output.1 from pretrained.pt, W: 768x2000, b: 2000)
4. Pin static shapes: x=[1, 1500, 80], x_lens=[1]
5. Compose with Fbank DSP MatMul front-end and Static CTC collapse + Byte Detokenizer
Output: outputs/zip150_full_npu/zip150_full_pipeline.onnx
"""
import os, sys, time, json, shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnx.compose as oc
import sentencepiece as spm

ROOT = r"d:\ChuyenNganhAI\AuraTranslateEdge-OneVoice"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "step4_quantization", "step1_asr", "zipformer", "step4_hardware"))
from prepare_zipformer_for_qnn import find_bool_slices, wrap_bool_slice
from fbank_matmul_verify import WINDOW, MEL_MAT, DFT_RE, DFT_IM, FRAME_LEN, FRAME_SHIFT, PREEMPH

D = os.path.join(ROOT, "third_party", "zipformer", "third_party_zipformer_150m_crctc")
OUT_CTC = os.path.join(ROOT, "outputs", "zip150_ctc")
OUT_FULL = os.path.join(ROOT, "outputs", "zip150_full_npu")
os.makedirs(OUT_CTC, exist_ok=True)
os.makedirs(OUT_FULL, exist_ok=True)

T = 1500
NFFT = 512
N_SAMPLES = (T - 1) * FRAME_SHIFT + FRAME_LEN  # 240240
T_OUT = 373  # Subsampling 4x down from 1500
VOCAB, MAX_LEN, BLANK_ID, MAX_BYTES = 2000, T_OUT, 0, 12
OPSET = 13

print(f"--- Step 1: Building Static Encoder CTC (1500 frames) ---")
bpe_path = os.path.join(D, "bpe.model")
head_path = os.path.join(D, "ctc_head.npz")
enc_raw_path = os.path.join(D, "encoder-epoch-11-avg-2.onnx")

assert os.path.exists(enc_raw_path), f"Missing {enc_raw_path}"
assert os.path.exists(head_path), f"Missing {head_path}"
assert os.path.exists(bpe_path), f"Missing {bpe_path}"

sp = spm.SentencePieceProcessor()
sp.load(bpe_path)

hd = np.load(head_path)
W, b = hd["W"].astype(np.float32), hd["b"].astype(np.float32)  # (2000, 768)

m = onnx.load(enc_raw_path)
print(f"Original encoder nodes: {len(m.graph.node)}")

# 1. Wrap bool slices
bs = find_bool_slices(m)
print(f"Found bool Slices: {bs}")
for n in bs:
    wrap_bool_slice(m.graph, n)
assert not find_bool_slices(m), "Still has bool Slice ops!"

# 2. Cut encoder_proj, attach CTC Head
byout = {o: n for n in m.graph.node for o in n.output}
mm, ad = byout["/encoder_proj/MatMul_output_0"], byout["encoder_out"]
assert mm.op_type == "MatMul" and ad.op_type == "Add"
pre = mm.input[0]
assert pre == "/Transpose_1_output_0", pre
proj_inits = {mm.input[1], "encoder_proj.bias"}
users = {i for n in m.graph.node if n not in (mm, ad) for i in n.input}
assert not (proj_inits & users), "encoder_proj inits reused elsewhere!"
m.graph.node.remove(mm)
m.graph.node.remove(ad)
for i in [x for x in m.graph.initializer if x.name in proj_inits]:
    m.graph.initializer.remove(i)

m.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W.T), "ctc_head.W"))  # (768, 2000)
m.graph.initializer.append(numpy_helper.from_array(b, "ctc_head.b"))
m.graph.node.append(helper.make_node("MatMul", [pre, "ctc_head.W"], ["ctc_head/mm"], name="ctc_head/MatMul"))
m.graph.node.append(helper.make_node("Add", ["ctc_head/mm", "ctc_head.b"], ["ctc_logits"], name="ctc_head/Add"))

del m.graph.output[:]
m.graph.output.append(helper.make_tensor_value_info("ctc_logits", TensorProto.FLOAT, ["N", "T", 2000]))
m.graph.output.append(helper.make_tensor_value_info("encoder_out_lens", TensorProto.INT64, ["N"]))

# 3. Pin static input shapes
for inp in m.graph.input:
    dims = inp.type.tensor_type.shape.dim
    if inp.name == "x":
        for d, v in zip(dims, (1, T, 80)):
            d.ClearField("dim_param")
            d.dim_value = v
    elif inp.name == "x_lens":
        dims[0].ClearField("dim_param")
        dims[0].dim_value = 1

enc_static_path = os.path.join(OUT_CTC, f"zip150_ctc_static_{T}.onnx")
onnx.save(m, enc_static_path)
print(f"Saved {enc_static_path} ({os.path.getsize(enc_static_path)/1e6:.1f} MB), nodes={len(m.graph.node)}")

print("\n--- Step 2: Building Full Pipeline ONNX (DSP Fbank + Encoder + CTC Collapse + Detokenizer) ---")
def make_model(nodes, inits, inputs, outputs):
    g = helper.make_graph(nodes, "g", inputs, outputs, initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 7
    return model

# A) FBANK model
nA, iA = [], []
def cA(name, arr):
    iA.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
    return name
def nA_(op, ins, outs, name, **kw):
    nA.append(helper.make_node(op, ins, outs, name=name, **kw))

frame_idx = (np.arange(T)[:, None] * FRAME_SHIFT + np.arange(FRAME_LEN)[None, :]).astype(np.int64)
cA("frame_idx", frame_idx.reshape(-1))
nA_("Gather", ["raw_wave_flat", "frame_idx"], ["frames_flat"], "fb/gather_frames", axis=0)
nA_("Reshape", ["frames_flat", "shape_T_400"], ["frames_raw"], "fb/reshape_frames")
cA("shape_T_400", np.array([T, FRAME_LEN], dtype=np.int64))

nA_("ReduceMean", ["frames_raw"], ["frame_mean"], "fb/mean", axes=[1], keepdims=1)
nA_("Sub", ["frames_raw", "frame_mean"], ["frames_dc"], "fb/dc_removed")

nA_("Slice", ["frames_dc", "sl0_s", "sl0_e", "sl0_a"], ["first_col"], "fb/first_col")
cA("sl0_s", np.array([0], np.int64)); cA("sl0_e", np.array([1], np.int64)); cA("sl0_a", np.array([1], np.int64))
nA_("Slice", ["frames_dc", "sl1_s", "sl1_e", "sl1_a"], ["all_but_last"], "fb/all_but_last")
cA("sl1_s", np.array([0], np.int64)); cA("sl1_e", np.array([FRAME_LEN - 1], np.int64)); cA("sl1_a", np.array([1], np.int64))
nA_("Concat", ["first_col", "all_but_last"], ["shifted"], "fb/shifted", axis=1)
cA("preemph_coeff", np.array(PREEMPH, np.float32))
nA_("Mul", ["shifted", "preemph_coeff"], ["shifted_scaled"], "fb/preemph_mul")
nA_("Sub", ["frames_dc", "shifted_scaled"], ["frames_pre"], "fb/preemph_sub")

cA("povey_window", WINDOW.astype(np.float32))
nA_("Mul", ["frames_pre", "povey_window"], ["frames_win"], "fb/window")

nA_("Pad", ["frames_win", "pad_amounts", "pad_value"], ["frames_padded"], "fb/pad_to_nfft")
cA("pad_amounts", np.array([0, 0, 0, NFFT - FRAME_LEN], dtype=np.int64))
cA("pad_value", np.array(0.0, np.float32))

cA("dft_re", DFT_RE); cA("dft_im", DFT_IM)
nA_("MatMul", ["frames_padded", "dft_re"], ["spec_re"], "fb/dft_re")
nA_("MatMul", ["frames_padded", "dft_im"], ["spec_im"], "fb/dft_im")
nA_("Mul", ["spec_re", "spec_re"], ["re2"], "fb/re2")
nA_("Mul", ["spec_im", "spec_im"], ["im2"], "fb/im2")
nA_("Add", ["re2", "im2"], ["power"], "fb/power")

cA("mel_mat", MEL_MAT.astype(np.float32))
nA_("MatMul", ["power", "mel_mat"], ["mel_energy"], "fb/mel")
cA("log_eps", np.array(1.1920929e-07, np.float32))
nA_("Max", ["mel_energy", "log_eps"], ["mel_clip"], "fb/clip")
nA_("Log", ["mel_clip"], ["fbank_feats"], "fb/log")
nA_("Unsqueeze", ["fbank_feats", "axis0"], ["x"], "fb/unsqueeze_batch")
cA("axis0", np.array([0], np.int64))

fbank_model = make_model(
    nA, iA,
    inputs=[helper.make_tensor_value_info("raw_wave_flat", TensorProto.FLOAT, [N_SAMPLES])],
    outputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, T, 80])])
onnx.checker.check_model(fbank_model, full_check=True)
print(f"fbank_model OK, nodes={len(fbank_model.graph.node)}")

# B) POST model: ctc_logits -> argmax -> collapse -> detokenize
nC, iC = [], []
def cC(name, arr):
    iC.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
    return name
def nC_(op, ins, outs, name, **kw):
    nC.append(helper.make_node(op, ins, outs, name=name, **kw))

nC_("ArgMax", ["ctc_logits"], ["frame_tokens_2d"], "post/argmax", axis=2, keepdims=0)
nC_("Reshape", ["frame_tokens_2d", "shape_T"], ["frame_tokens_raw"], "post/reshape_tok")
cC("shape_T", np.array([T_OUT], dtype=np.int64))

# Mask padding frames to BLANK to prevent hallucination
cC("arange_T_OUT", np.arange(T_OUT, dtype=np.int64))
nC_("Less", ["arange_T_OUT", "encoder_out_lens"], ["valid_mask"], "post/valid_mask")
cC("blank_id_tok", np.array(BLANK_ID, dtype=np.int64))
nC_("Where", ["valid_mask", "frame_tokens_raw", "blank_id_tok"], ["frame_tokens"], "post/mask_invalid")

cC("zero_i64", np.array(0, np.int64)); cC("blank_id", np.array(BLANK_ID, np.int64))
nC_("Equal", ["frame_tokens", "blank_id"], ["is_blank"], "col/is_blank")
nC_("Not", ["is_blank"], ["is_not_blank"], "col/is_not_blank")

nC_("Slice", ["frame_tokens", "sl2_s", "sl2_e", "sl2_a"], ["tok_head"], "col/tok_head")
cC("sl2_s", np.array([0], np.int64)); cC("sl2_e", np.array([1], np.int64)); cC("sl2_a", np.array([0], np.int64))
nC_("Slice", ["frame_tokens", "sl3_s", "sl3_e", "sl3_a"], ["tok_all_but_last"], "col/tok_prefix")
cC("sl3_s", np.array([0], np.int64)); cC("sl3_e", np.array([T_OUT - 1], np.int64)); cC("sl3_a", np.array([0], np.int64))
nC_("Concat", ["tok_head", "tok_all_but_last"], ["tok_prev"], "col/tok_prev", axis=0)
nC_("Equal", ["frame_tokens", "tok_prev"], ["same_as_prev"], "col/same_as_prev")

cC("pos0_mask", (np.arange(T_OUT) == 0))
nC_("Not", ["same_as_prev"], ["diff_from_prev"], "col/diff_from_prev")
nC_("Or", ["diff_from_prev", "pos0_mask"], ["is_new_token"], "col/is_new_token")
nC_("And", ["is_not_blank", "is_new_token"], ["keep_mask"], "col/keep_mask")

nC_("Cast", ["keep_mask"], ["keep_mask_i64"], "col/cast_keep", to=TensorProto.INT64)
nC_("CumSum", ["keep_mask_i64", "zero_i64"], ["cum_keep"], "col/cumsum")
cC("one_i64", np.array(1, np.int64))
nC_("Sub", ["cum_keep", "one_i64"], ["dest_idx"], "col/dest_idx")
cC("max_len_i64", np.array(MAX_LEN, np.int64))
nC_("Where", ["keep_mask", "dest_idx", "max_len_i64"], ["safe_dest"], "col/safe_dest")

cC("collapse_init", np.full((MAX_LEN + 1,), BLANK_ID, dtype=np.int64))
nC_("ScatterElements", ["collapse_init", "safe_dest", "frame_tokens"], ["collapsed_full"], "col/scatter", axis=0)
nC_("Slice", ["collapsed_full", "sl4_s", "sl4_e", "sl4_a"], ["collapsed_tokens"], "col/final_slice")
cC("sl4_s", np.array([0], np.int64)); cC("sl4_e", np.array([MAX_LEN], np.int64)); cC("sl4_a", np.array([0], np.int64))

byte_table = np.zeros((VOCAB, MAX_BYTES), dtype=np.int64)
len_table = np.zeros((VOCAB,), dtype=np.int64)
for i in range(VOCAB):
    if i in (0, 1, 2):
        continue
    txt = sp.id_to_piece(i).replace("\u2581", " ")
    b_bytes = txt.encode("utf-8")
    assert len(b_bytes) <= MAX_BYTES, (i, txt, len(b_bytes))
    byte_table[i, :len(b_bytes)] = list(b_bytes)
    len_table[i] = len(b_bytes)
cC("byte_table", byte_table)
cC("len_table", len_table)
nC_("Gather", ["byte_table", "collapsed_tokens"], ["out_byte_matrix"], "det/gather_bytes", axis=0)
nC_("Gather", ["len_table", "collapsed_tokens"], ["out_byte_len"], "det/gather_lens", axis=0)

post_model = make_model(
    nC, iC,
    inputs=[helper.make_tensor_value_info("ctc_logits", TensorProto.FLOAT, [1, T_OUT, VOCAB]),
            helper.make_tensor_value_info("encoder_out_lens", TensorProto.INT64, [1])],
    outputs=[helper.make_tensor_value_info("out_byte_matrix", TensorProto.INT64, [MAX_LEN, MAX_BYTES]),
             helper.make_tensor_value_info("out_byte_len", TensorProto.INT64, [MAX_LEN])])
onnx.checker.check_model(post_model, full_check=True)
print(f"post_model OK, nodes={len(post_model.graph.node)}")

# C) Compose A -> B -> C
enc_model = onnx.load(enc_static_path)
ab = oc.merge_models(fbank_model, enc_model, io_map=[("x", "x")], prefix1="fb_", prefix2="enc_")
print(f"Merged A+B OK, nodes={len(ab.graph.node)}, inputs={[i.name for i in ab.graph.input]}")
full = oc.merge_models(ab, post_model,
                       io_map=[("enc_ctc_logits", "ctc_logits"), ("enc_encoder_out_lens", "encoder_out_lens")],
                       prefix2="post_")
print(f"Merged A+B+C OK, total nodes={len(full.graph.node)}")
print("Final inputs :", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in full.graph.input])
print("Final outputs:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in full.graph.output])

onnx.checker.check_model(full, full_check=False)
full_pipeline_path = os.path.join(OUT_FULL, "zip150_full_pipeline.onnx")
onnx.save(full, full_pipeline_path)
print(f"SUCCESS! Saved {full_pipeline_path} ({os.path.getsize(full_pipeline_path)/1e6:.1f} MB)")
