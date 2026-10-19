# -*- coding: utf-8 -*-
"""Step 2: fuse tokenizer table + duration predictor + text encoder + 5-step ODE (shared weights) + vocoder into ONE static ONNX graph.
Inputs : cp int32 [1,N] (Unicode code points of the preprocessed text incl. <lang> tags, 0 = pad), style_dp [1,8,16], style_ttl [1,50,256]
Outputs: wav float [1, L*3072] (masked past the real length), lat_len float [1]  (valid samples = lat_len*3072; seconds = that/44100)
usage: python fuse.py N L [steps=5]"""
import sys, os, json, copy
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto as TP

N, L = int(sys.argv[1]), int(sys.argv[2]); STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); D = os.path.join(HERE, f"static_N{N}_L{L}")
IDX = json.load(open(os.path.expanduser("~/.cache/supertonic3/onnx/unicode_indexer.json")))
SPEED, SR, HOP = 1.05, 44100, 3072


def load(name):
    return onnx.load(os.path.join(D, f"{name}_static.onnx"))


def rename(model, prefix, io_map, share_inits=None, step_suffix=""):
    """prefix all tensor/node names. io_map: original IO name -> final name. Initializers get `prefix`; (shared across unrolled copies)."""
    inits = {i.name for i in model.graph.initializer}
    def nm(t, is_init=False):
        if t == "": return t
        if t in io_map: return io_map[t]
        if t in inits: return prefix + t
        return prefix + step_suffix + t
    nodes = []
    for k, n in enumerate(model.graph.node):
        n2 = copy.deepcopy(n); n2.name = prefix + step_suffix + (n.name or f"{n.op_type}_{k}")
        n2.input[:] = [nm(x) for x in n.input]; n2.output[:] = [nm(x) for x in n.output]
        nodes.append(n2)
    ini = []
    for i in model.graph.initializer:
        i2 = copy.deepcopy(i); i2.name = prefix + i.name; ini.append(i2)
    return nodes, ini


nodes, inits = [], []
def C(name, arr):
    inits.append(numpy_helper.from_array(arr, name))
def add(*ns): nodes.extend(ns)

# ---- tokenizer: cp -> text_ids / text_mask (integer path: ids up to 8321 are NOT representable exactly in fp16)
C("tok_table", np.array(IDX, dtype=np.int32))
C("c_zero_i32", np.array(0, np.int32)); C("c_ax1", np.array([1], np.int64))
add(helper.make_node("Gather", ["tok_table", "cp"], ["tok_g"], axis=0, name="tok_gather"),
    helper.make_node("Greater", ["cp", "c_zero_i32"], ["tok_valid"], name="tok_valid"),
    helper.make_node("Where", ["tok_valid", "tok_g", "c_zero_i32"], ["tok_ids32"], name="tok_where"),
    helper.make_node("Cast", ["tok_ids32"], ["text_ids"], to=TP.INT64, name="tok_cast_ids"),
    helper.make_node("Cast", ["tok_valid"], ["tok_validf"], to=TP.FLOAT, name="tok_cast_mask"),
    helper.make_node("Unsqueeze", ["tok_validf", "c_ax1"], ["text_mask"], name="tok_unsq"))

# ---- duration predictor
n_, i_ = rename(load("duration_predictor"), "dp_", {"text_ids": "text_ids", "style_dp": "style_dp", "text_mask": "text_mask", "duration": "dur_raw"})
nodes += n_; inits += i_
# ---- text encoder
n_, i_ = rename(load("text_encoder"), "te_", {"text_ids": "text_ids", "style_ttl": "style_ttl", "text_mask": "text_mask", "text_emb": "text_emb"})
nodes += n_; inits += i_

# ---- duration -> latent length / mask / noise (all in graph)
C("c_speed", np.array([SPEED], np.float32)); C("c_fps", np.array([SR / HOP], np.float32))
C("c_one", np.array([1.0], np.float32)); C("c_L", np.array([float(L)], np.float32))
C("c_lgrid", np.arange(L, dtype=np.float32).reshape(1, 1, L)); C("c_shape111", np.array([1, 1, 1], np.int64))
rng = np.random.RandomState(0); C("c_noise", rng.randn(1, 144, L).astype(np.float32))
add(helper.make_node("Div", ["dur_raw", "c_speed"], ["dur_s"], name="g_div"),
    helper.make_node("Mul", ["dur_s", "c_fps"], ["lat_f"], name="g_mul"),
    helper.make_node("Ceil", ["lat_f"], ["lat_c0"], name="g_ceil"),
    helper.make_node("Clip", ["lat_c0", "c_one", "c_L"], ["lat_len"], name="g_clip"),
    helper.make_node("Reshape", ["lat_len", "c_shape111"], ["lat_len3"], name="g_resh"),
    helper.make_node("Less", ["c_lgrid", "lat_len3"], ["lm_b"], name="g_less"),
    helper.make_node("Cast", ["lm_b"], ["latent_mask"], to=TP.FLOAT, name="g_cast_lm"),
    helper.make_node("Mul", ["c_noise", "latent_mask"], ["xt0"], name="g_noise_mask"))

# ---- 5-step ODE: weights shared (initializers prefixed ve_ once), nodes duplicated per step
ve = onnx.load(os.path.join(HERE, f"ve_rotR_L{L}.onnx")); seen = set()   # VE with HTP-safe ReduceSum (mask length) rewrite
for s in range(STEPS):
    C(f"c_cur{s}", np.array([float(s)], np.float32)); C(f"c_tot{s}", np.array([float(STEPS)], np.float32))
    io = {"noisy_latent": f"xt{s}", "text_emb": "text_emb", "style_ttl": "style_ttl", "latent_mask": "latent_mask", "text_mask": "text_mask",
          "current_step": f"c_cur{s}", "total_step": f"c_tot{s}", "denoised_latent": f"xt{s+1}"}
    n_, i_ = rename(ve, "ve_", io, step_suffix=f"s{s}_")
    nodes += n_
    for i in i_:
        if i.name not in seen: inits.append(i); seen.add(i.name)

# ---- vocoder + wav mask
n_, i_ = rename(load("vocoder"), "vo_", {"latent": f"xt{STEPS}", "wav_tts": "wav_raw"})
nodes += n_; inits += i_
C("c_exp_shape", np.array([1, L, HOP], np.int64)); C("c_out_shape", np.array([1, L * HOP], np.int64))
add(helper.make_node("Transpose", ["latent_mask"], ["lm_t"], perm=[0, 2, 1], name="w_tr"),
    helper.make_node("Expand", ["lm_t", "c_exp_shape"], ["lm_e"], name="w_exp"),
    helper.make_node("Reshape", ["lm_e", "c_out_shape"], ["wav_mask"], name="w_resh"),
    helper.make_node("Mul", ["wav_raw", "wav_mask"], ["wav"], name="w_mul"))

vi = lambda n, t, s: helper.make_tensor_value_info(n, t, s)
g = helper.make_graph(nodes, "supertonic_full_text_to_wav", [vi("cp", TP.INT32, [1, N]), vi("style_dp", TP.FLOAT, [1, 8, 16]), vi("style_ttl", TP.FLOAT, [1, 50, 256])],
                      [vi("wav", TP.FLOAT, [1, L * HOP]), vi("lat_len", TP.FLOAT, [1])] + ([vi("text_emb", TP.FLOAT, [1, 256, N]), vi("xt1", TP.FLOAT, [1, 144, L]), vi(f"xt{STEPS}", TP.FLOAT, [1, 144, L]), vi("dur_raw", TP.FLOAT, [1])] if os.environ.get("DEBUG") else []), initializer=inits)
opset = max(o.version for o in load("vocoder").opset_import if o.domain == "")
m = helper.make_model(g, producer_name="supertonic_fuse", opset_imports=[helper.make_opsetid("", opset)])
m.ir_version = load("vocoder").ir_version
out = os.path.join(HERE, f"supertonic_full_N{N}_L{L}_s{STEPS}{'_dbg' if os.environ.get('DEBUG') else ''}.onnx")
onnx.checker.check_model(m, full_check=False)
onnx.save(m, out)
print(f"saved {out} | opset {opset} | nodes {len(nodes)} | initializers {len(inits)} | {os.path.getsize(out)/1e6:.0f} MB")
