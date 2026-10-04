"""Regenerates outputs/nllb-onnx/* (deleted during disk cleanup -- gitignored,
reproducible build artifact per README). Reproduces the same 3-stage chain
this session already built and verified: (1) export encoder from HF NLLB
to ONNX, (2) host-compute the attention mask fix (fix_nllb_mask_hostcompute.py
logic), (3) pin static input_ids shape (fix_nllb_static_shapes.py logic).
Then runs onnxruntime's own graph optimizer to fold the -100 mask path
constants (the step that got NLLB compiling further into the QNN converter
earlier this session), matching encoder_model_hostmask_static_optimized.onnx.
"""
import os

import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "nllb-onnx")
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH = os.path.join(OUT_DIR, "encoder_model.onnx")
HOSTMASK_PATH = os.path.join(OUT_DIR, "encoder_model_hostmask.onnx")
STATIC_PATH = os.path.join(OUT_DIR, "encoder_model_hostmask_static.onnx")
OPTIMIZED_PATH = os.path.join(OUT_DIR, "encoder_model_hostmask_static_optimized.onnx")

MASK_OUTPUT = "/encoder/Where_1_output_0"
NEG_BIAS = -30000.0
FIXED_SEQ_LEN = 64


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def step1_export():
    if os.path.exists(RAW_PATH):
        print(f"[step1] {RAW_PATH} already exists, skipping export")
        return
    print("[step1] loading facebook/nllb-200-distilled-600M ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", use_safetensors=True, attn_implementation="eager")
    model.eval()
    wrapper = EncoderWrapper(model.model.encoder)

    dummy_ids = torch.randint(3, 25000, (1, 16), dtype=torch.long)
    dummy_mask = torch.ones((1, 16), dtype=torch.long)

    print(f"[step1] exporting to {RAW_PATH} ...")
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask), RAW_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"},
                       "last_hidden_state": {1: "seq_len"}},
        opset_version=17)
    print(f"[step1] wrote {RAW_PATH} ({os.path.getsize(RAW_PATH)/1e6:.1f} MB)")


def find_safe_to_delete_ancestors(model, target_output, stop_at=("attention_mask", "input_ids")):
    nodes = list(model.graph.node)
    by_output = {}
    for n in nodes:
        for o in n.output:
            by_output[o] = n
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


def host_mask(attention_mask, neg_bias=NEG_BIAS):
    inv = (1 - attention_mask).astype(np.float32)
    bias = inv * neg_bias
    return np.broadcast_to(bias[:, None, None, :], (bias.shape[0], 1, bias.shape[1], bias.shape[1])).copy()


def step2_hostmask():
    if os.path.exists(HOSTMASK_PATH):
        print(f"[step2] {HOSTMASK_PATH} already exists, skipping")
        return
    print(f"[step2] loading {RAW_PATH} ...")
    model = onnx.load(RAW_PATH, load_external_data=True)
    safe = find_safe_to_delete_ancestors(model, MASK_OUTPUT)
    print(f"[step2] {len(safe)} nodes safe to delete")
    kept = [n for n in model.graph.node if n.name not in safe]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    inp = helper.make_tensor_value_info(MASK_OUTPUT, TensorProto.FLOAT, [1, 1, "seq_len", "seq_len"])
    model.graph.input.append(inp)
    del model.graph.value_info[:]
    remaining_consumers = any("attention_mask" in n.input for n in model.graph.node)
    if not remaining_consumers:
        keep_inputs = [i for i in model.graph.input if i.name != "attention_mask"]
        del model.graph.input[:]
        model.graph.input.extend(keep_inputs)
        print("[step2] removed orphaned 'attention_mask' input")
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, HOSTMASK_PATH)
    print(f"[step2] wrote {HOSTMASK_PATH} ({os.path.getsize(HOSTMASK_PATH)/1e6:.1f} MB)")

    rng = np.random.default_rng(0)
    input_ids = rng.integers(3, 1000, size=(1, 20)).astype(np.int64)
    attention_mask = np.zeros((1, 20), dtype=np.int64)
    attention_mask[:, :15] = 1
    so = ort.SessionOptions(); so.log_severity_level = 3
    ref = ort.InferenceSession(RAW_PATH, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    mask_bias = host_mask(attention_mask)
    new = ort.InferenceSession(HOSTMASK_PATH, so, providers=["CPUExecutionProvider"])
    new_out = new.run(None, {"input_ids": input_ids, MASK_OUTPUT: mask_bias})[0]
    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64) - np.asarray(new_out, dtype=np.float64)).max())
    print(f"[step2] verify max_abs_diff={diff:.3e}")
    assert diff <= 1e-2, "hostmask fix mismatch"


def step3_static():
    if os.path.exists(STATIC_PATH):
        print(f"[step3] {STATIC_PATH} already exists, skipping")
        return
    print(f"[step3] loading {HOSTMASK_PATH} ...")
    model = onnx.load(HOSTMASK_PATH)
    new_inputs = []
    for i in model.graph.input:
        if i.name == "input_ids":
            new_inputs.append(helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, FIXED_SEQ_LEN]))
        elif i.name == MASK_OUTPUT:
            new_inputs.append(helper.make_tensor_value_info(MASK_OUTPUT, TensorProto.FLOAT,
                                                              [1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN]))
        else:
            new_inputs.append(i)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, STATIC_PATH)
    print(f"[step3] wrote {STATIC_PATH} ({os.path.getsize(STATIC_PATH)/1e6:.1f} MB)")

    rng = np.random.default_rng(0)
    ids = rng.integers(1, 25000, size=(1, FIXED_SEQ_LEN)).astype(np.int64)
    bias = np.zeros((1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=np.float32)
    so = ort.SessionOptions(); so.log_severity_level = 3
    ref = ort.InferenceSession(HOSTMASK_PATH, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": ids, MASK_OUTPUT: bias})[0]
    new = ort.InferenceSession(STATIC_PATH, so, providers=["CPUExecutionProvider"])
    new_out = new.run(None, {"input_ids": ids, MASK_OUTPUT: bias})[0]
    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64) - np.asarray(new_out, dtype=np.float64)).max())
    print(f"[step3] verify max_abs_diff={diff:.3e}")
    assert diff < 1e-4, "static shape pin mismatch"


def step4_optimize():
    if os.path.exists(OPTIMIZED_PATH):
        print(f"[step4] {OPTIMIZED_PATH} already exists, skipping")
        return
    print(f"[step4] running onnxruntime graph optimizer on {STATIC_PATH} ...")
    so = ort.SessionOptions()
    # ORT_ENABLE_EXTENDED (and ALL) apply transformer-specific fusions
    # (Attention/SkipLayerNorm/FusedMatMul) that live in the com.microsoft
    # contrib domain -- Qualcomm AI Hub's converter rejects those outright
    # ("not supported by Qualcomm AI Hub Workbench"). ORT_ENABLE_BASIC only
    # does constant folding / redundant-node elimination, no vendor ops,
    # and that's all we actually need here (the Reshape-shape-folding fix
    # from Mui 4 is a constant-folding result, not a fusion result).
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = OPTIMIZED_PATH
    so.log_severity_level = 3
    ort.InferenceSession(STATIC_PATH, so, providers=["CPUExecutionProvider"])
    print(f"[step4] wrote {OPTIMIZED_PATH} ({os.path.getsize(OPTIMIZED_PATH)/1e6:.1f} MB)")

    rng = np.random.default_rng(0)
    ids = rng.integers(1, 25000, size=(1, FIXED_SEQ_LEN)).astype(np.int64)
    bias = np.zeros((1, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN), dtype=np.float32)
    so2 = ort.SessionOptions(); so2.log_severity_level = 3
    ref = ort.InferenceSession(STATIC_PATH, so2, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": ids, MASK_OUTPUT: bias})[0]
    opt = ort.InferenceSession(OPTIMIZED_PATH, so2, providers=["CPUExecutionProvider"])
    opt_out = opt.run(None, {"input_ids": ids, MASK_OUTPUT: bias})[0]
    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64) - np.asarray(opt_out, dtype=np.float64)).max())
    print(f"[step4] verify max_abs_diff={diff:.3e}")
    assert diff < 1e-4, "optimizer changed numerics"
    print("[step4] OK (numerically equivalent)")


if __name__ == "__main__":
    step1_export()
    step2_hostmask()
    step3_static()
    step4_optimize()
    print("[regen] all 4 files regenerated successfully")
