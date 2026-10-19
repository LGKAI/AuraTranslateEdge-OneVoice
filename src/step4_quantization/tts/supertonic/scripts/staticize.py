# -*- coding: utf-8 -*-
"""Step 1: make the 4 upstream Supertonic-3 ONNX models fully static (fixed text length N and latent length L),
constant-fold shape arithmetic with ORT (BASIC level => standard ops only), and report leftover dynamic ops.
usage: python staticize.py N L"""
import sys, os, collections
import onnx, onnxruntime as ort
from onnx import shape_inference

N, L = int(sys.argv[1]), int(sys.argv[2])
SRC = os.path.expanduser("~/.cache/supertonic3/onnx")
OUT = os.path.join((os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")), f"static_N{N}_L{L}")
os.makedirs(OUT, exist_ok=True)

SPECS = {  # input name -> static shape
    "duration_predictor": {"text_ids": [1, N], "style_dp": [1, 8, 16], "text_mask": [1, 1, N]},
    "text_encoder": {"text_ids": [1, N], "style_ttl": [1, 50, 256], "text_mask": [1, 1, N]},
    "vector_estimator": {"noisy_latent": [1, 144, L], "text_emb": [1, 256, N], "style_ttl": [1, 50, 256],
                         "latent_mask": [1, 1, L], "text_mask": [1, 1, N], "current_step": [1], "total_step": [1]},
    "vocoder": {"latent": [1, 144, L]},
}


def fix(model, spec):
    for vi in model.graph.input:
        if vi.name in spec:
            dims = vi.type.tensor_type.shape.dim
            for d, v in zip(dims, spec[vi.name]):
                d.ClearField("dim_param"); d.dim_value = v
    # drop stale value_info / output dims so inference recomputes them
    del model.graph.value_info[:]
    for o in model.graph.output:
        for d in o.type.tensor_type.shape.dim:
            d.ClearField("dim_param"); d.ClearField("dim_value")
    return model


for name, spec in SPECS.items():
    m = onnx.load(os.path.join(SRC, f"{name}.onnx"))
    m = fix(m, spec)
    tmp = os.path.join(OUT, f"{name}_fixed.onnx"); onnx.save(m, tmp)
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = os.path.join(OUT, f"{name}_static.onnx")
    ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
    m2 = onnx.load(so.optimized_model_filepath)
    m2 = shape_inference.infer_shapes(m2)
    ops = collections.Counter(n.op_type for n in m2.graph.node)
    dyn = {k: v for k, v in ops.items() if k in ("Shape", "Range", "NonZero", "If", "Loop", "ConstantOfShape", "Where", "Expand", "TopK", "ScatterND", "GatherND", "CumSum", "Erf", "Resize")}
    dom = {i.domain for i in m2.opset_import}
    out_shapes = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m2.graph.output]
    print(f"{name:20s} nodes {len(m2.graph.node):5d} | domains {dom} | out {out_shapes}\n    risky ops: {dyn}")
    onnx.save(m2, so.optimized_model_filepath)
print("saved to", OUT)
