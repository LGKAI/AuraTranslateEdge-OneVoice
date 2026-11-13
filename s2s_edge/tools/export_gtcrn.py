# -*- coding: utf-8 -*-
"""Export GTCRN (48 k parameters, DNS3 checkpoint) to ONNX with a dynamic number of frames. STFT / iSTFT are done in numpy on the host (s2s_edge/denoise.py).
 python tools/export_gtcrn.py --repo ../third_party/gtcrn --out assets/gtcrn_dns3.onnx"""
import argparse, os, sys, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--out", default="assets/gtcrn_dns3.onnx"); a = ap.parse_args()
sys.path.insert(0, a.repo); from gtcrn import GTCRN
m = GTCRN().eval(); m.load_state_dict(torch.load(os.path.join(a.repo, "checkpoints", "model_trained_on_dns3.tar"), map_location="cpu")["model"])
x = torch.randn(1, 257, 100, 2)
torch.onnx.export(m, x, a.out, input_names=["spec"], output_names=["enh"], dynamic_axes={"spec": {2: "T"}, "enh": {2: "T"}}, opset_version=17, dynamo=False)
import onnxruntime as ort
s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
for T in (100, 333):
    xx = torch.randn(1, 257, T, 2)
    with torch.no_grad(): ref = m(xx).numpy()
    got = s.run(None, {"spec": xx.numpy()})[0]; print("T", T, "max abs diff", float(np.abs(ref - got).max()))
print("saved", a.out, os.path.getsize(a.out) // 1024, "KB")
