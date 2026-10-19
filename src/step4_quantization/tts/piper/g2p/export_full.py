import os, sys, time, numpy as np, torch, onnx, onnxruntime as ort, collections
sys.stdout.reconfigure(encoding="utf-8")
import full_graph as F
G_=int(sys.argv[1]); m, tab = F.build(G_)
x = torch.from_numpy(F.text_to_bytes("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!"))
os.makedirs("build", exist_ok=True); out = f"build/piper_vi_full_g{G_}.onnx"
t0=time.time()
torch.onnx.export(m, (x,), out, input_names=["text_bytes"], output_names=["audio","y_len","n_oov"], opset_version=15, do_constant_folding=True, dynamo=False)
print("exported", round(time.time()-t0), "s")
# inline external data if any
mod = onnx.load(out); import os
print("ops:", dict(collections.Counter(n.op_type for n in mod.graph.node).most_common(40)))
print("size MB", round(os.path.getsize(out)/1e6,1))
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
with torch.no_grad(): ta, ty, to = m(x)
oa, oy, oo = s.run(None, {"text_bytes": x.numpy()})
n = int(ty[0])*256
print("torch y_len", int(ty[0]), "ort y_len", int(oy[0]), "| max diff audio", float(np.abs(oa[0,:n]-ta[0,:n].numpy()).max()))
