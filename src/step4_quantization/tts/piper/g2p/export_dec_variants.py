import os, sys, torch
os.makedirs('build', exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
import full_graph as F
m, tab = F.build()
class DecB(torch.nn.Module):
    def __init__(s, m, nb): super().__init__(); s.m=m; s.nb=nb
    def forward(s, zb): return s.m.dec(zb)
class DecSeq(torch.nn.Module):
    def __init__(s, m, n): super().__init__(); s.m=m; s.n=n
    def forward(s, zb):
        outs=[s.m.dec(zb[k:k+1]) for k in range(s.n)]
        return torch.cat(outs, dim=0)
torch.onnx.export(DecB(m,8), (torch.zeros(8,192,64),), "build/piper_vi_dec_b8.onnx", input_names=["zb"], output_names=["audio"], opset_version=15, do_constant_folding=True, dynamo=False)
torch.onnx.export(DecSeq(m,39), (torch.zeros(39,192,64),), "build/piper_vi_dec_seq39.onnx", input_names=["zb"], output_names=["audio"], opset_version=15, do_constant_folding=True, dynamo=False)
print("ok")
