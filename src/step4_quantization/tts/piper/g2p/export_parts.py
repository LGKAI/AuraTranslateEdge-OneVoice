import os, sys, torch, numpy as np
os.makedirs('build', exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
import full_graph as F
m, tab = F.build()
x = torch.from_numpy(F.text_to_bytes("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!"))
torch.onnx.export(m.g2p, (x,), "build/piper_vi_g2p_only.onnx", input_names=["text_bytes"], output_names=["x","x_lengths","n_oov"], opset_version=15, do_constant_folding=True, dynamo=False)
class DecB(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m=m
    def forward(s, zb):
        a = s.m.dec(zb)
        return torch.cat([a[0:1,0,0:F.HOP*F.UP], a[1:,0,F.OVL*F.UP:(F.OVL+F.HOP)*F.UP].reshape(1,(F.NW-1)*F.HOP*F.UP)],1)
torch.onnx.export(DecB(m), (torch.zeros(F.NW,192,F.WIN),), "build/piper_vi_dec_b39.onnx", input_names=["zb"], output_names=["audio"], opset_version=15, do_constant_folding=True, dynamo=False)
print("ok")
