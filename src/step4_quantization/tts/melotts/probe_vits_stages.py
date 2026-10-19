# -*- coding: utf-8 -*-
"""Bisect VITS chip-vs-CPU gap: graph that outputs intermediate stages (logw, w_ceil, m_p, z) for the same inputs the chain used."""
import sys, numpy as np, torch, qai_hub as hub
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, ".")
from melo.api import TTS
from melo_static import MeloStatic
class Stages(torch.nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x, tone, lang, bert, x_len):
        s, m = self.s, self.s.m; ja = torch.zeros(1, 768, s.N)
        xx, m_p, logs_p, x_mask = m.enc_p(x.long(), x_len.long(), tone.long(), lang.long(), bert, ja, g=s.g)
        logw = m.dp(xx, x_mask, g=s.g); w = torch.exp(logw) * x_mask * s.length_scale
        z, y_mask, y_len = s.acoustic(x, tone, lang, bert, ja, x_len)
        return logw, w, m_p, z
tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
S = Stages(MeloStatic(tts.model, 96, 384, group=5)).eval()
a = hub.get_job("jp4e3qqlg").download_output_data(); k = {int(n.split("_")[-1]): n for n in a}
idx = [1, 15, 29, 31]; IN = {"x": [], "tone": [], "lang": [], "bert": [], "x_len": []}
for i in idx:
    IN["x"].append(np.asarray(a[k[0]][i])); IN["tone"].append(np.asarray(a[k[1]][i])); IN["lang"].append(np.zeros((1, 96), np.int32))
    IN["bert"].append(np.asarray(a[k[4]][i]).astype(np.float32)); IN["x_len"].append(np.asarray(a[k[3]][i]))
np.savez("work/stages_inputs.npz", **{kk: np.stack(v) for kk, v in IN.items()})
args = tuple(torch.from_numpy(IN[kk][0]) for kk in ("x", "tone", "lang", "bert", "x_len"))
torch.onnx.export(S, args, "work/vits_stages.onnx", input_names=["x", "tone", "lang", "bert", "x_len"], output_names=["output_0", "output_1", "output_2", "output_3"], opset_version=17, do_constant_folding=True, dynamo=False)
print("exported")
