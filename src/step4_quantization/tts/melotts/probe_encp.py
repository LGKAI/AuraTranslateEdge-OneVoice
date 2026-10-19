# -*- coding: utf-8 -*-
"""Bisect enc_p on chip: outputs = embedding sum, each encoder layer (attn+ffn), proj stats."""
import sys, math, numpy as np, torch
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, ".")
from melo.api import TTS
from melo import commons
from melo_static import MeloStatic
class P(torch.nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x, tone, lang, bert, x_len):
        s = self.s; e = s.m.enc_p; enc = e.encoder; g = s.g
        h = (e.emb(x.long()) + e.tone_emb(tone.long()) + e.language_emb(lang.long()) + e.bert_proj(bert).transpose(1, 2) + e.ja_bert_proj(torch.zeros(1, 768, s.N)).transpose(1, 2)) * math.sqrt(e.hidden_channels)
        h = h.transpose(1, -1); xm = torch.unsqueeze(commons.sequence_mask(x_len.long(), h.size(2)), 1).to(h.dtype)
        outs = [h * xm]; am = xm.unsqueeze(2) * xm.unsqueeze(-1); h = h * xm
        for i in range(enc.n_layers):
            if i == enc.cond_layer_idx and g is not None:
                gg = enc.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2); h = (h + gg) * xm
            y = enc.attn_layers[i](h, h, am); outs.append(y * xm)
            h = enc.norm_layers_1[i](h + y); y = enc.ffn_layers[i](h, xm); h = enc.norm_layers_2[i](h + y); outs.append(h * xm)
        outs.append(e.proj(h * xm) * xm)
        return tuple(outs)
tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
m = P(MeloStatic(tts.model, 96, 384, group=5)).eval(); d = np.load("work/stages_inputs.npz")
args = tuple(torch.from_numpy(d[k][0]) for k in ("x", "tone", "lang", "bert", "x_len"))
with torch.no_grad(): n = len(m(*args))
torch.onnx.export(m, args, "work/encp_probe.onnx", input_names=["x", "tone", "lang", "bert", "x_len"], output_names=[f"output_{i}" for i in range(n)], opset_version=17, do_constant_folding=True, dynamo=False)
print("exported outputs", n, "layers", m.s.m.enc_p.encoder.n_layers, "cond_layer", m.s.m.enc_p.encoder.cond_layer_idx)
