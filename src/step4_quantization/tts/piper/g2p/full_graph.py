# -*- coding: utf-8 -*-
"""ONE static graph: UTF-8 text bytes -> 22.05 kHz waveform (Piper vi), for Qualcomm HTP.
  text_bytes int32[1,512] -> G2P -> encoder -> SDP -> monotonic aligner -> flow -> windowed HiFi-GAN decoder (batch of 39 windows)
Outputs: audio float[1,399360] (masked beyond the real length), y_len int32[1] (frames; samples = y_len*256), n_oov float[1]
"""
import sys, os, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src", "step4_npu"))
import numpy as np, torch, torch.nn as nn
import g2p_graph as G
from export_piper_components import Encoder, SDP, Flow, Decoder, build_model_from_onnx
from alignment_pipeline import MonotonicAligner

NW, WIN, OVL, HOP, UP, MAXF = 40, 64, 12, 40, 256, 1536
ONNX = os.path.join(HERE, "..", "outputs", "piper_vi_npu", "vi_VN-vais1000-medium.onnx")


class FullTTS(nn.Module):
    def __init__(self, g2p, enc, sdp, align, flow, dec, length_scale=1.0, noise_w=0.8, noise=0.667, group=8):
        super().__init__()
        self.group = group
        self.g2p, self.enc, self.sdp, self.align, self.flow, self.dec = g2p, enc, sdp, align, flow, dec
        self.register_buffer("ls", torch.tensor([length_scale])); self.register_buffer("nw", torch.tensor([noise_w]))
        self.register_buffer("ns", torch.tensor([noise]))
        sel = np.zeros((MAXF, (NW - 1) * WIN), np.float32)
        for k in range(1, NW):
            for p_ in range(WIN):
                f = HOP * k - OVL + p_
                if f < MAXF: sel[f, (k - 1) * WIN + p_] = 1
        self.register_buffer("sel", torch.tensor(sel))
        self.register_buffer("fgrid", torch.arange(NW * HOP, dtype=torch.float32).view(NW * HOP, 1))

    def front(self, text_bytes):
        x, xl, oov = self.g2p(text_bytes)
        x_enc, m_p, logs_p, x_mask = self.enc(x, xl)
        y_len, w_ceil = self.sdp(x_enc, x_mask, self.ls, self.nw)
        y_len = torch.clamp(y_len, max=float(MAXF))
        attn, y_mask = self.align(w_ceil, x_mask, y_len)
        z = self.flow(m_p, logs_p, y_mask, attn, self.ns)          # [1,192,1536]
        return z, y_len, oov

    def forward(self, text_bytes):
        z, y_len, oov = self.front(text_bytes)
        # window 0: frames [0,52) (no left context); windows k>=1: frames [40k-12, 40k+52) via ONE constant 0/1 selection matmul
        win0 = torch.nn.functional.pad(z[:, :, 0:HOP + OVL], (0, WIN - (HOP + OVL)))          # [1,192,64]
        rest = (z[0] @ self.sel).reshape(192, NW - 1, WIN).permute(1, 0, 2)                  # [38,192,64]
        zb = torch.cat([win0, rest], dim=0)                                                   # [39,192,64]
        a = torch.cat([self.dec(c) for c in torch.split(zb, self.group, dim=0)], dim=0)      # decoder in sequential groups -> [40,1,16384]
        a0 = a[0:1, 0, 0:HOP * UP]                                                            # [1,10240]
        ar = a[1:, 0, OVL * UP:(OVL + HOP) * UP].reshape(1, (NW - 1) * HOP * UP)              # [1,38*10240]
        audio = torch.cat([a0, ar], dim=1)
        fmask = (self.fgrid < y_len.reshape(1, 1)).float()
        mask = fmask.expand(NW * HOP, UP).reshape(1, NW * HOP * UP)
        return audio * mask, y_len.to(torch.int32), oov


def build(group=8):
    torch.manual_seed(0)
    letters, classes, pats, pcls, keys, phs, tab = pickle.load(open(os.path.join(HERE, "build", "g2p_tables.pkl"), "rb"))
    g2p = G.VietG2P(letters, classes, pats, pcls, keys, phs).eval()
    gen = build_model_from_onnx(ONNX, ONNX + ".json")
    mods = [Encoder(gen), SDP(gen), MonotonicAligner(), Flow(gen), Decoder(gen)]
    m = FullTTS(g2p, *mods, group=group).eval()
    return m, tab


def text_to_bytes(text):
    import unicodedata
    b = unicodedata.normalize("NFC", text).encode("utf-8")[:G.LB]
    a = np.zeros((1, G.LB), np.int32); a[0, :len(b)] = list(b)
    return a
