# -*- coding: utf-8 -*-
"""MeloTTS-ZH VITS part as ONE static graph (phones -> waveform), HTP-friendly.
Inputs (static): x/tone/lang int32 [1,N], bert float [1,1024,N], x_len int32 [1]  (ZH uses only the 1024-d `bert` slot; ja_bert is zeros).
Differences from the reference infer(): sdp_ratio=0 (deterministic duration predictor only), constant noise, static length regulation
(aligner by cumsum + compare, no NonZero/Range), y length clamped to Y.
Outputs: audio float [1, Y*512] (masked beyond the real length), y_len int32 [1]  (valid samples = y_len*512)."""
import sys, math, torch, torch.nn as nn

sys.path.insert(0, "MeloTTS")
HOP = 512


class MeloStatic(nn.Module):
    def __init__(self, model, N=96, Y=384, speaker=1, noise_scale=0.6, length_scale=1.0, seed=0, group=0):
        super().__init__()
        self.group = group                      # 0 = decode all Y frames at once (CPU reference); >0 = windowed decoder, `group` windows per decoder call
        self.WIN, self.OVL, self.HOPF = 64, 12, 40
        self.NW = -(-Y // self.HOPF)
        import numpy as np
        sel = np.zeros((Y, (self.NW - 1) * self.WIN), np.float32)
        for k in range(1, self.NW):
            for p in range(self.WIN):
                f = self.HOPF * k - self.OVL + p
                if 0 <= f < Y: sel[f, (k - 1) * self.WIN + p] = 1
        self.register_buffer('sel', torch.tensor(sel))
        self.register_buffer('fgrid', torch.arange(self.NW * self.HOPF, dtype=torch.float32).view(self.NW * self.HOPF, 1))
        self.m, self.N, self.Y, self.noise_scale, self.length_scale = model, N, Y, noise_scale, length_scale
        with torch.no_grad():
            g = model.emb_g(torch.LongTensor([speaker])).unsqueeze(-1)          # [1,256,1] constant speaker embedding
        self.register_buffer("g", g)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("noise", torch.randn(1, model.inter_channels, Y, generator=gen))
        self.register_buffer("ygrid", torch.arange(Y, dtype=torch.float32).view(1, Y, 1))
        self.register_buffer("ones_n", torch.ones(N, 1))
        self.register_buffer("zero_pad", torch.zeros(1, 1, 1))

    def acoustic(self, x, tone, lang, bert, ja_bert, x_len):
        m, N, Y = self.m, self.N, self.Y
        x, m_p, logs_p, x_mask = m.enc_p(x.long(), x_len.long(), tone.long(), lang.long(), bert, ja_bert, g=self.g)
        logw = m.dp(x, x_mask, g=self.g)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * self.length_scale)                    # [1,1,N]
        y_len = torch.clamp(w_ceil.reshape(1, N) @ self.ones_n, min=1.0, max=float(Y))      # [1,1]  (MatMul, not multi-axis ReduceSum)
        cum = torch.cumsum(w_ceil, dim=-1)
        cum_prev = torch.cat([self.zero_pad, cum[:, :, :-1]], dim=-1)
        y_mask1 = (self.ygrid < y_len.reshape(1, 1, 1)).float()                              # [1,Y,1]
        attn = ((self.ygrid >= cum_prev).float() * (self.ygrid < cum).float()) * x_mask * y_mask1   # [1,Y,N]
        mm = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2)                         # [1,192,Y]
        ll = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2)
        y_mask = y_mask1.transpose(1, 2)                                                     # [1,1,Y]
        z_p = mm + self.noise * torch.exp(ll) * self.noise_scale
        z = m.flow(z_p, y_mask, g=self.g, reverse=True)
        return z * y_mask, y_mask, y_len

    def forward(self, x, tone, lang, bert, x_len):
        ja = torch.zeros(1, 768, self.N)
        z, y_mask, y_len = self.acoustic(x, tone, lang, bert, ja, x_len)
        if self.group == 0:
            o = self.m.dec(z, g=self.g)
            up = y_mask.reshape(1, self.Y, 1).expand(1, self.Y, HOP).reshape(1, 1, self.Y * HOP)
            return (o * up).reshape(1, self.Y * HOP), y_len.reshape(1).to(torch.int32)
        W, OV, H, NW = self.WIN, self.OVL, self.HOPF, self.NW
        win0 = torch.nn.functional.pad(z[:, :, 0:H + OV], (0, W - (H + OV)))                    # [1,192,64] first window: no left context
        rest = (z[0] @ self.sel).reshape(z.shape[1], NW - 1, W).permute(1, 0, 2)               # [NW-1,192,64]
        zb = torch.cat([win0, rest], dim=0)                                                     # [NW,192,64]
        gg = self.g.expand(NW, -1, -1)
        outs = [self.m.dec(c, g=gc) for c, gc in zip(torch.split(zb, self.group, dim=0), torch.split(gg, self.group, dim=0))]
        a_ = torch.cat(outs, dim=0)                                                             # [NW,1,64*512]
        a0 = a_[0:1, 0, 0:H * HOP]; ar = a_[1:, 0, OV * HOP:(OV + H) * HOP].reshape(1, (NW - 1) * H * HOP)
        audio = torch.cat([a0, ar], dim=1)                                                      # [1,NW*40*512]
        fm = (self.fgrid < y_len.reshape(1, 1)).float(); mask = fm.expand(NW * H, HOP).reshape(1, NW * H * HOP)
        return audio * mask, y_len.reshape(1).to(torch.int32)
