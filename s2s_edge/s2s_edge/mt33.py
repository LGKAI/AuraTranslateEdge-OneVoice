# -*- coding: utf-8 -*-
"""NLLB-200 3.3B on the NPU as four graphs (encA, encB, decA_b4, decB_b4; see tools/nllb33_export.py) behind the same beam search as the 600M model.
The embedding lookup (pruned vocabulary, fp16 table, assets/nllb33_host.npz) runs on the host, so no graph carries the 285 M-parameter table twice.
  Nllb33Beam(encA, encB, decA, decB, assets_dir)  ->  callable like NllbBeam: mt(text, src, tgt)"""
import os, numpy as np
from .mt import NllbBeam

PAD = 1


class Nllb33Beam(NllbBeam):
    def __init__(self, encA, encB, decA, decB, assets_dir, S=64, Tm=96, B=4, L=24, H=16, dk=128, host_npz="nllb33_host.npz", **kw):
        z = np.load(os.path.join(assets_dir, host_npz))
        self.emb = z["emb"]; self.fwd = z["fwd"].astype(np.int64); self.pos_enc = z["pos_enc"].astype(np.float32); self.pos_dec = z["pos_dec"][:Tm].astype(np.float32)
        self.scale = float(z["scale"]); self.half = L // 2; self.gA, self.gB, self.dA, self.dB = encA, encB, decA, decB
        NllbBeam.__init__(self, self._enc, self._dec, assets_dir, S=S, Tm=Tm, B=B, L=L, H=H, dk=dk, **kw)

    def _embed(self, ids):
        return self.emb[self.fwd[np.asarray(ids, np.int64).reshape(-1)]].astype(np.float32) * self.scale

    def _enc(self, src_ids):
        ids = np.asarray(src_ids).reshape(-1); S = len(ids)
        x0 = (self._embed(ids) + self.pos_enc[:S]).reshape(1, S, -1); bias = ((ids == PAD).astype(np.float32) * -1e4).reshape(1, 1, S)
        x = np.asarray(self.gA(x0=x0, bias=bias)[0]); ck, cv = [np.asarray(o) for o in self.gB(x=x, bias=bias)[:2]]
        return ck, cv, bias

    def _dec(self, tok, oh, mask, sk, sv, ck, cv, bias):
        h = self.half; Bn = len(np.asarray(tok).reshape(-1))
        x0 = (self._embed(tok) + (np.asarray(oh, np.float32) @ self.pos_dec)[None, :]).reshape(Bn, 1, -1)
        x, skA, svA = [np.asarray(o) for o in self.dA(x0=x0, oh=oh, mask=mask, sk=sk[:h], sv=sv[:h], ck=ck[:h], cv=cv[:h], bias=bias)]
        lg, skB, svB = [np.asarray(o) for o in self.dB(x=x.reshape(Bn, 1, -1), oh=oh, mask=mask, sk=sk[h:], sv=sv[h:], ck=ck[h:], cv=cv[h:], bias=bias)]
        return lg, np.concatenate([skA.reshape(sk[:h].shape), skB.reshape(sk[h:].shape)]), np.concatenate([svA.reshape(sv[:h].shape), svB.reshape(sv[h:].shape)])
