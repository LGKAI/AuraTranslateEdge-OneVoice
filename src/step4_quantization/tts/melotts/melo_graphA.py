# -*- coding: utf-8 -*-
"""Graph A = static-table text front + BERT (one-hot embedding over the 7335-token space) + word2ph expansion. Input: UTF-8 bytes. Outputs: x,tone,lang,x_len (int32) + bert [1,1024,N]."""
import torch, torch.nn as nn
from melo_front import MeloFront, N, L, LB
from melo_full import BertStatic

class GraphA(nn.Module):
    def __init__(self, T, bert_static, tokenizer):
        super().__init__()
        self.f = MeloFront(T); self.b = bert_static
        ids = [tokenizer.convert_tokens_to_ids(t) for t in T["toks"]]
        W = bert_static.emb.word_embeddings.weight.detach()[torch.tensor(ids)]               # [V,1024]
        self.register_buffer("W", W.clone()); self.V = len(ids)
        v = torch.arange(self.V, dtype=torch.float32); self.register_buffer("vhi", torch.floor(v / 128.0).view(1, 1, self.V)); self.register_buffer("vlo", (v - torch.floor(v / 128.0) * 128.0).view(1, 1, self.V))

    def forward(self, text_bytes):
        x, tone, lang, xl, E, rh, rl, tm = self.f(text_bytes)
        tid = rh * 128.0 + rl                                                                    # [1,L] token index (<=7334: exact only as hi/lo pair)
        oh = (torch.abs(rh.unsqueeze(-1) - self.vhi) < 0.5).float() * (torch.abs(rl.unsqueeze(-1) - self.vlo) < 0.5).float()   # [1,L,V], hi/lo compared separately (each < 2048 => exact in fp16)
        e = oh @ self.W                                                                          # [1,L,1024]
        b = self.b; pos = b.emb.position_embeddings(b.pos) + b.emb.token_type_embeddings(b.tt)
        xx = b.emb.LayerNorm(e + pos); Lh, H, D = b.L, b.heads, b.dk
        bias = ((1.0 - tm) * -1e4).reshape(1, 1, Lh)
        for lyr in b.layers:
            a = lyr.attention.self
            q, k, v = [t(xx).reshape(Lh, H, D).permute(1, 0, 2) for t in (a.query, a.key, a.value)]
            s = torch.matmul(q, k.transpose(1, 2)) * (D ** -0.5) + bias
            c = torch.matmul(torch.softmax(s, -1), v).permute(1, 0, 2).reshape(1, Lh, H * D)
            xx = lyr.attention.output.LayerNorm(lyr.attention.output.dense(c) + xx)
            xx = lyr.output.LayerNorm(lyr.output.dense(lyr.intermediate.intermediate_act_fn(lyr.intermediate.dense(xx))) + xx)
        bert = torch.matmul(xx[0].t(), E).unsqueeze(0)                                           # [1,1024,N]
        return x, tone, lang, xl, bert
