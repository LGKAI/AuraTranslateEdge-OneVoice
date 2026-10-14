# -*- coding: utf-8 -*-
"""NLLB-200-distilled-600M as ONE static graph: src_ids -> encoder -> T unrolled greedy decoder steps (static KV cache) -> token ids.
No Python loop at run time: the autoregressive loop is unrolled inside the graph.
forward(src_ids int32 [1,S], tgt_lang int32 [1]) -> out_ids int32 [1,T]   (tokens that follow </s>: [lang, y1, y2, ...]; host cuts at EOS=2)
Optional vocabulary pruning: ids are remapped in-graph (orig->pruned on input, pruned->orig on output)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

PAD, EOS, UNK = 1, 2, 3


class NLLBStatic(nn.Module):
    def __init__(self, hf, S=32, T=24, keep_ids=None, half_emb=False):
        super().__init__()
        m = hf.model; self.S, self.T = S, T; cfg = hf.config
        self.d, self.h = cfg.d_model, cfg.encoder_attention_heads; self.dk = self.d // self.h
        self.enc_layers, self.dec_layers = m.encoder.layers, m.decoder.layers
        self.enc_ln, self.dec_ln = m.encoder.layer_norm, m.decoder.layer_norm
        W = m.shared.weight.detach()
        if keep_ids is not None:                                    # vocabulary pruning
            keep = torch.tensor(sorted(set(keep_ids)), dtype=torch.long); V0 = W.shape[0]
            fwd = torch.full((V0,), UNK, dtype=torch.long); fwd[keep] = torch.arange(len(keep))          # orig -> pruned (others -> UNK's slot)
            fwd[UNK] = int((keep == UNK).nonzero()[0]); self.register_buffer("fwd", fwd.to(torch.int32)); self.register_buffer("inv", keep.to(torch.int32))
            self.register_buffer("emb", W[keep].clone().half() if half_emb else W[keep].clone()); self.pruned = True; self.eos_p = int((keep == EOS).nonzero()[0])
        else:
            self.register_buffer("emb", W.clone().half() if half_emb else W.clone()); self.pruned = False; self.eos_p = EOS
        self.scale = math.sqrt(self.d)
        pos = m.encoder.embed_positions.weights.detach()           # sinusoidal table, row = position id (offset already 2 for first token)
        self.register_buffer("pos_enc", pos[2:2 + S].clone()); self.register_buffer("pos_dec", pos[2:2 + T + 1].clone())
        self.register_buffer("start_emb", (self.emb[self.eos_p].float() * self.scale).clone())

    # -- helpers ------------------------------------------------------------------
    def _split(self, x, L):                                          # [1,L,d] -> [h,L,dk]
        return x.reshape(L, self.h, self.dk).permute(1, 0, 2)

    def _attn(self, q, k, v, bias=None):                             # q [h,Lq,dk] k,v [h,Lk,dk]
        s = torch.matmul(q * (self.dk ** -0.5), k.transpose(1, 2))
        if bias is not None: s = s + bias
        return torch.matmul(torch.softmax(s, dim=-1), v)

    def _merge(self, x, Lq):                                         # [h,Lq,dk] -> [1,Lq,d]
        return x.permute(1, 0, 2).reshape(1, Lq, self.d)

    def encode(self, src_ids):
        S = self.S
        ids = src_ids.to(torch.long).reshape(S); valid = (ids != PAD).float()
        if self.pruned: ids = self.fwd[ids].to(torch.long)
        x = (self.emb[ids].float() * self.scale + self.pos_enc).unsqueeze(0)  # [1,S,d]
        bias = ((1.0 - valid) * -1e4).reshape(1, 1, S)               # key padding mask, additive
        for L in self.enc_layers:
            r = x; y = L.self_attn_layer_norm(x)
            a = L.self_attn
            q, k, v = self._split(a.q_proj(y), S), self._split(a.k_proj(y), S), self._split(a.v_proj(y), S)
            x = r + a.out_proj(self._merge(self._attn(q, k, v, bias), S))
            r = x; y = L.final_layer_norm(x); x = r + L.fc2(F.relu(L.fc1(y)))
        return self.enc_ln(x), bias                                  # [1,S,d]

    def forward(self, src_ids, tgt_lang):
        S, T = self.S, self.T
        enc, bias = self.encode(src_ids)
        ck = [self._split(L.encoder_attn.k_proj(enc), S) for L in self.dec_layers]    # cross K/V once
        cv = [self._split(L.encoder_attn.v_proj(enc), S) for L in self.dec_layers]
        sk = [[] for _ in self.dec_layers]; sv = [[] for _ in self.dec_layers]
        lang = tgt_lang.to(torch.long).reshape(1)
        lang_p = self.fwd[lang].to(torch.long) if self.pruned else lang
        x = (self.start_emb + self.pos_dec[0]).reshape(1, 1, self.d)
        outs = []
        for t in range(T):
            for li, L in enumerate(self.dec_layers):
                r = x; y = L.self_attn_layer_norm(x); a = L.self_attn
                q = self._split(a.q_proj(y), 1); kk = self._split(a.k_proj(y), 1); vv = self._split(a.v_proj(y), 1)
                sk[li].append(kk); sv[li].append(vv)
                K = torch.cat(sk[li], dim=1) if t > 0 else kk; V = torch.cat(sv[li], dim=1) if t > 0 else vv
                x = r + a.out_proj(self._merge(self._attn(q, K, V), 1))
                r = x; y = L.encoder_attn_layer_norm(x); c = L.encoder_attn
                x = r + c.out_proj(self._merge(self._attn(self._split(c.q_proj(y), 1), ck[li], cv[li], bias), 1))
                r = x; y = L.final_layer_norm(x); x = r + L.fc2(F.relu(L.fc1(y)))
            h_ = self.dec_ln(x).reshape(1, self.d); logits = F.linear(h_.to(self.emb.dtype), self.emb)                 # tied lm_head, [1,V']
            nxt = lang_p if t == 0 else torch.argmax(logits, dim=-1)                       # step 0 is forced to the target-language token
            outs.append(nxt)
            x = (self.emb[nxt].float() * self.scale + self.pos_dec[t + 1]).reshape(1, 1, self.d)
        ids = torch.cat(outs).reshape(1, T)
        if self.pruned: ids = self.inv[ids]
        return ids.to(torch.int32)
