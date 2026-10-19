# -*- coding: utf-8 -*-
"""MeloTTS-ZH end-to-end static graph: BERT(22 layers, hidden_states[-3]) -> phoneme-level expansion (matmul with E) -> VITS part -> wav.
Host side only does text normalisation / g2p / wordpiece ids / building the 0/1 expansion matrix E (word2ph) -- all integer bookkeeping.
Inputs : bert_ids int32 [1,L], bert_mask float [1,L], E float [L,N], x/tone/lang int32 [1,N], x_len int32 [1]
Outputs: audio float [1, NW*40*512], y_len int32 [1]"""
import sys, torch, torch.nn as nn, numpy as np
sys.path.insert(0, "MeloTTS")


class BertStatic(nn.Module):
    def __init__(self, bert_mlm, L=32, layers=22):
        super().__init__()
        b = bert_mlm.bert; self.emb = b.embeddings; self.layers = b.encoder.layer[:layers]; self.L = L
        self.register_buffer("tt", torch.zeros(1, L, dtype=torch.long)); self.register_buffer("pos", torch.arange(L, dtype=torch.long).view(1, L))
        self.heads = bert_mlm.config.num_attention_heads; self.dk = bert_mlm.config.hidden_size // self.heads

    def forward(self, ids, mask):                                  # ids int32 [1,L], mask float [1,L]
        e = self.emb.word_embeddings(ids.long()) + self.emb.position_embeddings(self.pos) + self.emb.token_type_embeddings(self.tt)
        x = self.emb.dropout(self.emb.LayerNorm(e)); L, H, D = self.L, self.heads, self.dk
        bias = ((1.0 - mask) * -1e4).reshape(1, 1, L)
        for lyr in self.layers:
            a = lyr.attention.self
            q, k, v = [t(x).reshape(L, H, D).permute(1, 0, 2) for t in (a.query, a.key, a.value)]
            s = torch.matmul(q, k.transpose(1, 2)) * (D ** -0.5) + bias
            c = torch.matmul(torch.softmax(s, -1), v).permute(1, 0, 2).reshape(1, L, H * D)
            x = lyr.attention.output.LayerNorm(lyr.attention.output.dense(c) + x)
            x = lyr.output.LayerNorm(lyr.output.dense(lyr.intermediate.intermediate_act_fn(lyr.intermediate.dense(x))) + x)
        return x                                                   # [1,L,1024]  == hidden_states[22]


class MeloFull(nn.Module):
    def __init__(self, bert_static, melo_static):
        super().__init__(); self.b, self.v = bert_static, melo_static

    def forward(self, bert_ids, bert_mask, E, x, tone, lang, x_len):
        h = self.b(bert_ids, bert_mask)                            # [1,L,1024]
        bert = torch.matmul(h[0].t(), E).unsqueeze(0)              # [1,1024,N]  token features repeated per phoneme (word2ph)
        return self.v(x, tone, lang, bert, x_len)


def host_front(tts, tok, text, N, L):
    """text -> (bert_ids[1,L], bert_mask[1,L], E[L,N], x, tone, lang [1,N], x_len[1]) ; all integer bookkeeping (no model compute)."""
    from melo.text.cleaner import clean_text
    from melo.text import cleaned_text_to_sequence
    from melo import commons
    norm_text, phone, tone, word2ph = clean_text(text, "ZH")
    phone, tone, lang = cleaned_text_to_sequence(phone, tone, "ZH", tts.symbol_to_id)
    phone, tone, lang = commons.intersperse(phone, 0), commons.intersperse(tone, 0), commons.intersperse(lang, 0)
    word2ph = [w * 2 for w in word2ph]; word2ph[0] += 1
    ids = tok(norm_text)["input_ids"]; assert len(ids) == len(word2ph) <= L, (len(ids), len(word2ph), L)
    n = len(phone); assert n <= N and sum(word2ph) == n
    E = np.zeros((L, N), np.float32); off = 0
    for i, w in enumerate(word2ph): E[i, off:off + w] = 1; off += w
    pad = lambda v: np.concatenate([np.array(v, np.int64), np.zeros(N - n, np.int64)]).astype(np.int32)[None]
    bi = np.zeros((1, L), np.int32); bi[0, :len(ids)] = ids; bm = np.zeros((1, L), np.float32); bm[0, :len(ids)] = 1
    return bi, bm, E, pad(phone), pad(tone), pad(lang), np.array([n], np.int32)
