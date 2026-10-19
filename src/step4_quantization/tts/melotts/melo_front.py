# -*- coding: utf-8 -*-
"""MeloTTS-ZH text front as ONE static tensor graph (NPU-friendly): UTF-8 bytes -> {phone ids, tones, E (word2ph expansion), BERT token rows}.
No string ops / dictionaries at run time: everything is table lookup by exact small-integer comparisons (<2048, exact in fp16) and matmul.
Inputs : text_bytes int32 [1,LB]  (zero padded NFC UTF-8)
Outputs: x,tone,lang int32 [1,N] ; x_len int32 [1] ; E float [L,N] ; tok_hi,tok_lo,tok_mask float [1,L]  (BERT token = hi*128+lo in the shared token space)
"""
import sys, pickle, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "MeloTTS")

LB, LC, N, L = 112, 24, 96, 32
NEAR = lambda a, b: (torch.abs(a - b) < 0.5).float()


def load_tables(path=None):
    import os
    path = path or ("work/front_tables.pkl" if os.path.exists("work/front_tables.pkl") else "tables/front_tables.pkl")
    return pickle.load(open(path, "rb"))


def build_patterns(T):
    """pattern rows: (b0,b1,b2) bytes (-1 = absent), attrs [hi, lo, p1, p2, tone, valid]"""
    from melo.text.chinese import rep_map
    idx, D, sid, PUN = T["idx"], T["D"], T["sid"], T["pun"]
    rows = []
    def tokattr(tok, ph, tn):
        i = idx[tok]; p = [sid[x] for x in ph] + [0] * (2 - len(ph))
        return [i // 128, i % 128, p[0], p[1], tn[-1] if tn else 0, 1]
    def addpat(ch, attrs):
        b = list(ch.encode("utf-8")) + [-1, -1]; rows.append((b[:3], attrs))
    for c, d in D.items():
        if any(x not in sid for x in d[0]): continue
        if c in PUN: addpat(c, tokattr(c, (c,), (0,)))                         # ASCII punctuation already in the symbol set
        else: addpat(c, tokattr(c, d[0], d[1]))
    for raw, tgt in rep_map.items():
        if len(raw) == 1 and tgt in PUN and raw not in D: addpat(raw, tokattr(tgt, (tgt,), (0,)))
    addpat("…", tokattr("…", ("…",), (0,)))
    # text_normalize special cases: 嗯->恩, 呣->母
    for src, dst in (("嗯", "恩"), ("呣", "母")):
        if dst in D: addpat(src, tokattr(dst, D[dst][0], D[dst][1]))
    P = np.array([r[0] for r in rows], np.float32); A = np.array([r[1] for r in rows], np.float32)
    return P, A


def key_fields(entry_key, idx):
    """entry key elements: token string or ('T', tone) -> (hi, lo) pairs"""
    out = []
    for e in entry_key:
        if isinstance(e, tuple): out.append((100 + e[1] + 1, 0))          # tone field: hi = 100 + (tone+1)
        else: i = idx[e]; out.append((i // 128, i % 128))
    return out


class MeloFront(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T; idx = T["idx"]; self.order = T["order"]; self.S = T["pad_sym"]
        P, A = build_patterns(T)
        self.register_buffer("P0", torch.tensor(P[:, 0])); self.register_buffer("P1", torch.tensor(P[:, 1])); self.register_buffer("P2", torch.tensor(P[:, 2]))
        self.register_buffer("abs1", torch.tensor((P[:, 1] < 0).astype(np.float32))); self.register_buffer("abs2", torch.tensor((P[:, 2] < 0).astype(np.float32)))
        self.register_buffer("A", torch.tensor(A)); self.Np = len(P)
        self.register_buffer("ones_np", torch.ones(self.Np, 1))
        self.register_buffer("U_lb", torch.tensor(np.triu(np.ones((LB, LB), np.float32))))
        self.register_buffer("jgrid", torch.arange(LC, dtype=torch.float32).view(1, LC))
        self.register_buffer("U_lc", torch.tensor(np.triu(np.ones((LC, LC), np.float32), k=1)))          # strictly upper: exclusive cumsum when used as x @ U
        self.register_buffer("pos", torch.arange(N, dtype=torch.float32).view(1, N))
        self.register_buffer("rgrid", torch.arange(L, dtype=torch.float32).view(L, 1))
        # override tables
        self.tabs = {}
        for k in self.order:
            ent = T["over"][k]; keys = list(ent.keys()); nf = len(keys[0]) if keys else 0
            F = np.array([[v for pair in key_fields(key, idx) for v in pair] for key in keys], np.float32).reshape(len(keys), nf * 2)
            O = np.array([[sid_ for sid_ in ([T["sid"][x] for x in ent[key][0]] + [0] * (2 - len(ent[key][0])))] + [ent[key][1][-1]] for key in keys], np.float32)
            self.register_buffer(f"K_{k}", torch.tensor(F).t().contiguous()); self.register_buffer(f"O_{k}", torch.tensor(O)); self.register_buffer(f"one_{k}", torch.ones(len(keys), 1)); self.tabs[k] = nf
        # sentinel token ids
        s = lambda t: (idx[t] // 128, idx[t] % 128)
        self.register_buffer("sent", torch.tensor([s("^"), s("$"), s("[PAD]"), s("[CLS]"), s("[SEP]")], dtype=torch.float32))

    def fields(self, hi, lo, ex):                                                # compact arrays [LC] -> shifted views (prev, prev2, next, next2, cur) with sentinels
        sp, sn = self.sent[0], self.sent[1]
        def sh(a, k, fill):                                                      # k>0: previous k, k<0: next |k|
            f = fill.reshape(1).expand(abs(k))
            return torch.cat([f, a[:-k]]) if k > 0 else torch.cat([a[-k:], f])
        out = {"cur": (hi, lo)}
        for name, k, fi in (("prev", 1, 0), ("prev2", 2, 0), ("next", -1, 1), ("next2", -2, 1)):
            out[name] = (sh(hi, k, self.sent[fi][0]), sh(lo, k, self.sent[fi][1]))
        return out

    def forward(self, text_bytes):
        b0 = text_bytes.to(torch.float32).reshape(LB); z1, z2 = torch.zeros(1), torch.zeros(2)
        b1 = torch.cat([b0[1:], z1]); b2 = torch.cat([b0[2:], z2])
        m = NEAR(b0.unsqueeze(1), self.P0.unsqueeze(0)) * torch.clamp(NEAR(b1.unsqueeze(1), self.P1.unsqueeze(0)) + self.abs1.unsqueeze(0), max=1.0) \
            * torch.clamp(NEAR(b2.unsqueeze(1), self.P2.unsqueeze(0)) + self.abs2.unsqueeze(0), max=1.0)           # [LB,Np]
        start = (m @ self.ones_np).reshape(LB)                                                                    # 1 at the first byte of each recognised char
        cs = start.unsqueeze(0) @ self.U_lb                                                                       # inclusive cumsum
        ordi = cs.reshape(LB) - 1.0
        Comp = start.unsqueeze(1) * NEAR(ordi.unsqueeze(1), self.jgrid)                                           # [LB,LC]
        attrs = Comp.t() @ (m @ self.A)                                                                           # [LC,6]: hi lo p1 p2 tone valid
        valid = attrs[:, 5]
        nch = valid.sum().reshape(1) if False else (valid.unsqueeze(0) @ torch.ones(LC, 1)).reshape(1)
        hi = valid * attrs[:, 0] + (1 - valid) * self.sent[1][0]; lo = valid * attrs[:, 1] + (1 - valid) * self.sent[1][1]
        base_tone = attrs[:, 4]
        fs = self.fields(hi, lo, None)
        vprev = torch.cat([torch.zeros(1), valid[:-1]]); vnext = torch.cat([valid[1:], torch.zeros(1)])
        tone_enc = lambda t, ok: torch.stack([ok * (100.0 + t + 1.0) + (1 - ok) * 100.0, torch.zeros_like(t)])    # (hi, lo) of a tone key field
        tp = tone_enc(torch.cat([torch.zeros(1), base_tone[:-1]]), vprev); tn = tone_enc(torch.cat([base_tone[1:], torch.zeros(1)]), vnext)
        fs["Tprev"] = (tp[0], tp[1]); fs["Tnext"] = (tn[0], tn[1])
        layout = {"tri": ("prev", "cur", "next"), "quad_l": ("prev2", "prev", "cur"), "quad_r": ("cur", "next", "next2"), "tone3": ("Tprev", "cur", "Tnext"),
                  "tone_r": ("cur", "Tnext"), "bil": ("prev", "cur"), "bir": ("cur", "next")}
        res = attrs[:, 2:5]                                                                                       # default (p1,p2,tone)
        for k in reversed(self.order):
            Kk = getattr(self, f"K_{k}"); eq = None
            for fi, name in enumerate(layout[k]):
                h, l = fs[name]
                e = NEAR(h.unsqueeze(1), Kk[2 * fi].unsqueeze(0)) * NEAR(l.unsqueeze(1), Kk[2 * fi + 1].unsqueeze(0))
                eq = e if eq is None else eq * e                                                                  # [LC,Ek]
            hit = torch.clamp(eq @ getattr(self, f"one_{k}"), max=1.0)                                            # [LC,1]
            res = hit * (eq @ getattr(self, f"O_{k}")) + (1 - hit) * res
        p1, p2, tone = res[:, 0] * valid, res[:, 1] * valid, res[:, 2] * valid
        has2 = (p2 > 0.5).float(); slots = 2.0 * valid * (1.0 + has2)                                              # slots per char (2 per phone: phone + blank)
        off = 3.0 + slots.unsqueeze(0) @ self.U_lc                                                             # exclusive cumsum of slots, start after the 3-slot [0,_,0] head
        off = off.reshape(LC); tot = 3.0 + (slots.unsqueeze(0) @ torch.ones(LC, 1)).reshape(1)
        oh1 = valid.unsqueeze(1) * NEAR(self.pos, off.unsqueeze(1)); oh2 = (valid * has2).unsqueeze(1) * NEAR(self.pos, (off + 2).unsqueeze(1))
        S = float(self.S)
        x = p1.unsqueeze(0) @ oh1 + p2.unsqueeze(0) @ oh2 + S * (NEAR(self.pos, torch.ones(1, 1)) + NEAR(self.pos, tot.reshape(1, 1)))
        tn_ = tone.unsqueeze(0) @ oh1 + tone.unsqueeze(0) @ oh2
        x_len = tot + 2.0
        # BERT rows: row0=[CLS], rows 1..LC = chars, row nch+1 = [SEP], rest [PAD]
        pad_h, pad_l = self.sent[2][0], self.sent[2][1]
        rows_h = torch.cat([self.sent[3][0].reshape(1), valid * attrs[:, 0] + (1 - valid) * pad_h, pad_h.reshape(1).expand(L - LC - 1)])
        rows_l = torch.cat([self.sent[3][1].reshape(1), valid * attrs[:, 1] + (1 - valid) * pad_l, pad_l.reshape(1).expand(L - LC - 1)])
        sep = NEAR(self.rgrid.reshape(L), (nch + 1.0).expand(L)); rows_h = (1 - sep) * rows_h + sep * self.sent[4][0]; rows_l = (1 - sep) * rows_l + sep * self.sent[4][1]
        tmask = (self.rgrid.reshape(L) <= nch + 1.0 + 0.5).float()
        st = torch.cat([torch.zeros(1), off * valid, torch.zeros(L - LC - 1)]); en = torch.cat([torch.full((1,), 3.0), (off + slots) * valid, torch.zeros(L - LC - 1)])
        st = (1 - sep) * st + sep * tot; en = (1 - sep) * en + sep * (tot + 2.0)
        E = (self.pos >= st.unsqueeze(1) - 0.5).float() * (self.pos < en.unsqueeze(1) - 0.5).float()                 # [L,N]
        zl = torch.zeros(1, N)
        return (x.to(torch.int32), tn_.to(torch.int32), zl.to(torch.int32), x_len.to(torch.int32), E, rows_h.reshape(1, L), rows_l.reshape(1, L), tmask.reshape(1, L))
