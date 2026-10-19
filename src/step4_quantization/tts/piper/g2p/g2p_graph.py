# -*- coding: utf-8 -*-
"""Vietnamese text -> Piper phoneme ids as a STATIC tensor graph (NPU-friendly).

Input : UTF-8 bytes of the (NFC) sentence, zero padded, int32 [1, LB]
Output: x [1,512] int32 (Piper encoder input incl. BOS/pad/EOS), x_lengths [1] int32, n_oov [1] float

Only Sub/Abs/Less/MatMul/Mul/Add/Clip ops with small exact integers (<2048 -> exact in fp16).
No string ops, no Gather/NonZero/ScatterND/Loop. Everything is a constant table lookup:
espeak-ng (vi) phonemes are context free per written syllable (verified: 0 conflicts / 2070 syllables),
so  token -> phoneme-id string  is a dictionary lookup, implemented as a one-hot match matmul.
"""
import json, re, unicodedata, os, sys
import numpy as np
import torch
import torch.nn as nn

LB, LC, NT, S, NPH, XL = 512, 256, 128, 10, 254, 512
NFC = lambda s: unicodedata.normalize("NFC", s)
VOW12 = "aăâeêioôơuưy"
TONES = ["", "̀", "́", "̉", "̃", "̣"]
PUNCT_CLASSES = [".", ",", "!", "?", ";", ":"]
PUNCT_ALIAS = {"–": ";", "—": ";"}          # espeak reads en/em dash as ';'
HERE = os.path.dirname(os.path.abspath(__file__))


def letter_list():
    L = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["đ"]
    for v in VOW12:
        for t in TONES:
            ch = NFC(v + t)
            if ch not in L:
                L.append(ch)
    return L


def utf8(ch):
    b = list(ch.encode("utf-8")) + [-1, -1]
    return b[:3]


def build_patterns():
    """classes: letters..., punct... ; patterns: (B0,B1,B2) -> class index"""
    letters = letter_list()
    classes = letters + PUNCT_CLASSES
    pats, cls = [], []
    for ci, ch in enumerate(classes):
        vs = {ch, ch.upper()} if ch.isalpha() else {ch}
        if ci >= len(letters):
            vs |= {a for a, b in PUNCT_ALIAS.items() if b == ch}
        for v in sorted(vs):
            pats.append(utf8(NFC(v))); cls.append(ci)
    return letters, classes, np.array(pats, np.float32), np.array(cls)


def token_key_chars(tok, letters, classes):
    """tok string (lowercase letters or one punct or ' ') -> list of class indices (len<=S) or None"""
    out = []
    for ch in NFC(tok):
        if ch == " ":
            out.append(len(classes))            # separator channel
        elif ch in classes:
            out.append(classes.index(ch))
        else:
            return None
    return out if 0 < len(out) <= S else None


def build_tables(vocab, phon_of, classes, letters, id_map):
    """vocab: list of tokens; phon_of(tok)->list of phoneme ids"""
    C = len(classes)
    Cp = C + 2                                    # classes + separator + pad
    keys, phs = [], []
    for tok in vocab:
        k = token_key_chars(tok, letters, classes)
        if k is None:
            continue
        ids = phon_of(tok)
        if not ids:
            continue
        keys.append(k); phs.append(ids)
    return keys, phs, Cp


def make_W(keys, Cp, C):
    V = len(keys)
    W = np.zeros((S * Cp, V), np.float16)
    for v, k in enumerate(keys):
        for p in range(S):
            c = k[p] if p < len(k) else Cp - 1     # pad channel
            W[p * Cp + c, v] = 1
    return W


class VietG2P(nn.Module):
    def __init__(self, letters, classes, pats, pcls, keys, phs, id_sp=3):
        super().__init__()
        C = len(classes); self.C = C; self.Cp = C + 2
        KP = max(len(p) for p in phs); self.KP = KP
        V = len(keys); self.V = V
        npat = len(pats)
        M = np.zeros((npat, C), np.float32); M[np.arange(npat), pcls] = 1
        self.register_buffer("B0", torch.tensor(pats[:, 0])); self.register_buffer("B1", torch.tensor(pats[:, 1]))
        self.register_buffer("B2", torch.tensor(pats[:, 2]))
        self.register_buffer("abs1", torch.tensor((pats[:, 1] < 0).astype(np.float32)))
        self.register_buffer("abs2", torch.tensor((pats[:, 2] < 0).astype(np.float32)))
        self.register_buffer("M", torch.tensor(M))
        self.nl = len(letters)
        U = np.triu(np.ones((LB, LB), np.float32)); self.register_buffer("U_lb", torch.tensor(U))
        self.register_buffer("U_lc", torch.tensor(np.triu(np.ones((LC, LC), np.float32))))
        self.register_buffer("jgrid", torch.arange(LC, dtype=torch.float32))
        self.register_buffer("tgrid", torch.arange(NT, dtype=torch.float32))
        self.register_buffer("pgrid", torch.arange(S, dtype=torch.float32))
        self.register_buffer("lcgrid", torch.arange(LC, dtype=torch.float32).view(1, LC))
        W = make_W(keys, self.Cp, C)
        self.register_buffer("W", torch.tensor(W.astype(np.float32)))
        PH = np.zeros((V, KP), np.float32)
        for v, p in enumerate(phs):
            PH[v, :len(p)] = p
        self.register_buffer("PH", torch.tensor(PH))
        NK = NT * KP
        self.register_buffer("U_nk", torch.tensor(np.triu(np.ones((NK, NK), np.float32))))
        self.register_buffer("qgrid", torch.arange(NPH, dtype=torch.float32).view(1, NPH))
        place = np.zeros((NPH, XL), np.float32)
        for q in range(NPH):
            place[q, 2 + 2 * q] = 1
        self.register_buffer("Place", torch.tensor(place))
        self.register_buffer("xgrid", torch.arange(XL, dtype=torch.float32).view(1, XL))
        bos = np.zeros((1, XL), np.float32); bos[0, 0] = 1
        self.register_buffer("BOS", torch.tensor(bos))
        self.sep_phone = float(id_sp)

    @staticmethod
    def near(a, b):
        return (torch.abs(a - b) < 0.5).float()

    def forward(self, text_bytes):
        C, Cp, KP = self.C, self.Cp, self.KP
        b0 = text_bytes.to(torch.float32).reshape(LB)
        z1 = torch.zeros(1); z2 = torch.zeros(2)
        b1 = torch.cat([b0[1:], z1]); b2 = torch.cat([b0[2:], z2])
        m0 = self.near(b0.unsqueeze(1), self.B0.unsqueeze(0))
        m1 = torch.clamp(self.near(b1.unsqueeze(1), self.B1.unsqueeze(0)) + self.abs1.unsqueeze(0), max=1.0)
        m2 = torch.clamp(self.near(b2.unsqueeze(1), self.B2.unsqueeze(0)) + self.abs2.unsqueeze(0), max=1.0)
        ch_all = (m0 * m1 * m2) @ self.M                                   # [LB, C]
        is_start = (b0 > 0.5).float() * ((b0 < 127.5).float() + (b0 > 191.5).float())
        sep = (is_start - ch_all.sum(1)).clamp(min=0.0)
        ch_ext = torch.cat([ch_all, sep.unsqueeze(1)], dim=1)               # [LB, C+1]
        # compact: drop UTF-8 continuation bytes
        cum_start = is_start.unsqueeze(0) @ self.U_lb                       # [1,LB]
        ordi = cum_start.reshape(LB) - 1.0
        Cmat = is_start.unsqueeze(1) * self.near(ordi.unsqueeze(1), self.lcgrid)   # [LB, LC]
        ch = Cmat.t() @ ch_ext                                              # [LC, C+1]
        valid = ch.sum(1)
        let = ch[:, :self.nl].sum(1)
        pun = ch[:, self.nl:C].sum(1)
        sp = ch[:, C]
        let_prev = torch.cat([torch.zeros(1), let[:-1]])
        let_next = torch.cat([let[1:], torch.zeros(1)]); sp_next = torch.cat([sp[1:], torch.zeros(1)])
        nonsp = let + pun
        cumn = (nonsp.unsqueeze(0) @ self.U_lc).reshape(LC)
        # one space token per separator run, only if a LETTER directly follows the run (espeak: no space before punctuation / at the end)
        sp_start = sp * (1.0 - sp_next) * let_next * (cumn > 0.5).float()
        start = let * (1.0 - let_prev) + pun + sp_start
        belongs = let + pun + sp_start
        tok_idx = (start.unsqueeze(0) @ self.U_lc).reshape(LC) - 1.0
        OH = belongs.unsqueeze(1) * self.near(tok_idx.unsqueeze(1), self.tgrid.unsqueeze(0))   # [LC, NT]
        start_idx = ((OH * (start * self.jgrid).unsqueeze(1)).sum(0))                          # [NT]
        pos = self.jgrid - (OH * start_idx.unsqueeze(0)).sum(1)                                # [LC]
        POS1H = belongs.unsqueeze(1) * self.near(pos.unsqueeze(1), self.pgrid.unsqueeze(0))    # [LC, S]
        A3 = (OH.unsqueeze(2) * POS1H.unsqueeze(1)).reshape(LC, NT * S)
        F = A3.t() @ ch                                                                       # [NT*S, C+1]
        padc = (1.0 - F.sum(1, keepdim=True)).clamp(min=0.0)
        K = torch.cat([F, padc], dim=1).reshape(NT, S * Cp)
        length = OH.sum(0)                                                                     # [NT]
        ok = (length > 0.5).float() * (length < S + 0.5).float()
        score = K @ self.W                                                                     # [NT, V]
        match = (score > (S - 0.5)).float() * ok.unsqueeze(1)
        phon = match @ self.PH                                                                 # [NT, KP]
        n_oov = ((length > 0.5).float() * (1.0 - match.sum(1)).clamp(min=0.0)).sum().reshape(1)
        valid_ph = (phon > 0.5).float().reshape(1, NT * KP)
        posk = (valid_ph @ self.U_nk).reshape(NT * KP) - 1.0
        PM = valid_ph.reshape(NT * KP, 1) * self.near(posk.unsqueeze(1), self.qgrid)           # [NT*KP, NPH]
        seq = phon.reshape(1, NT * KP) @ PM                                                    # [1, NPH]
        n = torch.clamp(valid_ph.sum(), max=float(NPH))
        x = seq @ self.Place + self.BOS * 1.0
        eos = self.near(self.xgrid, (2.0 + 2.0 * n).reshape(1, 1)) * 2.0
        x = x + eos
        x_len = (3.0 + 2.0 * n).reshape(1)
        return x.to(torch.int32), x_len.to(torch.int32), n_oov
