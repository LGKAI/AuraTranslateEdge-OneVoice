# -*- coding: utf-8 -*-
"""TTS back ends. Each class: synthesize(text) -> (float32 waveform, sample_rate). Long text is split at punctuation / word limits
so every chunk fits the static graph, and chunks are joined with a short pause.
 TtsVi : Piper vi_VN-vais1000 fused graph   text_bytes[1,512] i32 -> audio[1,409600] f32 (22.05 kHz), y_len i32 (samples = y_len*256), n_oov f32
 TtsZh : MeloTTS-ZH + static-table front    text_bytes[1,112] i32 -> audio[1,204800] f32 (44.1 kHz), y_len i32 (samples = y_len*512)
 TtsEnKo: Supertonic-3 fused graph          cp[1,96] i32 (unicode ids), style_dp, style_ttl -> wav[1,460800] f32 (44.1 kHz), lat_len f32 (samples = lat_len*3072)"""
import os, re, unicodedata, numpy as np
from .textnorm import normalize
from .names import VietnameseNames, latin_to_vi

def split_outputs(out):
    """(audio, length, others) from a TTS graph's outputs, independent of their order: the AI Hub inference check on QCS8550 showed the
    MeloTTS context binary returns (y_len, audio) while the ONNX reference returns (audio, y_len). Audio = the largest output; length = the
    first integer-typed scalar (Piper / Melo y_len), else the first scalar (Supertonic's float lat_len)."""
    arrs = [np.asarray(o) for o in out]; ia = max(range(len(arrs)), key=lambda i: arrs[i].size)
    small = [i for i in range(len(arrs)) if i != ia and arrs[i].size == 1]
    ints = [i for i in small if np.issubdtype(arrs[i].dtype, np.integer)]; il = (ints or small)[0]
    return arrs[ia].reshape(-1), float(arrs[il].reshape(-1)[0]), [arrs[i] for i in small if i != il]


SENT = re.compile(r"(?<=[\.\!\?;。！？；\n])\s*")
CLAUSE = re.compile(r"(?<=[,，、:：])\s*")


def _merge(parts, count, max_units, unit):
    out = []; cur = ""
    for p in parts:
        nxt = (cur + (" " if unit == "word" else "") + p) if cur else p
        if cur and count(nxt) > max_units: out.append(cur); cur = p
        else: cur = nxt
    if cur: out.append(cur)
    return out


def split_text(text, max_units, unit="word"):
    """Split into pieces of at most max_units (words or chars): sentence ends first, then clauses, then hard cuts at word/char boundaries."""
    count = (lambda s: len(s.split())) if unit == "word" else (lambda s: len(s))
    pieces = [text.strip()] if text.strip() else []
    for rx in (SENT, CLAUSE):
        nxt = []
        for pc in pieces:
            if count(pc) <= max_units: nxt.append(pc); continue
            parts = [q for q in rx.split(pc) if q.strip()]
            nxt += _merge(parts, count, max_units, unit) if len(parts) > 1 else [pc]
        pieces = nxt
    out = []
    for pc in pieces: out += [pc] if count(pc) <= max_units else _hard(pc, max_units, unit)
    return out


def _hard(text, n, unit):
    if unit == "word":
        w = text.split(); return [" ".join(w[i:i + n]) for i in range(0, len(w), n)]
    return [text[i:i + n] for i in range(0, len(text), n)]


def adaptive(text, max_units, unit, synth, depth=0):
    """Synthesize `text` in pieces of at most max_units; when the graph reports that a piece hit its static audio window (the end would be cut off), halve the
    piece and synthesize the halves. synth(piece) -> (wave, capped)."""
    out = []
    for piece in split_text(text, max_units, unit):
        w, capped = synth(piece)
        n = len(piece.split()) if unit == "word" else len(piece)
        if capped and depth < 4 and n >= 2:
            out += adaptive(piece, max(1, n // 2), unit, synth, depth + 1)
        else: out.append(w)
    return out


def _join(waves, sr, pause=0.18):
    if not waves: return np.zeros(0, np.float32), sr
    gap = np.zeros(int(sr * pause), np.float32); out = []
    for i, w in enumerate(waves): out.append(w); out.append(gap) if i < len(waves) - 1 else None
    return np.concatenate(out).astype(np.float32), sr


class TtsVi:
    sr = 22050; unit, max_units = "word", 26

    def __init__(self, graph, assets_dir=None, glossary=None):
        self.g = graph; self.oov = 0; self.glossary = glossary; self.vn = None
        if assets_dir and os.path.exists(os.path.join(assets_dir, "piper_vi_words.txt")):
            known = set(open(os.path.join(assets_dir, "piper_vi_words.txt"), encoding="utf-8").read().splitlines())
            self.vn = VietnameseNames(os.path.join(assets_dir, "cmudict_min.json"), known)

    def _synth(self, piece):
        b = unicodedata.normalize("NFC", piece).encode("utf-8")[:512]; x = np.zeros((1, 512), np.int32); x[0, :len(b)] = list(b)
        audio, yl, rest = split_outputs(self.g(text_bytes=x)); yl = int(round(yl)); self.oov += int(rest[0].reshape(-1)[0]) if rest else 0
        return audio[: max(yl, 1) * 256].astype(np.float32), yl >= 1533

    def synthesize(self, text):
        if self.vn is not None: text = latin_to_vi(text, self.vn, self.glossary)     # words outside Piper's table would be silently dropped
        text = normalize(text.lower(), "vi")
        return _join(adaptive(text, self.max_units, "word", self._synth), self.sr)


class TtsZh:
    sr = 44100; unit, max_units = "char", 20

    def __init__(self, graph): self.g = graph

    def _synth(self, piece):
        b = unicodedata.normalize("NFC", piece).encode("utf-8")[:112]
        while True:                                            # never cut inside a UTF-8 character
            try: b.decode("utf-8"); break
            except UnicodeDecodeError: b = b[:-1]
        x = np.zeros((1, 112), np.int32); x[0, :len(b)] = list(b)
        audio, yl, _ = split_outputs(self.g(text_bytes=x)); yl = int(round(yl))
        return audio[: max(yl, 1) * 512].astype(np.float32), yl >= 381

    def synthesize(self, text):
        return _join(adaptive(normalize(text, "zh"), self.max_units, "char", self._synth), self.sr)


class TtsEnKo:
    sr = 44100; N = 96

    def __init__(self, graph, lang, assets_dir, voice="M1"):
        import json
        from .vendor.supertonic_text import UnicodeProcessor
        self.g = graph; self.lang = lang; self.p = UnicodeProcessor(os.path.join(assets_dir, "supertonic_unicode_indexer.json"))
        st = np.load(os.path.join(assets_dir, "supertonic_styles.npz")); self.dp = st[voice + "_dp"].astype(np.float32); self.ttl = st[voice + "_ttl"].astype(np.float32)
        self.ok = np.array([0 <= i for i in self.p.indexer])         # characters the model can embed (index -1 = unsupported)

    def _clean(self, s):
        return "".join(c for c in s if ord(c) < len(self.ok) and self.ok[ord(c)])

    def _synth(self, piece):
        s = self._clean(self.p._preprocess_text(piece, self.lang))[: self.N]
        if not s.strip(): return np.zeros(0, np.float32), False
        cp = np.zeros((1, self.N), np.int32); cp[0, :len(s)] = [ord(c) for c in s]
        wav, lat, _ = split_outputs(self.g(cp=cp, style_dp=self.dp, style_ttl=self.ttl)); lat = int(round(lat))
        return wav[: max(lat, 1) * 3072].astype(np.float32), lat >= len(wav) // 3072 - 2

    def synthesize(self, text):
        text = normalize(text, self.lang); units = self.N - 14 if self.lang == "en" else 52            # Korean is spoken slowly: ~52 characters fill the 10 s window
        return _join([w for w in adaptive(text, units, "char", self._synth) if len(w)], self.sr)
