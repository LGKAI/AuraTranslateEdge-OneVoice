# -*- coding: utf-8 -*-
"""Korean punctuation restoration on the CPU: Horizon-Labs/punctuation-restoration-small (mmBERT-small token classification, Apache-2.0,
assets/ko_punct/model_quantized.onnx 268 MB + tokenizer.json). Labels per word, on the word's last token: 0 . , ? - :
Pure-Python port of the reference (punctuate.py of the model repo): no torch / transformers / tokenizers needed on the box.
  - tokenizer: the BPE of tokenizer.json (Metaspace "▁" before every word, merges by rank, byte fallback <0xXX>); parity with the `tokenizers`
    library is checked by tests/test_punct_ko.py when that library is installed (laptop)
  - existing marks are removed first (except inside numbers: 3.5, 1,000, 10:30), as in the reference
  - Korean policy (conversation.restore_ko): ":" -> "," (NLLB reads "X:" as a dialogue line, as with the Vietnamese colons), "-" dropped,
    a sentence end without a mark gets "."
S2S_KO_PUNCT=1 (default) uses it when the files are present; S2S_KO_PUNCT=0 keeps SenseVoice's own punctuation."""
import json, os, re
from functools import lru_cache
import numpy as np

MARKS = re.compile(r"[.,;:!?。，、；：！？…]")


def strip_marks(t):
    """Remove punctuation marks, except . , : between two digits (3.5, 1,000, 10:30)."""
    return MARKS.sub(lambda m: m.group() if m.group() in ".,:" and 0 < m.start() < len(t) - 1 and t[m.start() - 1].isdigit()
                     and t[m.start() + 1].isdigit() else "", t)


class BpeTokenizer:
    """tokenizer.json of type BPE with Metaspace pre-tokenizer (prepend "▁", split on spaces) and byte fallback, as used by mmBERT (Gemma vocab)."""

    def __init__(self, path):
        t = json.load(open(path, encoding="utf-8")); m = t["model"]
        assert m["type"] == "BPE", m["type"]
        self.vocab = m["vocab"]; self.byte_fallback = bool(m.get("byte_fallback"))
        merges = m.get("merges", [])
        self.ranks = {(tuple(p) if isinstance(p, list) else tuple(p.split(" ", 1))): i for i, p in enumerate(merges)}
        self.unk = self.vocab.get(m.get("unk_token") or "<unk>", 0)
        self.bos, self.eos = self.vocab["<bos>"], self.vocab["<eos>"]

    @lru_cache(maxsize=20000)
    def word(self, w):
        """ids of one pre-tokenized word (already starting with "▁")."""
        syms = []
        for ch in w:
            if ch in self.vocab: syms.append(ch)
            elif self.byte_fallback: syms += [f"<0x{b:02X}>" for b in ch.encode("utf-8")]
            else: syms.append(None)
        while len(syms) > 1:
            best, pos = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best): best, pos = r, i
            if best is None: break
            syms[pos:pos + 2] = [syms[pos] + syms[pos + 1]]
        return tuple(self.vocab.get(s, self.unk) if s is not None else self.unk for s in syms)

    def words(self, text):
        """[(word, ids)] for the space-separated words of text (Metaspace, prepend_scheme=always)."""
        return [(w, self.word("▁" + w)) for w in text.split(" ") if w]


class KoPunct:
    LABELS = ["0", ".", ",", "?", "-", ":"]

    def __init__(self, onnx_path, tokenizer_path, max_tokens=400):
        from . import cpu_ort
        ort = cpu_ort(); so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.tok = BpeTokenizer(tokenizer_path); self.max_tokens = max_tokens

    def labels(self, words):
        """one label per word (label of its last token); long inputs in chunks of whole words."""
        out, i = [], 0
        while i < len(words):
            ids, ends, j = [], [], i
            while j < len(words) and (not ids or len(ids) + len(words[j][1]) <= self.max_tokens):
                ids += words[j][1]; ends.append(len(ids) - 1); j += 1
            x = np.array([[self.tok.bos] + ids + [self.tok.eos]], np.int64)
            lg = self.s.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0, 1:-1]
            pred = lg.argmax(-1)
            out += [self.LABELS[int(pred[e])] for e in ends]; i = j
        return out

    def __call__(self, text):
        """raw model output (reference behaviour for non-CJK scripts): marks after words, '-' as ' -'."""
        text = re.sub(r"\s+", " ", strip_marks(text)).strip()
        if not text: return text
        words = self.tok.words(text); labs = self.labels(words)
        return " ".join(w + ({"0": "", "-": " -"}.get(l, l)) for (w, _), l in zip(words, labs))


_INSTANCE = None


def get(assets_dir):
    """Lazy singleton; None when assets/ko_punct is not in the package (the caller keeps the ASR's own punctuation)."""
    global _INSTANCE
    if _INSTANCE is None:
        d = os.path.join(assets_dir, "ko_punct"); onnx, tok = os.path.join(d, "model_quantized.onnx"), os.path.join(d, "tokenizer.json")
        if not (os.path.exists(onnx) and os.path.exists(tok)): return None
        _INSTANCE = KoPunct(onnx, tok)
    return _INSTANCE
