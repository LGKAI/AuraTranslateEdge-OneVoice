# -*- coding: utf-8 -*-
"""Vietnamese punctuation + capitalisation restoration for the Zipformer ASR output (which is lower case, no punctuation), on the CPU with
onnxruntime + numpy only (no torch / transformers on the box).

Model: dragonSwing/vibert-capu (viBERT-base token tagger, GECToR style; CC-BY-SA-4.0), exported to ONNX and int8-quantised
(assets/vi_capu/vibert_capu_int8.onnx, 111 MiB). Labels per word: keep / capitalise / upper / append "," "." ":" "?" ...
This file re-implements the reference wrapper (gec_model.py, utils.py of that repository) without torch:
  * tokenisation = what transformers' BertTokenizerFast actually does there (the wrapper asks for do_basic_tokenize=False, which the fast
    tokenizer ignores): clean text, pad CJK chars, split on whitespace + punctuation, no lower-casing / accent stripping, greedy WordPiece;
  * up to 3 tagging iterations, sentences of < 3 words left to the rule-based vi_punct (the reference returns them unchanged);
  * long inputs split into 48-word chunks with 12 words overlap and merged with difflib, as in the reference.
Measured (text level, BTC sentences with punctuation removed, NLLB-600M): chrF vi->en 52.0 vs 50.4 (vi_punct), vi->ko 28.9 vs 25.9; int8 = fp32 quality.
Parity of this port with the reference implementation: tools/check_punct_vi.py."""
import os, re, unicodedata
from difflib import SequenceMatcher
import numpy as np

START = "$START"
LABELS = ["$KEEP", "$TRANSFORM_CASE_CAPITAL", "$APPEND_,", "$APPEND_.", "$TRANSFORM_VERB_VB_VBN", "$TRANSFORM_CASE_UPPER", "$APPEND_:", "$APPEND_?",
          "$TRANSFORM_VERB_VB_VBC", "$TRANSFORM_CASE_LOWER", "$TRANSFORM_CASE_CAPITAL_1", "$TRANSFORM_CASE_UPPER_-1", "$MERGE_SPACE", "@@UNKNOWN@@", "@@PADDING@@"]
INCORRECT = 1                                                         # d_tags: CORRECT, INCORRECT, @@UNKNOWN@@, @@PADDING@@
PUNC = {":", ".", ",", "?"}


def _ort_session(path, so=None, providers=None):
    """The REAL onnxruntime.InferenceSession. qai_appbuilder replaces the onnxruntime module with its QNN wrapper (breaks plain CPU models: "backend library does not exist"); s2s_edge/__init__ keeps the real one."""
    from . import _ort_real
    return _ort_real.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])


def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126: return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F
            or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """BertNormalizer(clean_text, handle_chinese_chars, no lowercase, no accent strip) + BertPreTokenizer + WordPiece('##', [UNK], 100 chars)."""

    def __init__(self, vocab_path):
        self.vocab = {w: i for i, w in enumerate(open(vocab_path, encoding="utf-8").read().split("\n")) if w}
        self.unk = self.vocab["[UNK]"]; self.start_id = len(self.vocab)          # "$START" was added after the vocabulary (id 38168)

    def _pieces(self, word):
        out = []
        for ch in word:                                                         # normaliser
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"): continue
            out.append(f" {ch} " if _is_cjk(cp) else (" " if ch.isspace() else ch))
        text = "".join(out); toks = []
        for w in text.split():                                                  # pre-tokeniser: whitespace, then every punctuation char alone
            cur = ""
            for ch in w:
                if _is_punct(ch):
                    if cur: toks.append(cur); cur = ""
                    toks.append(ch)
                else: cur += ch
            if cur: toks.append(cur)
        ids = []
        for t in toks:                                                          # WordPiece, greedy longest match
            if len(t) > 100: ids.append(self.unk); continue
            start, sub = 0, []
            while start < len(t):
                end = len(t); cur = None
                while start < end:
                    s = t[start:end] if start == 0 else "##" + t[start:end]
                    if s in self.vocab: cur = self.vocab[s]; break
                    end -= 1
                if cur is None: sub = [self.unk]; break
                sub.append(cur); start = end
            ids.extend(sub)
        return ids

    def encode_words(self, words):
        """-> input_ids, offsets (index of the first sub-token of $START and of every word)."""
        ids, offsets = [self.start_id], [0]
        for w in words:
            p = self._pieces(w)
            if not p: continue                                                  # a word that normalises to nothing gets no sub-token (no word id)
            offsets.append(len(ids)); ids.extend(p)
        return ids, offsets


def _case(token, action):
    if action.endswith("LOWER"): return token.lower()
    if action.endswith("UPPER"): return token.upper()
    if action.endswith("CAPITAL"): return token.capitalize()
    if action.endswith("CAPITAL_1"): return token[0] + token[1:].capitalize()
    if action.endswith("UPPER_-1"): return token[:-1].upper() + token[-1]
    return token


def load_verb_forms(path):
    """verb-form-vocab.txt ('ba_3:VB_VBN' ...): the reference repurposes GECToR's English verb-form labels for Vietnamese number words -> digits."""
    dec = {}
    for line in open(path, encoding="utf-8"):
        if ":" not in line: continue
        words, tags = line.split(":"); w1, w2 = words.split("_"); t1, t2 = tags.split("_"); k = f"{w1}_{t1}_{t2.strip()}"
        if k not in dec: dec[k] = w2
    return dec


def _apply(tokens, edits, verbs=None):
    """utils.get_target_sent_by_edits restricted to the labels this model has (case / verb-form transforms, appends, merge)."""
    t = tokens[:]; shift = 0
    for start, end, label in edits:
        pos = start + shift
        if start < 0: continue
        src = t[pos] if len(t) > pos else ""
        if start == end:                                                       # append
            word = label.replace("$APPEND_", "")
            if (pos < len(t) and t[pos] == word) or (pos > 0 and t[pos - 1] == word): continue
            t[pos:pos] = [word]; shift += 1
        elif label.startswith("$TRANSFORM_CASE"):
            t[pos] = _case(src, label) if src else src
        elif label.startswith("$TRANSFORM_VERB_"):                              # number word -> digits ("bảy" -> "7"), unknown -> unchanged
            w = (verbs or {}).get(f"{src}_{label[len('$TRANSFORM_VERB_'):]}")
            t[pos] = w if w is not None else src
        elif label.startswith("$MERGE_"):
            t[pos + 1: pos + 1] = [label]; shift += 1
    if any(x.startswith("$MERGE_") for x in t):
        if t and t[0].startswith("$MERGE_"): t = t[1:]
        if t and t[-1].startswith("$MERGE_"): t = t[:-1]
        line = " ".join(t).replace(" $MERGE_HYPHEN ", "-").replace(" $MERGE_SPACE ", "")
        t = re.sub(r"([\.\,\?\:]\s+)+", r"\1", line).split()
    return t


def _softmax(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class ViCapu:
    def __init__(self, onnx_path, vocab_path, verb_path=None, threads=4, iterations=3, min_len=3, chunk=48, overlap=12, min_words_cut=6, max_len=64):
        from . import _ort_real as ort
        verb_path = verb_path or os.path.join(os.path.dirname(vocab_path), "verb-form-vocab.txt"); self.verbs = load_verb_forms(verb_path) if os.path.exists(verb_path) else {}
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.s = _ort_session(onnx_path, so, providers=["CPUExecutionProvider"]); self.names = [i.name for i in self.s.get_inputs()]
        self.wp = WordPiece(vocab_path); self.iterations, self.min_len, self.chunk, self.overlap, self.min_words_cut, self.max_len = iterations, min_len, chunk, overlap, min_words_cut, max_len

    def _tag(self, words):
        ids, offs = self.wp.encode_words(words[: self.max_len])
        feed = {"input_ids": np.array([ids], np.int64), "token_type_ids": np.zeros((1, len(ids)), np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
                "input_offsets": np.array([offs], np.int64)}
        lg, dl = self.s.run(None, {n: feed[n] for n in self.names})
        p = _softmax(lg[0]); return p.max(-1), p.argmax(-1)

    def _one(self, words):
        """Up to `iterations` tagging passes on one chunk (handle_batch for a batch of one)."""
        final, prev = words[:], [words[:]]
        for _ in range(self.iterations):
            probs, idx = self._tag(final)
            if idx.max() == 0: break
            edits = []
            for i in range(min(len(final), self.max_len) + 1):
                if i >= len(idx) or idx[i] == 0: continue
                lab = LABELS[idx[i]]
                if lab in ("@@UNKNOWN@@", "@@PADDING@@", "$KEEP"): continue
                s, e = (i, i + 1) if lab.startswith("$TRANSFORM_") else (i + 1, i + 1)
                edits.append((s - 1, e - 1, lab if (lab.startswith("$TRANSFORM_") or lab.startswith("$MERGE_")) else lab[lab.index("_") + 1:]))
            pred = _apply(final, edits, self.verbs)
            if pred == final: break
            stop = pred in prev; final = pred; prev.append(pred)
            if stop: break
        return final

    def _chunks(self, w):
        n, c, o = len(w), self.chunk, self.overlap
        if n <= c: return [w]
        if n < c * 2 - o: k = (n + o + 1) // 2; return [w[:k], w[k - o:]]
        return [w[i: i + c] for i in range(0, n - o, c - o)]

    def _merge(self, tokens, nxt):
        if not tokens: return nxt
        si, ti, st, tt = [], [], [], []; keep = self.overlap - self.min_words_cut; i = 0
        while len(si) < self.overlap and -i < len(tokens):
            i -= 1
            if tokens[i] not in PUNC: si.insert(0, i); st.insert(0, tokens[i].lower())
        i = 0
        while len(ti) < self.overlap and i < len(nxt):
            if nxt[i] not in PUNC: ti.append(i); tt.append(nxt[i].lower())
            i += 1
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, st, tt).get_opcodes():
            if tag == "equal":
                if i1 >= keep: return tokens[: si[i1]] + nxt[ti[j1]:]
                if i2 > keep: return tokens[: si[keep]] + nxt[ti[j2 - i2 + keep]:]
            elif tag == "delete" and i1 == 0: keep += i2 // 2
        raise ValueError("no overlap")                                          # the reference prints and keeps the previous result

    def __call__(self, text):
        words = text.split()
        if len(words) < self.min_len:
            from .textnorm import vi_punct
            return vi_punct(text) if words else text
        parts = [self._one(c) if len(c) >= self.min_len else c for c in self._chunks(words)]
        out = []
        if len(parts) == 1: out = parts[0]
        else:
            for p in parts:
                try: out = self._merge(out, p)
                except Exception: pass
        return re.sub(r"\s+([:.,?])", r"\1", " ".join(out))


_INSTANCE = None


def get(assets_dir):
    """Lazy singleton; None when the model files are not in the package (the caller then falls back to vi_punct)."""
    global _INSTANCE
    if _INSTANCE is None:
        d = os.path.join(assets_dir, "vi_capu"); onnx, vocab = os.path.join(d, "vibert_capu_int8.onnx"), os.path.join(d, "vocab.txt")
        if not (os.path.exists(onnx) and os.path.exists(vocab)): return None
        _INSTANCE = ViCapu(onnx, vocab)
    return _INSTANCE
