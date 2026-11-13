# -*- coding: utf-8 -*-
"""ASR front ends. AsrVi: Zipformer-150M CTC (fbank on host, CTC collapse on host). AsrXx: SenseVoice-Small for en/zh/ko (byte output decoded on host).

Static windows ("buckets"). NPU graphs have fixed input shapes, so with one graph every utterance pays for the whole window (Zipformer 15 s,
SenseVoice 29 s) although most utterances are 2-9 s. Both classes accept several compiled windows of the same model, as a list of
(window, graph), and run the smallest window that holds the utterance; padding is masked inside the encoders (x_lens / speech_lengths), so the
transcript does not change (tools/eval_asr.py, build_zip150_buckets.py --check-wavs). One graph = the previous behaviour.

SenseVoice front end. The deployed frontend graph (fp16 NPU, per-frame scaling to stay in fp16 range) approximates FunASR's WavFrontend and only
exists for the 29 s window. `sv_features` computes the exact training-time features on the host (kaldi fbank on int16-scaled samples, LFR 7/6
with the last LFR frame padded like FunASR, global CMVN of am.mvn) in a few ms; with it the frontend graph and its 464000-sample I/O are skipped."""
import os, re, numpy as np
from .audio import SR

T_FRAMES = 1500                                                  # Zipformer largest static window: 1500 fbank frames = 15 s
PAD_LOGMEL = 0.0                                                 # value of the padded fbank frames of the Zipformer window
SV_FRAMES = 500                                                  # SenseVoice largest static window: 500 LFR frames (60 ms each) = 30 s
LID = {"auto": 0, "zh": 3, "en": 4, "ko": 12}
LFR_M, LFR_N = 7, 6


def _frames(f):
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32) if f.num_frames_ready else np.zeros((0, 80), np.float32)


def _fbank(w):
    import kaldi_native_fbank as knf
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = SR; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = -400.0
    f = knf.OnlineFbank(o); f.accept_waveform(SR, w.tolist()); f.input_finished()
    return _frames(f)


def load_cmvn(path):
    """am.mvn (Kaldi nnet text: <AddShift> means, <Rescale> scales) -> (2, 560) float32, the same parse as FunASR load_cmvn."""
    lines = open(path, encoding="utf-8").read().splitlines(); rows = {}
    for i, l in enumerate(lines):
        t = l.split()
        if t and t[0] in ("<AddShift>", "<Rescale>") and i + 1 < len(lines) and lines[i + 1].split()[:1] == ["<LearnRateCoef>"]:
            rows[t[0]] = np.array(lines[i + 1].split()[3:-1], np.float32)
    return np.stack([rows["<AddShift>"], rows["<Rescale>"]])


def sv_features(w, cmvn):
    """FunASR WavFrontend for SenseVoice (dither 0): kaldi fbank of int16-scaled audio (25/10 ms, hamming, snip_edges) -> LFR(7, 6) -> CMVN.
    Returns (n_lfr, 560) float32, n_lfr = ceil(n_frames / 6) (FunASR pads the last LFR frame with copies of the last fbank frame)."""
    import kaldi_native_fbank as knf
    o = knf.FbankOptions(); o.frame_opts.samp_freq = SR; o.frame_opts.dither = 0.0; o.frame_opts.window_type = "hamming"
    o.frame_opts.snip_edges = True; o.mel_opts.num_bins = 80; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = 0.0
    f = knf.OnlineFbank(o); f.accept_waveform(SR, (np.asarray(w, np.float32) * 32768.0).tolist()); f.input_finished(); m = _frames(f)
    if not len(m): return np.zeros((0, 80 * LFR_M), np.float32)
    n = -(-len(m) // LFR_N); p = np.concatenate([np.repeat(m[:1], (LFR_M - 1) // 2, 0), m, np.repeat(m[-1:], LFR_M, 0)])
    x = p[np.arange(n)[:, None] * LFR_N + np.arange(LFR_M)[None, :]].reshape(n, -1)
    return ((x + cmvn[0]) * cmvn[1]).astype(np.float32)


_GLUED = re.compile(r"\b(at|by|to|till|until|from|before|after|around|about|since|past|is|was)(\d)", re.IGNORECASE)
_HHMM = re.compile(r"\b([01]?\d)([0-5]\d)(\s*)(a\.?m\.?|p\.?m\.?)(?![A-Za-z])", re.IGNORECASE)


def fix_itn(text, lang):
    """Repair SenseVoice inverse text normalisation for English times. "at 5:30 AM" needs two tokens that carry no sound of their own
    (a lone "▁" and ":"); greedy CTC at 60 ms frames drops exactly those, giving "at530 AM" (BTC EV-3775), which MT cannot read.
    Only words that are followed by a time are split from the digits, and only 3-4 digit numbers before AM/PM become h:mm."""
    if lang not in ("en", "auto") or not any(c.isdigit() for c in text): return text      # auto: the patterns only match Latin words / AM-PM
    text = _GLUED.sub(r"\1 \2", text)
    return _HHMM.sub(lambda m: f"{m.group(1)}:{m.group(2)}{m.group(3) or ' '}{m.group(4)}" if int(m.group(1)) <= 12 else m.group(0), text)


def windows(graphs, default):
    """A Graph or a list of (window, Graph) -> list sorted by window."""
    return sorted(graphs, key=lambda x: x[0]) if isinstance(graphs, (list, tuple)) else [(default, graphs)]


def pick(ws, need):
    """Smallest (window, graph) with window >= need; the largest one when nothing is big enough (the caller truncates)."""
    for w, g in ws:
        if w >= need: return w, g
    return ws[-1]


def zip_outputs(out, T, vocab=2000, dim=512):
    """Zipformer graph outputs -> (ctc_logits (1, T', vocab), lens, encoder_out (T', dim) or None), recognised by SIZE, not order or shape:
    a graph of a multi-graph context may come back flat (no per-graph shape query), and the CTC-only graph has no encoder_out.
    T' = ((T - 7) // 2 + 1) // 2 (Conv2dSubsampling then the output downsampling by 2)."""
    tp = ((T - 7) // 2 + 1) // 2; lg = el = eo = None
    for o in out:
        o = np.asarray(o)
        if o.size == 1: el = o
        elif o.size == tp * vocab and lg is None: lg = o.reshape(1, tp, vocab)
        elif o.size == tp * dim: eo = o.reshape(tp, dim)
    if lg is None or el is None: raise ValueError(f"unexpected Zipformer outputs {[np.asarray(o).shape for o in out]} for a {T}-frame window")
    return lg, el, eo


class Transducer:
    """Host-side RNN-T decoding for Zipformer-150M-CR-CTC-RNNT: the model was trained with CR-CTC + CTC + RNN-T losses, and its transducer head is
    the stronger one (icefall / CR-CTC paper: the RNN-T head of a jointly trained model beats its CTC head by ~10 % relative WER), but only the
    CTC head was deployed. Needs the encoder graph to also output "encoder_out" (T', 512, after encoder_proj; build_zip150_buckets.py
    --with-encoder-out). Stateless decoder (context 2) = decoder-*.onnx on the CPU, cached per context; joiner = tanh(enc + dec) @ W + b in numpy
    (~1 MMAC per frame). Greedy search with at most one symbol per frame, as icefall greedy_search / sherpa-onnx."""

    def __init__(self, decoder_onnx, joiner_onnx):
        from . import cpu_ort; ort = cpu_ort()                           # the REAL onnxruntime (qai_appbuilder replaces the module on the box); no `onnx` offline
        so = ort.SessionOptions(); so.log_severity_level = 3; so.intra_op_num_threads = 1
        self.dec = ort.InferenceSession(decoder_onnx, so, providers=["CPUExecutionProvider"]); self.dec_in = self.dec.get_inputs()[0].name
        self.ctx = int(self.dec.get_modelmeta().custom_metadata_map.get("context_size", 2)); self.cache = {}; self.W = self.joi = None
        try:                                                             # numpy joiner (fast) when the onnx package is there to read the weights
            import onnx
            from onnx import numpy_helper
            jm = onnx.load(joiner_onnx); init = {i.name: numpy_helper.to_array(i) for i in jm.graph.initializer}
            gemm = [n for n in jm.graph.node if n.op_type == "Gemm"]
            if len(gemm) == 1 and [n.op_type for n in jm.graph.node if n.op_type != "Constant"][:2] == ["Add", "Tanh"]:
                tb = {a.name: a.i for a in gemm[0].attribute}.get("transB", 0); W = init[gemm[0].input[1]].astype(np.float32)
                self.W = np.ascontiguousarray(W.T if tb else W); self.b = init[gemm[0].input[2]].astype(np.float32)
        except ImportError:
            pass
        if self.W is None:                                               # otherwise the joiner graph itself on onnxruntime
            self.joi = ort.InferenceSession(joiner_onnx, so, providers=["CPUExecutionProvider"]); self.joi_in = [i.name for i in self.joi.get_inputs()]

    def dec_out(self, ctx):
        k = tuple(ctx)
        if k not in self.cache:
            if len(self.cache) > 4096: self.cache.clear()
            self.cache[k] = self.dec.run(None, {self.dec_in: np.array([k], np.int64)})[0][0]
        return self.cache[k]

    def join(self, e, d):
        if self.W is not None: return np.tanh(e + d) @ self.W + self.b
        return self.joi.run(None, {self.joi_in[0]: e, self.joi_in[1]: np.ascontiguousarray(np.broadcast_to(d, e.shape), np.float32)})[0]

    def greedy(self, enc, blank=0, unk=2, block=8, return_frames=False):
        """enc: (T', 512) -> token ids (or (token_ids, emission_frames) when return_frames=True). Frames are scored in blocks of `block` with the
        current decoder output: most frames are blank, so a block is one matmul; after a symbol, the block restarts at the next frame with the
        new decoder output. Start context [-1, blank] and <unk> treated as blank, as icefall greedy_search / sherpa-onnx (the decoder maps -1 to
        a zero embedding, the training left padding)."""
        hyp = [-1] * (self.ctx - 1) + [blank]; d = self.dec_out(hyp[-self.ctx:]); t = 0; T = len(enc); enc = np.asarray(enc, np.float32)
        out, frames = [], []
        while t < T:
            ids = self.join(enc[t:t + block], d).argmax(-1); nb = np.flatnonzero((ids != blank) & (ids != unk))
            if not len(nb): t += block; continue
            j = int(nb[0]); hyp.append(int(ids[j])); out.append(int(ids[j])); frames.append(t + j); d = self.dec_out(hyp[-self.ctx:]); t += j + 1
        return (out, frames) if return_frames else out


def _word_ends(sp, pieces, frames):
    """End frame index (40 ms Zipformer encoder frames) for each whitespace-delimited word in `sp.decode(pieces).lower().strip()`."""
    ends, prev = [], ""
    for i, fr in enumerate(frames):
        cur = sp.decode(pieces[: i + 1]).lower().strip()
        if cur == prev: continue
        prev = cur; nw = len(cur.split())
        if nw > len(ends): ends.extend([int(fr)] * (nw - len(ends)))
        elif ends: ends[-1] = int(fr)
    return ends


class AsrVi:
    def __init__(self, graph, bpe_path, known_words=None, transducer=None):
        import sentencepiece as spm
        self.ws = windows(graph, T_FRAMES); self.g = self.ws[-1][1]; self.max_s = self.ws[-1][0] / 100.0 - 0.5
        self.sp = spm.SentencePieceProcessor(model_file=bpe_path); self.known = known_words; self.stats = {}; self.last = {}; self.rnnt = transducer

    def __call__(self, wav):
        if len(wav) < int(0.1 * SR):                                             # under 100 ms: no fbank frames, nothing to recognise
            self.stats = dict(conf_nb=0.0, nb_ratio=0.0, conf_all=0.0, blank_p=1.0, known=0.0, nwords=0); self.last = {}; return ""
        f = _fbank(wav[: int(self.max_s * SR)]); T, g = pick(self.ws, len(f)); n = min(len(f), T); x = np.full((1, T, 80), PAD_LOGMEL, np.float32); x[0, :n] = f[:n]
        lg, el, eo = zip_outputs(g(x=x, x_lens=np.array([n], np.int64)), T); self.last = dict(window=T, frames=n)
        real = int(el.reshape(-1)[0]); lgr = lg[0, :real].astype(np.float64); ids = lgr.argmax(-1)
        if self.rnnt is not None and eo is not None:
            pieces, tok_frames = self.rnnt.greedy(eo[:real], return_frames=True); self.last["decode"] = "rnnt"
        else:
            if self.rnnt is not None and not getattr(self, "_warned", False):   # decoder/joiner configured but the graph is the old CTC-only one
                import sys; self._warned = True
                print(f"[s2s] asr_vi: the {T}-frame graph has no encoder_out -> CTC greedy (rebuild with build_zip150_buckets.py --with-encoder-out)", file=sys.stderr)
            pieces, tok_frames, prev = [], [], None
            for fr, t in enumerate(ids):
                if t != 0 and t != prev: pieces.append(int(t)); tok_frames.append(fr)
                prev = t
            self.last["decode"] = "ctc"
        text = self.sp.decode(pieces).lower().strip()
        self.last["word_ends"] = _word_ends(self.sp, pieces, tok_frames)
        e = np.exp(lgr - lgr.max(-1, keepdims=True)); s = e.sum(-1); pmax = 1.0 / s; nb = ids != 0                  # confidence features for language identification
        words = text.split()                                                     # (max softmax prob = 1 / sum exp(l - max); no full softmax)
        self.stats = dict(conf_nb=float(pmax[nb].mean()) if nb.any() else 0.0, nb_ratio=float(nb.mean()) if real else 0.0, conf_all=float(pmax.mean()) if real else 0.0,
                          blank_p=float((e[:, 0] / s).mean()) if real else 1.0,
                          known=float(np.mean([w in self.known for w in words])) if (words and self.known) else 0.0, nwords=len(words))
        return text

    def warmup(self):
        """One call per window (the first call of a graph pays for its allocation / HTP setup)."""
        for T, _ in self.ws: self(np.full(int(T * SR / 100) - 400, 1e-4, np.float32))


class AsrXx:
    """SenseVoice-Small (en / zh / ko). As deployed on the NPU: frontend graph (wav -> fbank) + encoder+CTC graph (fbank -> UTF-8 bytes);
    or one combined graph (frontend=combined, encoder=None); or host features (frontend=None, cmvn from am.mvn) + encoder window(s).
    language ids: zh 3, en 4, ko 12; textnorm 14 = with inverse text normalisation/punctuation."""
    max_s = 28.5

    def __init__(self, frontend, encoder, lang, cmvn=None):
        self.fe = frontend; self.lang = lang; self.cmvn = cmvn; self.last = {}
        self.ws = windows(encoder, SV_FRAMES) if encoder is not None else []
        self.enc = self.ws[-1][1] if self.ws else None
        if self.fe is None and (self.enc is None or self.cmvn is None): raise ValueError("SenseVoice: host features need encoder graph(s) and am.mvn")
        if self.fe is None: self.max_s = min(28.5, self.ws[-1][0] * LFR_N * 0.01 - 0.5)

    def __call__(self, wav):
        if len(wav) < int(0.1 * SR): self.last = {}; return ""
        lid = np.array([LID[self.lang]], np.int32); tn = np.array([14], np.int32)
        if self.fe is None:                                               # host features: exact FunASR front end, no frontend graph
            x = sv_features(wav[: int(self.max_s * SR)], self.cmvn); L, g = pick(self.ws, len(x)); n = min(len(x), L)
            big = np.zeros((1, L, 560), np.float32); big[0, :n] = x[:n]; sl = np.array([n], np.int32)
            o = g(fbank=big, speech_lengths=sl, language=lid, textnorm=tn); a = np.asarray(o[0]); self.last = dict(window=L, frames=n)
        elif self.enc is None:                                            # combined graph
            n = min(len(wav), 464000); b = np.zeros((1, 464000), np.float32); b[0, :n] = wav[:n]
            o = self.fe(wav=b, wav_len=np.array([n], np.int32), language=lid, textnorm=tn); a = np.asarray(o[0]); self.last = dict(window=SV_FRAMES)
        else:                                                             # frontend graph (29 s) + the smallest encoder window that fits
            n = min(len(wav), 464000); b = np.zeros((1, 464000), np.float32); b[0, :n] = wav[:n]
            f = self.fe(wav=b, wav_len=np.array([n], np.int32)); fb = [np.asarray(x) for x in f]
            big = [x for x in fb if x.size > 1000][0].reshape(1, SV_FRAMES, 560).astype(np.float32); sl = [x for x in fb if x.size == 1][0].reshape(1).astype(np.int32)
            L, g = pick(self.ws, int(sl[0])); sl = np.minimum(sl, L).astype(np.int32)
            o = g(fbank=np.ascontiguousarray(big[:, :L]), speech_lengths=sl, language=lid, textnorm=tn); a = np.asarray(o[0]); self.last = dict(window=L, frames=int(sl[0]))
        text = a.astype(np.int64).reshape(-1).astype(np.uint8).tobytes().replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
        return fix_itn(text, self.lang)

    def warmup(self):
        """One call per encoder window (host features or frontend graph), or one call of the combined graph."""
        if not self.ws: self(np.full(SR, 1e-4, np.float32)); return
        for L, _ in self.ws: self(np.full(min(max(int((L * LFR_N - 4) * 160) + 240, 1600), 464000), 1e-4, np.float32))
