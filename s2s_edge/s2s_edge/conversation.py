# -*- coding: utf-8 -*-
"""Multilingual conversation: every utterance is routed on its own, so speakers may switch language between turns.
 pair mode : two languages (one is Vietnamese): vi -> X and X -> vi.   e.g. --pair vi,ko
 auto mode : Vietnamese plus any of en / zh / ko; the partner language is detected from the other speaker (sticky, default `partner`).
Language identification per utterance (cheap, no extra model):
 1. the Vietnamese CTC model always runs first (it is the cheapest ASR): high confidence and mostly valid Vietnamese words  =>  Vietnamese;
 2. otherwise, with one candidate left that language is used; with several (auto mode) SenseVoice runs in `auto` language mode and the script of its
    text decides (Hangul -> ko, Han -> zh, Latin -> en)."""
import os, re, time
from bisect import bisect_right
from difflib import SequenceMatcher
import numpy as np
from .audio import SR, normalize_level, SEG_MAX_S, segment_with_breaks
from .asr import AsrXx
from .names import NameTransliterator
from .textnorm import vi_punct
from . import neural_textnorm
from .asr_clean import clean_asr, clean_mt
from .ko_polite import polite as ko_polite
MT_SPLIT = os.environ.get("S2S_MT_SPLIT", "1") != "0"

# Punctuation/casing for the Vietnamese ASR text (Zipformer outputs lower case, no punctuation) before MT:
#   S2S_VI_PUNCT=model : (default) vibert-capu int8 on the CPU (s2s_edge/punct_vi.py), falls back to the rule when assets/vi_capu is missing.
#                        Measured 11/11, NLLB chrF on real ASR transcripts (30 clips each): vi->ko 34.1 vs 25.6 (rule), vi->zh 36.3 vs 31.0,
#                        vi->en 55.1 vs 55.6; on 280 gold sentences without punctuation: vi->en 52.0 vs 50.4, vi->ko 28.9 vs 25.9.
#   S2S_VI_PUNCT=1     : rule-based vi_punct (final "?" / "."),   S2S_VI_PUNCT=0 : off
VI_PUNCT_MODE = os.environ.get("S2S_VI_PUNCT", "model").lower()
VI_PUNCT = VI_PUNCT_MODE != "0"


def _align_breaks(src_words, dst_words, breaks):
    """Map 1-indexed word boundary positions across tokenization/edit differences (e.g. ViCapu $MERGE_SPACE or glossary _asr_fix)."""
    if not breaks: return []
    if len(src_words) == len(dst_words): return list(breaks)
    norm = lambda ws: [re.sub(r"\W+", "", w).lower() for w in ws]
    sm = SequenceMatcher(None, norm(src_words), norm(dst_words)); mp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1): mp[i1 + d + 1] = j1 + d + 1
        elif i2 > i1 and j2 > j1:
            mp[i2] = j2
    return sorted({mp[k] for k in breaks if k in mp and 0 < mp[k] < len(dst_words)})


def _break_positions(breaks, word_ends, frame_s=0.04):
    """Map pause sample offsets (relative to segment start) to 1-indexed word boundary positions using Zipformer encoder frames."""
    if not breaks or not word_ends: return []
    n = len(word_ends); pos = set()
    for s in breaks:
        k = bisect_right(word_ends, s / (frame_s * SR))
        if 0 < k < n: pos.add(k)
    return sorted(pos)


def restore_vi(text, breaks=None):
    if VI_PUNCT_MODE == "model":
        from .punct_vi import get
        from .registry import ASSETS
        m = get(ASSETS)
        if m is not None:
            out = m(text)
            if breaks: breaks = _align_breaks(text.split(), out.split(), breaks)
            return vi_punct(out, breaks=breaks)
    return vi_punct(text, breaks=breaks)

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ]"); HAN = re.compile(r"[一-鿿]"); LATIN = re.compile(r"[A-Za-z]")


def script_lang(text):
    h, c, l = len(HANGUL.findall(text)), len(HAN.findall(text)), len(LATIN.findall(text))
    if h + c + l == 0: return None
    return "ko" if h >= max(c, l) else ("zh" if c >= l else "en")


class Engine:
    """Shared models for all directions (loaded lazily): ASR per language, one MT, TTS per language."""

    def __init__(self, registry, denoise=False):
        self.r = registry; self._asr = {}; self._tts = {}; self._mt = None; self.names = None; self.denoise = denoise; self._den = None
        prof = getattr(registry, "profile", None)
        if getattr(registry, "source_mode", None) is not None:
            self.source_mode = str(registry.source_mode)
        elif isinstance(prof, dict) and prof.get("source_mode") is not None:
            self.source_mode = str(prof["source_mode"])
        else:
            self.source_mode = None

    def prep(self, wav):
        """Level normalisation (+ optional GTCRN speech enhancement on the CPU) before ASR."""
        if self.denoise:
            if self._den is None:
                import os
                from .denoise import Gtcrn
                from .registry import ASSETS
                self._den = Gtcrn(os.path.join(ASSETS, "gtcrn_dns3.onnx"))
            wav = self._den(wav)
        return normalize_level(wav)

    def asr(self, lang):
        if lang not in self._asr: self._asr[lang] = self.r.asr(lang)
        return self._asr[lang]

    def tts(self, lang):
        if lang not in self._tts: self._tts[lang] = self.r.tts(lang)
        return self._tts[lang]

    @property
    def mt(self):
        if self._mt is None: self._mt = self.r.mt(); self.names = NameTransliterator(self._mt, self.r.glossary)
        return self._mt

    def graphs_for(self, src, tgt): return [self.r.graph(k) for k in self.r.components_for(src, tgt)]

    def gtime(self, graphs): return sum(g.total_s for g in graphs), sum(g.calls for g in graphs)

    def _asr_stage(self, wav, src, text=None):
        asr = self.asr(src)
        if text is None: text = asr(wav)
        raw_words = text.split(); we = getattr(asr, "last", {}).get("word_ends") if src == "vi" else None
        word_ends = we if (we and len(we) == len(raw_words)) else None
        for pat, rep in (self.r.glossary.get("_asr_fix", {}).get(src, {}) or {}).items():      # domain fixes: how the ASR (mis)spells a term -> the term
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text, raw_words, word_ends

    def neural_enabled(self):
        if self.source_mode is not None:
            want = self.source_mode == "1"
        elif getattr(self.r, "source_mode", None) is not None:
            want = str(self.r.source_mode) == "1"
        else:
            want = neural_textnorm.enabled()
        if want and not neural_textnorm.usable()[0]:                  # e.g. on the AIBOX 8550: no MLX -> vi punctuation / plain MT path
            neural_textnorm.warn_unusable("source_mode=1"); return False
        return want

    def _punct_stage(self, text, src, raw_words, word_ends, breaks, min_chars=2):
        """-> (text, parts). parts is the list of MT units from the neural source normalizer (source_mode=1), else None."""
        if len(re.sub(r"[\W_]+", "", text)) < min_chars: return "", None                           # nothing intelligible (noise, breath, a lone "I.")
        if src in ("vi", "en") and self.neural_enabled():
            parts = neural_textnorm.get().prepare_source(text, src, self.mt.S - 2, self.mt.encode)
            return " ".join(parts), parts
        if src == "vi" and VI_PUNCT:
            wb = _break_positions(breaks, word_ends)
            if wb and text.split() != raw_words: wb = _align_breaks(raw_words, text.split(), wb)
            text = restore_vi(text, breaks=wb)                                             # the CTC/RNN-T model outputs no punctuation: restore "?" so MT keeps questions
        return text, None

    def prepare_text(self, text, src):
        """Plan-01 API (tools/evaluate_asr_integration.py): source preparation without pause hints."""
        return self._punct_stage(text, src, text.split(), None, None) if text else (text, None)

    def translate_text_tagged(self, text, src, tgt, parts=None):
        """Text -> (target text, tags): the one MT path of every caller (file, live, stream, web batch, typed text, web demo).
        Units: `parts` from the neural source normalizer (source_mode=1, translated atomically), else split_sentences (the same splitter the MT class
        uses, so a lone "Vâng." stays attached to its neighbour; S2S_MT_SPLIT=0 skips it). Per unit: MT -> clean_mt (block symbols, copied "问:" label,
        leading dash, "。" in vi/en) -> Latin-name transliteration (zh/ko) -> Korean politeness (S2S_KO_POLITE=0 disables)."""
        if not text or not text.strip(): return "", []
        if parts is not None: units, run = list(parts), (lambda p: self.mt.translate_atomic(p, src, tgt))
        else: units, run = (split_sentences(text, src) if MT_SPLIT else [text]), (lambda p: self.mt(p, src, tgt))
        outs, tags = [], []
        for p in units:
            m, tg = clean_mt(run(p), p, tgt)
            if tgt in ("zh", "ko") and m and self.names: m = self.names(m, tgt)
            if tgt == "ko" and m: m = ko_polite(m)
            if m: outs.append(m)
            tags += [x for x in tg if x not in tags]
        if len(units) > 1: tags.append("mt_split")
        return ("" if tgt == "zh" else " ").join(outs), tags

    def translate_text(self, text, src, tgt, parts=None):
        """Plan-01 API: target text only (see translate_text_tagged)."""
        return self.translate_text_tagged(text, src, tgt, parts)[0]

    def translate_segment(self, wav, src, tgt, text=None, prepped=False, breaks=None):
        """text: pre-computed ASR text (language identification already ran the ASR)."""
        wav = wav if prepped else self.prep(wav); gr = self.graphs_for(src, tgt); g0, c0 = self.gtime(gr); t = {}; a = time.perf_counter()
        text, raw_words, word_ends = self._asr_stage(wav, src, text)
        b = time.perf_counter(); t["asr_s"] = b - a
        asr_raw = text; text, clean_tags = clean_asr(text, src)                               # stray label ("Q. "), leading "Um.", stuck repeats, nothing intelligible (noise, breath, a lone "I.")
        asr_text = text
        text, prepared = self._punct_stage(text, src, raw_words, word_ends, breaks, min_chars=1) if text else ("", None)   # clean_asr already dropped junk; "好。" is a real answer
        t["source_normalize_s"] = time.perf_counter() - b
        mt_text, mt_tags = self.translate_text_tagged(text, src, tgt, prepared)              # sentence by sentence + output guard + names + ko politeness
        c = time.perf_counter(); t["mt_s"] = c - b - t["source_normalize_s"]; tts = self.tts(tgt)
        out, sr = tts.synthesize(mt_text) if mt_text.strip() else (np.zeros(0, np.float32), tts.sr); d = time.perf_counter(); t["tts_s"] = d - c
        g1, c1 = self.gtime(gr); t["total_s"] = d - a; t["graph_s"] = g1 - g0; t["graph_calls"] = c1 - c0; t["host_s"] = t["total_s"] - t["graph_s"]
        t["audio_in_s"] = len(wav) / SR; t["audio_out_s"] = len(out) / sr
        return dict(src=src, tgt=tgt, asr=text, asr_before_normalization=asr_text, asr_raw=asr_raw, asr_clean=clean_tags + mt_tags, mt=mt_text,
                    tts_text=getattr(tts, "tts_input_text", mt_text) if mt_text.strip() else "", tts_format=getattr(tts, "format_trace", {}) if mt_text.strip() else {},
                    wav=out, sr=sr, timing=t)

    def translate_segment_stream(self, wav, src, tgt, text=None, prepped=False, breaks=None):
        """Pipelined version for live output: ASR once, split the (punctuated) transcript into sentences, then MT + TTS sentence by sentence and
        yield each result as soon as its audio is ready, so playback of sentence 1 overlaps MT/TTS of sentence 2 (the NPU runs graphs one at a
        time, so this saves waiting time, not compute). Same fields as translate_segment, plus "part" = (index, count)."""
        wav = wav if prepped else self.prep(wav); gr = self.graphs_for(src, tgt); a = time.perf_counter()
        text, raw_words, word_ends = self._asr_stage(wav, src, text)
        asr_s = time.perf_counter() - a
        asr_raw = text; text, clean_tags = clean_asr(text, src)
        text, prepared = self._punct_stage(text, src, raw_words, word_ends, breaks, min_chars=1) if text else ("", None)   # clean_asr already dropped junk; "好。" is a real answer
        if not text: return
        parts = prepared if prepared is not None else split_sentences(text, src); tts = self.tts(tgt)
        for i, sent in enumerate(parts):
            g0, c0 = self.gtime(gr); b = time.perf_counter()
            mt_text, mt_tags = self.translate_text_tagged(sent, src, tgt, [sent] if prepared is not None else None)
            c = time.perf_counter(); out, sr = tts.synthesize(mt_text) if mt_text.strip() else (np.zeros(0, np.float32), tts.sr); d = time.perf_counter()
            g1, c1 = self.gtime(gr); t = dict(asr_s=asr_s if i == 0 else 0.0, mt_s=c - b, tts_s=d - c, total_s=(d - a) if i == 0 else (d - b), graph_s=g1 - g0, graph_calls=c1 - c0)
            t["host_s"] = t["total_s"] - t["graph_s"]; t["audio_in_s"] = len(wav) / SR if i == 0 else 0.0; t["audio_out_s"] = len(out) / sr
            yield dict(src=src, tgt=tgt, asr=sent, asr_raw=asr_raw, asr_clean=clean_tags + mt_tags, mt=mt_text,
                       tts_text=getattr(tts, "tts_input_text", mt_text) if mt_text.strip() else "", tts_format=getattr(tts, "format_trace", {}) if mt_text.strip() else {},
                       wav=out, sr=sr, timing=t, part=(i, len(parts)))


from .mt import split_sentences                                                    # one sentence splitter for the pipeline and the MT class


class LanguageID:
    """vi / not-vi from confidence statistics of the Vietnamese CTC model: logistic regression on six features (assets/lid_vi.json, trained on FLEURS
    clean speech: 10-fold CV 100 % on 120 clips, see docs/conversation.md). `bias` shifts the decision (positive = more willing to call it Vietnamese)."""

    def __init__(self, engine, model_path=None, bias=0.0):
        import json, os
        from .registry import ASSETS
        self.e = engine; self.bias = bias; self.m = json.load(open(model_path or os.path.join(ASSETS, "lid_vi.json"), encoding="utf-8"))

    def vi_prob(self, st):
        x = np.array([st["conf_nb"], st["known"], st["nb_ratio"], st.get("conf_all", st["conf_nb"]), st.get("blank_p", 1 - st["nb_ratio"]), min(st["nwords"], 30) / 30.0])
        z = float(((x - np.array(self.m["mean"])) / np.array(self.m["scale"])) @ np.array(self.m["coef"]) + self.m["intercept"] + self.bias)
        return 1.0 / (1.0 + np.exp(-z))

    def is_vi(self, stats):
        return stats["nwords"] >= 1 and self.vi_prob(stats) >= 0.5

    def other(self, wav, candidates):
        if len(candidates) == 1: return candidates[0], None
        sv = self.e.asr("en"); lang0 = sv.lang; sv.lang = "auto"
        try:
            text = sv(wav)
        finally: sv.lang = lang0
        return (script_lang(text) or candidates[0]), text


class Conversation:
    def __init__(self, registry, pair=None, auto=False, partner="en", denoise=False, vad_profile=None):
        from .registry import resolve_runtime_profile
        self.e = Engine(registry, denoise); self.lid = LanguageID(self.e); self.auto = auto; self.partner = partner
        self.vad_profile = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else getattr(registry, "vad_profile", None))["vad_profile"]
        self.others = [l for l in (pair or ("vi", partner)) if l != "vi"] if not auto else ["en", "zh", "ko"]
        if not auto: self.partner = self.others[0]

    def max_seg(self, max_seg_s=None):
        """Every segment goes through the Vietnamese ASR first (LID), then maybe SenseVoice: cap by both windows. join_gap_s=0 in segment():
        two speakers' turns 0.6-1 s apart must stay separate segments, each routed by its own language (the in-sentence join is for one speaker)."""
        caps = [max_seg_s or SEG_MAX_S, getattr(self.e.asr("vi"), "max_s", SEG_MAX_S)] + [getattr(self.e.asr(l), "max_s", SEG_MAX_S) for l in self.others[:1]]
        return min(caps)

    def route(self, wav):
        """-> (src, tgt, vi_text_or_None)"""
        asr_vi = self.e.asr("vi"); text = asr_vi(wav); st = dict(asr_vi.stats)
        if self.lid.is_vi(st): return "vi", self.partner, text, st
        src, text = self.lid.other(wav, self.others)                   # auto mode: SenseVoice text is reused, no second pass
        if self.auto and text and len(re.sub(r"[\W_]+", "", text)) >= 3: self.partner = src       # junk fragments must not change the partner language
        return src, "vi", text, st

    def _segments(self, wav, max_seg_s=None, vad_profile=None):
        from .registry import resolve_runtime_profile
        prof = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else self.vad_profile)["vad_profile"]
        return segment_with_breaks(wav, max_s=self.max_seg(max_seg_s), vad_profile=prof, join_gap_s=0.0)

    def translate(self, wav, max_seg_s=None, vad_profile=None):
        out = []
        for (a, b), brk in self._segments(wav, max_seg_s, vad_profile):
            seg = self.e.prep(wav[a:b]); t0 = time.perf_counter(); src, tgt, text, st = self.route(seg); dt = time.perf_counter() - t0
            r = self.e.translate_segment(seg, src, tgt, text=text, prepped=True, breaks=brk)
            tm = r.setdefault("timing", {}); tm["asr_s"] = tm.get("asr_s", 0.0) + dt; tm["total_s"] = tm.get("total_s", 0.0) + dt   # route() = ASR + language id (graph_s excludes it)
            r["span"] = (a / SR, b / SR); r["lid"] = st
            if r["mt"]: out.append(r)
        return out

    def translate_stream(self, wav, max_seg_s=None, vad_profile=None):
        """Generator: one result per sentence as soon as its audio is ready (see Engine.translate_segment_stream)."""
        for (a, b), brk in self._segments(wav, max_seg_s, vad_profile):
            seg = self.e.prep(wav[a:b]); t0 = time.perf_counter(); src, tgt, text, st = self.route(seg); dt = time.perf_counter() - t0
            for r in self.e.translate_segment_stream(seg, src, tgt, text=text, prepped=True, breaks=brk):
                if r.get("part", (0, 1))[0] == 0: tm = r.setdefault("timing", {}); tm["asr_s"] = tm.get("asr_s", 0.0) + dt; tm["total_s"] = tm.get("total_s", 0.0) + dt   # route() = ASR + language id
                r["span"] = (a / SR, b / SR); r["lid"] = st
                if r["mt"]: yield r
