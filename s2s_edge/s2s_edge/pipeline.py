# -*- coding: utf-8 -*-
"""Fixed-direction speech-to-speech pipeline: audio -> utterance segments -> ASR -> MT -> TTS, with per-stage timing (host wall time and graph time).
The work is done by `conversation.Engine` (shared with the multilingual conversation router)."""
from .audio import SR, segment
from .conversation import Engine


class Translator:
    def __init__(self, registry, src, tgt, denoise=False):
        self.r = registry; self.src, self.tgt = src, tgt; self.e = Engine(registry, denoise)
        self.asr = self.e.asr(src); self.mt = self.e.mt; self.tts = self.e.tts(tgt)
        self.graphs = self.e.graphs_for(src, tgt)

    def translate_segment(self, wav):
        return self.e.translate_segment(wav, self.src, self.tgt)

    def translate(self, wav, max_seg_s=None):
        max_seg = max_seg_s or min(getattr(self.asr, "max_s", 12.0), 12.0)
        out = []
        for a, b in segment(wav, max_s=max_seg):
            r = self.translate_segment(wav[a:b]); r["span"] = (a / SR, b / SR); out.append(r)
        return out

    def translate_stream(self, wav, max_seg_s=None):
        """Generator for live output: one result per sentence as soon as its audio is ready (Engine.translate_segment_stream)."""
        max_seg = max_seg_s or min(getattr(self.asr, "max_s", 12.0), 12.0)
        for a, b in segment(wav, max_s=max_seg):
            for r in self.e.translate_segment_stream(wav[a:b], self.src, self.tgt):
                r["span"] = (a / SR, b / SR); yield r
