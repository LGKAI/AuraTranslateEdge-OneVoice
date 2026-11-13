# -*- coding: utf-8 -*-
"""Fixed-direction speech-to-speech pipeline: audio -> utterance segments -> ASR -> MT -> TTS, with per-stage timing (host wall time and graph time).
The work is done by `conversation.Engine` (shared with the multilingual conversation router)."""
from .audio import SR, SEG_MAX_S, LEGACY_PROFILES, segment_with_breaks
from .conversation import Engine
from .registry import resolve_runtime_profile


class Translator:
    def __init__(self, registry, src, tgt, denoise=False, vad_profile=None):
        self.r = registry; self.src, self.tgt = src, tgt; self.e = Engine(registry, denoise)
        self.vad_profile = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else getattr(registry, "vad_profile", None))["vad_profile"]
        self.asr = self.e.asr(src); self.mt = self.e.mt; self.tts = self.e.tts(tgt)
        self.graphs = self.e.graphs_for(src, tgt)

    def translate_segment(self, wav, breaks=None):
        return self.e.translate_segment(wav, self.src, self.tgt, breaks=breaks)

    def _segments(self, wav, max_seg_s=None, vad_profile=None):
        prof = resolve_runtime_profile(vad_profile=vad_profile if vad_profile is not None else self.vad_profile)["vad_profile"]
        cap = 12.0 if prof in LEGACY_PROFILES else SEG_MAX_S          # legacy profiles keep the 12 s cap of the Plan-01 baselines
        max_seg = max_seg_s or min(getattr(self.asr, "max_s", cap), cap)
        return segment_with_breaks(wav, max_s=max_seg, vad_profile=prof)

    def translate(self, wav, max_seg_s=None, vad_profile=None):
        out = []
        for (a, b), brk in self._segments(wav, max_seg_s, vad_profile):
            r = self.translate_segment(wav[a:b], breaks=brk); r["span"] = (a / SR, b / SR); out.append(r)
        return out

    def translate_stream(self, wav, max_seg_s=None, vad_profile=None):
        """Generator for live output: one result per sentence as soon as its audio is ready (Engine.translate_segment_stream)."""
        for (a, b), brk in self._segments(wav, max_seg_s, vad_profile):
            for r in self.e.translate_segment_stream(wav[a:b], self.src, self.tgt, breaks=brk):
                r["span"] = (a / SR, b / SR); yield r
