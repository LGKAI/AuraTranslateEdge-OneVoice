# -*- coding: utf-8 -*-
"""Integration: the filter is in BOTH paths (translate_segment and translate_segment_stream); fake ASR/MT/TTS record what reaches MT."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge import conversation as C

seen = []
class FakeMT:
    last = "mt600"
    def wanted(self, s, t): return "mt600"
    def __call__(self, text, s, t): seen.append(text); return MT_OUT[0] if MT_OUT[0] is not None else "T(" + text + ")"
spoken = []
MT_OUT = [None]
class FakeTTS:
    sr = 16000
    def synthesize(self, t): spoken.append(t); return np.ones(1600, np.float32), 16000
class FakeReg:
    glossary = {}
    def graph(self, k): return type("G", (), dict(total_s=0.0, calls=0))()
    def components_for(self, s, t): return []
    def asr(self, l): return lambda wav: ASR_TEXT[0]
    def tts(self, l): return FakeTTS()
    def mt(self): return FakeMT()
ASR_TEXT = [""]
C.VI_PUNCT = False
E = C.Engine(FakeReg()); E.names = None
E.__class__.mt = property(lambda self: FakeMT())
wav = np.zeros(16000, np.float32)

def run(text, src="vi", tgt="en", stream=False):
    seen.clear(); ASR_TEXT[0] = text
    if stream: res = list(E.translate_segment_stream(wav, src, tgt, prepped=True))
    else: res = [E.translate_segment(wav, src, tgt, prepped=True)]
    return res, list(seen)

def test_both_paths():
    for stream in (False, True):
        r, s = run("Q. Sách tiếng Trung đó là của tôi.", stream=stream); assert s == ["Sách tiếng Trung đó là của tôi."], (stream, s)
        assert r[0]["asr_raw"].startswith("Q. ") and "label" in r[0]["asr_clean"], r[0]
        r, s = run("Có. Tôi thêm một viên pin.", stream=stream); assert s and s[0].startswith("Có."), (stream, s)       # a real "Có." is not cut
        r, s = run("Um.", stream=stream)
        assert s == [], (stream, s)                                                                                       # filler only: MT is never called
        r, s = run("。", "zh", "vi", stream=stream); assert s == []
        r, s = run("好。", "zh", "vi", stream=stream); assert s == ["好。"], (stream, s)                                 # a one-character answer is a real answer
        r, s = run("여름.", "ko", "vi", stream=stream); assert s == ["여름."], (stream, s)

def test_empty_result_still_well_formed():
    r, s = run("Um.")                      # non-stream returns a segment with empty mt and silent audio, no crash
    assert r[0]["asr"] == "" and r[0]["mt"] == "" and len(r[0]["wav"]) == 0 and r[0]["asr_raw"] == "Um."

def test_mt_guard_both_paths():
    for stream in (False, True):
        try:
            MT_OUT[0] = "❚ Ông có hơn 70 chiếc răng. ‖"; spoken.clear(); r, s = run("他有七十多颗牙齿。", "zh", "vi", stream=stream)
            assert r[0]["mt"] == "Ông có hơn 70 chiếc răng." and spoken == ["Ông có hơn 70 chiếc răng."] and "mt_symbol" in r[0]["asr_clean"], (stream, r[0]["mt"], spoken)
            MT_OUT[0] = "- ."; spoken.clear(); r, s = run("Xin chào.", "vi", "en", stream=stream)
            assert r == [] or (r[0]["mt"] == "" and spoken == []), (stream, r)                 # nothing to speak -> TTS is not called
        finally: MT_OUT[0] = None

def test_disable_switch():
    from s2s_edge import asr_clean as A
    A.ENABLED = False
    try:
        r, s = run("Q. Sách đó."); assert s == ["Q. Sách đó."], s
    finally: A.ENABLED = True

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
