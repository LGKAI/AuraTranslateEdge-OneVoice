# -*- coding: utf-8 -*-
"""Engine.translate_text: multi-sentence input goes to MT one sentence at a time, in order, nothing lost; every path (non-stream, stream, typed text)
uses it; S2S_MT_SPLIT=0 restores one call. Plus the ko politeness additions (measured on BTC vi-ko: 5/5 changed sentences closer to the reference)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_asr_clean_pipeline as P
from s2s_edge import conversation as C
from s2s_edge.ko_polite import polite
P.E.names = lambda m, t: m                                                                 # no name transliteration in the fake

def test_split_non_stream():
    r, s = P.run("Vâng. Nếu cần giúp đỡ, hãy gọi tôi. Tôi là Suji.", "vi", "en")
    assert s == ["Vâng. Nếu cần giúp đỡ, hãy gọi tôi.", "Tôi là Suji."], s                   # "Vâng." is merged into its neighbour (no lone fragment)
    assert r[0]["mt"] == "T(Vâng. Nếu cần giúp đỡ, hãy gọi tôi.) T(Tôi là Suji.)" and "mt_split" in r[0]["asr_clean"], r[0]

def test_zh_join_without_space():
    r, s = P.run("Trước đây tôi là nhân viên. Bây giờ tôi là giáo viên.", "vi", "zh")
    assert len(s) == 2 and r[0]["mt"] == "T(Trước đây tôi là nhân viên.)T(Bây giờ tôi là giáo viên.)", r[0]["mt"]

def test_single_sentence_one_call():
    r, s = P.run("Anh chờ tôi một chút.", "vi", "en"); assert s == ["Anh chờ tôi một chút."] and "mt_split" not in r[0]["asr_clean"]

def test_stream_same_sentences():
    r, s = P.run("Vâng. Nếu cần giúp đỡ, hãy gọi tôi. Tôi là Suji.", "vi", "en", stream=True)
    assert s == ["Vâng. Nếu cần giúp đỡ, hãy gọi tôi.", "Tôi là Suji."] and len(r) == 2, s

def test_empty_part_dropped():
    try:
        P.MT_OUT[0] = "- ."; r, s = P.run("Một. Hai ba bốn.", "vi", "en"); assert r[0]["mt"] == "", r[0]["mt"]
    finally: P.MT_OUT[0] = None

def test_switch_off():
    try:
        C.MT_SPLIT = False; r, s = P.run("Vâng, được. Tôi là Suji.", "vi", "en"); assert s == ["Vâng, được. Tôi là Suji."], s
    finally: C.MT_SPLIT = True

def test_ko_polite_additions():
    cases = {"날 속이지 마": "날 속이지 마세요", "미안하지만 안 돼": "미안하지만 안 돼요", "아침 식사는 어때?": "아침 식사는 어때요?", "어디로 내려갈까?": "어디로 내려갈까요?",
             "신호등 앞에 멈춰": "신호등 앞에 멈춰요",
             # untouched: already polite / formal
             "한국 사람입니까?": "한국 사람입니까?", "그럴까요?": "그럴까요?", "감사합니다.": "감사합니다.", "괜찮아요.": "괜찮아요.", "하지 마세요.": "하지 마세요."}
    for a, b in cases.items(): assert polite(a) == b, (a, polite(a), b)
    for b in cases.values(): assert polite(b) == b, (b, polite(b))                          # idempotent

def test_split_abbreviations():
    S = C.split_sentences
    assert S("Please wake me at 7 a.m. tomorrow, okay?", "en") == ["Please wake me at 7 a.m. tomorrow, okay?"]
    assert S("I met Mr. Kim yesterday. He was nice.", "en") == ["I met Mr. Kim yesterday.", "He was nice."]
    assert S("Tôi ở TP. Hồ Chí Minh. Còn bạn thì sao?", "vi") == ["Tôi ở TP. Hồ Chí Minh.", "Còn bạn thì sao?"]
    assert S("The answer is no. I will go home.", "en") == ["The answer is no.", "I will go home."]        # "no." is a word here, not "No." (number)
    assert S("今天很好。我们走吧。", "zh") == ["今天很好。", "我们走吧。"]

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
