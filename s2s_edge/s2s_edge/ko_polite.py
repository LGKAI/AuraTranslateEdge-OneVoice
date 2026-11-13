# -*- coding: utf-8 -*-
"""Korean politeness post-edit for MT output (vi -> ko). NLLB often ends sentences in banmal (해체: "뭐야?", "어딨어?", "괜찮아"), which sounds rude to a
listener who is a customer or a stranger. This turns sentence-final banmal into 해요체 (polite informal: -요). Only the LAST word of each sentence is
touched, and only for a small set of unambiguous endings; formal endings (-니다, -요, -까요, -세요, -죠) are left as they are.
Switch off with S2S_KO_POLITE=0."""
import os, re

ON = os.environ.get("S2S_KO_POLITE", "1") != "0"
_HANGUL = re.compile(r"[가-힣]")


def _has_batchim(ch):
    return _HANGUL.match(ch) is not None and (ord(ch) - 0xAC00) % 28 != 0


_FIXED = {"어딨어": "어디 있어요", "뭐야": "뭐예요", "왜야": "왜요", "가자": "가요", "하자": "해요", "먹자": "먹어요", "그래": "그래요", "알았어": "알았어요"}
# final syllables of 해체 that become polite by adding 요 (있어 -> 있어요, 괜찮아 -> 괜찮아요, 도와줘 -> 도와줘요, 할게 -> 할게요, 해 -> 해요)
_ADD_YO = set("어아여워와줘봐해돼래게대네때춰")          # 때: 어때 -> 어때요; 춰: 멈춰 -> 멈춰요
# one-syllable sentence-final verbs (only reached when they are the last word of a sentence)
_SINGLE = {"돼": "돼요", "해": "해요", "마": "마세요", "줘": "줘요", "봐": "봐요"}


def _word(w):
    m = re.match(r"^(.*?)([가-힣]+)$", w)
    if not m: return w
    pre, core = m.groups()
    if core in _FIXED: return pre + _FIXED[core]
    if core.endswith(("요", "니다", "세요", "죠", "니까")): return w
    if core.endswith("까") and len(core) > 1: return pre + core + "요"                              # 갈까 -> 갈까요, 할까 -> 할까요 (banmal -ㄹ까; 합니까 kept above)
    if core in _SINGLE: return pre + _SINGLE[core]                                                  # (안) 돼 -> 돼요, (하지) 마 -> 마세요
    if core.endswith("이야") and len(core) > 2: return pre + core[:-2] + "이에요"                     # 일이야 -> 일이에요
    if core.endswith("거야"): return pre + core[:-2] + "거예요"
    if core.endswith("야") and len(core) > 1: return pre + core[:-1] + ("이에요" if _has_batchim(core[-2]) else "예요")
    if core.endswith("지") and len(core) > 1: return pre + core[:-1] + "죠"                          # 맞지 -> 맞죠
    if core.endswith(("니", "냐")) and len(core) > 1: return pre + core[:-1] + "나요"               # 가니 -> 가나요
    if core.endswith("나") and len(core) > 1: return pre + core + "요"                              # 있나 -> 있나요
    if core[-1] in _ADD_YO and len(core) > 1: return pre + core + "요"
    return w


def polite(text):
    """Apply to each sentence: the last Hangul word before the final punctuation."""
    if not ON or not text or not _HANGUL.search(text): return text
    parts = re.split(r"([\.\?\!。？！]+\s*)", text); out = []
    for i in range(0, len(parts), 2):
        seg = parts[i]; punct = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.match(r"^(.*?)(\S+)(\s*)$", seg, re.S)
        if m:
            head, last, tail = m.groups(); seg = head + _word(last) + tail
        out.append(seg + punct)
    return "".join(out)
