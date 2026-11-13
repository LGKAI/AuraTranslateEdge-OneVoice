# -*- coding: utf-8 -*-
"""Rule-based clean-up of the ASR text BEFORE machine translation (S2S_ASR_CLEAN=0 disables it).
It only removes things that are not speech: a stray label letter ("Q. ", "H. ", "问:"), a leading hesitation ("Um. "), runs of one unit repeated
by a stuck decoder, and text with nothing in it. It never touches real words: "Có.", "Vâng.", "여름.", "好。", "A.", "I." stay (see tests/test_asr_clean.py).
clean_asr(text, lang) -> (clean_text, tags) ; tags is a list of what was removed, [] when nothing changed (the raw text is kept by the caller)."""
import os, re

ENABLED = os.environ.get("S2S_ASR_CLEAN", "1") != "0"
# Single letters that are NOT a word in vi/en/zh-pinyin/ko: only these can be a label ("Q. Sách..."). Vowel letters (A, E, I, O, U, Y) are words
# or interjections ("A, tôi hiểu rồi"), so they are never cut.
_LABEL_LETTERS = "BCDFGHJKLMNPQRSTVWXZ"
_LABEL = re.compile(r"^\s*[" + _LABEL_LETTERS + r"]\s*[.:：]\s+(?=\S)")                 # needs a space after, so "H.264", "Q.1" and "K.O" stay
_CJK_LABEL = re.compile(r"^\s*[问答問]\s*[:：]\s*(?=\S)")                               # 问: / 答:   (only with a colon: 问 alone is a word)
_FILLER_WORDS = r"(?:um+|uh+m*|erm+|hmm+|hm+|ừm+|ưm+|ờm+|ơm+)"
_FILLER_LEAD = re.compile(r"^\s*" + _FILLER_WORDS + r"\s*[.,!?…]*\s+(?=\S)", re.I)    # "Um. Tôi muốn..." (needs content after it)
_FILLER_ONLY = re.compile(r"^\s*" + _FILLER_WORDS + r"\s*[.,!?…]*\s*$", re.I)
_CJK_FILLER_ONLY = re.compile(r"^\s*[嗯呃额음어으]+\s*[。.,，!！?？…]*\s*$")              # a lone 嗯 / 呃 / 음 is breath, not a sentence
_NONWORD = re.compile(r"[\W_]+", re.U)
CJK = re.compile(r"[㐀-鿿가-힣]")


def _norm(toks): return [_NONWORD.sub("", t).lower() for t in toks]


def _collapse_runs(text):
    """A unit of 1-4 tokens repeated >= 4 times in a row (stuck decoder) keeps 2 copies; 2-3 repeats are speech ("no no no", "xin lỗi xin lỗi")."""
    toks = text.split(); changed = False
    for size in (1, 2, 3, 4):
        i = 0; out = []
        while i < len(toks):
            unit = _norm(toks[i:i + size])
            if len(unit) == size and all(unit):
                k = 1
                while _norm(toks[i + k * size:i + (k + 1) * size]) == unit: k += 1
                if k >= 4: out += toks[i:i + 2 * size]; i += k * size; changed = True; continue
            out.append(toks[i]); i += 1
        toks = out
    return " ".join(toks), changed


# Hesitation syllables that the ASR writes as a sentence of their own under noise (adv_asr_cpu.json: "嗯。 1963年大坝建成后…。 嗯。",
# "그. 지하철의…", "… 생성되었습니다. 그."). Removed only when they are a WHOLE sentence at the start or the end AND other content remains;
# with a comma they are speech ("嗯，我知道了。", "음, 알겠어요.") and stay. Lexical words (没 "no", 네 "yes", Có, Vâng) are never in these lists.
# Only at the END (at the start they are an answer: "没。我没去。"): adv_asr_cpu.json "…沉积物不再冲散到河流里。 没。" (2 noisy clips, 0 in gold text).
_TRAIL_ONLY = {"zh": {"没"}}
_EDGE_FILLERS = {"zh": {"嗯", "呃", "额", "嗯嗯"}, "ko": {"그", "음", "으", "어", "에"}, "vi": {"um", "uh", "ừm", "ưm", "ờm", "ơm"}, "en": {"um", "uh", "uhm", "erm", "hmm"}}


def _strip_edge_fillers(t, lang):
    """Remove whole-sentence fillers at the start / end by cutting the original string (spacing of the rest is kept exactly)."""
    fill = _EDGE_FILLERS.get(lang, set()) | _EDGE_FILLERS["zh"] | _EDGE_FILLERS["en"]      # zh/en fillers also show up in mixed-language ASR
    alt = "|".join(sorted(map(re.escape, fill), key=len, reverse=True))
    lead = re.compile(r"^\s*(?:" + alt + r")\s*[.。!?！？]+\s*(?=\S)", re.I)
    alt_t = "|".join(sorted(map(re.escape, fill | _TRAIL_ONLY.get(lang, set())), key=len, reverse=True))
    trail = re.compile(r"(?<=[.。!?！？])\s*(?:" + alt_t + r")\s*[.。!?！？]+\s*$", re.I)
    t0 = t
    while True:
        u = lead.sub("", t, 1)
        u = trail.sub("", u, 1) if _NONWORD.sub("", trail.sub("", u, 1)) else u               # never strip the last real content
        if u == t: break
        t = u
    return t.strip(), t != t0


def clean_asr(text, lang="vi"):
    """Applies every rule until nothing changes (so cleaning twice == cleaning once)."""
    if not ENABLED or not text: return (text or "").strip(), []
    t, tags = text, []
    for _ in range(10):
        u, tg = _clean_once(t, lang)
        tags += [x for x in tg if x not in tags]
        if u == t.strip(): return u, tags
        t = u
        if not t: return "", tags
    return t, tags


def _clean_once(text, lang):
    if not ENABLED or not text: return (text or "").strip(), []
    t = text.strip(); tags = []
    for _ in range(20):                                       # a label can sit behind a filler: "Um. Q. Sách"
        before = t
        if _CJK_LABEL.match(t): t = _CJK_LABEL.sub("", t, 1).strip(); tags.append("label")
        elif _LABEL.match(t): t = _LABEL.sub("", t, 1).strip(); tags.append("label")
        elif _FILLER_LEAD.match(t): t = _FILLER_LEAD.sub("", t, 1).strip(); tags.append("filler")
        if t == before: break
    if _FILLER_ONLY.match(t) or _CJK_FILLER_ONLY.match(t): return "", tags + ["filler_only"]
    t2, ch = _strip_edge_fillers(t, lang)
    if ch: t, tags = t2, tags + ["edge_filler"]
    if " " in t:
        t2, ch = _collapse_runs(t)
        if ch: t, tags = t2, tags + ["repeat"]
    letters = _NONWORD.sub("", t)
    if not letters or re.fullmatch(r"[A-Za-z]", letters): return "", tags + ["no_text"]    # a lone ASCII letter ("I.", "A") is noise; a lone 好 / 네 / ừ / à is a word
    return t, tags


# ---------------- MT output guard (after MT, before TTS) ----------------
# Seen in real outputs (outputOnevoice2/mt_compare, asr_clean/mt_effect.csv): "❚ Ông có…‖", a leading "。", "问:" copied from a label,
# a CJK full stop inside Vietnamese, a subtitle dash "- Yes, I did.".
# NOT done (measured, outputOnevoice2/asr_clean/README.md): collapsing a doubled interjection ("그래, 그래." for "Vâng.") - the BTC references
# themselves double it ("Không có." -> "아니, 아니."), corpus chrF vi-ko went 26.74 -> 26.69, so there is no evidence it helps.
_JUNK_SYM = re.compile(r"[▀-▟─-╿❘-❚‖‗■-◿•]")   # block / box / bar / bullet symbols (NOT "·": 雷杰普·塔伊普)
_LEAD_PUNCT = re.compile(r"^(?:[\s.,;:!?。，、；：！？…]+|[\-–—]+\s+)+")                     # a dash only with a space after: "- Yes." goes, "-5 độ" stays
_OUT_LABEL = re.compile(r"^\s*(?:问|答|問|Q|A|H)\s*[:：]\s*")


def clean_mt(out, src_text, tgt):
    """-> (clean, tags). Only removes what is never part of a translation; the words themselves are not touched. Runs to a fixed point."""
    t, tags = out or "", []
    for _ in range(10):
        u, tg = _clean_mt_once(t, src_text, tgt); tags += [x for x in tg if x not in tags]
        if u == t.strip() or not u: return u, tags
        t = u
    return t, tags


def _clean_mt_once(out, src_text, tgt):
    if not ENABLED or not out: return out or "", []
    t = out; tags = []
    u = _JUNK_SYM.sub("", t)
    if u != t: t, tags = re.sub(r"\s{2,}", " ", u).strip(), tags + ["mt_symbol"]
    if not re.search(r"\w", t): return "", tags + ["mt_no_text"]                       # only punctuation / symbols: nothing to speak
    if _OUT_LABEL.match(t) and not _OUT_LABEL.match(src_text or ""): t, tags = _OUT_LABEL.sub("", t, 1), tags + ["mt_label"]
    u = _LEAD_PUNCT.sub("", t)
    if u != t and u: t, tags = u, tags + ["mt_lead_punct"]
    if tgt in ("vi", "en"):
        u = re.sub(r"\s*。\s*$", ".", re.sub(r"\s*。\s*(?=\S)", ". ", t))
        if u != t: t, tags = u, tags + ["mt_cjk_stop"]
    return t.strip(), tags
