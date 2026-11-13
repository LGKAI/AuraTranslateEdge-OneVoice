# -*- coding: utf-8 -*-
"""Spoken-form number normalisation for TTS input (vi / en / zh / ko). Pure Python, no dependencies; runs on the device CPU."""
import re

# ---------------- Vietnamese ----------------
_VD = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _vi3(n, full):
    h, r = divmod(n, 100); t, u = divmod(r, 10); s = []
    if h or full: s.append(_VD[h] + " trăm")
    if t > 1:
        s.append(_VD[t] + " mươi")
        if u == 1: s.append("mốt")
        elif u == 5: s.append("lăm")
        elif u: s.append(_VD[u])
    elif t == 1:
        s.append("mười")
        if u == 5: s.append("lăm")
        elif u: s.append(_VD[u])
    elif u:
        if h or full: s.append("lẻ")
        s.append(_VD[u])
    return " ".join(s)


def vi_num(n):
    if n == 0: return "không"
    parts = []; i = 0; unit = ["", " nghìn", " triệu", " tỷ"]
    while n:
        n, r = divmod(n, 1000)
        if r: parts.append((_vi3(r, full=bool(n)) + unit[i % 4]).strip())
        i += 1
    return " ".join(reversed(parts))


# ---------------- Chinese ----------------
_ZD = "零一二三四五六七八九"


def _zh4(n, first):
    s = ""; u = ["", "十", "百", "千"]; z = False
    for i, ch in enumerate(str(n).zfill(4)):
        d = int(ch); p = 3 - i
        if d == 0:
            z = bool(s); continue
        if z: s += "零"
        z = False
        s += ("" if (d == 1 and p == 1 and not s and first) else _ZD[d]) + u[p]
    return s


def zh_num(n):
    if n == 0: return "零"
    groups = []; units = ["", "万", "亿"]; i = 0
    while n:
        n, r = divmod(n, 10000); groups.append((r, i)); i += 1
    s = ""
    for k, (r, i) in enumerate(reversed(groups)):
        if r == 0: continue
        if s and r < 1000 and not s.endswith("零"): s += "零"
        s += _zh4(r, first=(k == 0)) + units[i % 3] + ("亿" * (i // 3))
    return s


# ---------------- Korean (Sino-Korean reading) ----------------
_KD = "영일이삼사오육칠팔구"


def ko_num(n):
    if n == 0: return "영"
    parts = []; i = 0; units = ["", "만", "억", "조"]; for_u = ["", "십", "백", "천"]
    while n:
        n, r = divmod(n, 10000)
        if r:
            s = ""
            for p, ch in enumerate(str(r).zfill(4)):
                d = int(ch); pos = 3 - p
                if d == 0: continue
                s += ("" if (d == 1 and pos > 0) else _KD[d]) + for_u[pos]
            parts.append(s + units[i])
        i += 1
    return "".join(reversed(parts))


# ---------------- English ----------------
_E1 = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_E10 = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def en_num(n):
    if n < 20: return _E1[n]
    if n < 100: return _E10[n // 10] + ("-" + _E1[n % 10] if n % 10 else "")
    if n < 1000: return _E1[n // 100] + " hundred" + (" " + en_num(n % 100) if n % 100 else "")
    for v, w in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= v: return en_num(n // v) + " " + w + (" " + en_num(n % v) if n % v else "")


_FN = {"vi": vi_num, "zh": zh_num, "ko": ko_num, "en": en_num}
_DEC = {"vi": " phẩy ", "zh": "点", "ko": " 점 ", "en": " point "}
_PCT = {"vi": " phần trăm", "zh": "百分之", "ko": " 퍼센트", "en": " percent"}
_DIGIT = {"vi": _VD, "zh": list(_ZD), "ko": list(_KD), "en": _E1[:10]}


def normalize(text, lang):
    f = _FN[lang]
    text = re.sub(r"(?<=\d)[,.](?=\d{3}(?!\d))", "", text)                      # thousands separators 1,000 / 1.000

    def number(s):
        if "." in s or "," in s:
            a, b = re.split(r"[.,]", s, 1)
            return f(int(a)) + _DEC[lang] + " ".join(_DIGIT[lang][int(c)] for c in b)
        return f(int(s))

    def pct(m):
        v = number(m.group(1))
        return (_PCT[lang] + v) if lang == "zh" else v + _PCT[lang]

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", pct, text)
    text = re.sub(r"\d+(?:[.,]\d+)?", lambda m: number(m.group(0)) if lang == "zh" else " " + number(m.group(0)) + " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Vietnamese ASR output is lower case without punctuation; NLLB then reads questions as statements ("anh cần thêm dây không" -> "I need more wires.").
_VI_Q_LAST = {"không", "chưa", "à", "ư", "hả", "hở", "nhỉ", "sao", "gì", "đâu", "nào", "chứ", "hử", "ai", "mấy", "chăng"}
_VI_Q_WORDS = re.compile(r"\b(bao nhiêu|bao lâu|bao giờ|thế nào|như nào|ở đâu|đi đâu|đến đâu|tại sao|vì sao|làm sao|khi nào|lúc nào|mấy|có phải|phải không|được không|hay là)\b"
                         r"|\b\w+ nào\b(?! cũng)|\bgì\b|\b\w+ hay \w+")
_VI_NOT_Q = re.compile(r"\b(không sao|không có gì|chẳng sao|có gì đâu|gì cả|gì đâu|gì cũng|nào cũng|rất hay|thường hay|hay quá|hay lắm)\b"
                       r"|\b(biết|hiểu|nhớ|hỏi)\b.*\b(gì|đâu|nào|sao|mấy|bao|không)\b")      # "tôi không biết khi nào ..." is a statement
_VI_NEG_DAU = re.compile(r"\b(không|chưa|chẳng|gì)\b.*\bđâu$")                                 # "chưa quá đông đâu": negation, not "where"


def vi_punct(text):
    """Restore sentence case and a final '?' or '.' on a Vietnamese ASR transcript (rule based: question particles and question words).
    Measured on the 431 Vietnamese sentences of the organisers' sample set (punctuation removed, then restored): see docs/aibox8550_report.md."""
    t = re.sub(r"\s+", " ", text).strip()
    if not t: return t
    if t[-1] in ".?!": return t[:1].upper() + t[1:]
    words = t.lower().split(); core = words[:-1] if len(words) > 1 and words[-1] in ("ạ", "nhé", "vậy", "thế", "ha", "hả") and words[-2] in _VI_Q_LAST else words
    low = " ".join(words); last = core[-1]
    tail = low if len(words) <= 12 else " ".join(words[-5:])                    # question words inside long sentences are mostly relative ("điều gì thúc đẩy ...")
    q = last in _VI_Q_LAST or bool(_VI_Q_WORDS.search(tail))
    if _VI_NOT_Q.search(low) and last not in ("không", "chưa", "à", "ư", "hả", "nhỉ", "chứ"): q = False
    if last == "đâu" and _VI_NEG_DAU.search(" ".join(core)): q = False
    if last in ("không", "chưa") and len(core) <= 2 and core[0] in ("không", "chưa", "tôi", "em"): q = False        # bare "không" / "tôi chưa" answers
    return t[:1].upper() + t[1:] + ("?" if q else ".")
