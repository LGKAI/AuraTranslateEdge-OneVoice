# -*- coding: utf-8 -*-
"""Proper names and foreign words for TTS.
 * Vietnamese target (Piper reads only Vietnamese syllables and silently drops anything else): Latin words are respelled as Vietnamese syllables,
   from CMUdict phonemes (English) or letter rules (other names), the way Vietnamese speakers read them ("Trump" -> "trăm pơ").
 * Chinese / Korean target (Melo-zh drops Latin letters): leftover Latin names are transliterated by the translation model itself
   (translate the name alone, English -> target), cached.
 * A user glossary (JSON {"Samsung": {"vi": "sam sung", "zh": "三星", "ko": "삼성"}}) overrides both.
Pure Python; the only data file is assets/cmudict_min.json (words -> ARPAbet)."""
import json, os, re

WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")                       # whole word, any script
ASCII_WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
VI_CHARS = set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ")

ONSET = {"B": "b", "CH": "ch", "D": "đ", "DH": "đ", "F": "ph", "G": "g", "HH": "h", "JH": "gi", "K": "c", "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r",
         "S": "s", "SH": "s", "T": "t", "TH": "th", "V": "v", "W": "u", "Y": "d", "Z": "d", "ZH": "gi"}
CODA = {"P": "p", "T": "t", "K": "c", "M": "m", "N": "n", "NG": "ng", "CH": "ch"}
VOWEL = {"AA": "a", "AE": "e", "AH": "ơ", "AO": "o", "AW": "ao", "AY": "ai", "EH": "e", "ER": "ơ", "EY": "ây", "IH": "i", "IY": "i", "OW": "ô", "OY": "oi", "UH": "u", "UW": "u"}
FRONT = ("e", "i", "ê", "ây")

# letter -> pseudo ARPAbet for words missing from CMUdict (names of other origins): digraphs first
DIGRAPH = [("sch", "SH"), ("tch", "CH"), ("sh", "SH"), ("ch", "CH"), ("th", "TH"), ("ph", "F"), ("ng", "NG"), ("ck", "K"), ("qu", "K W"), ("gh", "G"), ("kh", "K"), ("dj", "JH"), ("tz", "T S"),
           ("oo", "UW"), ("ee", "IY"), ("ea", "IY"), ("ou", "AW"), ("ai", "EY"), ("ay", "EY"), ("ei", "EY"), ("oi", "OY"), ("au", "AO"), ("ie", "IY")]
LETTER = {"a": "AA", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "o": "OW", "p": "P", "q": "K",
          "r": "R", "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K S", "y": "IY", "z": "Z", "à": "AA", "á": "AA", "â": "AA", "ä": "AE", "é": "EH", "è": "EH", "ö": "ER", "ü": "UW",
          "ç": "S", "ñ": "N"}


def _letters_to_arpa(w):
    w = w.lower(); out = []; i = 0
    while i < len(w):
        for d, ph in DIGRAPH:
            if w.startswith(d, i): out += ph.split(); i += len(d); break
        else:
            ch = w[i]
            if ch == "c" and i + 1 < len(w) and w[i + 1] in "eiy": out.append("S")
            elif ch == "g" and i + 1 < len(w) and w[i + 1] in "ei" and i > 0: out.append("JH")
            elif ch == "e" and i == len(w) - 1 and i > 1: pass                      # silent final e
            elif ch in LETTER: out += LETTER[ch].split()
            i += 1
    return out


def _syllables(phones):
    """ARPAbet phones -> list of Vietnamese-looking syllables (no tones)."""
    ph = [re.sub(r"\d", "", p) for p in phones]; stress = [("1" in p or "2" in p) for p in phones]
    syl = []; i = 0; n = len(ph)
    while i < n:
        onset = "";
        cons = []
        while i < n and ph[i] not in VOWEL:
            cons.append(ph[i]); i += 1
        if i >= n:                                                                  # trailing consonants
            for c in cons:
                if syl and c in CODA and not syl[-1][-1:] in "pctmn" and not syl[-1].endswith(("ng", "ch")): syl[-1] += CODA[c]
                else: syl.append(ONSET.get(c, "") + "ơ")
            break
        # onset cluster: keep a valid pair, otherwise epenthetic "ơ" after the first consonant
        if len(cons) >= 2 and not (cons[-2:] in (["T", "R"], ["K", "R"], ["P", "R"], ["B", "R"], ["D", "R"], ["G", "R"], ["F", "R"], ["TH", "R"], ["S", "T"], ["K", "L"])):
            for c in cons[:-1]: syl.append(ONSET.get(c, "") + "ơ")
            cons = cons[-1:]
        if cons[-2:] in (["T", "R"], ["K", "R"], ["P", "R"], ["B", "R"], ["D", "R"], ["G", "R"], ["F", "R"], ["TH", "R"]): on = {"T": "tr", "K": "c", "P": "p", "B": "b", "D": "đ", "G": "g", "F": "ph", "TH": "th"}[cons[-2]] + "r" if cons[-2] != "T" else "tr"
        elif cons[-2:] == ["S", "T"]: on = "x" + "t"
        elif cons[-2:] == ["K", "L"]: on = "c" + "l"
        elif cons: on = ONSET.get(cons[-1], "")
        else: on = ""
        v = VOWEL[ph[i]]
        if ph[i] == "AH" and stress[i]: v = "a"
        if on == "c" and v.startswith(FRONT): on = "k"
        if on == "u" and v == "u": on = ""
        i += 1; coda = ""
        if i < n and ph[i] not in VOWEL:
            nxt_is_vowel = i + 1 < n and ph[i + 1] in VOWEL
            if not nxt_is_vowel:                                                    # consonant that cannot start the next syllable -> coda
                c = ph[i]
                if c in CODA: coda = CODA[c]; i += 1
                elif c == "R": coda = "c" if (i + 1 < n and v in ("e", "ơ", "o")) else ""; i += 1
                elif c == "L": coda = "u" if False else "";
        syl.append(on + v + coda)
    return syl


class VietnameseNames:
    def __init__(self, cmudict_path=None, known_syllables=None):
        self.cmu = {}
        if cmudict_path and os.path.exists(cmudict_path): self.cmu = json.load(open(cmudict_path, encoding="utf-8"))
        self.known = {_fold(k) for k in known_syllables} if known_syllables is not None else None

    def word(self, w):
        lw = w.lower().strip("'’-")
        arpa = self.cmu.get(lw)
        phones = arpa.split() if arpa else _letters_to_arpa(lw)
        return " ".join(_syllables(phones))

    def is_vietnamese(self, w):
        """True for words Piper reads as they are: Vietnamese syllables (with or without tone marks)."""
        lw = w.lower()
        if not ASCII_WORD.fullmatch(w): return True                                                    # has diacritics / non-Latin letters: leave alone
        return self.known is not None and _fold(lw) in self.known


def _fold(w):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", w.lower().replace("đ", "d")) if unicodedata.category(c) != "Mn")


def latin_to_vi(text, vn, glossary=None):
    """Replace Latin words Piper cannot read by Vietnamese respellings."""
    def sub(m):
        w = m.group(0)
        if glossary and w in glossary and "vi" in glossary[w]: return glossary[w]["vi"]
        if vn.is_vietnamese(w): return w
        return vn.word(w)
    return WORD.sub(sub, text)


def latin_spans(text):
    """Runs of consecutive Latin words (a name like 'Pamela Ferguson') in a CJK / Hangul sentence."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z][A-Za-z'’\-\.]*(?: [A-Z][A-Za-z'’\-\.]*)*", text)]


class NameTransliterator:
    """Latin names inside zh / ko output -> target script, using the NLLB model itself (cached)."""

    def __init__(self, mt, glossary=None):
        self.mt = mt; self.cache = {}; self.glossary = glossary or {}

    def __call__(self, text, tgt):
        out = text; spans = latin_spans(text)
        for a, b, name in reversed(spans):
            key = (name, tgt)
            if key not in self.cache:
                g = self.glossary.get(name, {}).get(tgt)
                self.cache[key] = g if g else self.mt(name, "en", tgt).strip(" .")
            out = out[:a] + self.cache[key] + out[b:]
        return out
