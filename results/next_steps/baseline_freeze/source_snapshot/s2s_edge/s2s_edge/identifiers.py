"""Typed identifier spans for VI/EN TTS, protected before name transliteration.

Phone/ID values are spelled, while ordinary quantities keep their number grammar.
All spans use one compositional character verbalizer; no per-value substitutions.
"""
from dataclasses import dataclass
import re

_DIGITS = {
    "vi": "không một hai ba bốn năm sáu bảy tám chín".split(),
    "en": "zero one two three four five six seven eight nine".split(),
}
_VI_LETTERS = "a bê xê đê e ép giê hát i giây ca e-lờ em en o pê quy e-rờ ét tê u vê đúp-vê ích i-dài dét".split()
_LETTERS = {
    "vi": dict(zip("abcdefghijklmnopqrstuvwxyz", _VI_LETTERS)),
    "en": dict(zip("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz")),
}
_PHONE_CUES = (
    r"số điện thoại", r"điện thoại", r"gọi(?: số(?: khẩn cấp)?)?", r"hotline",
    r"phone(?: number)?", r"telephone(?: number)?", r"call(?: emergency number)?",
    r"emergency number", r"tel\.?", r"fax",
)
_CODE_CUES = (
    r"mã(?: số| bưu chính| zip| đơn(?: hàng)?| chuyến bay| xác nhận| định danh)?",
    r"mật khẩu", r"chuyến bay", r"zip(?: code)?", r"postal(?: code)?",
    r"post(?:al)?code", r"pin", r"otp", r"password", r"flight(?: number)?",
    r"order(?: number| code)?", r"serial(?: number)?", r"tracking(?: number| code)?", r"code",
)
def _cue(patterns):
    return re.compile(r"(?:^|\b)(?:" + "|".join(patterns) + r")\s*(?:(?:là|is|number|no\.)\s*)?[:#=]?\s*$", re.I)

_PHONE_CONTEXT = _cue(_PHONE_CUES)
_CODE_CONTEXT = _cue(_CODE_CUES)
_NUMERIC = re.compile(r"(?<![\w])\+?\d(?:[\d ()\u00a0.\-]*\d)?(?![\w])")
_ALNUM = re.compile(r"(?<![\w])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?![\w])")
_CLOCK = re.compile(r"\d{1,2}h\d{0,2}(?:p)?", re.I)
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)", re.I)
_SCIENTIFIC = re.compile(r"\d+(?:[.,]\d+)?e[+-]?\d+", re.I)
_MEASURE = re.compile(r"\d+(?:[.,]\d+)?(?:mm|cm|dm|km|m|mg|kg|g|ml|cl|dl|l|ms|s|hz|khz|mhz|ghz|kb|mb|gb|tb|v|w|kw|kwh|usd|vnd)", re.I)
_PHONE_SHAPE = re.compile(r"(?:\(\d{3}\)[ -]?)?\d{3}[-.]\d{4}|\d{3}[-.]\d{3}[-.]\d{4}")

@dataclass(frozen=True)
class Identifier:
    start: int
    end: int
    kind: str
    value: str

def find_identifiers(text):
    """Recognize typed spans using format and local semantic cues, before casing.

    Ambiguous bare integers stay cardinal unless a phone/code cue is present.
    Time, units, ordinals and scientific notation take precedence over IDs.
    """
    found = []
    for m in _NUMERIC.finditer(text):
        value = m.group(); digits = re.sub(r"\D", "", value)
        prefix = text[max(0, m.start() - 70):m.start()]
        if _CODE_CONTEXT.search(prefix):
            kind = "code"
        elif _PHONE_CONTEXT.search(prefix) and 3 <= len(digits) <= 15:
            kind = "telephone"
        elif 7 <= len(digits) <= 15 and (
            value.startswith("+") or
            (digits.startswith("0") and 9 <= len(digits) <= 11) or
            _PHONE_SHAPE.fullmatch(value)
        ):
            kind = "telephone"
        else:
            continue
        # A decimal/amount is not a phone just because its digits start with 0.
        if not _PHONE_CONTEXT.search(prefix) and not _CODE_CONTEXT.search(prefix):
            if re.fullmatch(r"\d+[.,]\d+", value) and not _PHONE_SHAPE.fullmatch(value):
                continue
        found.append(Identifier(m.start(), m.end(), kind, value))
    for m in _ALNUM.finditer(text):
        value = m.group(); prefix = text[max(0, m.start() - 70):m.start()]
        contextual = bool(_CODE_CONTEXT.search(prefix))
        mixed = any(c.isalpha() for c in value) and any(c.isdigit() for c in value)
        if not contextual and not mixed:
            continue
        if value.isalpha() and _CODE_CONTEXT.search(text[max(0, m.start() - 70):m.end()]):
            continue  # This word completes the cue itself (e.g. "ZIP code").
        if any(p.fullmatch(value) for p in (_CLOCK, _ORDINAL, _SCIENTIFIC, _MEASURE)) and not contextual:
            continue
        # Identifier recognition must not split an email/URL into invented IDs.
        left = text[max(0, m.start() - 1):m.start()]
        right = text[m.end():m.end() + 1]
        if left in ("@", "/") or right == "@":
            continue
        found.append(Identifier(m.start(), m.end(), "code", value))
    accepted = []
    for span in sorted(found, key=lambda s: (s.start, -(s.end - s.start))):
        if not accepted or span.start >= accepted[-1].end:
            accepted.append(span)
    return accepted

def spell(value, lang):
    """Read all letters/digits in order, retaining zeros and a leading plus."""
    out = []
    for ch in value:
        if ch in "0123456789":
            out.append(_DIGITS[lang][int(ch)])
        elif ch.lower() in _LETTERS[lang]:
            out.append(_LETTERS[lang][ch.lower()].replace("-", " "))
        elif ch == "+":
            out.append("cộng" if lang == "vi" else "plus")
    return " ".join(out)

def prepare_for_tts(text, lang, normalizer, foreign_words=None):
    """Protect IDs before the existing names -> lower-case -> TN sequence.

    `normalizer(text, lang)` can be the baseline, WFST, CRF+WFST or WFST+LM.
    Only non-ID spans pass through the foreign-name respeller.
    """
    if lang not in _DIGITS:
        return normalizer(text, lang)
    spans = find_identifiers(text); parts = []; pos = 0
    def ordinary(s):
        return foreign_words(s) if foreign_words is not None else s
    for span in spans:
        parts.append(ordinary(text[pos:span.start]))
        parts.append(spell(span.value, lang))
        pos = span.end
    parts.append(ordinary(text[pos:]))
    protected = "".join(parts)
    if lang == "vi":
        protected = protected.lower()
    return normalizer(protected, lang)
