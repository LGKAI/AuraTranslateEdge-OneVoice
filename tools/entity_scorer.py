"""Typed, boundary-aware semantic entity scorer with tri-state output (correct/wrong/unknown) and label coverage."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

NON_FLIGHT_PREFIXES = {
    "am", "an", "as", "at", "by", "do", "go", "if", "in", "is", "it",
    "my", "no", "of", "on", "or", "pm", "so", "to", "up", "us", "we",
    "kg", "km", "cm", "mm", "mg", "ms", "hz", "zip", "and", "the", "for",
}

SPOKEN_NUMERAL_WORDS_EN = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
}

SPOKEN_NUMERAL_WORDS_VI = {
    "không", "một", "mốt", "hai", "ba", "tư", "bốn", "năm", "lăm", "nhăm",
    "sáu", "bảy", "bẩy", "tám", "chín", "mười", "mươi", "linh", "lẻ",
    "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "rưỡi",
}

SPOKEN_NUMERAL_CHARS_CJK = set("零一二两兩三四五六七八九十百千万萬亿億")

SPOKEN_NUMERAL_WORDS_KO = {
    "영", "공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구", "십", "백", "천", "만",
    "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열", "스물", "서른",
}

_CURRENCY_WORD_MAP = {
    "$": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "đô": "USD",
    "đô la": "USD",
    "đồng": "VND",
    "vnd": "VND",
    "đ": "VND",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "원": "KRW",
    "달러": "USD",
    "元": "CNY",
    "块": "CNY",
    "美元": "USD",
}


@dataclass(frozen=True)
class EntityToken:
    kind: str
    norm: str
    raw: str
    span: tuple[int, int]


def _norm_decimal(raw_num: str, *, allow_vi_comma: bool = False) -> str | None:
    s = raw_num.strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", s) and allow_vi_comma:
        # Vietnamese thousands grouping like 50.000
        s = s.replace(".", "")
    elif "," in s and "." not in s:
        if allow_vi_comma and re.fullmatch(r"-?\d+,\d{1,2}", s):
            s = s.replace(",", ".")
        else:
            return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    norm = format(d.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".") or "0"
    if norm == "-0":
        norm = "0"
    return norm


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    a, b = span
    return any(not (b <= ua or a >= ub) for ua, ub in used)


def _norm_meridiem(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower().replace(".", "").strip()
    if low in ("am", "in the morning", "sáng"):
        return "am"
    if low in ("pm", "in the afternoon", "in the evening", "at night", "chiều", "tối", "đêm", "trưa"):
        return "pm"
    return None


def _norm_temp_unit(raw: str) -> str:
    low = raw.lower().strip()
    if "f" in low or "fahrenheit" in low:
        return "F"
    return "C"


def _norm_unit(raw: str) -> str:
    low = raw.lower().strip()
    mapping = {
        "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
        "gram": "g", "grams": "g",
        "meter": "m", "meters": "m",
        "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
        "minute": "min", "minutes": "min", "min": "min", "mins": "min",
    }
    return mapping.get(low, low)


def _parse_hhmm(h_str: str, m_str: str, meridiem: str | None) -> str | None:
    h, m = int(h_str), int(m_str)
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None
    return f"{h:d}:{m:02d}" + (f":{meridiem}" if meridiem else "")


def extract_entities(text: str, lang: str = "en") -> list[EntityToken]:
    """Extract typed, boundary-aware entity tokens from text in specificity order."""
    if not text:
        return []
    tokens: list[EntityToken] = []
    used: list[tuple[int, int]] = []
    allow_vi_comma = (lang == "vi")

    def add(kind: str, norm: str | None, m: re.Match) -> None:
        span = m.span()
        if norm is None or _overlaps(span, used):
            return
        used.append(span)
        tokens.append(EntityToken(kind=kind, norm=norm, raw=m.group(0), span=span))

    # 1. Dates (numeric DD/MM/YYYY or ISO YYYY-MM-DD, EN Month Day Year, CJK/KO dates)
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d{4})-(\d{1,2})-(\d{1,2})(?![A-Za-z0-9])", text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            add("date", f"{y:04d}-{mo:02d}-{d:02d}", m)
    for m in re.finditer(r"(?<![A-Za-z0-9])(\d{1,2})/(\d{1,2})/(\d{4})(?![A-Za-z0-9])", text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            add("date", f"{y:04d}-{mo:02d}-{d:02d}", m)
    month_pat = "|".join(MONTHS)
    for m in re.finditer(rf"(?<![A-Za-z])\b({month_pat})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+|\s+)(\d{{4}})\b", text, re.I):
        mo = MONTHS[m.group(1).lower()]
        d, y = int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31:
            add("date", f"{y:04d}-{mo:02d}-{d:02d}", m)
    for m in re.finditer(r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]", text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            add("date", f"{y:04d}-{mo:02d}-{d:02d}", m)

    # 2. Times (HH:MM [am/pm], 14h45, 'at 330 pm' / 'at530 AM', '12 o'clock', CJK/KO times)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(\d{1,2}):(\d{2})(?:\s*(a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening)|at night|sáng|trưa|chiều|tối|đêm))?(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        add("time", _parse_hhmm(m.group(1), m.group(2), _norm_meridiem(m.group(3))), m)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(\d{1,2})h(\d{2})?(?:\s*(sáng|trưa|chiều|tối|đêm))?(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        add("time", _parse_hhmm(m.group(1), m.group(2) or "00", _norm_meridiem(m.group(3))), m)
    for m in re.finditer(
        r"(?:at\s*|lúc\s*)(\d{3,4})\s*(a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening)|at night)(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        digits = m.group(1)
        add("time", _parse_hhmm(digits[:-2], digits[-2:], _norm_meridiem(m.group(2))), m)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(\d{3,4})\s*(a\.?m\.?|p\.?m\.?)(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        digits = m.group(1)
        add("time", _parse_hhmm(digits[:-2], digits[-2:], _norm_meridiem(m.group(2))), m)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(\d{1,2})\s+o['’]?clock(?:\s+(a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening)))?",
        text,
        re.I,
    ):
        add("time", _parse_hhmm(m.group(1), "00", _norm_meridiem(m.group(2))), m)
    for m in re.finditer(r"(\d{1,2})\s*[点點時시](?:\s*(\d{1,2})\s*[分분]?)?", text):
        add("time", _parse_hhmm(m.group(1), m.group(2) or "00", None), m)

    # 3. Flight / alphanumeric identifiers (VN123, VN 123, VN-123)
    for m in re.finditer(r"(?<![A-Za-z0-9])([A-Za-z]{2,3})(?:([-\s])(\d{2,5})|(\d{2,5}))(?![A-Za-z0-9])", text):
        prefix = m.group(1)
        sep = m.group(2)
        digits = m.group(3) if m.group(3) is not None else m.group(4)
        if prefix.lower() in NON_FLIGHT_PREFIXES:
            continue
        if sep is not None and prefix != prefix.upper():
            continue
        add("flight_id", f"{prefix.upper()}{digits}", m)

    # 4. Temperature & explicit units
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(-?\d+(?:[.,]\d+)?)\s*(?:°\s*([CF])|degrees?\s*(celsius|fahrenheit|c|f)|độ\s*([cf])?|(摄氏度)|(도))(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        val = _norm_decimal(m.group(1), allow_vi_comma=allow_vi_comma or ("," in m.group(1)))
        unit_raw = next((g for g in m.groups()[1:] if g), "C")
        if val is not None:
            add("temperature", f"{val}|{_norm_temp_unit(unit_raw)}", m)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(-?\d+(?:[.,]\d+)?)\s*(kilograms?|kgs?|grams?|meters?|hours?|hrs?|minutes?|mins?|kg|km|cm|mm|mg|ms|gb|mb|kb)\b(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        val = _norm_decimal(m.group(1), allow_vi_comma=allow_vi_comma)
        if val is not None:
            add("unit_number", f"{val}|{_norm_unit(m.group(2))}", m)

    # 5. Percentages
    for m in re.finditer(r"(?<![A-Za-z0-9])(-?\d+(?:[.,]\d+)?)\s*(%|percent\b|phần trăm\b|퍼센트)", text, re.I):
        val = _norm_decimal(m.group(1), allow_vi_comma=allow_vi_comma)
        if val is not None:
            add("percent", f"{val}%", m)

    # 6. Currency
    for m in re.finditer(r"([$£€¥])\s*(-?\d+(?:[.,]\d+)?)(?![A-Za-z0-9])", text):
        val = _norm_decimal(m.group(2), allow_vi_comma=allow_vi_comma)
        if val is not None:
            add("currency", f"{val}|{_CURRENCY_WORD_MAP[m.group(1)]}", m)
    for m in re.finditer(
        r"(?<![A-Za-z0-9])(-?\d+(?:[.,]\d+)?)\s*(dollars?|usd|đồng|vnd|đô la|원|달러|美元|元|块)(?![A-Za-z0-9])",
        text,
        re.I,
    ):
        val = _norm_decimal(m.group(1), allow_vi_comma=allow_vi_comma)
        cur = _CURRENCY_WORD_MAP.get(m.group(2).lower(), m.group(2).lower())
        if val is not None:
            add("currency", f"{val}|{cur}", m)

    # 7. Leading-zero codes / phone numbers (preserving exact leading 0, optional single spaces/hyphens inside one segment)
    for m in re.finditer(r"(?<![A-Za-z0-9-])(?<!\d[.,])(0\d(?:[ -]?\d){1,})(?![A-Za-z0-9])(?![.,]\d)", text):
        digits = re.sub(r"[ -]", "", m.group(1))
        add("code_or_phone", digits, m)

    # 8. Standalone signed or unsigned integers / decimals (strict boundaries!)
    for m in re.finditer(
        r"(?<![A-Za-z0-9-])(?<!\d[.,])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])(?![.,]\d)",
        text,
    ):
        val = _norm_decimal(m.group(1), allow_vi_comma=False)
        if val is not None:
            add("number", val, m)

    tokens.sort(key=lambda t: t.span[0])
    return tokens


def _parse_expected_item(item: str | dict, lang: str = "en") -> EntityToken | None:
    if isinstance(item, dict):
        kind = str(item["kind"])
        norm = str(item.get("norm") or item["value"])
        raw = str(item.get("raw") or item.get("value") or norm)
        return EntityToken(kind=kind, norm=norm, raw=raw, span=(0, len(raw)))
    raw = str(item).strip()
    if not raw:
        return None
    parsed = extract_entities(raw, lang=lang)
    if len(parsed) == 1 and parsed[0].raw.strip() == raw:
        return parsed[0]
    if parsed and len(parsed) == 1:
        return parsed[0]
    # Pure alphabetic flight carrier fragment like "VN" shouldn't coerce into a full typed token
    return EntityToken(kind="untyped", norm=raw.lower(), raw=raw, span=(0, len(raw)))


def _has_spoken_numeral_words(text: str) -> bool:
    if any(ch in SPOKEN_NUMERAL_CHARS_CJK for ch in text):
        return True
    words = {w.lower() for w in re.findall(r"[^\W\d_]+", text, re.UNICODE)}
    return bool(
        (words & SPOKEN_NUMERAL_WORDS_EN)
        or (words & SPOKEN_NUMERAL_WORDS_VI)
        or (words & SPOKEN_NUMERAL_WORDS_KO)
    )


def _match_written_coverage(written_tokens: list[EntityToken], expected_tokens: list[EntityToken]) -> tuple[int, list[EntityToken]]:
    """Measure how many entities detected in written text are actually covered by expected labels."""
    if not written_tokens:
        return 0, []
    used_exp: set[int] = set()
    covered = 0
    uncovered: list[EntityToken] = []
    for wt in written_tokens:
        match_idx = None
        for idx, et in enumerate(expected_tokens):
            if idx in used_exp:
                continue
            if wt.kind == et.kind and wt.norm == et.norm:
                match_idx = idx
                break
        if match_idx is not None:
            used_exp.add(match_idx)
            covered += 1
        else:
            uncovered.append(wt)
    return covered, uncovered


def score_text_entities(
    written: str | None,
    asr_text: str,
    expected_entities: Iterable[str | dict] | None = None,
    lang: str = "en",
) -> dict:
    """Score ASR output against expected entities and written ground truth with tri-state & coverage reporting."""
    written_clean = (written or "").strip()
    is_synthetic_stub = written_clean.startswith("Synthetic ")
    written_tokens = [] if (not written_clean or is_synthetic_stub) else extract_entities(written_clean, lang=lang)

    expected_list = list(expected_entities) if expected_entities is not None else []
    expected_tokens: list[EntityToken] = []
    for item in expected_list:
        tok = _parse_expected_item(item, lang=lang)
        if tok is not None:
            expected_tokens.append(tok)

    asr_tokens = extract_entities(asr_text or "", lang=lang)
    consumed_asr: set[int] = set()
    spoken_in_asr = _has_spoken_numeral_words(asr_text or "")

    # Pre-match expected tokens first to find exact typed boundary matches (consuming each ASR token once)
    matched_idx_by_exp: dict[int, int] = {}
    for e_idx, et in enumerate(expected_tokens):
        for a_idx, at in enumerate(asr_tokens):
            if a_idx in consumed_asr:
                continue
            if et.kind == at.kind and et.norm == at.norm:
                consumed_asr.add(a_idx)
                matched_idx_by_exp[e_idx] = a_idx
                break

    consumed_spans = [asr_tokens[idx].span for idx in consumed_asr]
    has_unconsumed_asr_digits = any(a_idx not in consumed_asr for a_idx in range(len(asr_tokens))) or any(
        not any(sa <= m.start() < sb for sa, sb in consumed_spans)
        for m in re.finditer(r"\d", asr_text or "")
    )

    items: list[dict] = []
    correct_count = 0
    wrong_count = 0
    unknown_count = 0

    for e_idx, et in enumerate(expected_tokens):
        if e_idx in matched_idx_by_exp:
            a_tok = asr_tokens[matched_idx_by_exp[e_idx]]
            correct_count += 1
            items.append({
                "expected": et.raw,
                "kind": et.kind,
                "norm": et.norm,
                "status": "correct",
                "matched_raw": a_tok.raw,
                "reason": "exact_typed_match",
            })
        elif spoken_in_asr and not has_unconsumed_asr_digits:
            unknown_count += 1
            items.append({
                "expected": et.raw,
                "kind": et.kind,
                "norm": et.norm,
                "status": "unknown",
                "matched_raw": None,
                "reason": "unparsed_spoken",
            })
        else:
            wrong_count += 1
            items.append({
                "expected": et.raw,
                "kind": et.kind,
                "norm": et.norm,
                "status": "wrong",
                "matched_raw": None,
                "reason": "missing_or_mismatched_entity",
            })

    # Coverage against typed entities extracted from written text
    covered_written, uncovered_written = _match_written_coverage(written_tokens, expected_tokens)
    for wt in uncovered_written:
        unknown_count += 1
        items.append({
            "expected": wt.raw,
            "kind": wt.kind,
            "norm": wt.norm,
            "status": "unknown",
            "matched_raw": None,
            "reason": f"missing_{wt.kind}_label",
        })

    if is_synthetic_stub and not expected_tokens:
        unknown_count += 1
        items.append({
            "expected": None,
            "kind": "unlabeled_case",
            "norm": None,
            "status": "unknown",
            "matched_raw": None,
            "reason": "synthetic_stub_without_written_text",
        })

    if written_tokens:
        coverage: float | None = round(covered_written / len(written_tokens), 4)
    elif is_synthetic_stub:
        coverage = 0.0
    elif written_clean:
        coverage = 1.0 if expected_tokens or not _has_spoken_numeral_words(written_clean) else 0.0
    else:
        coverage = None

    return {
        "scorer": "typed_boundary_entity_v1",
        "coverage": coverage,
        "labeled_detected": covered_written,
        "total_detected": len(written_tokens),
        "labeled_expected": len(expected_tokens),
        "correct": correct_count,
        "wrong": wrong_count,
        "unknown": unknown_count,
        "all_labeled_correct": len(expected_tokens) > 0 and wrong_count == 0 and unknown_count == 0,
        "items": items,
    }


def rescore_vad_report(vad_json_path, split_path, out_path) -> dict:
    """Read-only rescore of historical vad_ablation.json without mutating historical artifacts."""
    import json
    from pathlib import Path

    vad_data = json.loads(Path(vad_json_path).read_text(encoding="utf-8"))
    split_data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    by_id = {}
    for sec in ("native_en_dev", "native_en_holdout", "synthetic_dev"):
        for c in split_data.get(sec, []):
            by_id[c["id"]] = c

    rescored = {"source_vad_json": str(vad_json_path), "source_split_json": str(split_path), "profiles": {}}
    for prof in ("legacy", "upstream"):
        rows_out = []
        for row in vad_data.get("full_audit", {}).get(prof, []):
            case = by_id.get(row["id"], {})
            ev = score_text_entities(
                case.get("written"),
                row.get("asr_raw", ""),
                case.get("expected_entities"),
                lang=row.get("src", "en"),
            )
            rows_out.append({
                "id": row["id"],
                "src": row.get("src"),
                "input_kind": row.get("input_kind"),
                "asr_raw": row.get("asr_raw", ""),
                "legacy_lexical_proxy": row.get("entity_check"),
                "entity_eval": ev,
            })
        covs = [r["entity_eval"]["coverage"] for r in rows_out if r["entity_eval"]["coverage"] is not None]
        rescored["profiles"][prof] = {
            "cases_total": len(rows_out),
            "cases_all_labeled_correct": sum(1 for r in rows_out if r["entity_eval"]["all_labeled_correct"]),
            "entities_correct": sum(r["entity_eval"]["correct"] for r in rows_out),
            "entities_wrong": sum(r["entity_eval"]["wrong"] for r in rows_out),
            "entities_unknown": sum(r["entity_eval"]["unknown"] for r in rows_out),
            "written_entities_detected": sum(r["entity_eval"]["total_detected"] for r in rows_out),
            "written_entities_labeled": sum(r["entity_eval"]["labeled_detected"] for r in rows_out),
            "mean_coverage": round(sum(covs) / len(covs), 4) if covs else None,
            "cases": rows_out,
        }
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rescored, indent=2, ensure_ascii=False), encoding="utf-8")
    return rescored

