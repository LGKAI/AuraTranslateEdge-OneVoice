"""Typed, value-preserving TN. Grammars recognize classes, never fixture strings.

An optional existing CRF proposes identifier spans; it cannot generate text.
All rendering is deterministic from immutable numeric fields. Unknown/ambiguous
input is retained or spelled conservatively and reported, rather than guessed.
"""
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
import json
import os
import re
import time
import unicodedata

from .format_locales import (CURRENCY, CURRENCY_ALIASES, DIGITS, IDENTIFIER_CUES,
                             KO_HOURS, LETTERS, MONTHS_EN, UNITS)
from .textnorm import _vi3, _zh4, en_num

LANGS = ('vi', 'en', 'ko', 'zh')
NUM = r'[+\-−]?\d+(?:[.,，．]\d+)*'
_CRF = None


def join(lang, words):
    return ('' if lang == 'zh' else ' ').join(w for w in words if w)


def digits(raw, lang):
    return join(lang, [DIGITS[lang][int(c)] for c in raw])


def cardinal(n, lang):
    if n == 0:
        return DIGITS[lang][0]
    if n < 0:
        return join(lang, [dict(vi='âm', en='minus', ko='마이너스', zh='负')[lang], cardinal(-n, lang)])
    if lang == 'en':
        for scale, word in [(10**18, 'quintillion'), (10**15, 'quadrillion'), (10**12, 'trillion')]:
            if n >= scale:
                return cardinal(n//scale, lang) + ' ' + word + (' ' + cardinal(n%scale, lang) if n%scale else '')
        return en_num(n)
    if lang == 'zh':
        groups = []; i = 0
        while n:
            n, r = divmod(n, 10000)
            if r: groups.append((i, r))
            i += 1
        out = ''; previous = None
        for i, r in reversed(groups):
            if previous is not None and (previous-i > 1 or r < 1000): out += '零'
            scale = ('万' if i % 2 else '') + '亿'*(i//2)
            out += _zh4(r, first=not out) + scale
            previous = i
        return out
    if lang == 'vi':
        parts = []; i = 0
        while n:
            n, r = divmod(n, 1000)
            unit = (('', 'nghìn', 'triệu')[i % 3] + (' tỷ' * (i//3))).strip()
            if r:
                parts.append((_vi3(r, bool(n)) + ' ' + unit).strip())
            i += 1
        return ' '.join(reversed(parts))
    parts = []; i = 0
    scales = ['', '만', '억', '조', '경', '해']
    while n:
        n, r = divmod(n, 10000)
        if r:
            group = ''
            for p, c in enumerate(str(r).zfill(4)):
                d = int(c); pos = 3-p
                if d:
                    group += ('' if d == 1 and pos else DIGITS['ko'][d]) + ['', '십', '백', '천'][pos]
            parts.append(group + scales[i])
        i += 1
    return ''.join(reversed(parts))


def characters(raw, lang):
    table = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', LETTERS[lang]))
    symbols = {
        '+': ('cộng', 'plus', '플러스', '加'), '-': ('gạch ngang', 'dash', '하이픈', '杠'),
        '/': ('gạch chéo', 'slash', '슬래시', '斜杠'), ':': ('hai chấm', 'colon', '콜론', '冒号'),
        '.': ('chấm', 'dot', '점', '点'), ',': ('phẩy', 'comma', '쉼표', '逗号'),
        '(': ('mở ngoặc', 'open parenthesis', '괄호 열기', '左括号'),
        ')': ('đóng ngoặc', 'close parenthesis', '괄호 닫기', '右括号'),
    }
    out = []
    for c in raw:
        if c.isdecimal(): out.append(DIGITS[lang][int(c)])
        elif c.upper() in table: out.append(table[c.upper()])
        elif c in symbols: out.append(symbols[c][LANGS.index(lang)])
        elif not c.isspace(): out.append(c)
    return join(lang, out)


def numeric_fields(raw, lang):
    """Parse separators without float conversion; preserve fractional precision."""
    raw = ''.join(str(int(c)) if c.isdecimal() else c for c in raw).translate(str.maketrans({'，':',', '．':'.'}))
    sign = '-' if raw.startswith(('-', '−')) else '+' if raw.startswith('+') else ''
    s = raw.lstrip('+-−')
    decimal = ',' if lang == 'vi' else '.'
    if '.' in s and ',' in s:
        sep = '.' if s.rfind('.') > s.rfind(',') else ','
        other = ',' if sep == '.' else '.'
        whole, frac = s.rsplit(sep, 1)
        if sep in whole or not re.fullmatch(r'\d{1,3}(?:' + re.escape(other) + r'\d{3})+', whole):
            raise ValueError('invalid mixed number separators')
        whole = whole.replace(other, '')
    elif '.' in s or ',' in s:
        sep = '.' if '.' in s else ','
        groups = s.split(sep)
        if sep != decimal and all(len(g) == 3 for g in groups[1:]) and len(groups[0]) <= 3:
            whole, frac = ''.join(groups), ''
        elif len(groups) == 2:
            whole, frac = groups
        elif sep != decimal and all(len(g) == 3 for g in groups[1:]):
            whole, frac = ''.join(groups), ''
        else:
            raise ValueError('ambiguous or invalid repeated decimal separator')
    else:
        whole, frac = s, ''
    return {'sign': sign, 'whole': whole, 'fraction': frac}


def number_words(value, lang):
    whole, frac = value['whole'], value['fraction']
    # Very large values are intentionally spelled rather than overrun scale tables.
    out = cardinal(int(whole), lang) if len(whole) <= 18 else digits(whole, lang)
    if frac:
        out = join(lang, [out, dict(vi='phẩy', en='point', ko='점', zh='点')[lang], digits(frac, lang)])
    if value['sign']:
        prefix = dict(vi='âm', en='minus', ko='마이너스', zh='负') if value['sign'] == '-' else dict(vi='dương', en='plus', ko='플러스', zh='正')
        out = join(lang, [prefix[lang], out])
    return out


def ordinal_en(n):
    special = {1:'first', 2:'second', 3:'third', 5:'fifth', 8:'eighth', 9:'ninth', 12:'twelfth'}
    if n in special: return special[n]
    if n < 20: return cardinal(n, 'en') + 'th'
    if n % 10 == 0: return cardinal(n, 'en')[:-1] + 'ieth'
    return cardinal(n//10*10, 'en') + ' ' + ordinal_en(n%10)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    raw: str
    value: dict
    spoken: str
    source: str = 'grammar'


class HybridNormalizer:
    def __init__(self, date_order=None, learned_identifiers=True):
        self.date_order = date_order
        self.learned_identifiers = learned_identifiers
        self.records = []
        self.identifier_model_status = 'not_requested'

    @staticmethod
    def _context_identifier(text, start):
        prefix = re.split(r'[.!?。！？;；\n]', text[:start])[-1].casefold()[-64:]
        # Limit proximity; a cue in an earlier clause must not turn amounts into IDs.
        for cue in IDENTIFIER_CUES:
            cjk = any('\u3400' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in cue)
            pattern = re.escape(cue) if cjk else r'(?<!\w)' + re.escape(cue) + r'(?!\w)'
            if re.search(pattern + r'[^\d,，;；]{0,16}$', prefix): return True
        return False

    def _learned_spans(self, text, lang):
        global _CRF
        if not self.learned_identifiers or lang not in ('en', 'vi'):
            return []
        try:
            if _CRF is None:
                from pathlib import Path
                from .identifier_tagger import IdentifierTagger
                _CRF = IdentifierTagger(Path(__file__).resolve().parents[2] / 'models/normalization/identifier.crfsuite')
            self.identifier_model_status = 'CRF loaded'
            found = []; cursor = 0
            for raw in _CRF(text, lang):
                a = text.find(raw, cursor)
                if a < 0: continue
                cursor = a+len(raw)
                if re.fullmatch(r'[A-Za-z\d+() -]+', raw) and any(c.isdecimal() for c in raw):
                    found.append((a, cursor))
            return found
        except (ImportError, OSError):
            self.identifier_model_status = 'unavailable; context grammars only'
            return []

    def render(self, kind, raw, m, lang):
        if kind == 'unsupported':
            raise ValueError('unsupported numeric notation; do not infer its semantic value')
        if kind == 'date':
            nums = re.findall(r'\d+', raw)
            order = self.date_order or os.environ.get('S2S_DATE_ORDER', '')
            month_word = next((i+1 for i, name in enumerate(MONTHS_EN) if name.casefold() in raw.casefold()), None)
            if month_word:
                month = month_word; day = int(nums[0]); y = int(nums[1]) if len(nums) == 2 else 2000
                has_year = len(nums) == 2
            elif len(nums[0]) == 4:
                y, month, day = map(int, nums)
                has_year = True
            else:
                if 'tháng' in raw.casefold(): order = 'DMY'
                if not order:
                    if int(nums[0]) > 12: order = 'DMY'
                    elif int(nums[1]) > 12: order = 'MDY'
                    else: raise ValueError('ambiguous date; configure S2S_DATE_ORDER=DMY or MDY')
                if order not in ('DMY', 'MDY'): raise ValueError('unsupported date order')
                day, month = map(int, nums[:2]) if order == 'DMY' else map(int, reversed(nums[:2]))
                y = int(nums[2]) if len(nums) == 3 else 2000  # validation only for yearless dates
                if len(nums) == 3 and len(nums[2]) != 4: raise ValueError('two-digit year is ambiguous')
                has_year = len(nums) == 3
            date(y, month, day)
            if lang == 'vi': spoken = join(lang, ['ngày', cardinal(day, lang), 'tháng', cardinal(month, lang), 'năm' if has_year else '', cardinal(y, lang) if has_year else ''])
            elif lang == 'en': spoken = MONTHS_EN[month-1] + ' ' + ordinal_en(day) + (' ' + cardinal(y, lang) if has_year else '')
            elif lang == 'ko': spoken = (cardinal(y, lang)+' 년 ' if has_year else '') + cardinal(month, lang)+' 월 '+cardinal(day, lang)+' 일'
            else: spoken = (digits(str(y), lang)+'年' if has_year else '') + cardinal(month, lang)+'月'+cardinal(day, lang)+'日'
            return {'year': y if has_year else None, 'month': month, 'day': day}, spoken
        if kind == 'year':
            year = re.search(r'\d+', raw).group()
            return {'year': int(year)}, (digits(year, lang) if lang == 'zh' else cardinal(int(year), lang)) + ('年' if lang == 'zh' else ' 년')
        if kind == 'time':
            h, minute = int(m['hour']), int(m['minute'])
            second = int(m['second']) if m.groupdict().get('second') else None
            ampm = (m.groupdict().get('ampm') or '').replace('.', '').upper()
            if not 0 <= minute < 60 or not 0 <= h < 24 or (ampm and not 1 <= h <= 12) or (second is not None and not 0 <= second < 60):
                raise ValueError('invalid clock time')
            tail = {'vi': {'AM':'sáng', 'PM':'chiều'}, 'en': {'AM':'ay em', 'PM':'pee em'}, 'ko': {'AM':'오전', 'PM':'오후'}, 'zh': {'AM':'上午', 'PM':'下午'}}[lang].get(ampm, '')
            if lang == 'vi' and h == 12 and ampm: tail = 'trưa' if ampm == 'PM' else 'đêm'
            if lang == 'vi': spoken = join(lang, [cardinal(h, lang), 'giờ', cardinal(minute, lang) if minute else '', 'phút' if minute else '', cardinal(second, lang)+' giây' if second is not None else '', tail])
            elif lang == 'en': spoken = join(lang, [cardinal(h, lang), ('oh ' if minute < 10 else '')+cardinal(minute, lang) if minute else "o'clock", cardinal(second, lang)+' seconds' if second is not None else '', tail])
            elif lang == 'ko': spoken = join(lang, [tail, KO_HOURS[h] if 0 < h <= 12 else cardinal(h, lang), '시', cardinal(minute, lang)+' 분' if minute else '', cardinal(second, lang)+' 초' if second is not None else ''])
            else: spoken = join(lang, [tail, cardinal(h, lang), '点', cardinal(minute, lang)+'分' if minute else '', cardinal(second, lang)+'秒' if second is not None else ''])
            return {'hour': h, 'minute': minute, 'second': second, 'ampm': ampm}, spoken
        if kind == 'identifier':
            # Only whitespace is grouping; retain plus/dash/parentheses as symbols.
            canonical = ''.join(c for c in raw if not c.isspace())
            return {'characters': canonical}, characters(canonical, lang)
        if kind == 'money':
            alias = m['currency']; currency = {k.casefold():v for k,v in CURRENCY_ALIASES.items()}[alias.casefold()]
            value = numeric_fields(m['number'], lang)
            major, minor = CURRENCY[currency][lang]
            frac = value['fraction']
            if minor and len(frac) in (1, 2):
                n, cents = int(value['whole']), int(frac.ljust(2, '0'))
                if lang == 'en':
                    major = major if n == 1 else major+'s'
                    minor = ('pence' if minor == 'penny' else minor+'s') if cents != 1 else minor
                unsigned = dict(value, fraction='')
                spoken = join(lang, [number_words(unsigned, lang), major, cardinal(cents, lang), minor])
            else:
                if lang == 'en' and Decimal(value['whole']+'.'+(frac or '0')) != 1 and currency in ('USD', 'EUR', 'GBP'): major += 's'
                spoken = join(lang, [number_words(value, lang), major])
            return {'currency': currency, 'number': value}, spoken
        if kind in ('percent', 'measure'):
            value = numeric_fields(m['number'], lang)
            if kind == 'percent':
                unit = ('phần trăm', 'percent', '퍼센트', '百分之')[LANGS.index(lang)]
                spoken = join(lang, [unit, number_words(value, lang)] if lang == 'zh' else [number_words(value, lang), unit])
            else:
                unit = m['unit'].casefold().replace(' ', '')
                label = UNITS[unit][LANGS.index(lang)]
                if lang == 'en' and Decimal(value['whole']+'.'+(value['fraction'] or '0')) == 1:
                    head, *rest = label.split()
                    if head.endswith('s'): head = head[:-1]
                    label = ' '.join([head]+rest)
                spoken = join(lang, [number_words(value, lang), label])
            return {'number': value, 'unit': '%' if kind == 'percent' else unit}, spoken
        value = numeric_fields(raw, lang)
        return value, number_words(value, lang)

    def __call__(self, text, lang):
        if lang not in LANGS: raise ValueError('unsupported language')
        start_time = time.perf_counter()
        # Keep offsets relative to the supplied string. Unicode Nd digits are parsed by int().
        currency = '|'.join(re.escape(s) for s in sorted(CURRENCY_ALIASES, key=len, reverse=True))
        units = '|'.join(re.escape(s) for s in sorted(UNITS, key=len, reverse=True))
        patterns = [
            ('date', r'(?<!\d)(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)|\d{4}年\s*\d{1,2}月\s*\d{1,2}日|\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}/\d{1,2}(?:/\d{2,4})?(?!\d)|\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?)'),
            ('date', r'\b(?:'+'|'.join(MONTHS_EN)+r')\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b'),
            ('year', r'(?<!\d)\d{4}\s*(?:年|년)'),
            ('time', r'(?<![\d:：])(?P<hour>\d{1,2})\s*[:：hH]\s*(?P<minute>\d{2})(?:[:：](?P<second>\d{2}))?(?:\s*(?P<ampm>[aApP]\.?[mM]\.?))?(?!\d)'),
            ('money', r'(?P<currency>'+currency+r')\s*(?P<number>'+NUM+r')(?!\d)'),
            ('money', r'(?P<number>'+NUM+r')\s*(?P<currency>'+currency+r')(?![A-Za-z])'),
            ('percent', r'(?P<number>'+NUM+r')\s*[%％]'),
            ('measure', r'(?P<number>'+NUM+r')\s*(?P<unit>'+units+r')(?![A-Za-z])'),
            ('unsupported', r'(?<![\w-])\d+(?:[.,]\d+)?[eE][+\-]?\d+(?!\w)'),
            ('unsupported', r'(?<![\w-])\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?(?![\w-])'),
        ]
        candidates = []
        for kind, pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidates.append((m.start(), m.end(), kind, m, 'grammar'))
        # An identifier is a whole code/sequence, not a substring of a number.
        for m in re.finditer(r'(?<![A-Za-z\d])(?:\+?\d[\d ()-]*\d|[A-Za-z]+[A-Za-z\d-]*\d[A-Za-z\d-]*|\d+[A-Za-z][A-Za-z\d]*)(?![A-Za-z\d])', text):
            raw = m.group().rstrip(' -')
            context = self._context_identifier(text, m.start())
            mixed = bool(re.search('[A-Za-z]', raw)) and bool(re.search(r'\d', raw))
            # Single letter suffixes can be units; grammar candidates take priority.
            if context or (mixed and len(re.findall('[A-Za-z]', raw)) >= 2):
                candidates.append((m.start(), m.start()+len(raw), 'identifier', m, 'context' if context else 'alphanumeric grammar'))
        for a, b in self._learned_spans(text, lang):
            candidates.append((a, b, 'identifier', None, 'CRF proposal'))
        for m in re.finditer(NUM, text):
            candidates.append((m.start(), m.end(), 'number', m, 'grammar'))
        priority = {'date': 0, 'year': 1, 'time': 1, 'money': 2, 'percent': 2, 'measure': 2, 'identifier': 3, 'unsupported': 4, 'number': 5}
        selected = []
        for a, b, kind, match, source in sorted(candidates, key=lambda c: (priority[c[2]], c[0], -(c[1]-c[0]))):
            if any(a < s.end and b > s.start for s in selected): continue
            raw = text[a:b]
            try:
                value, spoken = self.render(kind, raw, match, lang)
                if kind == 'date' and lang == 'vi' and re.search(r'\bngày\s*$', text[:a], re.IGNORECASE):
                    spoken = spoken.removeprefix('ngày ')
                if kind == 'number' and value['whole'].startswith('0') and len(value['whole']) > 1 and not value['fraction']:
                    kind = 'identifier'
                    value = {'characters': raw, 'warning': 'untyped leading-zero sequence; preserve characters'}
                    spoken = characters(raw, lang)
            except (ValueError, IndexError, KeyError) as exc:
                value, spoken = {'reason': str(exc)}, characters(raw, lang)
                kind = 'unresolved'
            selected.append(Span(a, b, kind, raw, value, spoken, source))
        selected.sort(key=lambda s: s.start)
        cursor = 0; parts = []; spoken_ranges = []; warnings = []
        for s in selected:
            assert s.start >= cursor and text[s.start:s.end] == s.raw
            parts.append(text[cursor:s.start]); begin = sum(len(p) for p in parts)
            parts.append(s.spoken); spoken_ranges.append((begin, begin+len(s.spoken)))
            if s.kind == 'unresolved': warnings.append({'raw': s.raw, 'reason': s.value['reason'], 'policy': 'spell characters, do not infer value'})
            if 'warning' in s.value:
                warnings.append({'raw': s.raw, 'reason': s.value['warning'], 'policy': 'spell characters'})
            cursor = s.end
        if self.identifier_model_status.startswith('unavailable'):
            warnings.append({'reason': self.identifier_model_status, 'policy': 'install python-crfsuite for learned identifier proposals'})
        parts.append(text[cursor:]); output = ''.join(parts)
        record = {'stage': 'tts_hybrid', 'lang': lang, 'input': text, 'output': output,
                  'spans': [asdict(s) for s in selected], 'protected_ranges': spoken_ranges,
                  'warnings': warnings, 'seconds': time.perf_counter()-start_time,
                  'identifier_model': self.identifier_model_status,
                  'validation': 'raw span identity; typed deterministic renderer; no generative numeric rewriting'}
        self.records.append(record)
        if len(self.records) > 128: del self.records[:-128]
        return output, record


_NORMALIZER = None


def normalize(text, lang):
    global _NORMALIZER
    if _NORMALIZER is None: _NORMALIZER = HybridNormalizer()
    result, record = _NORMALIZER(text, lang)
    log = os.environ.get('S2S_TN_AUDIT')
    if log:
        with open(log, 'a', encoding='utf-8') as f: f.write(json.dumps(record, ensure_ascii=False)+'\n')
    return result, record
