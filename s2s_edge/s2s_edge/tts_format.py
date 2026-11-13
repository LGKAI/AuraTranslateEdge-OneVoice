"""Normalization dispatch and chunk planning shared by every TTS language."""
import os
import re
import unicodedata
from . import hybrid_textnorm, neural_textnorm
from .textnorm import normalize as legacy_normalize


def resolve_mode(tts_format=None):
    value = os.environ.get('S2S_TTS_FORMAT', 'hybrid') if tts_format is None else str(tts_format)
    if value not in ('hybrid', 'legacy', 'neural'):
        raise ValueError('S2S_TTS_FORMAT must be hybrid, legacy or neural')
    return value


def mode(tts_format=None):
    return resolve_mode(tts_format)


def prepare(text, lang, mode=None):
    selected = resolve_mode(mode)
    if selected == 'neural' and not neural_textnorm.usable()[0]:
        neural_textnorm.warn_unusable('tts_format=neural'); selected = 'hybrid'
    if selected == 'hybrid':
        return hybrid_textnorm.normalize(text, lang)
    if selected == 'neural' and lang in ('en', 'vi'):
        result = neural_textnorm.get()(text, lang, stage='tts')
    else:
        result = legacy_normalize(text, lang)
    return result, {'stage': 'tts_'+selected, 'input': text, 'output': result,
                    'spans': [], 'protected_ranges': [], 'warnings': []}


def chunks(text, lang, max_units, unit, record=None, counter=None):
    """Preserve tokens; prefer sentence breaks and keep typed spans when they fit.

    Oversized numeric spans can exceed graph limits and are split at word/CJK
    character boundaries. These boundaries are reported, never silently truncated.
    English words are never split; an oversized word is an explicit error.
    """
    count = counter or ((lambda s: len(s.split())) if unit == 'word' else len)
    ranges = []
    cursor = 0
    for span in (record or {}).get('spans', []):
        spoken = span['spoken'].lower() if lang == 'vi' else span['spoken']
        a = text.find(spoken, cursor)
        if a >= 0:
            b = a+len(spoken); cursor = b
            if count(spoken) <= max_units: ranges.append((a, b))
    boundaries = {0, len(text)}
    for i in range(1, len(text)):
        left, right = text[i-1], text[i]
        if unicodedata.combining(right): continue
        if left.isspace() or right.isspace() or left in '.!?。！？;；,，':
            boundaries.add(i)
        elif lang == 'zh' and not (left.isascii() and right.isascii() and left.isalnum() and right.isalnum()):
            boundaries.add(i)
    boundaries = sorted(i for i in boundaries if not any(a < i < b for a, b in ranges))
    out = []; start = 0
    while start < len(text):
        options = [b for b in boundaries if b > start and count(text[start:b]) <= max_units]
        if not options:
            raise ValueError('An indivisible TTS token exceeds the graph input budget')
        end = max(options)
        sentence = [b for b in options if b > start and text[start:b].rstrip().endswith(('.', '!', '?', '。', '！', '？', ';', '；'))]
        if end < len(text) and sentence: end = max(sentence)
        piece = text[start:end].strip()
        if piece: out.append(piece)
        start = end
    if record is not None:
        record['planned_chunks'] = out
        record['chunk_policy'] = 'never cut Latin/Hangul tokens; keep typed spans if graph budget permits'
    return out


def adaptive(text, lang, max_units, unit, synth, record=None, depth=0, counter=None):
    out = []
    for piece in chunks(text, lang, max_units, unit, record, counter):
        wav, capped = synth(piece)
        if capped:
            smaller = max_units//2
            if depth >= 8 or smaller < 1:
                raise RuntimeError('TTS graph audio cap reached for an indivisible token')
            if record is not None: record.setdefault('audio_cap_retries', []).append(piece)
            out += adaptive(piece, lang, smaller, unit, synth, None, depth+1, counter)
        else:
            out.append(wav)
    return out
