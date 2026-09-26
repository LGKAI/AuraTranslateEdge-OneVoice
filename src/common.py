"""Shared helpers for Step-0 (Audio Front-end) tests: VAD, denoise, beamform."""
import os
import re
import glob
import time

import numpy as np
import soundfile as sf
import librosa
import torch

SR = 16000
AUDIO_EXT = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")


def _ensure_utf8_stdout():
    """Windows' console often leaves Python's stdout/stderr on a legacy
    codepage (cp1252), which can't encode Vietnamese/Chinese/Korean text --
    print() then raises UnicodeEncodeError on the first non-Latin1 character
    (e.g. 'a with breve'), crashing a run after all the actual work (model
    inference, WER/CER scoring, CSV write) already succeeded. Force UTF-8 so
    printing a transcribed hypothesis for a live progress log can't do that."""
    import sys
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


_ensure_utf8_stdout()


def _stub_broken_torchaudio():
    """torchaudio's native extension can be ABI-incompatible with the
    installed torch build (seen on this machine as WinError 127 / "procedure
    not found"). Nothing here calls torchaudio directly, but some libraries
    `import torchaudio` unconditionally at import time -- silero's hubconf,
    and transformers>=5.13's audio_utils (pulled in by any AutoProcessor
    import, needed for Qwen3-ASR). Stub it out before that first real import
    happens; once a module is in sys.modules, every later `import torchaudio`
    anywhere else in the process just reuses this harmless stub instead of
    re-running torchaudio's real (crashing) init code.

    The stub needs a real __spec__: transformers' own availability check
    (`importlib.util.find_spec("torchaudio")`, used by is_torchaudio_available)
    reads __spec__ off whatever's already sitting in sys.modules, and raises
    ValueError if that attribute is missing or None -- which a bare
    types.ModuleType(...) leaves it as by default."""
    try:
        import torchaudio  # noqa: F401
    except Exception:
        import sys
        import types
        import importlib.machinery
        stub = types.ModuleType("torchaudio")
        stub.__version__ = "0.0.0-stub"
        stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        sys.modules["torchaudio"] = stub


_stub_broken_torchaudio()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_audio_files(folder):
    files = []
    if not os.path.isdir(folder):
        return files
    for ext in AUDIO_EXT:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)


def load_wav(path, sr=SR):
    """Load audio as mono float32 at target sample rate."""
    wav, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def save_wav(path, wav, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > 0.99:
        wav = wav / peak * 0.99
    sf.write(path, wav.astype(np.float32), sr)


class Timer:
    """`with Timer() as t: ...` then read t.elapsed (seconds)."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def rtf(process_seconds, audio_seconds):
    """Real-Time Factor = processing time / audio duration. <1 = faster than real-time."""
    return process_seconds / max(audio_seconds, 1e-9)


def normalize_text(s):
    """Lowercase + strip punctuation/extra whitespace for fair WER/CER comparison.
    \\w is Unicode-aware in Python 3, so accented/tonal letters (Vietnamese, etc.)
    are kept -- only punctuation and casing are normalized away."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bleu(hyp, ref, tgt_lang):
    """Sentence-level sacreBLEU, tokenized appropriately for the target
    language ("zh" tokenizer for Chinese -- it has no whitespace word
    boundaries, so the default 13a tokenizer would undercount matches;
    13a is fine for Vietnamese/English/Korean, which are whitespace-delimited)."""
    import sacrebleu
    tokenize = "zh" if tgt_lang == "zh" else "13a"
    return sacrebleu.sentence_bleu(hyp, [ref], tokenize=tokenize).score


def normalize_text_for_cer(s):
    """normalize_text() then strip ALL whitespace, for character-level (CER)
    scoring only. Some CJK reference transcripts (FLEURS zh, confirmed by
    inspection) insert a space between every character; jiwer.cer() counts
    those spaces as real characters, so a hypothesis with normal (no
    inter-character space) output gets scored as if every character were a
    deletion -- inflating CER from ~5% to ~50%+ on otherwise near-perfect
    transcriptions. Word-level WER scoring must NOT use this: spaces are the
    actual word boundaries there."""
    return re.sub(r"\s+", "", normalize_text(s))
