# -*- coding: utf-8 -*-
"""Audio I/O and utterance segmentation (pure numpy; VAD is energy based so it runs anywhere, no extra model)."""
import os, subprocess, shutil, wave, numpy as np

SR = 16000
# ALSA devices for the live demo. The KIMQ8550 box has no built-in mic/speaker, so the USB device is often NOT the ALSA default:
# find it with `arecord -l` / `aplay -l` (tools/audio_check.sh) and set e.g. S2S_MIC=plughw:1,0 S2S_SPK=plughw:1,0.
_MIC = ["-D", os.environ["S2S_MIC"]] if os.environ.get("S2S_MIC") else []
_SPK = ["-D", os.environ["S2S_SPK"]] if os.environ.get("S2S_SPK") else []


def load_audio(path, sr=SR):
    import soundfile as sf
    w, r = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1: w = w.mean(axis=1)
    if r != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(r), sr); w = resample_poly(w, sr // g, int(r) // g).astype(np.float32)
    return w


def save_wav(path, x, sr):
    x = np.clip(np.asarray(x, np.float32), -1, 1)
    with wave.open(path, "wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(int(sr)); f.writeframes((x * 32767).astype(np.int16).tobytes())


def record(seconds, sr=SR):
    """Capture from the default microphone (ALSA arecord, or sounddevice if available)."""
    if shutil.which("arecord"):
        p = subprocess.run(["arecord", "-q", *_MIC, "-f", "S16_LE", "-r", str(sr), "-c", "1", "-d", str(int(seconds)), "-t", "raw"], capture_output=True, check=True)
        return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0
    import sounddevice as sd
    return sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", blocking=True)[:, 0]


def play(x, sr):
    import tempfile, os
    p = tempfile.mktemp(suffix=".wav"); save_wav(p, x, sr)
    for c in (["aplay", "-q", *_SPK, p], ["paplay", p], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", p]):
        if shutil.which(c[0]): subprocess.run(c); break
    os.remove(p)


def active_rms(x, sr=SR, frame_ms=20, rel_db=35.0):
    """RMS over the 'active' frames: those within rel_db of the loudest 20 ms frame (pauses and room noise do not dilute the speech level)."""
    n = int(sr * frame_ms / 1000); nf = len(x) // n
    if nf == 0: return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    e = (x[:nf * n].reshape(nf, n).astype(np.float64) ** 2).mean(1)
    return float(np.sqrt(e[e >= e.max() * 10 ** (-rel_db / 10)].mean()))


def normalize_level(x, target_db=-24.0, max_gain_db=35.0, loud_db=-14.0, peak_ceiling=0.9):
    """Bring the active speech to a fixed level before ASR. Both ASR models are level sensitive: a gain shifts every log-mel bin by the same
    amount and neither model removes it (SenseVoice: fixed global CMVN of am.mvn; Zipformer: no CMVN). The old rule (scale to peak 0.5 only
    when the peak is under 0.2) left quiet speech with one loud click, and every recording with a peak over 0.2, at their raw level.
    gain = target / active RMS, at most +max_gain_db; speech louder than the target is left alone up to loud_db and brought down to loud_db
    above it (the level map is continuous). The peak limit uses the 99.9th percentile of |x|, so one click does not block the gain; the few
    samples above it are clipped to +-1. Digital silence is returned unchanged."""
    if not len(x): return x
    a = np.abs(x); pk = float(np.percentile(a, 99.9)) if len(x) >= 1000 else float(a.max()); r = active_rms(x)
    if not np.isfinite(r) or a.max() < 1e-5 or r < 1e-7: return x
    t, loud = 10 ** (target_db / 20), 10 ** (loud_db / 20)
    g = t / r if r < t else (1.0 if r <= loud else loud / r)
    g = min(g, 10 ** (max_gain_db / 20), peak_ceiling / max(pk, 1e-9))
    return np.clip(x * g, -1.0, 1.0).astype(np.float32) if abs(g - 1.0) > 1e-3 else x


def _env_float(name, default):
    try: return float(os.environ.get(name, default))
    except ValueError: return float(default)


def vad_mode():
    """S2S_VAD_MODE: 'hybrid' (default: join short pauses for ASR, pass >= break_ms breaks to vi punct), 'split' (branch params), 'join' (no break hints)."""
    m = os.environ.get("S2S_VAD_MODE", "hybrid").strip().lower()
    return m if m in ("hybrid", "split", "join") else "hybrid"


SEG_MAX_S = _env_float("S2S_SEG_MAX_S", "14.0")    # longest segment sent to ASR. Was 12 s: FLEURS-en clips of 11.5-14 s cut in two at
                                                               # 12 s had WER 9.4 % vs 8.0 % whole (Zipformer window 14.5 s, SenseVoice 28.5 s)
_UNSET = object()


def _voiced_spans(voiced, es, rel, k, ext, nf):
    """Group voiced frames bridging gaps <= k frames, then extend edges over weak onsets/codas above `rel` (never into neighbours)."""
    out = []; i = 0
    while i < nf:
        if not voiced[i]: i += 1; continue
        j = i; last = i
        while j < nf and j - last <= k:
            if voiced[j]: last = j
            j += 1
        out.append((i, last + 1)); i = j
    spans = []
    for idx, (a, b) in enumerate(out):
        lo = out[idx - 1][1] if idx else 0; hi = out[idx + 1][0] if idx + 1 < len(out) else nf
        a2 = a
        while a2 > lo and a - a2 < ext and es[a2 - 1] > rel[a2 - 1]: a2 -= 1
        b2 = b
        while b2 < hi and b2 - b < ext and es[b2] > rel[b2]: b2 += 1
        spans.append((a2, b2))
    return spans


def _segment_hybrid(x, sr=SR, max_s=None, min_s=0.3, frame_ms=20, pad_ms=_UNSET, min_sil_ms=_UNSET, merge_gap_s=_UNSET, short_s=_UNSET,
            head_pad_ms=_UNSET, tail_pad_ms=_UNSET, ext_ms=400, join_gap_s=_UNSET, return_breaks=False, break_ms=320):
    """Split a recording into utterances at pauses. Adaptive threshold from the local level (works with fan noise and quiet microphones).
    Segments longer than max_s are cut at the quietest frame in their last 40% (never mid-word if a pause exists).
    Boundaries (hysteresis): a segment STARTS where the energy crosses the detection threshold, but its edges are then extended over the
    neighbouring frames that are still above a much lower release threshold (about 30 dB under the local peak instead of 18 dB, at most
    ext_ms), then padded by head_pad_ms / tail_pad_ms. Word-final fricatives and released stops ("-s", "-ers", "-t") are 20-30 dB weaker than
    vowels; cutting them at the detection threshold with a 150 ms pad deleted last words ("followers" -> "follower", BTC EV-4294).
    Neighbours closer than join_gap_s (an in-sentence pause) are joined while the result fits max_s, so a sentence is not cut in two
    before MT (BTC EV-0383 "...we could do is. Maybe if..."). pad_ms is kept for callers of the old signature (used when head/tail are None).
    When return_breaks=True, returns (segs, breaks) where breaks[i] lists sample offsets (relative to segs[i][0]) of pauses >= break_ms inside segs[i]."""
    split = vad_mode() == "split"
    if pad_ms is _UNSET: pad_ms = 120 if split else 150
    if min_sil_ms is _UNSET: min_sil_ms = 320 if split else 500
    if merge_gap_s is _UNSET: merge_gap_s = 0.32 if split else 1.5
    if short_s is _UNSET: short_s = 1.1 if split else 3.0
    if head_pad_ms is _UNSET: head_pad_ms = None if split else 250
    if tail_pad_ms is _UNSET: tail_pad_ms = None if split else 350
    if join_gap_s is _UNSET: join_gap_s = 0.0 if split else 0.8
    max_s = max_s or SEG_MAX_S
    n = int(sr * frame_ms / 1000); nf = len(x) // n
    if nf == 0: return ([], []) if return_breaks else []
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    e = np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1) + 1e-12)
    if np.percentile(e, 95) < 3e-5: return ([], []) if return_breaks else []      # digital silence
    es = uniform_filter1d(e, 3)
    ceil = maximum_filter1d(es, int(3.0 * 1000 / frame_ms)); floor = np.maximum(minimum_filter1d(es, int(6.0 * 1000 / frame_ms)), 1e-6)   # LOCAL level: speakers at different distances / volumes
    thr = np.maximum.reduce([floor * 2.5, floor + 0.12 * (ceil - floor), np.full_like(e, 2e-5)])
    rel = np.maximum.reduce([floor * 1.6, floor + 0.03 * (ceil - floor), np.full_like(e, 1.5e-5)])         # release threshold (hysteresis)
    voiced = e > thr; k = max(1, int(min_sil_ms / frame_ms)); ext = max(0, int(ext_ms / frame_ms))
    spans = _voiced_spans(voiced, es, rel, k, ext, nf)
    hp = int(sr * (pad_ms if head_pad_ms is None else head_pad_ms) / 1000); tp = int(sr * (pad_ms if tail_pad_ms is None else tail_pad_ms) / 1000)
    fs = frame_ms / 1000.0; extra = (hp + tp) / sr; old_pad = 2 * pad_ms / 1000.0; joined = []
    for a, b in spans:                  # join neighbours of one meaning unit (gaps measured between the extended spans, before padding)
        if joined:
            pa, pb = joined[-1]; gap = (a - pb) * fs; total = (b - pa) * fs + extra
            short = (pb - pa) * fs + old_pad < short_s or (b - a) * fs + old_pad < short_s
            if total <= max_s and (gap < join_gap_s or (gap < merge_gap_s + old_pad and short)):
                joined[-1] = (pa, b); continue
        joined.append((a, b))
    segs = []
    for idx, (a, b) in enumerate(joined):                          # pad, but never past the middle of the pause to the neighbour
        a = max(0, a * n - hp, (joined[idx - 1][1] * n + a * n) // 2 if idx else 0)
        b = min(len(x), b * n + tp, (b * n + joined[idx + 1][0] * n) // 2 if idx + 1 < len(joined) else len(x))
        while (b - a) / sr > max_s:
            lo = a + int(0.6 * max_s * sr); hi = min(a + int(max_s * sr), b - int(min_s * sr)); w = e[lo // n: max(hi // n, lo // n + 1)]
            cut = (lo // n + int(np.argmin(w))) * n; segs.append((a, cut)); a = cut
        if (b - a) / sr >= min_s: segs.append((a, b))
    segs = [(a, b) for a, b in segs if voiced[a // n: max(b // n, a // n + 1)].any()]      # a cut remainder of pure pad/noise is not speech
    if not return_breaks: return segs
    kb = max(1, int(break_ms / frame_ms))
    bspans = spans if kb >= k else _voiced_spans(voiced, es, rel, kb, ext, nf)
    mids = [((pb + a) * n) // 2 for (_, pb), (a, _) in zip(bspans, bspans[1:]) if (a - pb) * frame_ms >= break_ms]
    breaks = [[m - sa for m in mids if sa < m < sb and voiced[sa // n: m // n].any() and voiced[m // n: max(sb // n, m // n + 1)].any()] for sa, sb in segs]
    return segs, breaks


# ---- VAD profiles -------------------------------------------------------------------------------------------------------------------
# hybrid   : (default) hysteresis VAD of the ASR speed/accuracy work: edges extended over weak onsets/codas, in-sentence pauses joined,
#            pauses >= 320 ms passed to the Vietnamese punctuation as sentence-break hints. S2S_VAD_MODE=split|join still tunes it.
# legacy   : the original energy VAD (pad 150 ms, min silence 500 ms, merge short neighbours < 1.5 s). Kept bit-exact for the Plan-01
#            measurement baselines (results/next_steps, reports/asr_integration).
# upstream : the original VAD with the shorter "split" parameters (fragments ZIP codes / digit strings, see EXECUTION_REPORT.md).
# Select with vad_profile=... (callers, Registry, --vad-profile) or S2S_VAD_PROFILE.
VAD_PROFILES = {
    "legacy": {"pad_ms": 150, "min_sil_ms": 500, "merge_gap_s": 1.5, "short_s": 3.0},
    "upstream": {"pad_ms": 120, "min_sil_ms": 320, "merge_gap_s": 0.32, "short_s": 1.1},
    "hybrid": {"pad_ms": 150, "min_sil_ms": 500, "merge_gap_s": 1.5, "short_s": 3.0, "head_pad_ms": 250, "tail_pad_ms": 350,
               "join_gap_s": 0.8, "ext_ms": 400, "break_ms": 320},
}
LEGACY_PROFILES = ("legacy", "upstream")
DEFAULT_VAD_PROFILE = "hybrid"


def get_vad_options(profile: str) -> dict:
    """Return VAD segmentation parameters for the named profile ('hybrid', 'legacy' or 'upstream')."""
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(f"Unsupported vad_profile {profile!r}; expected one of {tuple(VAD_PROFILES)}")
    key = profile.strip().lower()
    if key not in VAD_PROFILES:
        raise ValueError(f"Unsupported vad_profile {profile!r}; expected one of {tuple(VAD_PROFILES)}")
    return dict(VAD_PROFILES[key])


def resolve_vad_options(vad_profile=None, pad_ms=None, min_sil_ms=None, merge_gap_s=None, short_s=None):
    """Resolve effective VAD profile and parameter dict without altering explicit caller values."""
    if vad_profile is None:
        env_prof = os.environ.get("S2S_VAD_PROFILE")
        prof = env_prof if (env_prof is not None and env_prof != "") else DEFAULT_VAD_PROFILE
    else:
        prof = vad_profile
    opts = get_vad_options(prof)
    norm_prof = prof.strip().lower()
    for k, v in (("pad_ms", pad_ms), ("min_sil_ms", min_sil_ms), ("merge_gap_s", merge_gap_s), ("short_s", short_s)):
        if v is not None: opts[k] = v
    return norm_prof, opts


def _segment_legacy(x, sr=SR, max_s=12.0, min_s=0.3, frame_ms=20, pad_ms=150, min_sil_ms=500, merge_gap_s=1.5, short_s=3.0):
    """Original energy VAD (profiles 'legacy' / 'upstream'). Returns a list of (start_sample, end_sample)."""
    n = int(sr * frame_ms / 1000); nf = len(x) // n
    if nf == 0: return []
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    e = np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1) + 1e-12)
    if np.percentile(e, 95) < 3e-5: return []                                    # digital silence
    es = uniform_filter1d(e, 3)
    ceil = maximum_filter1d(es, int(3.0 * 1000 / frame_ms)); floor = np.maximum(minimum_filter1d(es, int(6.0 * 1000 / frame_ms)), 1e-6)
    thr = np.maximum.reduce([floor * 2.5, floor + 0.12 * (ceil - floor), np.full_like(e, 2e-5)])
    voiced = e > thr; k = max(1, int(min_sil_ms / frame_ms)); out = []; i = 0
    while i < nf:
        if not voiced[i]: i += 1; continue
        j = i; last = i
        while j < nf and j - last <= k:
            if voiced[j]: last = j
            j += 1
        out.append((i, last + 1)); i = j
    pad = int(sr * pad_ms / 1000); segs = []
    for a, b in out:
        a, b = max(0, a * n - pad), min(len(x), b * n + pad)
        while (b - a) / sr > max_s:
            lo = a + int(0.6 * max_s * sr); hi = a + int(max_s * sr); w = e[lo // n: hi // n]; cut = (lo // n + int(np.argmin(w))) * n
            segs.append((a, cut)); a = cut
        if (b - a) / sr >= min_s: segs.append((a, b))
    merged = []                                                    # join short neighbours separated by a short pause (one meaning unit)
    for a, b in segs:
        if merged:
            pa, pb = merged[-1]
            if (a - pb) / sr < merge_gap_s and (b - pa) / sr <= max_s and ((pb - pa) / sr < short_s or (b - a) / sr < short_s):
                merged[-1] = (pa, b); continue
        merged.append((a, b))
    return merged


def segment(x, sr=SR, max_s=None, min_s=0.3, frame_ms=20, pad_ms=None, min_sil_ms=None, merge_gap_s=None, short_s=None,
            vad_profile=None, return_breaks=False, **hybrid_kw):
    """Split a recording into utterances at pauses. vad_profile: 'hybrid' (default), 'legacy' or 'upstream' (S2S_VAD_PROFILE when None).
    Explicit caller values (pad_ms, min_sil_ms, ...) always win over the profile. hybrid_kw (head_pad_ms, tail_pad_ms, ext_ms, join_gap_s,
    break_ms) only apply to 'hybrid'. return_breaks=True returns (segs, breaks); the legacy profiles give no break hints (empty lists)."""
    prof, opts = resolve_vad_options(vad_profile, pad_ms, min_sil_ms, merge_gap_s, short_s)
    if prof in LEGACY_PROFILES:
        segs = _segment_legacy(x, sr, max_s or 12.0, min_s, frame_ms, opts["pad_ms"], opts["min_sil_ms"], opts["merge_gap_s"], opts["short_s"])
        return (segs, [[] for _ in segs]) if return_breaks else segs
    kw = {k: v for k, v in (("pad_ms", pad_ms), ("min_sil_ms", min_sil_ms), ("merge_gap_s", merge_gap_s), ("short_s", short_s)) if v is not None}
    kw.update(hybrid_kw)
    return _segment_hybrid(x, sr, max_s, min_s, frame_ms, return_breaks=return_breaks, **kw)


def segment_with_breaks(x, max_s=None, vad_profile=None, join_gap_s=None):
    """[((start, end), breaks_or_None), ...] for the translators. Break hints only in the 'hybrid' profile with S2S_VAD_MODE=hybrid."""
    prof, _ = resolve_vad_options(vad_profile)
    kw = {} if join_gap_s is None or prof in LEGACY_PROFILES else {"join_gap_s": join_gap_s}
    if prof not in LEGACY_PROFILES and vad_mode() == "hybrid":
        segs, brks = segment(x, max_s=max_s, vad_profile=prof, return_breaks=True, **kw)
        return list(zip(segs, brks))
    return [(s, None) for s in segment(x, max_s=max_s, vad_profile=prof, **kw)]


LISTEN_MIN_RMS = _env_float("S2S_LISTEN_MIN_RMS", "2e-4")     # about -74 dBFS. It was 0.01 (-40 dBFS): speech from a low-sensitivity
                                                                         # USB microphone (RMS ~5e-4, see README) never started a capture.


def listen(max_s=12.0, silence_s=0.9, start_timeout_s=8.0, sr=SR):
    """Push-to-talk capture without extra dependencies: reads raw 16 kHz mono from `arecord` (or sounddevice) and stops after
    `silence_s` of silence following speech, or at max_s. Returns float32 audio (may be empty if nothing was said)."""
    n = int(sr * 0.02); chunks = []; spoken = False; quiet = 0; floor = None; t = 0.0
    if shutil.which("arecord"):
        p = subprocess.Popen(["arecord", "-q", *_MIC, "-f", "S16_LE", "-r", str(sr), "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        read = lambda: p.stdout.read(n * 2)
        stop = p.terminate
        def frame():
            b = read(); return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0 if b and len(b) == n * 2 else None
    else:
        import sounddevice as sd
        st = sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=n); st.start(); stop = st.stop
        def frame():
            d, _ = st.read(n); return d[:, 0]
    try:
        while t < max_s:
            f = frame()
            if f is None: break
            t += 0.02; chunks.append(f); e = float(np.sqrt((f ** 2).mean() + 1e-10))
            if floor is None or t < 0.3: floor = e if floor is None else min(floor, e) * 0.5 + floor * 0.5
            loud = e > max(floor * 4.0, LISTEN_MIN_RMS)                  # relative to the room; the absolute floor only rejects digital silence
            if loud: spoken = True; quiet = 0
            elif spoken: quiet += 1
            if spoken and quiet * 0.02 >= silence_s: break
            if not spoken and t >= start_timeout_s: return np.zeros(0, np.float32)
    finally:
        stop()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def concat_results(results, gap_s=0.25):
    """Join per-utterance outputs that may have different sample rates (Piper 22.05 kHz, Melo / Supertonic 44.1 kHz) into one wav. Returns (wave, sr)."""
    from math import gcd
    from scipy.signal import resample_poly
    ws = [(r["wav"], r["sr"]) for r in results if len(r["wav"])]
    if not ws: return np.zeros(0, np.float32), 16000
    sr = max(s for _, s in ws); out = []
    for w, s in ws:
        if s != sr: g = gcd(int(sr), int(s)); w = resample_poly(w, int(sr) // g, int(s) // g).astype(np.float32)
        out += [w, np.zeros(int(sr * gap_s), np.float32)]
    return np.concatenate(out[:-1]), sr
