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


def normalize_level(x, target_peak=0.5, below=0.2):
    """Bring quiet recordings up to a normal level before ASR (never attenuates loud ones)."""
    pk = float(np.abs(x).max()) if len(x) else 0.0
    return (x * (target_peak / pk)).astype(np.float32) if 1e-5 < pk < below else x


def segment(x, sr=SR, max_s=12.0, min_s=0.3, frame_ms=20, pad_ms=120, min_sil_ms=320, merge_gap_s=0.32, short_s=1.1):
    """Split a recording into utterances at pauses. Adaptive threshold from the quietest 10% of frames (works with fan noise).
    Segments longer than max_s are cut at the quietest frame in their last 40% (never mid-word if a pause exists).
    Returns a list of (start_sample, end_sample)."""
    n = int(sr * frame_ms / 1000); nf = len(x) // n
    if nf == 0: return []
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    e = np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1) + 1e-12)
    if np.percentile(e, 95) < 3e-5: return []                                    # digital silence
    es = uniform_filter1d(e, 3)
    ceil = maximum_filter1d(es, int(3.0 * 1000 / frame_ms)); floor = np.maximum(minimum_filter1d(es, int(6.0 * 1000 / frame_ms)), 1e-6)   # LOCAL level: speakers at different distances / volumes
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
            loud = e > max(floor * 4.0, 0.01)
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
