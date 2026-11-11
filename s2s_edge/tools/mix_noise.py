# -*- coding: utf-8 -*-
"""Make a noisy copy of the organisers' sample set (same folder tree, same file names, same xlsx) so tools/eval_btc.py runs on it unchanged.
Office / enterprise noise is simulated three ways, none of which needs a download:
  babble : sum of 5 other clips from the same set (other talkers in the room), each level-normalised
  pink   : synthetic 1/f noise (HVAC, fans, road hum)
  file   : a recording you supply, e.g. 2 minutes of real office ambience recorded ON THE BOX with
           `arecord -D <usb mic> -f S16_LE -r 16000 -c 1 -d 120 office.wav` (the most realistic option; it also contains the mic's own response)
SNR is measured against the active-speech level (frames within 40 dB of the loudest frame), so silence does not inflate it.
  python tools/mix_noise.py --data /path/onevoice_btc_data --out /path/btc_babble10 --kind babble --snr 10
  python tools/mix_noise.py --data ... --out ... --kind file --noise office.wav --snr 5
Output audio is 16 kHz mono PCM16 (the Korean MP3-in-.wav files become real WAV, file names unchanged)."""
import argparse, hashlib, os, shutil, subprocess, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.audio import load_audio

SR = 16000


def load_any(path):
    try: return load_audio(path)
    except Exception:
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def active_rms(x, frame=400):
    n = len(x) // frame
    if n == 0: return float(np.sqrt(np.mean(x ** 2) + 1e-12))
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1); db = 10 * np.log10(e + 1e-12); act = e[db > db.max() - 40]
    return float(np.sqrt(act.mean() + 1e-12))


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1; y = np.fft.irfft(X / np.sqrt(f), n); return (y / (np.std(y) + 1e-9)).astype(np.float32)


def loop_to(x, n, rng):
    if len(x) == 0: return np.zeros(n, np.float32)
    reps = int(np.ceil((n + len(x)) / len(x))); y = np.tile(x, reps); o = int(rng.integers(0, len(x))); return y[o: o + n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["babble", "pink", "file"], default="babble"); ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--noise", default="", help="noise recording for --kind file"); ap.add_argument("--talkers", type=int, default=5); ap.add_argument("--limit", type=int, default=0, help="only the first N audio files (for a quick test)")
    a = ap.parse_args()
    audio, other = [], []
    for root, _, files in os.walk(a.data):
        for f in files:
            p = os.path.join(root, f); rel = os.path.relpath(p, a.data)
            (audio if f.lower().endswith((".wav", ".mp3", ".flac")) else other).append(rel)
    audio.sort(); audio = audio[: a.limit] if a.limit else audio
    for rel in other:                                                   # xlsx and anything else: copied as is
        os.makedirs(os.path.dirname(os.path.join(a.out, rel)) or a.out, exist_ok=True); shutil.copy2(os.path.join(a.data, rel), os.path.join(a.out, rel))
    rng0 = np.random.default_rng(0); pool = []
    if a.kind == "babble":
        for rel in rng0.choice(audio, size=min(40, len(audio)), replace=False):
            x = load_any(os.path.join(a.data, rel)); pool.append((rel, x / (active_rms(x) + 1e-9)))
    if a.kind == "file":
        nz = load_any(a.noise); nz = nz / (np.std(nz) + 1e-9)
    for k, rel in enumerate(audio):
        rng = np.random.default_rng(int(hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8], 16))
        x = load_any(os.path.join(a.data, rel)); n = len(x)
        if a.kind == "pink": z = pink(n, rng)
        elif a.kind == "file": z = loop_to(nz, n, rng)
        else:
            cand = [p for p in pool if p[0] != rel]; pick = rng.choice(len(cand), size=min(a.talkers, len(cand)), replace=False)
            z = np.sum([loop_to(cand[i][1], n, rng) for i in pick], axis=0)
        s = active_rms(x); zr = np.sqrt(np.mean(z ** 2) + 1e-12); y = x + z * (s / zr) * 10 ** (-a.snr / 20)
        peak = np.abs(y).max()
        if peak > 0.99: y = y * (0.99 / peak)                           # avoid clipping; the SNR is unchanged by a common gain
        dst = os.path.join(a.out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); sf.write(dst, y.astype(np.float32), SR, subtype="PCM_16", format="WAV")
        if k % 100 == 0: print(f"{k}/{len(audio)} {rel}", flush=True)
    print(f"done: {len(audio)} files, kind={a.kind}, snr={a.snr} dB -> {a.out}")


if __name__ == "__main__": main()
