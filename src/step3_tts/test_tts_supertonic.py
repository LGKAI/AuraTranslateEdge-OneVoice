"""Step 3 test -- Supertonic (chosen candidate for Vi/Ko/En TTS).
Synthesizes the same FLORES-200 sentences already fetched for Step 2 (MT)
-- reusing text keeps this comparable across steps and languages. Measures
RTF (synth_time / audio_duration) and saves WAVs for the round-trip
intelligibility check (test_tts_eval_quality.py) and for a human to listen to.
Maps to Technical Proposal SS4.2/SS4.3 (TTS row) for Vi/Ko/En.
"""
import os
import sys
import csv
import json
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout, rtf  # noqa: F401 -- side effect import

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_supertonic")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_supertonic_results.csv")

LANGS = ["vi", "ko", "en"]
N_SENTENCES = 5
SUPERTONIC_SR = 44100


def main():
    from supertonic import TTS

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    print("[test_tts_supertonic] loading model (auto-download on first run)...")
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    results = []
    for lang in LANGS:
        for i, row in enumerate(rows):
            text = row[lang]
            t0 = time.perf_counter()
            wav, duration = tts.synthesize(text=text, lang=lang, voice_style=style,
                                            total_steps=8, speed=1.0)
            elapsed = time.perf_counter() - t0
            audio_sec = float(duration[0]) if hasattr(duration, "__getitem__") else float(duration)

            wav_path = os.path.join(OUT_AUDIO_DIR, f"{lang}_{i}.wav")
            sf.write(wav_path, np.asarray(wav).squeeze(), SUPERTONIC_SR)

            r = rtf(elapsed, audio_sec)
            results.append({"lang": lang, "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                             "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
            print(f"[test_tts_supertonic] {lang}[{i}] RTF={r:.3f}  audio={audio_sec:.2f}s  "
                  f"synth={elapsed:.2f}s  '{text[:40]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[test_tts_supertonic] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
