"""Step 3 test -- Piper (baseline for comparison; this was the OLD tentative
pick before research found real gaps -- no viable Korean voice, weak
Vietnamese voice quality. Testing it for real numbers instead of relying on
the web-research claims alone). Vi + En only (no viable Ko voice to test).
Maps to Technical Proposal SS4.2/SS4.3 (TTS row) -- baseline comparison.
"""
import os
import csv
import json
import time
import wave
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout, rtf  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_piper")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_piper_results.csv")

# (short_lang, piper voice name)
VOICES = [("vi", "vi_VN-vais1000-medium"), ("en", "en_US-amy-medium")]
N_SENTENCES = 5
PIPER_SR = 22050


def synth_voice(short_lang, voice_name, rows):
    from piper.voice import PiperVoice

    # PiperVoice.load() needs a local .onnx/.json pair -- it does not
    # auto-download despite some docs implying otherwise. download_voices
    # is a no-op if the voice is already present.
    print(f"[test_tts_piper] downloading voice {voice_name} if not cached...")
    subprocess.run([sys.executable, "-m", "piper.download_voices", voice_name], check=True)

    # download_voices saves <name>.onnx + <name>.onnx.json -- PiperVoice.load()
    # derives the config path by appending ".json" to whatever we pass it, so
    # it must be given the .onnx filename itself, not the bare voice name.
    onnx_path = f"{voice_name}.onnx"
    print(f"[test_tts_piper] loading voice {onnx_path}...")
    voice = PiperVoice.load(onnx_path)

    results = []
    for i, row in enumerate(rows):
        text = row[short_lang]
        wav_path = os.path.join(OUT_AUDIO_DIR, f"{short_lang}_{i}.wav")
        t0 = time.perf_counter()
        audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
        elapsed = time.perf_counter() - t0

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PIPER_SR)
            wf.writeframes(audio_bytes)

        audio_sec = len(audio_bytes) / 2 / PIPER_SR  # 16-bit mono
        r = rtf(elapsed, audio_sec)
        results.append({"lang": short_lang, "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                         "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
        print(f"[test_tts_piper] {short_lang}[{i}] RTF={r:.3f}  audio={audio_sec:.2f}s  "
              f"synth={elapsed:.2f}s  '{text[:40]}'")
    return results


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    all_results = []
    for short_lang, voice_name in VOICES:
        all_results.extend(synth_voice(short_lang, voice_name, rows))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[test_tts_piper] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
