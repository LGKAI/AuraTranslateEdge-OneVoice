"""Step 3 test -- MeloTTS (chosen candidate for Mandarin TTS; also has an
English checkpoint, tested here too as a second data point since it's free
once the library is installed). Uses the same FLORES-200 sentences as
test_tts_supertonic.py for a fair comparison. Measures RTF and saves WAVs.
Maps to Technical Proposal SS4.2/SS4.3 (TTS row) for Zh (+En reference).
"""
import os
import sys
import csv
import json
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout, get_device, rtf  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_melotts")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_melotts_results.csv")

# MeloTTS language codes differ from our short codes.
MELO_LANG = {"zh": "ZH", "en": "EN"}
N_SENTENCES = 5


def synth_lang(model_lang, short_lang, rows, device_str):
    from melo.api import TTS

    print(f"[test_tts_melotts] loading {model_lang} model...")
    model = TTS(language=model_lang, device=device_str)
    speaker_ids = model.hps.data.spk2id
    # Pick the first available speaker id for this language checkpoint.
    speaker_id = list(speaker_ids.values())[0]
    sr = model.hps.data.sampling_rate

    results = []
    for i, row in enumerate(rows):
        text = row[short_lang]
        wav_path = os.path.join(OUT_AUDIO_DIR, f"{short_lang}_{i}.wav")
        t0 = time.perf_counter()
        model.tts_to_file(text, speaker_id, wav_path, speed=1.0, quiet=True)
        elapsed = time.perf_counter() - t0

        wav, out_sr = sf.read(wav_path)
        audio_sec = len(wav) / out_sr
        r = rtf(elapsed, audio_sec)
        results.append({"lang": short_lang, "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                         "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
        print(f"[test_tts_melotts] {short_lang}[{i}] RTF={r:.3f}  audio={audio_sec:.2f}s  "
              f"synth={elapsed:.2f}s  '{text[:40]}'")
    del model
    return results


def main():
    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[test_tts_melotts] device = {device_str}")

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    all_results = []
    for short_lang, model_lang in MELO_LANG.items():
        all_results.extend(synth_lang(model_lang, short_lang, rows, device_str))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[test_tts_melotts] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
