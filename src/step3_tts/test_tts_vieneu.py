"""Step 3 test -- VieNeu-TTS (Vietnamese-specific candidate). Unlike
Supertonic (multilingual, but real-tested Vietnamese quality was bad --
repeated words, 35% round-trip WER), VieNeu-TTS is purpose-built for
Vietnamese only: ~0.3B params (~200MB Q4-quantized), ONNX-native/torch-free
CPU path. Tested here as the dedicated-Vietnamese half of a 3-model split
(VieNeu-TTS Vi + Supertonic Ko/En + MeloTTS-ZH Zh) instead of forcing one
multilingual model to cover a language it's weak at.
"""
import os
import sys
import csv
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout, rtf  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_vieneu")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_vieneu_results.csv")

N_SENTENCES = 5
VIENEU_SR = 24000


def main():
    from vieneu import Vieneu

    # mode="standard" uses the PyTorch+Transformers backend, not GGUF/llama_cpp
    # -- the default "turbo" mode needs llama-cpp-python, whose prebuilt Windows
    # wheel failed to load its native DLL on this machine (missing dependency,
    # a known llama-cpp-python-on-Windows issue). Standard avoids that entirely.
    print("[test_tts_vieneu] loading model, mode=standard (auto-downloads on first run)...")
    model = Vieneu(mode="standard")
    print("[test_tts_vieneu] preset voices:", model.list_preset_voices())
    voice = model.get_preset_voice("Ngoc")  # Ngọc, nữ miền Bắc -- standard Northern female voice

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    results = []
    for i, row in enumerate(rows):
        text = row["vi"]
        t0 = time.perf_counter()
        audio = model.infer(text, voice=voice)
        elapsed = time.perf_counter() - t0
        audio_sec = len(audio) / VIENEU_SR

        wav_path = os.path.join(OUT_AUDIO_DIR, f"vi_{i}.wav")
        model.save(audio, wav_path)

        r = rtf(elapsed, audio_sec)
        results.append({"lang": "vi", "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                         "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
        print(f"[test_tts_vieneu] vi[{i}] RTF={r:.3f}  audio={audio_sec:.2f}s  "
              f"synth={elapsed:.2f}s  '{text[:40]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[test_tts_vieneu] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
