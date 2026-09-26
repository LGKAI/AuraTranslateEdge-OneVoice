"""Step 3 test -- Confucius4-TTS (NetEase Youdao), a candidate that claims to
cover Vietnamese + Korean + Chinese + English in ONE model -- unlike every
other candidate tested so far, which needed a 2-model split. Zero-shot voice
cloning architecture (w2v-bert-2.0 speaker conditioning + 24-layer T2S +
flow-matching S2A + BigVGAN vocoder) -- meaningfully heavier than Supertonic/
MeloTTS/Piper, so this test also answers "is it realistically on-device at
all" alongside speed/quality.

Not pip-installed as a package (repo has no pyproject.toml yet per its own
example.py comment) -- sys.path-inserted directly, matching example.py.
Uses one of Step 1's own clean Vietnamese clips as the zero-shot voice
reference (prompt_wav), reusing infrastructure already in this project
rather than sourcing a new sample voice.
"""
import os
import sys
import csv
import json
import time

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout, rtf  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_confucius")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_confucius_results.csv")

CONFUCIUS_REPO = os.path.join(os.path.dirname(ROOT), "Confucius4-TTS")
PROMPT_WAV = os.path.join(ROOT, "data", "asr", "vi", "vi_0.wav")
LANGS = ["vi", "ko", "zh", "en"]
N_SENTENCES = 5


def main():
    sys.path.insert(0, CONFUCIUS_REPO)
    os.chdir(CONFUCIUS_REPO)  # config paths in inference_config.yaml are relative
    from confuciustts.cli.inference import ConfuciusTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test_tts_confucius] device = {device}")
    print("[test_tts_confucius] loading model (auto-downloads checkpoints on first run)...")
    model = ConfuciusTTS(config_path="config/inference_config.yaml", device=device)
    print(f"[test_tts_confucius] loaded. sample_rate={model.sample_rate}")

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    results = []
    for lang in LANGS:
        for i, row in enumerate(rows):
            text = row[lang]
            t0 = time.perf_counter()
            audio = model.generate(text=text, lang=lang, prompt_wav=PROMPT_WAV, verbose=False)
            elapsed = time.perf_counter() - t0
            audio_sec = audio.shape[-1] / model.sample_rate

            wav_path = os.path.join(OUT_AUDIO_DIR, f"{lang}_{i}.wav")
            torchaudio.save(wav_path, audio.cpu(), model.sample_rate)

            r = rtf(elapsed, audio_sec)
            results.append({"lang": lang, "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                             "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
            print(f"[test_tts_confucius] {lang}[{i}] RTF={r:.3f}  audio={audio_sec:.2f}s  "
                  f"synth={elapsed:.2f}s  '{text[:40]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[test_tts_confucius] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
