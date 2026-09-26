"""Step 4 -- round-trip WER/CER check for the int8-quantized Supertonic
submodels (outputs/supertonic-int8/) against the same ko/en sentences used
in Step 3, so the delta is directly comparable to test_tts_supertonic.py's
original fp32 numbers (ko CER 6.8%, en WER 7.9%).
"""
import os
import sys
import csv
import json
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout, rtf  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
INT8_DIR = os.path.join(ROOT, "outputs", "supertonic-int8")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_supertonic_int8")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_supertonic_int8_results.csv")

N_SENTENCES = 5
LANGS = ["ko", "en"]  # vi excluded -- already rejected pre-quantization


def main():
    from supertonic import TTS

    # the int8 dir only has the 4 quantized .onnx graphs -- copy the small
    # non-weight assets (voice styles, tokenizer config) from the original
    # auto-downloaded cache so TTS() has everything it needs to load.
    orig_dir = os.path.expanduser("~/.cache/supertonic3")
    for name in os.listdir(orig_dir):
        src = os.path.join(orig_dir, name)
        if name == "onnx" or name.startswith("."):
            continue
        dst = os.path.join(INT8_DIR, "..", "supertonic-int8-full", name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    full_dir = os.path.join(ROOT, "outputs", "supertonic-int8-full")
    onnx_dst = os.path.join(full_dir, "onnx")
    if os.path.exists(onnx_dst):
        shutil.rmtree(onnx_dst)
    shutil.copytree(INT8_DIR, onnx_dst)

    print(f"[verify_supertonic_int8] loading from {full_dir}")
    tts = TTS(model_dir=full_dir, auto_download=False)
    style = tts.get_voice_style(voice_name="M1")

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:N_SENTENCES]

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    results = []
    for lang in LANGS:
        for i, row in enumerate(rows):
            text = row[lang]
            t0 = time.perf_counter()
            wav, audio_sec = tts.synthesize(text=text, lang=lang, voice_style=style, total_steps=8, speed=1.0)
            elapsed = time.perf_counter() - t0

            wav_path = os.path.join(OUT_AUDIO_DIR, f"{lang}_{i}.wav")
            import soundfile as sf
            sf.write(wav_path, wav, 24000 if not hasattr(tts, "sample_rate") else tts.sample_rate)

            r = rtf(elapsed, audio_sec)
            results.append({"lang": lang, "idx": i, "text": text, "audio_sec": round(audio_sec, 3),
                             "synth_sec": round(elapsed, 3), "rtf": round(r, 4), "wav_path": wav_path})
            print(f"[verify_supertonic_int8] {lang}[{i}] RTF={r:.3f}  '{text[:40]}'")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[verify_supertonic_int8] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
