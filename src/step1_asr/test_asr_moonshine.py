"""Step 1 test -- Moonshine (Vi/En/Zh/Ko ASR, alt candidate to PhoWhisper+
SenseVoice split). Surfaced after a deeper 2026 landscape check (see
step1.md SS12): tiny (~27M param) MONOLINGUAL models per language -- the
"Flavors of Moonshine" paper found monolingual beats multilingual at this
size. Natively supported in `transformers`, so this uses the same pipeline
pattern as test_asr_vi.py. Tests all 4 languages so it can be compared
directly against the current PhoWhisper (Vi) + SenseVoice (En/Zh/Ko) split.
"""
import os
import sys
import csv
import json
import time

import torch
import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, rtf, normalize_text, normalize_text_for_cer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_moonshine_results.csv")

MODEL_IDS = {
    "vi": "UsefulSensors/moonshine-tiny-vi",
    "en": "UsefulSensors/moonshine-tiny",
    "zh": "UsefulSensors/moonshine-tiny-zh",
    "ko": "UsefulSensors/moonshine-tiny-ko",
}
CER_LANGS = {"zh", "ko"}  # word boundaries ambiguous/absent, see step1.md SS2.4


def load_items():
    items = []
    for manifest_path, path_key, default_snr in [
        (os.path.join(ASR_DIR, "manifest.json"), "path", "clean"),
        (os.path.join(ASR_MIXED_DIR, "manifest.json"), "mixed_path", None),
    ]:
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            for it in json.load(f):
                items.append({
                    "lang": it["lang"],
                    "path": it[path_key],
                    "transcript": it["transcript"],
                    "snr_db": it["snr_db"] if default_snr is None else default_snr,
                })
    return items


def main():
    device = get_device()
    print(f"[test_asr_moonshine] device = {device}")

    items = load_items()
    if not items:
        print("[test_asr_moonshine] No ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    # NOTE: deliberately NOT using transformers.pipeline() -- see the note
    # in test_asr_vi.py: its preprocess() unconditionally imports torchcodec
    # even for already-decoded arrays, and torchcodec's FFmpeg DLL failed to
    # load on this machine. Direct processor/model calls skip that path.
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    models = {}  # lazy-load one (processor, model) pair per language

    rows = []
    for item in items:
        lang = item["lang"]
        if lang not in MODEL_IDS:
            continue
        if lang not in models:
            print(f"[test_asr_moonshine] loading {MODEL_IDS[lang]} for '{lang}'...")
            proc = AutoProcessor.from_pretrained(MODEL_IDS[lang])
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_IDS[lang]).to(device)
            if device.type == "cuda":
                mdl = mdl.half()
            models[lang] = (proc, mdl)
        processor, model = models[lang]

        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        inputs = processor(wav, sampling_rate=SR, return_tensors="pt")
        # NOTE: Moonshine feeds the raw waveform straight to its encoder (no
        # mel-spectrogram front-end like Whisper), so the processor output
        # exposes `.input_values`, not `.input_features` -- confirmed against
        # the official model doc after a KeyError: 'input_features' crash.
        input_values = inputs.input_values.to(device)
        if device.type == "cuda":
            input_values = input_values.half()
        with torch.no_grad():
            predicted_ids = model.generate(input_values)
        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - t0
        ref = item["transcript"]

        metric = "cer" if lang in CER_LANGS else "wer"
        if metric == "cer":
            score = jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
        else:
            score = jiwer.wer(normalize_text(ref), normalize_text(hyp))
        r = rtf(elapsed, len(wav) / SR)

        row = {
            "lang": lang,
            "file": os.path.basename(item["path"]),
            "snr_db": item["snr_db"],
            "metric": metric,
            "score": round(score, 4),
            "rtf": round(r, 5),
            "device": str(device),
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"[test_asr_moonshine] {lang} SNR={str(row['snr_db']):>6}  "
              f"{metric.upper()}={score:.3f}  RTF={r:.4f}  hyp='{hyp[:50]}'")

    if not rows:
        print("[test_asr_moonshine] No matching-language items found.")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_moonshine] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
