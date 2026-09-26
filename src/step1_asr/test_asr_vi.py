"""Step 1 test -- PhoWhisper (Vietnamese ASR).
Loads the OFFICIAL pretrained checkpoint from VinAI Research via Hugging
Face (vinai/PhoWhisper-small) -- zero fine-tuning, as decided in step1.md.
Measures WER (jiwer, on normalized text) and RTF on clean + noisy
(SNR-mixed) Vietnamese speech. Maps to Technical Proposal SS4.2/SS4.3
(ASR row) for Vietnamese.
"""
import os
import sys
import csv
import json
import time

import torch
import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, rtf, normalize_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
OUT_DIR = os.path.join(ROOT, "outputs", "asr_vi")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_vi_results.csv")

# Swap to vinai/PhoWhisper-{tiny,base,medium,large} to trade speed for accuracy.
MODEL_ID = "vinai/PhoWhisper-small"


def load_items():
    items = []
    clean_manifest = os.path.join(ASR_DIR, "manifest.json")
    if os.path.exists(clean_manifest):
        with open(clean_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] == "vi":
                    items.append({"path": it["path"], "transcript": it["transcript"],
                                  "snr_db": "clean"})
    mixed_manifest = os.path.join(ASR_MIXED_DIR, "manifest.json")
    if os.path.exists(mixed_manifest):
        with open(mixed_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] == "vi":
                    items.append({"path": it["mixed_path"], "transcript": it["transcript"],
                                  "snr_db": it["snr_db"]})
    return items


def main():
    device = get_device()
    print(f"[test_asr_vi] device = {device}, model = {MODEL_ID}")

    items = load_items()
    if not items:
        print("[test_asr_vi] No Vietnamese ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    # NOTE: deliberately NOT using transformers.pipeline() here -- its
    # AutomaticSpeechRecognitionPipeline.preprocess() unconditionally
    # `import torchcodec` even when given an already-decoded array, and
    # torchcodec needs a matching FFmpeg DLL install that failed on this
    # machine (same failure as fetch_asr_data.py hit). Calling the
    # processor/model directly skips that code path entirely -- the
    # feature extractor only needs the raw waveform we already have.
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(device)
    if device.type == "cuda":
        model = model.half()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for item in items:
        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        inputs = processor(wav, sampling_rate=SR, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        if device.type == "cuda":
            input_features = input_features.half()
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - t0
        ref = item["transcript"]
        wer = jiwer.wer(normalize_text(ref), normalize_text(hyp))
        r = rtf(elapsed, len(wav) / SR)

        tag = os.path.splitext(os.path.basename(item["path"]))[0]
        with open(os.path.join(OUT_DIR, f"{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(f"REF: {ref}\nHYP: {hyp}\nWER: {wer:.4f}\n")

        row = {
            "file": os.path.basename(item["path"]),
            "snr_db": item["snr_db"],
            "wer": round(wer, 4),
            "rtf": round(r, 5),
            "device": str(device),
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"[test_asr_vi] SNR={str(row['snr_db']):>6}  WER={wer:.3f}  RTF={r:.4f}  "
              f"hyp='{hyp[:50]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_vi] wrote {RESULTS_CSV}")
    print(f"[test_asr_vi] per-file ref/hyp text in {OUT_DIR}/ for manual inspection")


if __name__ == "__main__":
    main()
