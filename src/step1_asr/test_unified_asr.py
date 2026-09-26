"""Test the Unified ASR Pipeline (Zipformer + SenseVoice) end-to-end.
Evaluates WER/CER and RTF across all 4 languages (Vi, En, Zh, Ko) on both
clean and mixed-noise datasets, writing results to outputs/asr_unified_results.csv.
Now includes evaluation of the GTCRN denoiser on noisy data to demonstrate robustness.
"""
import os
import sys
import csv
import json
import time

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SR, load_wav, rtf, normalize_text, normalize_text_for_cer
from step1_asr.unified_asr import UnifiedASRPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_unified_results.csv")

LANGS = ["vi", "en", "zh", "ko"]
CER_LANGS = {"zh", "ko"}

def load_all_items():
    items = []
    
    # Load clean data
    clean_manifest = os.path.join(ASR_DIR, "manifest.json")
    if os.path.exists(clean_manifest):
        with open(clean_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] in LANGS:
                    items.append({
                        "lang": it["lang"],
                        "path": it["path"],
                        "transcript": it["transcript"],
                        "snr_db": "clean"
                    })
                    
    # Load mixed-noise data
    mixed_manifest = os.path.join(ASR_MIXED_DIR, "manifest.json")
    if os.path.exists(mixed_manifest):
        with open(mixed_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] in LANGS:
                    items.append({
                        "lang": it["lang"],
                        "path": it["mixed_path"],
                        "transcript": it["transcript"],
                        "snr_db": it["snr_db"]
                    })
                    
    return items

def main():
    items = load_all_items()
    if not items:
        print("[test_unified_asr] No ASR data found. Please run fetch_asr_data.py and mix_asr_noise.py first.")
        return

    print("\n[test_unified_asr] Loading Standard Pipeline...")
    pipeline = UnifiedASRPipeline(use_denoiser=False)
    
    print("\n[test_unified_asr] Loading Denoised Pipeline...")
    pipeline_denoised = UnifiedASRPipeline(use_denoiser=True)
    
    rows = []
    print("\n[test_unified_asr] Starting evaluation...")
    for item in items:
        lang = item["lang"]
        path = os.path.join(ROOT, item["path"])
        is_clean = item["snr_db"] == "clean"
        
        # We load wav here only to compute the actual duration for RTF.
        wav = load_wav(path)
        duration_s = max(len(wav) / SR, 1e-9)
        
        # Test baseline on everything. Test denoised on noisy data to see improvement.
        pipelines_to_run = [(pipeline, False)] if is_clean else [(pipeline, False), (pipeline_denoised, True)]
        
        for p, is_denoised in pipelines_to_run:
            t0 = time.perf_counter()
            try:
                hyp = p.transcribe(path, lang)
            except Exception as e:
                print(f"[test_unified_asr] FAILED on {path}: {e}")
                continue
            elapsed = time.perf_counter() - t0
            
            ref = item["transcript"]
            metric = "cer" if lang in CER_LANGS else "wer"
            
            if metric == "cer":
                score = jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
            else:
                score = jiwer.wer(normalize_text(ref), normalize_text(hyp))
                
            r = rtf(elapsed, duration_s)
            
            row = {
                "lang": lang,
                "file": os.path.basename(path),
                "snr_db": item["snr_db"],
                "denoised": is_denoised,
                "metric": metric,
                "score": round(score, 4),
                "rtf": round(r, 5),
                "device": p.device_str,
                "reference": ref,
                "hypothesis": hyp,
            }
            rows.append(row)
            print(f"[test_unified_asr] {lang.upper()} | SNR: {str(row['snr_db']):>5} | Denoised: {is_denoised} | "
                  f"{metric.upper()}: {score:.3f} | RTF: {r:.4f} | hyp: '{hyp[:50]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"\n[test_unified_asr] Evaluation complete. Results saved to: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
