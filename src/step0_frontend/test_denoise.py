"""Step 0 test -- GTCRN denoiser.
Uses their OWN repo code (third_party/gtcrn/gtcrn.py, MIT licensed, kept
verbatim) + their official DNS3-trained checkpoint -- zero fine-tuning, as
decided in step0.md. Reports PESQ/STOI before vs after denoise, and RTF,
per language and per SNR level. Maps to Technical Proposal SS4.2 / SS4.4.
"""
import os
import sys
import csv
import json
import time

import torch
import numpy as np
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, save_wav, rtf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "gtcrn"))
from gtcrn import GTCRN  # noqa: E402  (their official model definition, MIT licensed)

CKPT_PATH = os.path.join(ROOT, "third_party", "gtcrn", "checkpoints", "model_trained_on_dns3.tar")
MIXED_DIR = os.path.join(ROOT, "data", "mixed")
MANIFEST_PATH = os.path.join(MIXED_DIR, "manifest.json")
OUT_DIR = os.path.join(ROOT, "outputs", "denoise")
RESULTS_CSV = os.path.join(ROOT, "outputs", "denoise_results.csv")

N_FFT, HOP, WIN = 512, 256, 512


def load_gtcrn(device):
    model = GTCRN().eval().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def denoise(model, wav_np, device):
    """Same STFT/model/ISTFT flow as third_party/gtcrn/infer.py, but using
    torch.view_as_real/view_as_complex instead of the now-deprecated
    return_complex=False path (their infer.py targets torch==1.11.0; on
    newer torch, istft(return_complex=False) hard-errors)."""
    window = torch.hann_window(WIN).pow(0.5).to(device)
    wav = torch.from_numpy(wav_np).to(device)
    spec_complex = torch.stft(wav, N_FFT, HOP, WIN, window, return_complex=True)
    spec = torch.view_as_real(spec_complex)  # (F, T, 2) -- what GTCRN expects
    with torch.no_grad():
        out_spec = model(spec[None])[0]  # (F, T, 2)
    out_complex = torch.view_as_complex(out_spec.contiguous())
    enh = torch.istft(out_complex, N_FFT, HOP, WIN, window)
    return enh.detach().cpu().numpy()


def safe_pesq(clean, test, sr=SR):
    try:
        return pesq(sr, clean, test, "wb")  # wideband PESQ (needs 16k)
    except Exception:
        return float("nan")


def main():
    if not os.path.exists(MANIFEST_PATH):
        print("[test_denoise] No manifest found. Run mix_noise.py first.")
        return
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest:
        print("[test_denoise] Manifest empty -- add clean audio under data/clean/<lang>/ first.")
        return

    device = get_device()
    print(f"[test_denoise] device = {device}")
    model = load_gtcrn(device)

    rows = []
    for item in manifest:
        clean = load_wav(os.path.join(ROOT, item["clean_path"]))
        mixed = load_wav(os.path.join(ROOT, item["mixed_path"]))
        n = min(len(clean), len(mixed))
        clean, mixed = clean[:n], mixed[:n]

        t0 = time.perf_counter()
        enhanced = denoise(model, mixed, device)
        elapsed = time.perf_counter() - t0
        if len(enhanced) < n:
            enhanced = np.pad(enhanced, (0, n - len(enhanced)))
        enhanced = enhanced[:n]

        out_path = os.path.join(OUT_DIR, item["lang"], os.path.basename(item["mixed_path"]))
        save_wav(out_path, enhanced)

        row = {
            "lang": item["lang"], "snr_db": item["snr_db"],
            "file": os.path.basename(item["mixed_path"]),
            "pesq_before": round(safe_pesq(clean, mixed), 3),
            "pesq_after": round(safe_pesq(clean, enhanced), 3),
            "stoi_before": round(stoi(clean, mixed, SR, extended=False), 3),
            "stoi_after": round(stoi(clean, enhanced, SR, extended=False), 3),
            "rtf": round(rtf(elapsed, n / SR), 5),
            "device": str(device),
        }
        rows.append(row)
        print(f"[test_denoise] {row['lang']} SNR={row['snr_db']:>2}dB  "
              f"PESQ {row['pesq_before']:.2f}->{row['pesq_after']:.2f}  "
              f"STOI {row['stoi_before']:.2f}->{row['stoi_after']:.2f}  RTF={row['rtf']:.4f}")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_denoise] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
