"""Mix clean speech (VI/EN/ZH/KO, put under data/clean/<lang>/) with noise
(data/noise/ -- MUSAN clips or self-recorded factory noise; if empty, a
synthetic factory-noise fallback is used so the pipeline still runs) at
several SNR levels. Writes data/mixed/<lang>/*.wav + manifest.json (needed
by test_denoise.py / test_beamform.py for clean-reference PESQ/STOI).
"""
import os
import sys
import json
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import list_audio_files, load_wav, save_wav

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
NOISE_DIR = os.path.join(ROOT, "data", "noise")
MIXED_DIR = os.path.join(ROOT, "data", "mixed")
LANGS = ["vi", "en", "zh", "ko"]
SNR_DB_LIST = [0, 5, 10, 15, 20]


def synthetic_factory_noise(n_samples, sr=16000, seed=0):
    """Fallback noise (motor hum + band-limited broadband + random bursts),
    used only when data/noise/ is empty -- lets the pipeline run with zero setup."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    hum = 0.15 * np.sin(2 * np.pi * 100 * t) + 0.08 * np.sin(2 * np.pi * 150 * t)
    broadband = rng.normal(0, 1, n_samples)
    kernel = np.ones(9) / 9
    broadband = np.convolve(broadband, kernel, mode="same")
    bursts = np.zeros(n_samples)
    n_bursts = max(n_samples // sr, 1)
    for _ in range(n_bursts):
        start = int(rng.integers(0, max(n_samples - sr // 10, 1)))
        length = int(rng.integers(sr // 40, sr // 10))
        length = min(length, n_samples - start)
        bursts[start:start + length] += rng.normal(0, 0.6, length)
    noise = hum + 0.3 * broadband + bursts
    peak = np.max(np.abs(noise)) + 1e-8
    return (noise / peak * 0.5).astype(np.float32)


def get_noise_segment(length, noise_files, seed):
    if not noise_files:
        return synthetic_factory_noise(length, seed=seed)
    rng = random.Random(seed)
    path = rng.choice(noise_files)
    noise = load_wav(path)
    if len(noise) < length:
        reps = length // len(noise) + 1
        noise = np.tile(noise, reps)
    start = rng.randint(0, max(len(noise) - length, 0))
    return noise[start:start + length]


def mix_at_snr(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = clean + noise * scale
    return mixed.astype(np.float32)


def main():
    noise_files = list_audio_files(NOISE_DIR)
    print(f"[mix_noise] noise files found: {len(noise_files)}"
          + ("" if noise_files else "  -> using synthetic factory-noise fallback"))

    manifest = []
    for lang in LANGS:
        clean_files = list_audio_files(os.path.join(CLEAN_DIR, lang))
        if not clean_files:
            print(f"[mix_noise] SKIP '{lang}': no files in data/clean/{lang}/")
            continue
        for cpath in clean_files:
            clean = load_wav(cpath)
            for snr in SNR_DB_LIST:
                seed = abs(hash((cpath, snr))) % (2**31)
                noise = get_noise_segment(len(clean), noise_files, seed=seed)
                mixed = mix_at_snr(clean, noise, snr)
                out_name = f"{os.path.splitext(os.path.basename(cpath))[0]}_snr{snr}.wav"
                out_path = os.path.join(MIXED_DIR, lang, out_name)
                save_wav(out_path, mixed)
                manifest.append({
                    "lang": lang, "snr_db": snr,
                    "clean_path": os.path.relpath(cpath, ROOT).replace("\\", "/"),
                    "mixed_path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
                })
        print(f"[mix_noise] {lang}: {len(clean_files)} clean file(s) x {len(SNR_DB_LIST)} SNR levels")

    os.makedirs(MIXED_DIR, exist_ok=True)
    manifest_path = os.path.join(MIXED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[mix_noise] wrote manifest with {len(manifest)} mixed files -> {manifest_path}")
    if not manifest:
        print("[mix_noise] NOTHING to mix -- add files under data/clean/{vi,en,zh,ko}/ first.")


if __name__ == "__main__":
    main()
