"""Mix noise into the labeled ASR test set (data/asr/, from fetch_asr_data.py)
at several SNR levels, carrying the transcript through so WER/CER can be
measured per SNR level by test_asr_vi.py / test_asr_multi.py. Reuses
data/noise/ and the mixing logic from mix_noise.py (Step 0) instead of
duplicating it.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import list_audio_files, load_wav, save_wav
from step0_frontend.mix_noise import get_noise_segment, mix_at_snr, SNR_DB_LIST

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOISE_DIR = os.path.join(ROOT, "data", "noise")
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
MANIFEST_IN = os.path.join(ASR_DIR, "manifest.json")


def main():
    if not os.path.exists(MANIFEST_IN):
        print("[mix_asr_noise] No manifest found. Run fetch_asr_data.py first.")
        return
    with open(MANIFEST_IN, "r", encoding="utf-8") as f:
        items = json.load(f)

    noise_files = list_audio_files(NOISE_DIR)
    print(f"[mix_asr_noise] noise files found: {len(noise_files)}"
          + ("" if noise_files else "  -> using synthetic factory-noise fallback"))

    manifest = []
    for item in items:
        clean_path = os.path.join(ROOT, item["path"])
        clean = load_wav(clean_path)
        for snr in SNR_DB_LIST:
            seed = abs(hash((clean_path, snr))) % (2**31)
            noise = get_noise_segment(len(clean), noise_files, seed=seed)
            mixed = mix_at_snr(clean, noise, snr)
            out_name = f"{os.path.splitext(os.path.basename(clean_path))[0]}_snr{snr}.wav"
            out_path = os.path.join(ASR_MIXED_DIR, item["lang"], out_name)
            save_wav(out_path, mixed)
            manifest.append({
                "lang": item["lang"],
                "snr_db": snr,
                "transcript": item["transcript"],
                "clean_path": item["path"],
                "mixed_path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
            })
        print(f"[mix_asr_noise] {item['path']}: x{len(SNR_DB_LIST)} SNR levels")

    os.makedirs(ASR_MIXED_DIR, exist_ok=True)
    manifest_path = os.path.join(ASR_MIXED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[mix_asr_noise] wrote {len(manifest)} mixed files -> {manifest_path}")


if __name__ == "__main__":
    main()
