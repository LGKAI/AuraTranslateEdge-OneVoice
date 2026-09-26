"""Step 0 test -- Silero VAD.
Loads the OFFICIAL pretrained model straight from their GitHub repo via
torch.hub (snakers4/silero-vad) -- zero fine-tuning, as decided in step0.md.
Measures RTF and dumps each detected speech segment as its own .wav so you
can listen and manually grade cut accuracy ("nghe/cham tay").
Maps to Technical Proposal SS4.2 (VAD row) / SS4.4 (Robustness).
"""
import os
import sys
import csv
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, list_audio_files, load_wav, save_wav, rtf


def _ensure_torchaudio_importable():
    """Silero's hubconf imports torchaudio at module level, but only USES it
    inside read_audio()/save_audio() -- functions we never call (we load
    audio ourselves via soundfile/librosa in common.py). If the local
    torch/torchaudio install is ABI-mismatched (common on Windows, shows up
    as `OSError: [WinError 127] The specified procedure could not be
    found`), stub the module out instead of requiring an environment fix."""
    try:
        import torchaudio  # noqa: F401
    except Exception as e:
        import sys
        import types
        stub = types.ModuleType("torchaudio")
        stub.__version__ = "0.0.0-stub"
        sys.modules["torchaudio"] = stub
        print(f"[test_vad] WARNING: real torchaudio failed to import ({e!r}); "
              f"using a stub so Silero VAD can still load "
              f"(not needed for this test path).")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIXED_DIR = os.path.join(ROOT, "data", "mixed")
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
OUT_DIR = os.path.join(ROOT, "outputs", "vad")
RESULTS_CSV = os.path.join(ROOT, "outputs", "vad_results.csv")


def load_silero():
    _ensure_torchaudio_importable()
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def run_vad_on_file(model, get_speech_timestamps, path, device):
    wav_np = load_wav(path)
    wav = torch.from_numpy(wav_np).to(device)
    model.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        # returns sample-index dicts {'start': int, 'end': int} -- most
        # version-stable form of the Silero API, convert to seconds ourselves
        raw_ts = get_speech_timestamps(wav, model, sampling_rate=SR)
    elapsed = time.perf_counter() - t0
    audio_seconds = len(wav_np) / SR
    segs = [{"start": t["start"] / SR, "end": t["end"] / SR} for t in raw_ts]
    return wav_np, segs, elapsed, audio_seconds


def main():
    device = get_device()
    print(f"[test_vad] device = {device}")
    model, get_speech_timestamps = load_silero()

    files = list_audio_files(MIXED_DIR) + list_audio_files(CLEAN_DIR)
    if not files:
        print("[test_vad] No audio found. Put clean files under data/clean/<lang>/ "
              "then run mix_noise.py, or just run this on clean files directly.")
        return

    rows = []
    for path in files:
        wav_np, segs, elapsed, audio_seconds = run_vad_on_file(
            model, get_speech_timestamps, path, device)
        r = rtf(elapsed, audio_seconds)
        tag = os.path.splitext(os.path.basename(path))[0]
        for i, seg in enumerate(segs):
            s, e = int(seg["start"] * SR), int(seg["end"] * SR)
            save_wav(os.path.join(OUT_DIR, f"{tag}_seg{i}.wav"), wav_np[s:e])
        speech_seconds = sum(seg["end"] - seg["start"] for seg in segs)
        print(f"[test_vad] {os.path.basename(path):40s} "
              f"dur={audio_seconds:6.2f}s  segments={len(segs):3d}  "
              f"speech={speech_seconds:6.2f}s  RTF={r:.4f}")
        rows.append({
            "file": os.path.relpath(path, ROOT).replace("\\", "/"),
            "duration_s": round(audio_seconds, 3),
            "n_segments": len(segs),
            "speech_s": round(speech_seconds, 3),
            "rtf": round(r, 5),
            "device": str(device),
        })

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_vad] wrote {RESULTS_CSV}")
    print(f"[test_vad] listen to cut segments in {OUT_DIR}/ to manually grade cut accuracy")


if __name__ == "__main__":
    main()
