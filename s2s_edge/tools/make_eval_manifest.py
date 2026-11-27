# -*- coding: utf-8 -*-
"""Build an OVAC Phase 3 eval-mode input directory (manifest.csv + 16 kHz mono PCM16 WAVs)
from the organisers' observe dataset (data/phase3_sample_dataset).

Usage:
  python tools/make_eval_manifest.py --data ../data/phase3_sample_dataset --out ./eval_input
  python tools/make_eval_manifest.py --data ../data/phase3_sample_dataset --out ./eval_input --dirs vi2en,en2vi --limit 5
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.eval_mode import load_audio_any


def _xlsx_rows(path):
    import openpyxl
    rows = list(openpyxl.load_workbook(path, read_only=True).active.values)
    return [r for r in rows[1:] if r and r[0]]


def _collect_entries(data_root, direction):
    j = lambda *p: os.path.join(data_root, *p)
    out = []
    if direction in ("vi2en", "en2vi"):
        sub, xlsx = ("Vie-Eng", "OneVoice_V_to_E_samples_100.xlsx") if direction == "vi2en" else ("Eng-Vie", "OneVoice_E_to_V_samples_100.xlsx")
        for r in _xlsx_rows(j("Vie-Eng dataset", sub, xlsx)):
            raw_id = str(r[0]).strip()
            wav = j("Vie-Eng dataset", sub, "wav", f"{raw_id}.wav")
            if os.path.isfile(wav):
                safe_id = raw_id.replace("-", "_")
                out.append((safe_id, wav, direction))
    elif direction in ("vi2ko", "ko2vi"):
        folder, xlsx = "Vie-Kor dataset", "OneVoice_V_to_K_samples_180.xlsx"
        for r in _xlsx_rows(j(folder, xlsx)):
            raw_id = str(r[0]).strip()
            rel = r[3] if direction == "vi2ko" else r[4]
            wav = j(folder, str(rel))
            if os.path.isfile(wav):
                safe_id = f"{direction}_{raw_id.replace('-', '_')}"
                out.append((safe_id, wav, direction))
    elif direction in ("vi2zh", "zh2vi"):
        folder, xlsx = "Vie-Man dataset", "OneVoice_V_to_M_samples_180.xlsx"
        for r in _xlsx_rows(j(folder, xlsx)):
            raw_id = str(r[0]).strip()
            rel = r[3] if direction == "vi2zh" else r[4]
            wav = j(folder, str(rel))
            if os.path.isfile(wav):
                safe_id = f"{direction}_{raw_id.replace('-', '_')}"
                out.append((safe_id, wav, direction))
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return out


def _convert_wav_16k(src_wav, dst_wav):
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src_wav, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst_wav],
            check=True,
        )
        return
    x = load_audio_any(src_wav, sr=16000)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(dst_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Build OVAC eval-mode input/manifest.csv + 16 kHz mono WAVs")
    ap.add_argument("--data", required=True, help="Path to phase3_sample_dataset")
    ap.add_argument("--out", required=True, help="Output input directory (will contain manifest.csv and audio/)")
    ap.add_argument("--dirs", default="vi2en,en2vi,vi2zh,zh2vi,vi2ko,ko2vi", help="Comma-separated directions")
    ap.add_argument("--limit", type=int, default=0, help="Optional max clips per direction (0 = all)")
    ap.add_argument("--manifest-only", action="store_true", help="Only write manifest.csv without copying/converting WAVs")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    audio_dir = os.path.join(a.out, "audio")
    if not a.manifest_only:
        os.makedirs(audio_dir, exist_ok=True)

    rows = []
    for d in [x.strip() for x in a.dirs.split(",") if x.strip()]:
        entries = _collect_entries(a.data, d)
        if a.limit > 0:
            entries = entries[: a.limit]
        for rid, src_wav, direction in entries:
            rel_audio = f"audio/{rid}.wav"
            if not a.manifest_only:
                _convert_wav_16k(src_wav, os.path.join(a.out, rel_audio))
            rows.append((rid, rel_audio, direction))

    manifest_path = os.path.join(a.out, "manifest.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",", lineterminator="\n")
        w.writerow(["id", "audio", "direction"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
