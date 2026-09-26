"""Step 1 test -- Zipformer-30M-RNNT-6000h (Vietnamese ASR, alt candidate).
Surfaced after a deeper 2026 landscape check (see step1.md SS12): trained on
6000h of Vietnamese speech, reported to beat PhoWhisper-large's WER despite
being ~50x smaller (30M vs 1.5B params) -- native RNN-T streaming
architecture too. License is CC-BY-NC-ND-4.0 (non-commercial, no
derivatives) -- confirm this is acceptable for OneVoice before using beyond
this comparison test. Runs via sherpa-onnx (ONNX Runtime, CPU/GPU).

NOTE: less proven than test_asr_vi.py -- the exact token-file layout on the
HF repo wasn't 100% confirmed ahead of time, so this script tries a couple
of fallbacks and prints clearly which one it used.
"""
import os
import sys
import csv
import json
import time
import glob

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, rtf, normalize_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_zipformer_results.csv")

MODEL_REPO = "hynt/Zipformer-30M-RNNT-6000h"
MODEL_CACHE_DIR = os.path.join(ROOT, "third_party", "zipformer")


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
                if it["lang"] == "vi":
                    items.append({
                        "path": it[path_key],
                        "transcript": it["transcript"],
                        "snr_db": it["snr_db"] if default_snr is None else default_snr,
                    })
    return items


def find_model_files():
    """Download the repo snapshot and locate the (non-int8) encoder/decoder/
    joiner ONNX files + a tokens file, deriving tokens.txt from bpe.model
    via sentencepiece if the repo doesn't ship one directly."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=MODEL_REPO, cache_dir=MODEL_CACHE_DIR)

    def pick(pattern):
        matches = sorted(f for f in glob.glob(os.path.join(local_dir, pattern))
                          if ".int8." not in f)
        if not matches:
            matches = sorted(glob.glob(os.path.join(local_dir, pattern)))
        return matches[0] if matches else None

    encoder = pick("encoder*.onnx")
    decoder = pick("decoder*.onnx")
    joiner = pick("joiner*.onnx")
    if not (encoder and decoder and joiner):
        raise FileNotFoundError(
            f"Couldn't find encoder/decoder/joiner .onnx under {local_dir} -- "
            f"check the repo file list manually: https://huggingface.co/{MODEL_REPO}/tree/main")

    tokens = pick("tokens.txt") or pick("*tokens*.txt")
    if not tokens:
        bpe_model = pick("bpe.model") or pick("*.model")
        if not bpe_model:
            raise FileNotFoundError(
                f"No tokens.txt and no bpe.model found under {local_dir} -- "
                f"can't build a vocabulary for sherpa-onnx.")
        print(f"[test_asr_zipformer] no tokens.txt in repo -- deriving one from "
              f"{os.path.basename(bpe_model)} via sentencepiece")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
        tokens = os.path.join(local_dir, "tokens.generated.txt")
        with open(tokens, "w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                f.write(f"{sp.id_to_piece(i)} {i}\n")

    print(f"[test_asr_zipformer] encoder={os.path.basename(encoder)}  "
          f"decoder={os.path.basename(decoder)}  joiner={os.path.basename(joiner)}  "
          f"tokens={os.path.basename(tokens)}")
    return encoder, decoder, joiner, tokens


def main():
    device = get_device()
    print(f"[test_asr_zipformer] device = {device}, model = {MODEL_REPO}")

    items = load_items()
    if not items:
        print("[test_asr_zipformer] No Vietnamese ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    import sherpa_onnx
    encoder, decoder, joiner, tokens = find_model_files()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=2,
        sample_rate=SR,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cuda" if device.type == "cuda" else "cpu",
    )

    rows = []
    for item in items:
        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SR, wav)
        recognizer.decode_stream(stream)
        hyp = stream.result.text.strip()
        elapsed = time.perf_counter() - t0
        ref = item["transcript"]
        wer = jiwer.wer(normalize_text(ref), normalize_text(hyp))
        r = rtf(elapsed, len(wav) / SR)

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
        print(f"[test_asr_zipformer] SNR={str(row['snr_db']):>6}  WER={wer:.3f}  "
              f"RTF={r:.4f}  hyp='{hyp[:50]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_zipformer] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
