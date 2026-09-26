"""Step 1 test -- Qwen3-ASR-0.6B (Vi/En/Zh/Ko ASR, alt candidate: ONE
unified multilingual model instead of the PhoWhisper+SenseVoice split).
Surfaced after a deeper 2026 landscape check (see step1.md SS12): Alibaba,
Jan 2026, covers 30+ languages incl. Vietnamese/Korean/Chinese, 92ms
time-to-first-token on the 0.6B variant. Open question this test answers:
does one generalist model hold up on Vietnamese specifically vs the
PhoWhisper specialist (generalists usually lose on lower-resource
languages -- see step1.md SS2.3's take on Option A).

NOTE: needs transformers>=5.13.0 (a big jump; may need a separate env if it
conflicts with funasr's pin). Uses ONLY the documented AutoProcessor /
AutoModelForMultimodalLM path -- an earlier version also tried the generic
transformers.pipeline() call first, but that silently used Whisper-style
mel-spectrogram preprocessing (padded to 3000 frames) which doesn't match
this model's actual multimodal input format and crashed on every single
file with a tensor-shape mismatch. Qwen3-ASR-0.6B-hf is a multimodal-LM
architecture, not a plain AutoModelForSpeechSeq2Seq, so the pipeline's
auto-detected preprocessing was simply wrong for it -- not fixable by
changing the input format, hence dropped entirely rather than kept as a
"try first" path.
"""
import os
import sys
import csv
import json
import time

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, rtf, normalize_text, normalize_text_for_cer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_qwen_results.csv")

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
CER_LANGS = {"zh", "ko"}


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


def load_qwen(device):
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    # trust_remote_code=True: qwen3_asr is new enough that its architecture
    # isn't registered in transformers' own CONFIG_MAPPING yet -- the repo
    # ships its own modeling code, which Auto* only loads with this flag.
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None)
    if device.type != "cuda":
        model = model.to(device)
    return processor, model


def transcribe(processor, model, path):
    inputs = processor.apply_transcription_request(audio=path).to(
        model.device, model.dtype)
    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, return_format="transcription_only")[0].strip()


def main():
    device = get_device()
    print(f"[test_asr_qwen] device = {device}, model = {MODEL_ID}")

    items = load_items()
    if not items:
        print("[test_asr_qwen] No ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    processor, model = load_qwen(device)

    rows = []
    for item in items:
        lang = item["lang"]
        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        try:
            hyp = transcribe(processor, model, path)
        except Exception as e:
            print(f"[test_asr_qwen] FAILED on {item['path']}: {e!r} -- skipping")
            continue
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
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"[test_asr_qwen] {lang} SNR={str(row['snr_db']):>6}  "
              f"{metric.upper()}={score:.3f}  RTF={r:.4f}  hyp='{hyp[:50]}'")

    if not rows:
        print("[test_asr_qwen] Nothing transcribed successfully.")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_qwen] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
