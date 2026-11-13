"""Step 4 -- round-trip-free DIRECT WER/CER check for the int8-quantized
SenseVoice-Small ONNX export (outputs/sensevoice-onnx/model_quant.onnx),
scored against the same FLEURS-based Step 1 test clips (data/asr/<lang>/),
so the numbers are directly comparable to test_asr_multi.py's original
fp32 pytorch numbers.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout, normalize_text, normalize_text_for_cer  # noqa: F401

import jiwer
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")
CER_LANGS = {"zh", "ko"}
LANGS = ["en", "zh", "ko"]


def main():
    print(f"[verify_sensevoice_int8] loading quantized model from {MODEL_DIR}")
    model = SenseVoiceSmall(model_dir=MODEL_DIR, quantize=True, batch_size=1)

    with open(os.path.join(DATA_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for lang in LANGS:
        rows = [r for r in manifest if r["lang"] == lang]
        scores = []
        for row in rows:
            wav_path = os.path.join(ROOT, row["path"])
            result = model(wav_content=wav_path, language=lang, use_itn=True)
            hyp = rich_transcription_postprocess(result[0])
            ref = row["transcript"]

            if lang in CER_LANGS:
                score = jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
            else:
                score = jiwer.wer(normalize_text(ref), normalize_text(hyp))
            scores.append(score)
            print(f"[verify_sensevoice_int8] {lang} '{ref[:40]}' -> '{hyp[:40]}'  score={score:.3f}")

        metric = "CER" if lang in CER_LANGS else "WER"
        print(f"[verify_sensevoice_int8] {lang}  avg {metric}={sum(scores) / len(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
