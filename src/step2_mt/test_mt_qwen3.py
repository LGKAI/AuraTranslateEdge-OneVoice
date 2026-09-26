"""Step 2 test -- Qwen3-0.6B and Qwen3-1.7B as prompted MT candidates.
Both are pre-quantized and hosted on Qualcomm AI Hub for Snapdragon --
unlike NLLB, which isn't there and would need our own ONNX/QNN conversion.
Tests whether a small general LLM, used via prompting, is competitive with
a dedicated seq2seq MT model for Vi<->En/Zh/Ko. enable_thinking=False since
an on-device live-mode translator can't afford reasoning-token latency.
Measures sentence BLEU + RTF on all 6 directions using the same FLORES-200
sample as test_mt_nllb.py, for a direct comparison.
Maps to Technical Proposal SS4.2/SS4.3 (MT row).
"""
import os
import csv
import sys
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import get_device, bleu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_DIR = os.path.join(ROOT, "outputs")

MODEL_IDS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
# Backgrounded runs of this script have repeatedly stalled indefinitely with
# no progress (transformers' generation-config warning spam seems to cause a
# stdout-buffering stall specific to this session's background-capture
# mechanism -- foreground runs of the same code complete in the expected
# time). Split by model and run each in the foreground as a workaround: pass
# a model id as argv[1] to run just that one, or no arg to run all of them
# sequentially in-process (fine for foreground; avoid backgrounding this).
if len(sys.argv) > 1:
    MODEL_IDS = [sys.argv[1]]
LANG_NAME = {"vi": "Vietnamese", "en": "English", "zh": "Chinese", "ko": "Korean"}
DIRECTIONS = [("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi")]


def translate(model, tokenizer, text, src, tgt, device):
    prompt = (f"Translate the following {LANG_NAME[src]} text to {LANG_NAME[tgt]}. "
              f"Output only the translation, nothing else.\n\n{text}")
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_model(model_id, rows, device):
    print(f"[test_mt_qwen3] loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    rows_out = []
    for src, tgt in DIRECTIONS:
        scores, secs = [], []
        for i, row in enumerate(rows):
            src_text, ref = row[src], row[tgt]
            t0 = time.perf_counter()
            hyp = translate(model, tokenizer, src_text, src, tgt, device)
            elapsed = time.perf_counter() - t0
            score = bleu(hyp, ref, tgt)
            scores.append(score)
            secs.append(elapsed)
            print(f"[test_mt_qwen3]   {src}->{tgt} {i+1}/{len(rows)}  {elapsed:.2f}s",
                  flush=True)
        avg_bleu = sum(scores) / len(scores)
        avg_time = sum(secs) / len(secs)
        rows_out.append({"model": model_id, "direction": f"{src}->{tgt}",
                          "bleu": round(avg_bleu, 2),
                          "avg_sec_per_sentence": round(avg_time, 4), "n": len(rows)})
        print(f"[test_mt_qwen3] {model_id} {src}->{tgt}  BLEU={avg_bleu:.2f}  "
              f"avg_sec/sentence={avg_time:.4f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows_out


def main():
    device = get_device()
    print(f"[test_mt_qwen3] device = {device}")

    if not os.path.exists(MANIFEST):
        print("[test_mt_qwen3] No MT data found. Run fetch_mt_data.py first.")
        return

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    for model_id in MODEL_IDS:
        results = run_model(model_id, rows, device)
        safe_name = model_id.split("/")[-1].replace(".", "_")
        out_csv = os.path.join(OUT_DIR, f"mt_qwen3_{safe_name}_results.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"[test_mt_qwen3] wrote {out_csv}")


if __name__ == "__main__":
    main()
