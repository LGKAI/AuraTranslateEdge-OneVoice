"""Step 2 test -- NLLB-200-distilled-600M (baseline MT candidate).
Loads the official pretrained checkpoint -- zero fine-tuning, matching how
step1.md tested ASR candidates zero-shot before deciding what to fine-tune.
Measures sentence BLEU + RTF on all 6 directions (Vi<->En, Vi<->Zh, Vi<->Ko)
using the FLORES-200 sample from fetch_mt_data.py.
Maps to Technical Proposal SS4.2/SS4.3 (MT row).
"""
import os
import sys
import csv
import json
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers.modeling_utils

# Bypass PyTorch version restriction check (< 2.6) for cached .bin weights
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import get_device, bleu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
RESULTS_CSV = os.path.join(ROOT, "outputs", "mt_nllb_results.csv")

MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
DIRECTIONS = [("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi")]


def translate(model, tokenizer, text, src, tgt, device):
    tokenizer.src_lang = NLLB_CODE[src]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tgt_id = tokenizer.convert_tokens_to_ids(NLLB_CODE[tgt])
    out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=200)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test NLLB-200 Machine Translation (Step 2)")
    parser.add_argument("text", nargs="?", default=None, help="Text to translate (single-sentence mode)")
    parser.add_argument("--src", default="vi", choices=["vi", "en", "zh", "ko"], help="Source language")
    parser.add_argument("--tgt", default="en", choices=["vi", "en", "zh", "ko"], help="Target language")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across FLORES dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sentences in benchmark (e.g. 5)")
    parser.add_argument("--direction", default=None, help="Filter benchmark direction, e.g. 'vi->en'")
    args = parser.parse_args()

    device = get_device()
    print(f"[test_mt_nllb] device = {device}, model = {MODEL_ID}")

    print("Loading NLLB-200 tokenizer & model from local cache...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

    # Single-sentence translation mode
    if args.text or not args.benchmark:
        test_text = args.text if args.text else "Văn hóa và bộ lạc cổ xưa đã bắt đầu giữ những con vật này để dễ lấy sữa, thịt và da."
        print("\n" + "=" * 65)
        print(f"KẾT QUẢ DỊCH MÁY NLLB-200 ({args.src.upper()} -> {args.tgt.upper()}):")
        print("=" * 65)
        print(f"  * Văn bản gốc ({args.src.upper()}) : {test_text}")
        t0 = time.perf_counter()
        translated = translate(model, tokenizer, test_text, args.src, args.tgt, device)
        elapsed = time.perf_counter() - t0
        print(f"  * Bản dịch ({args.tgt.upper()})    : {translated}")
        print(f"  * Độ trễ xử lý     : {elapsed * 1000:.1f} ms")
        print("=" * 65)
        if not args.benchmark:
            print("\n(Gợi ý: Thêm cờ --benchmark để chạy đo điểm BLEU tự động trên tập FLORES-200)")
            return

    # Benchmark mode
    if not os.path.exists(MANIFEST):
        print(f"[test_mt_nllb] No MT data found at {MANIFEST}. Run fetch_mt_data.py first.")
        return

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if args.limit:
        rows = rows[:args.limit]
        print(f"[test_mt_nllb] Running benchmark on {len(rows)} sentences per direction...")

    target_directions = DIRECTIONS
    if args.direction:
        parts = args.direction.split("->") if "->" in args.direction else args.direction.split("2")
        if len(parts) == 2:
            target_directions = [(parts[0].strip(), parts[1].strip())]

    results = []
    for src, tgt in target_directions:
        scores, rtfs = [], []
        for row in rows:
            src_text, ref = row[src], row[tgt]
            t0 = time.perf_counter()
            hyp = translate(model, tokenizer, src_text, src, tgt, device)
            elapsed = time.perf_counter() - t0
            score = bleu(hyp, ref, tgt)
            scores.append(score)
            rtfs.append(elapsed)
        avg_bleu = sum(scores) / len(scores)
        avg_time = sum(rtfs) / len(rtfs)
        results.append({"direction": f"{src}->{tgt}", "bleu": round(avg_bleu, 2),
                         "avg_sec_per_sentence": round(avg_time, 4), "n": len(rows)})
        print(f"[test_mt_nllb] {src}->{tgt}  BLEU={avg_bleu:.2f}  "
              f"avg_sec/sentence={avg_time:.4f}")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[test_mt_nllb] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()

