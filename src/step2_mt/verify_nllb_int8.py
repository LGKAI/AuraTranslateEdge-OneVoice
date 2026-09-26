"""Step 4 -- sanity + BLEU check for the CTranslate2 int8 NLLB conversion
against the original fp32 transformers pipeline, on the same FLORES-200
sentences already used in Step 2, so the BLEU delta is directly comparable
to test_mt_nllb.py's original numbers.
"""
import os
import sys
import json
import time

import ctranslate2
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout, rtf, bleu  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
CT2_DIR = os.path.join(ROOT, "outputs", "nllb-ct2-int8")

LANG_TO_FLORES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
DIRECTIONS = [("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi")]


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    translator = ctranslate2.Translator(CT2_DIR, device="cpu")

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)

    print(f"[verify_nllb_int8] {len(rows)} sentences, CTranslate2 int8 model at {CT2_DIR}")
    for src_lang, tgt_lang in DIRECTIONS:
        srcs = [r[src_lang] for r in rows]
        refs = [r[tgt_lang] for r in rows]
        tgt_flores = LANG_TO_FLORES[tgt_lang]

        t0 = time.perf_counter()
        hyps = []
        for text in srcs:
            tokenizer.src_lang = LANG_TO_FLORES[src_lang]
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            result = translator.translate_batch(
                [tokens], target_prefix=[[tgt_flores]], beam_size=4, max_decoding_length=256,
            )
            out_tokens = result[0].hypotheses[0][1:]  # drop the target-lang prefix token
            hyps.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True))
        elapsed = time.perf_counter() - t0

        scores = [bleu(h, r, tgt_lang) for h, r in zip(hyps, refs)]
        avg_bleu = sum(scores) / len(scores)
        print(f"[verify_nllb_int8] {src_lang}->{tgt_lang}  BLEU={avg_bleu:.2f}  "
              f"({len(srcs)} sents, {elapsed:.1f}s total, {1000 * elapsed / len(srcs):.0f}ms/sent)")
        print(f"    sample: '{srcs[0][:50]}' -> '{hyps[0][:60]}'")


if __name__ == "__main__":
    main()
