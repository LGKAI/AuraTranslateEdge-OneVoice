"""Step 2 data fetch -- FLORES-200 devtest sentences for MT candidate testing.
Downloads Meta's official tarball directly (dl.fbaipublicfiles.com) rather
than going through the HuggingFace `datasets` library: Muennighoff/flores200
uses a loading script no longer supported by current `datasets` versions,
and facebook/flores + openlanguagedata/flores_plus are both gated (require
HF auth + accepting terms). The tarball itself is a plain public download,
no auth needed, and its file's language codes (vie_Latn, zho_Hans, ...)
match NLLB's own tokenizer codes exactly -- one less mapping to get wrong.

Writes a manifest all test_mt_*.py scripts share: N aligned sentences (same
line number = same source sentence, different language).
Maps to Technical Proposal SS4.2/SS4.3 (MT row) for Vi<->En/Zh/Ko.
"""
import os
import sys
import json
import random
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import _ensure_utf8_stdout  # noqa: F401 -- side effect import

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "mt")
RAW_DIR = os.path.join(OUT_DIR, "raw")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
TARBALL_PATH = os.path.join(OUT_DIR, "flores200_dataset.tar.gz")
LANG_CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
N_SENTENCES = 30
SEED = 0


def download_and_extract():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TARBALL_PATH):
        print(f"[fetch_mt_data] downloading {TARBALL_URL} (~200MB, all 200 languages)...")
        urllib.request.urlretrieve(TARBALL_URL, TARBALL_PATH)
    if not os.path.isdir(RAW_DIR):
        print("[fetch_mt_data] extracting devtest files...")
        os.makedirs(RAW_DIR, exist_ok=True)
        with tarfile.open(TARBALL_PATH, "r:gz") as tf:
            members = [m for m in tf.getmembers() if "/devtest/" in m.name]
            tf.extractall(RAW_DIR, members=members)


def find_devtest_file(lang_code):
    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            if fname == f"{lang_code}.devtest":
                return os.path.join(root, fname)
    raise FileNotFoundError(f"{lang_code}.devtest not found under {RAW_DIR}")


def load_lang_sentences(lang_code):
    path = find_devtest_file(lang_code)
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    download_and_extract()

    per_lang = {}
    for short, code in LANG_CODES.items():
        sents = load_lang_sentences(code)
        per_lang[short] = sents
        print(f"[fetch_mt_data] {short} ({code}): {len(sents)} sentences")

    n_total = min(len(v) for v in per_lang.values())
    rng = random.Random(SEED)
    indices = rng.sample(range(n_total), min(N_SENTENCES, n_total))

    rows = []
    for i in indices:
        rows.append({short: per_lang[short][i] for short in LANG_CODES})

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[fetch_mt_data] wrote {len(rows)} aligned sentences -> {MANIFEST}")


if __name__ == "__main__":
    main()
