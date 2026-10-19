# -*- coding: utf-8 -*-
"""Download what is not committed: the Piper vi voice, FLEURS-vi transcripts (CC-BY 4.0) and the PhoBERT syllable vocabulary.
Run from piper_npu/g2p."""
import os, shutil, unicodedata
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)
ROOT = os.path.join("..", "outputs", "piper_vi_npu")
os.makedirs(ROOT, exist_ok=True)
for f in ["vi_VN-vais1000-medium.onnx", "vi_VN-vais1000-medium.onnx.json"]:
    if not os.path.exists(os.path.join(ROOT, f)):
        shutil.copy(hf_hub_download("rhasspy/piper-voices", f"vi/vi_VN/vais1000/medium/{f}"), os.path.join(ROOT, f))
        print("voice", f)
rows = []
for sp in ["train", "dev", "test"]:
    p = hf_hub_download("google/fleurs", f"data/vi_vn/{sp}.tsv", repo_type="dataset")
    rows += [l.rstrip("\n").split("\t")[2] for l in open(p, encoding="utf-8") if l.count("\t") > 3]
open("data/fleurs_vi.txt", "w", encoding="utf-8").write("\n".join(rows))
print("fleurs sentences", len(rows))
toks = [l.split()[0] for l in open(hf_hub_download("vinai/phobert-base", "vocab.txt"), encoding="utf-8") if l.strip()]
VOW = set("aăâeêioôơuưy")
ALPH = set("abcdđeghiklmnopqrstuvxy") | set("ăâêôơư")
nfc = lambda s: unicodedata.normalize("NFC", s)


def base(c):
    d = unicodedata.normalize("NFD", c)
    return unicodedata.normalize("NFC", d[0] + "".join(ch for ch in d[1:] if ch in "\u0306\u0302\u031b"))


def ok(t):
    t = t.replace("@@", "")
    if not t.isalpha() or len(t) > 7:
        return False
    t = nfc(t.lower())
    return all(base(c) in ALPH for c in t) and any(base(c) in VOW for c in t)


syl = sorted({nfc(t.replace("@@", "").lower()) for t in toks if ok(t)})
open("data/syllables_phobert.txt", "w", encoding="utf-8").write("\n".join(syl))
print("syllables", len(syl))
