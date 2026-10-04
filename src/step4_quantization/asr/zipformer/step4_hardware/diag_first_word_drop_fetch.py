# -*- coding: utf-8 -*-
"""Lay lai ket qua tu job jgol8zoqg (da submit, cho chay xong)."""
import json, os, re, sys, time
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT); sys.path.insert(0, "src/step4_hardware")
from fbank_matmul_verify import fbank_matmul
import sentencepiece as spm

T = 1500
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()
sp = spm.SentencePieceProcessor(); sp.load("third_party_zipformer_150m_crctc/bpe.model")


def collapse(ids):
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev: out.append(int(t))
        prev = t
    return sp.decode(out)


def pad(feats):
    n = min(feats.shape[0], T); x = np.zeros((T, 80), np.float32); x[:n] = feats[:n]
    return x, n


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for r in rows:
    w, _ = sf.read(r["path"], dtype="float32")
    x, n = pad(fbank_matmul(w))
    items.append((r["transcript"], x, n))

ijob = hub.get_job("jgol8zoqg")
while True:
    st = ijob.get_status(); print("status:", st.code, flush=True)
    if st.code in ("SUCCESS", "FAILED", "CANCELLED"): break
    time.sleep(15)
if st.code != "SUCCESS":
    print("FAILED:", st.message); sys.exit(1)

out = ijob.download_output_data()
print("keys:", list(out.keys()), flush=True)
for k, v in out.items(): print(k, np.asarray(v[0]).shape)
key = [k for k in out if np.asarray(out[k][0]).shape[-1] == 2000][0]
lkey = [k for k in out if k != key][0]

sys.stdout.reconfigure(encoding="utf-8")
res = []
for (ref, x, n), lg, el in zip(items, out[key], out[lkey]):
    real = int(np.asarray(el).reshape(-1)[0])
    lg = np.asarray(lg).reshape(1, -1, 2000)
    hyp = collapse(lg[0, :real].argmax(-1))
    w = jiwer.wer(norm(ref), norm(hyp))
    res.append(w)
    print(f"REF: {ref[:60]}")
    print(f"HYP: {hyp[:60]}   (WER={w*100:.1f}%)")
print(f"\nmean WER (encoder+CTC-only, fbank matmul DUNG, KHONG co fbank/collapse NPU) = {np.mean(res)*100:.2f}%")
