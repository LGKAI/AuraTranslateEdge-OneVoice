# -*- coding: utf-8 -*-
"""Chan doan RE (khong compile lai, chi submit inference tren target model DA CO SAN tu job jg9zx94qp
= encoder+CTC-only w8a16, KHONG co fbank/collapse NPU). Dung fbank matmul dung (khong scale 32768) tinh
local, dua thang logits vao encoder that tren chip. Neu van mat tu dau -> loi cu (CTC head quantize),
KHONG phai loi moi (fbank/collapse toi vua them). Neu KHONG mat tu dau -> loi that nam o phan moi.
Chay trong tools/venv_nllb_export."""
import json, os, re, sys
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
for tag in ("clean",):
    for r in rows:
        w, _ = sf.read(r["path"], dtype="float32")
        x, n = pad(fbank_matmul(w))   # KHONG scale 32768 -- da xac nhan dung
        items.append((tag, r["transcript"], x, n))

tm = hub.get_job("jg9zx94qp").get_target_model()
device = hub.Device("Dragonwing IQ-9075 EVK")
ijob = hub.submit_inference_job(
    model=tm, device=device,
    inputs={"x": [x[None].astype(np.float32) for _, _, x, n in items],
            "x_lens": [np.array([n], np.int32) for _, _, x, n in items]},
    name="diag-first-word-drop")
print("INFER", ijob.job_id, ijob.url, flush=True)
import time
while True:
    st = ijob.get_status(); print("status:", st.code, flush=True)
    if st.code in ("SUCCESS", "FAILED", "CANCELLED"): break
    time.sleep(15)
out = ijob.download_output_data()
print("keys:", list(out.keys()), flush=True)
key = [k for k in out if "logit" in k.lower()][0]
lkey = [k for k in out if k != key][0]

sys.stdout.reconfigure(encoding="utf-8")
res = []
for (tag, ref, x, n), lg, el in zip(items, out[key], out[lkey]):
    real = int(np.asarray(el).reshape(-1)[0])
    lg = np.asarray(lg).reshape(1, -1, 2000)
    hyp = collapse(lg[0, :real].argmax(-1))
    w = jiwer.wer(norm(ref), norm(hyp))
    res.append(w)
    print(f"REF: {ref[:60]}")
    print(f"HYP: {hyp[:60]}   (WER={w*100:.1f}%)")
print(f"\nmean WER (encoder+CTC-only, KHONG fbank/collapse NPU, cung logic fbank matmul dung) = {np.mean(res)*100:.2f}%")
