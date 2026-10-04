# -*- coding: utf-8 -*-
"""Lay day du 15 mau (khong chi 10 cai dau) tu job w16a16 da SUCCESS, tinh WER tung cau + loai springboks
de xem con lai co that su het loi khong."""
import json, os, re
import numpy as np
import soundfile as sf
import jiwer
import qai_hub as hub

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()


def decode_output(byte_matrix, byte_len):
    out = bytearray()
    for row, l in zip(byte_matrix, byte_len):
        l = int(l)
        if l <= 0:
            continue
        out += bytes(int(b) & 0xFF for b in row[:l])
    return out.decode("utf-8", errors="replace")


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f):
            items.append((tag, r["transcript"]))

ijob = hub.get_job("jp8e1008p")
out = ijob.download_output_data()
bm_key = [k for k in out if np.asarray(out[k][0]).shape == (373, 12)][0]
bl_key = [k for k in out if k != bm_key][0]

import sys
sys.stdout.reconfigure(encoding="utf-8")
per, per_no_springboks = {}, {}
for (tag, ref), bm, bl in zip(items, out[bm_key], out[bl_key]):
    hyp = decode_output(np.asarray(bm), np.asarray(bl))
    w = jiwer.wer(norm(ref), norm(hyp))
    per.setdefault(tag, []).append(w)
    is_springboks = "springboks" in ref
    if not is_springboks:
        per_no_springboks.setdefault(tag, []).append(w)
    mark = "  <-- SPRINGBOKS (cau kho da biet)" if is_springboks else ""
    print(f"[{tag:5s}] WER={w*100:5.1f}%  REF: {ref[:70]}{mark}")
    print(f"          HYP: {hyp[:70]}")

print("\n=== w16a16, TAT CA 15 mau ===")
for t, v in per.items():
    print(f"  {t:6s} n={len(v)}  WER={np.mean(v)*100:.2f}%")
print("\n=== w16a16, LOAI CAU SPRINGBOKS (con 4/5 moi dieu kien) ===")
for t, v in per_no_springboks.items():
    print(f"  {t:6s} n={len(v)}  WER={np.mean(v)*100:.2f}%")
