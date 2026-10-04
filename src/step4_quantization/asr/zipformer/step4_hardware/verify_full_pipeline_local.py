# -*- coding: utf-8 -*-
"""Chay THAT graph gop (raw wave -> chu) bang ONNX Runtime CPU, so WER voi pipeline cu (fbank rieng bang
kaldi_native_fbank + sp.decode rieng) tren 15 mau (5 sach + 5 nhieu5dB + 5 nhieu0dB)."""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import onnxruntime as ort
import soundfile as sf
import jiwer

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
ONNX = "outputs/zip150_full_npu/zip150_full_pipeline.onnx"
N_SAMPLES = 240240
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])


def prep_wave(wav):
    n = min(wav.shape[0], N_SAMPLES)
    x = np.zeros((N_SAMPLES,), dtype=np.float32)
    x[:n] = wav[:n]     # KHONG scale 32768 -- da xac nhan bug that (dbg9), pipeline goc dung thang [-1,1]
    n_frames = 1 + (n - 400) // 160
    return x, n_frames


def decode_output(byte_matrix, byte_len):
    out = bytearray()
    for row, l in zip(byte_matrix, byte_len):
        l = int(l)
        if l <= 0:
            continue
        out += bytes(int(b) for b in row[:l])
    return out.decode("utf-8", errors="replace")


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f):
            w, _ = sf.read(f, dtype="float32"); items.append((tag, r["transcript"], w))

per = {}
for tag, ref, wav in items:
    x, n_frames = prep_wave(wav)
    out = sess.run(None, {"fb_raw_wave_flat": x, "enc_x_lens": np.array([n_frames], np.int64)})
    outmap = {o.name: v for o, v in zip(sess.get_outputs(), out)}
    hyp = decode_output(outmap["post_out_byte_matrix"], outmap["post_out_byte_len"])
    w_ = jiwer.wer(norm(ref), norm(hyp))
    per.setdefault(tag, []).append(w_)
    if len(per[tag]) <= 2:
        print(f"[{tag}] REF: {ref[:80]}")
        print(f"      HYP: {hyp[:80]}   (WER={w_*100:.1f}%)")

print("\n=== WER pipeline gop (raw wave -> chu, 1 graph, CPU/ORT local) ===")
for tag, v in per.items():
    print(f"  {tag:6s} n={len(v)}  WER={np.mean(v)*100:.2f}%")
print("\nSo sanh voi CTC 150M cu (tach rieng tung buoc, config snip_edges=True): clean 6.22%, snr5 8.72%, snr0 4.17%")
