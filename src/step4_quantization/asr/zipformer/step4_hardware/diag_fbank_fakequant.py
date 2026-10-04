# -*- coding: utf-8 -*-
"""Gia lap luong tu hoa int8 (naive, round-to-int8-then-dequant) cho CAC MA TRAN HANG SO cua fbank
(window, DFT_RE, DFT_IM, mel_mat) -- KHONG dung AI Hub, chi de kiem chung RE (vai giay) truoc khi
compile lai (~1h). Neu tai hien duoc "mat tu dau cau" -> dung la do fbank bi quantize, khong phai do
collapse/detokenize."""
import json, os, re, sys
import numpy as np
import soundfile as sf
import jiwer

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT); sys.path.insert(0, "src/step4_hardware")
from fbank_matmul_verify import WINDOW, MEL_MAT, DFT_RE, DFT_IM, FRAME_LEN, FRAME_SHIFT, PREEMPH
NFFT = 512
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()


def fake_quant_int8(w):
    """Symmetric per-tensor int8 fake-quant -- dung giong cach QNN PTQ mac dinh xu ly weight."""
    scale = np.abs(w).max() / 127.0
    if scale == 0: return w
    q = np.round(w / scale).clip(-127, 127)
    return (q * scale).astype(np.float32)


def fbank_matmul_fq(wav, quantize_weights):
    n = wav.shape[0]
    num_frames = 1 + (n - FRAME_LEN) // FRAME_SHIFT
    idx = (np.arange(num_frames)[:, None] * FRAME_SHIFT + np.arange(FRAME_LEN)[None, :])
    frames = wav[idx].astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)
    shifted = np.concatenate([frames[:, 0:1], frames[:, 0:-1]], axis=1)
    frames = frames - PREEMPH * shifted
    win = fake_quant_int8(WINDOW) if quantize_weights else WINDOW
    frames = frames * win[None, :]
    padded = np.zeros((num_frames, NFFT), dtype=np.float64)
    padded[:, :FRAME_LEN] = frames
    dre = fake_quant_int8(DFT_RE) if quantize_weights else DFT_RE
    dim = fake_quant_int8(DFT_IM) if quantize_weights else DFT_IM
    re = padded @ dre.astype(np.float64)
    im = padded @ dim.astype(np.float64)
    power = re * re + im * im
    mel = fake_quant_int8(MEL_MAT) if quantize_weights else MEL_MAT
    mel_energy = power @ mel.astype(np.float64)
    return np.log(np.maximum(mel_energy, 1.1920929e-07)).astype(np.float32)


import onnxruntime as ort
import sentencepiece as spm
sp = spm.SentencePieceProcessor(); sp.load("third_party_zipformer_150m_crctc/bpe.model")


def collapse(ids):
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev: out.append(int(t))
        prev = t
    return sp.decode(out)


T = 1500
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession("outputs/zip150_ctc/zip150_ctc_static_1500.onnx", so, providers=["CPUExecutionProvider"])

rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
sys.stdout.reconfigure(encoding="utf-8")
for qw in (False, True):
    print(f"\n=== fake-quant int8 tren trong so fbank: {qw} ===")
    res = []
    for r in rows:
        w, _ = sf.read(r["path"], dtype="float32")
        feats = fbank_matmul_fq(w, qw)
        n = min(feats.shape[0], T); x = np.zeros((T, 80), np.float32); x[:n] = feats[:n]
        lg, el = sess.run(None, {"x": x[None], "x_lens": np.array([n], np.int64)})
        real = int(el[0]); hyp = collapse(lg[0, :real].argmax(-1))
        wr = jiwer.wer(norm(r["transcript"]), norm(hyp))
        res.append(wr)
        print(f"REF: {r['transcript'][:55]}")
        print(f"HYP: {hyp[:55]}   (WER={wr*100:.1f}%)")
    print(f"mean WER = {np.mean(res)*100:.2f}%")
