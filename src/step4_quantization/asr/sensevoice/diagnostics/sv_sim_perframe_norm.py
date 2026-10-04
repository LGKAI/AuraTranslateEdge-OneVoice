# -*- coding: utf-8 -*-
"""Mo phong fp16 (lam tron sau MOI buoc, ca weights DFT/mel_fb) cho 3 cach xu ly scale:
  A) k toan cuc (nhu hien tai/2^4)   B) chuan hoa THEO TUNG FRAME: frame/peak*P, bu 2*log(peak/P*...) sau log.
So voi tham chieu fp32 chinh xac (wav*32768, clamp 1e-10)."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import build_mel_filterbank, FRAME_SAMPLES, HOP_SAMPLES, N_FFT

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
calib = np.load(os.path.join(OUT_DIR, "calib_data_unified_fixed.npz"))
labels = ["en_0","en_1","en_2","en_3","en_4","zh_0","zh_1","zh_2","zh_3","zh_4","ko_0","ko_1","ko_2","ko_3","ko_4"]

h = lambda t: t.half().float()
window = torch.hamming_window(FRAME_SAMPLES)
dft = torch.fft.rfft(torch.eye(N_FFT)); DR, DI = dft.real, dft.imag
MEL = build_mel_filterbank()
SC = float(1 << 15)

def frames_of(w):
    fr = F.unfold(w.view(1, 1, -1, 1), kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1)).squeeze(0).T
    return fr

def ref_log(wav):
    fr = frames_of(torch.from_numpy(wav).float() * SC)
    pe = fr.clone(); pe[:, 1:] = fr[:, 1:] - 0.97 * fr[:, :-1]; pe[:, 0] = fr[:, 0] * 0.03
    pad = F.pad(pe * window, (0, N_FFT - FRAME_SAMPLES))
    pw = (pad @ DR)[:, :-1] ** 2 + (pad @ DI)[:, :-1] ** 2
    return torch.clamp(pw @ MEL.T, min=1e-10).log()

def fp16_global(wav, k):
    fr = h(frames_of(h(torch.from_numpy(wav).float() * k)))
    pe = fr.clone(); pe[:, 1:] = h(fr[:, 1:] - 0.97 * fr[:, :-1]); pe[:, 0] = h(fr[:, 0] * 0.03)
    pad = F.pad(h(pe * h(window)), (0, N_FFT - FRAME_SAMPLES))
    re_, im_ = h(pad @ h(DR)), h(pad @ h(DI))
    pw = h(h(re_[:, :-1] ** 2) + h(im_[:, :-1] ** 2))
    mel = h(pw @ h(MEL).T)
    return torch.clamp(mel, min=6.2e-5).log() - 2 * np.log(k) + 2 * np.log(SC)

def fp16_perframe(wav, P):
    fr = frames_of(h(torch.from_numpy(wav).float()))
    peak = h(fr.abs().amax(dim=1, keepdim=True).clamp(min=1e-4))      # [T,1]
    fr_peak_raw = fr.abs().amax(dim=1, keepdim=True)
    g = h(P / peak)                                                   # he so rieng tung frame
    fr = h(fr * g)
    pe = fr.clone(); pe[:, 1:] = h(fr[:, 1:] - 0.97 * fr[:, :-1]); pe[:, 0] = h(fr[:, 0] * 0.03)
    pad = F.pad(h(pe * h(window)), (0, N_FFT - FRAME_SAMPLES))
    re_, im_ = h(pad @ h(DR)), h(pad @ h(DI))
    pw = h(h(re_[:, :-1] ** 2) + h(im_[:, :-1] ** 2))
    mel = h(pw @ h(MEL).T)
    # mel_that = mel / g^2 * SC^2  ->  log = log(mel) - 2*log(g) + 2*log(SC)
    out = torch.clamp(mel, min=6.2e-5).log() - 2 * torch.log(g) + 2 * np.log(SC)
    silent = (fr_peak_raw < 3e-8)
    return torch.where(silent, torch.full_like(out, float(np.log(1e-10))), out)

def score(fn, *args):
    ds, bad = [], 0
    per = []
    for i in range(len(calib["wav"])):
        n = int(calib["wav_len"][i].reshape(-1)[0])
        wav = calib["wav"][i].reshape(-1)[:n].astype(np.float32)
        r = ref_log(wav); o = fn(wav, *args)
        d = (o - r).abs()
        # bo qua bin gan im lang tuyet doi (ref < log(1e-6*SC^2) ~ -> khong anh huong sau CMVN/model)
        per.append((d.mean().item(), (d > 0.5).float().mean().item()))
        ds.append(d.flatten())
    d = torch.cat(ds)
    return d.mean().item(), (d > 0.5).float().mean().item(), (d > 2.0).float().mean().item(), per

print(f"{'phuong an':24s} {'mean|dlog|':>11s} {'%bin>0.5':>9s} {'%bin>2':>8s}")
for k in [16.0, 8.0]:
    m, p5, p2, _ = score(fp16_global, k)
    print(f"{'global k='+str(int(k)):24s} {m:11.4f} {100*p5:8.2f}% {100*p2:7.2f}%")
best = None
for P in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    m, p5, p2, per = score(fp16_perframe, P)
    print(f"{'per-frame P='+str(P):24s} {m:11.4f} {100*p5:8.2f}% {100*p2:7.2f}%")
    if best is None or m < best[0]: best = (m, P, per)
print(f"\nTot nhat: P={best[1]}  mean|dlog|={best[0]:.4f}")
print("Tung mau (mean|dlog|, %bin>0.5) voi P tot nhat:")
for lbl, (m, f) in zip(labels, best[2]):
    print(f"  {lbl}: {m:.4f}  {100*f:.2f}%")
