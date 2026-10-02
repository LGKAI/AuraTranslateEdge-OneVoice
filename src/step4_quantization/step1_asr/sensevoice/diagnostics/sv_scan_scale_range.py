# -*- coding: utf-8 -*-
"""Do dai gia tri trung gian cua frontend (wav KHONG scale) tren CPU, de chon he so scale k
sao cho: |real|,|imag|*k < ~200 (de real^2 khong tran 65504) VA mel*k^2 > 6.1e-5 (fp16
normal min) o phan lon bin. Sau do mo phong lam tron fp16 tai tung buoc de chon k tot nhat."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import build_mel_filterbank, FRAME_SAMPLES, HOP_SAMPLES, N_FFT

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
calib = np.load(os.path.join(OUT_DIR, "calib_data_unified_fixed.npz"))

window = torch.hamming_window(FRAME_SAMPLES)
dft = torch.fft.rfft(torch.eye(N_FFT))
dr, di = dft.real, dft.imag
mel_fb = build_mel_filterbank()

def stages(wav_1d, k, fp16=False):
    cast = (lambda t: t.half().float()) if fp16 else (lambda t: t)
    w = cast(torch.from_numpy(wav_1d).float() * k)
    fr = F.unfold(w.view(1, 1, -1, 1), kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1)).squeeze(0).T
    pe = fr.clone(); pe[:, 1:] = fr[:, 1:] - 0.97 * fr[:, :-1]; pe[:, 0] = fr[:, 0] * 0.03
    pe = cast(pe * window)
    pad = F.pad(pe, (0, N_FFT - FRAME_SAMPLES))
    re_, im_ = cast(pad @ dr), cast(pad @ di)
    pw = cast(re_[:, :-1] ** 2 + im_[:, :-1] ** 2)
    mel = cast(pw @ mel_fb.T)
    return re_, im_, pw, mel

all_reim, all_pw, all_mel = [], [], []
for i in range(len(calib["wav"])):
    n = int(calib["wav_len"][i].reshape(-1)[0])
    wav = calib["wav"][i].reshape(-1)[:n].astype(np.float32)
    re_, im_, pw, mel = stages(wav, 1.0)
    all_reim.append(torch.cat([re_.abs().flatten(), im_.abs().flatten()]).numpy())
    all_pw.append(pw.flatten().numpy())
    all_mel.append(mel.flatten().numpy())
reim = np.concatenate(all_reim); pw = np.concatenate(all_pw); mel = np.concatenate(all_mel)
print(f"|real|,|imag| (k=1): max={reim.max():.3e}  p99.99={np.percentile(reim,99.99):.3e}")
print(f"power       (k=1): max={pw.max():.3e}")
print(f"mel         (k=1): max={mel.max():.3e}  p0.1={np.percentile(mel,0.1):.3e}  p1={np.percentile(mel,1):.3e}  min={mel.min():.3e}")

print("\nMo phong fp16 tung buoc, so mel_log voi fp32 (k=1, chinh xac):")
print(f"{'k':>8s} {'max|log diff|':>14s} {'mean|log diff|':>15s} {'%mel underflow':>15s} {'%overflow':>10s}")
for exp in [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
    k = float(2 ** exp)
    diffs, uf, of, tot = [], 0, 0, 0
    for i in range(len(calib["wav"])):
        n = int(calib["wav_len"][i].reshape(-1)[0])
        wav = calib["wav"][i].reshape(-1)[:n].astype(np.float32)
        _, _, _, mel_ref = stages(wav, 1.0, fp16=False)
        _, _, pw16, mel16 = stages(wav, k, fp16=True)
        log_ref = torch.clamp(mel_ref, min=1e-10 / (32768.0 ** 2)).log()
        log16 = torch.clamp(mel16, min=6.1e-5).log() - 2 * np.log(k)
        valid = mel_ref > 1e-14
        d = (log16 - log_ref)[valid].abs()
        diffs.append(d)
        uf += int((mel16 < 6.1e-5)[valid].sum()); of += int(torch.isinf(pw16).sum()); tot += int(valid.sum())
    d = torch.cat(diffs)
    print(f"{'2^'+str(exp):>8s} {d.max().item():14.3f} {d.mean().item():15.5f} {100*uf/tot:14.3f}% {of:10d}")
