# -*- coding: utf-8 -*-
"""Tach nho tung buoc DSP de tim CHINH XAC buoc nao gay lech so voi frontend that cua FunASR:
   buoc 1: fbank tho (truoc LFR/CMVN)
   buoc 2: LFR stacking (truoc CMVN)
   buoc 3: CMVN
So sanh tung buoc voi ham noi bo cua FunASR (kaldi.fbank, apply_lfr, apply_cmvn)."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import torchaudio.compliance.kaldi as K

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)

FS = 16000
N_MELS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
LFR_M = 7
LFR_N = 6
FRAME_SAMPLES = int(FS * FRAME_LENGTH_MS / 1000)
HOP_SAMPLES = int(FS * FRAME_SHIFT_MS / 1000)
N_FFT = 512

window = torch.hamming_window(FRAME_SAMPLES, periodic=False)
dft_complex = torch.fft.rfft(torch.eye(N_FFT))
dft_real, dft_imag = dft_complex.real, dft_complex.imag
mel_fb, _ = K.get_mel_banks(N_MELS, N_FFT, FS, 20.0, 0.0, 100.0, -500.0, 1.0)


def my_fbank(wav):
    wav = wav * float(1 << 15)
    wav_img = wav.unsqueeze(1).unsqueeze(-1)
    frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
    frames = frames.squeeze(0).transpose(0, 1)
    frames_pe = frames.clone()
    frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    windowed = frames_pe * window.unsqueeze(0)
    padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
    real = torch.matmul(padded, dft_real)
    imag = torch.matmul(padded, dft_imag)
    power_no_dc = real[:, 1:] ** 2 + imag[:, 1:] ** 2
    mel = torch.matmul(power_no_dc, mel_fb.T)
    mel_log = torch.clamp(mel, min=1e-10).log()
    return mel_log  # [T, 80], KHONG nhan pre-emphasis vao frame dau (frames_pe[:,0]=frames[:,0])


print("=== So sanh voi torchaudio.compliance.kaldi.fbank truc tiep (chuan Kaldi that) ===")
wav, sr = sf.read("data/asr/en/en_0.wav", dtype="float32")
wav_t = torch.from_numpy(wav).unsqueeze(0)

with torch.no_grad():
    my_mel = my_fbank(wav_t)

# ham fbank CHUAN cua torchaudio (dung dung config Kaldi: frame 25ms, shift 10ms, 80 mel)
kaldi_fbank = K.fbank(
    wav_t, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
    dither=0.0, energy_floor=0.0, window_type="hamming",
    use_energy=False, sample_frequency=16000,
)
print("my_mel shape:", my_mel.shape, "kaldi_fbank shape:", kaldi_fbank.shape)
n = min(my_mel.shape[0], kaldi_fbank.shape[0])
diff = (my_mel[:n] - kaldi_fbank[:n]).abs()
print(f"max_diff={diff.max().item():.5f}  mean_diff={diff.mean().item():.5f}")
print("my_mel[0,:8]    =", my_mel[0, :8].tolist())
print("kaldi_fbank[0,:8]=", kaldi_fbank[0, :8].tolist())
print("my_mel[50,:8]    =", my_mel[50, :8].tolist())
print("kaldi_fbank[50,:8]=", kaldi_fbank[50, :8].tolist())

print("\n=== Neu fbank tho da lech, thu bo pre-emphasis xem co khop khong ===")
def my_fbank_no_preemph(wav):
    wav = wav * float(1 << 15)
    wav_img = wav.unsqueeze(1).unsqueeze(-1)
    frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
    frames = frames.squeeze(0).transpose(0, 1)
    windowed = frames * window.unsqueeze(0)
    padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
    real = torch.matmul(padded, dft_real)
    imag = torch.matmul(padded, dft_imag)
    power_no_dc = real[:, 1:] ** 2 + imag[:, 1:] ** 2
    mel = torch.matmul(power_no_dc, mel_fb.T)
    return torch.clamp(mel, min=1e-10).log()

my_mel2 = my_fbank_no_preemph(wav_t)
diff2 = (my_mel2[:n] - kaldi_fbank[:n]).abs()
print(f"(no preemph) max_diff={diff2.max().item():.5f}  mean_diff={diff2.mean().item():.5f}")

print("\n=== Kiem tra pre-emphasis dung Kaldi that (preemphasis_coefficient) ===")
kaldi_fbank_pe = K.fbank(
    wav_t, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
    dither=0.0, energy_floor=0.0, window_type="hamming",
    use_energy=False, sample_frequency=16000, preemphasis_coefficient=0.97,
)
diff3 = (my_mel[:n] - kaldi_fbank_pe[:n]).abs()
print(f"(so voi kaldi preemph=0.97) max_diff={diff3.max().item():.5f}  mean_diff={diff3.mean().item():.5f}")

print("\n=== Kiem tra window: periodic=True vs False ===")
window_periodic = torch.hamming_window(FRAME_SAMPLES, periodic=True)
def my_fbank_win(win):
    def f(wav):
        wav = wav * float(1 << 15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        windowed = frames_pe * win.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, dft_real); imag = torch.matmul(padded, dft_imag)
        power_no_dc = real[:, 1:] ** 2 + imag[:, 1:] ** 2
        mel = torch.matmul(power_no_dc, mel_fb.T)
        return torch.clamp(mel, min=1e-10).log()
    return f

my_mel_periodic = my_fbank_win(window_periodic)(wav_t)
diff4 = (my_mel_periodic[:n] - kaldi_fbank_pe[:n]).abs()
print(f"(window periodic=True, vs kaldi preemph=0.97) max_diff={diff4.max().item():.5f}  mean_diff={diff4.mean().item():.5f}")
