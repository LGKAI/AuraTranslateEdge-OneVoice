# -*- coding: utf-8 -*-
"""Buoc 1: tai SenseVoice-Small that, so sanh TraceableFrontend (tu viet trong export.py cua
Le Gia Khanh) voi frontend GOC cua FunASR tren audio that -- kiem tra xem DSP/LFR co bi sai
thu tu/transpose khong, TRUOC KHI nghi ngo quantize.
Chay trong tools/venv_sensevoice_diag."""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)

FS = 16000
N_MELS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
LFR_M = 7
LFR_N = 6
MAX_LFR_FRAMES = 500
FRAME_SAMPLES = int(FS * FRAME_LENGTH_MS / 1000)
HOP_SAMPLES = int(FS * FRAME_SHIFT_MS / 1000)
N_FFT = 512


def build_mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sample_rate=FS):
    import torchaudio.compliance.kaldi as K
    mel_banks, _ = K.get_mel_banks(n_mels, n_fft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)
    return mel_banks


class TraceableFrontend(nn.Module):
    """COPY NGUYEN VAN tu step4_s1_export_sensevoice_e2e_unified.py cua Le Gia Khanh -- khong sua gi."""
    def __init__(self, cmvn: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)
        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("cmvn_mean", cmvn[0:1, :])
        self.register_buffer("cmvn_scale", cmvn[1:2, :])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav * float(1 << 15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        windowed = frames_pe * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        power_no_dc = real[:, 1:] ** 2 + imag[:, 1:] ** 2
        mel = torch.matmul(power_no_dc, self.mel_fb.T)
        mel_log = torch.clamp(mel, min=1e-10).log()
        left_pad = mel_log[0:1].expand(LFR_M // 2, -1)
        mel_padded = torch.cat([left_pad, mel_log], dim=0)
        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M, 1), stride=(LFR_N, 1))
        mel_uf = mel_uf.squeeze(0).view(80, LFR_M, -1)
        mel_uf = mel_uf.permute(1, 0, 2).reshape(560, -1).transpose(0, 1)
        mel_uf = (mel_uf + self.cmvn_mean) * self.cmvn_scale
        actual_len = mel_uf.shape[0]
        if actual_len >= MAX_LFR_FRAMES:
            mel_uf = mel_uf[:MAX_LFR_FRAMES, :]
        else:
            mel_uf = F.pad(mel_uf, (0, 0, 0, MAX_LFR_FRAMES - actual_len))
        return mel_uf.unsqueeze(0), actual_len


print("[1] Tai SenseVoice-Small qua FunASR (co the mat vai phut lan dau) ...", flush=True)
from funasr import AutoModel
t0 = time.time()
am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
print(f"    xong sau {time.time()-t0:.0f}s", flush=True)

orig_fe = am.kwargs.get("frontend")
print("    frontend that:", type(orig_fe), flush=True)
print("    cmvn shape:", orig_fe.cmvn.shape if hasattr(orig_fe, "cmvn") else "N/A", flush=True)

my_fe = TraceableFrontend(orig_fe.cmvn)
my_fe.eval()

for lang, path in [("en", "data/asr/en/en_0.wav"), ("zh", "data/asr/zh/zh_0.wav"), ("ko", "data/asr/ko/ko_0.wav")]:
    wav, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    wav_t = torch.from_numpy(wav).unsqueeze(0)

    with torch.no_grad():
        my_feat, actual_len = my_fe(wav_t)          # [1, 500, 560]

    # frontend that cua FunASR -- goi dung cach ho dung noi bo (extract_fbank hoac forward_fbank)
    print(f"\n=== {lang} ({path}) actual_len(tu TraceableFrontend)={actual_len} ===", flush=True)
    try:
        fbank_ref, fbank_len_ref = orig_fe.forward_fbank(wav_t, torch.tensor([wav_t.shape[1]]))
        feat_ref, feat_len_ref = orig_fe.forward_lfr_cmvn(fbank_ref, fbank_len_ref)
        print("  ref fbank shape:", fbank_ref.shape, "| ref feat (LFR+CMVN) shape:", feat_ref.shape, "len=", feat_len_ref, flush=True)
        n = min(my_feat.shape[1], feat_ref.shape[1], actual_len)
        diff = (my_feat[0, :n] - feat_ref[0, :n]).abs()
        print(f"  so voi ref (vung hop le {n} frame): max_diff={diff.max().item():.4f}  mean_diff={diff.mean().item():.4f}", flush=True)
        print(f"  my_feat[0,0,:6]  = {my_feat[0,0,:6].tolist()}", flush=True)
        print(f"  ref_feat[0,0,:6] = {feat_ref[0,0,:6].tolist()}", flush=True)
        print(f"  my_feat[0,{n//2},:6]  = {my_feat[0,n//2,:6].tolist()}", flush=True)
        print(f"  ref_feat[0,{n//2},:6] = {feat_ref[0,n//2,:6].tolist()}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  LOI khi goi frontend that:", e, flush=True)
