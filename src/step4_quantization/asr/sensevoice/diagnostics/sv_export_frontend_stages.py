# -*- coding: utf-8 -*-
"""Xuat 3 checkpoint nho cua frontend de bypass test tren NPU that, tim chinh xac
buoc tinh toan nao gay sai lech lon (FFT-matmul / log / LFR-unfold-reshape).

Stage A: wav -> power_no_dc (truoc mel+log)          shape [1, MAX_RAW_FRAMES, 256]
Stage B: wav -> mel_log (sau mel+log, truoc LFR/CMVN)  shape [1, MAX_RAW_FRAMES, 80]
Stage C: wav -> fbank full (sau LFR+CMVN)              da co san (model_sv_frontend_inline.onnx)
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import (
    build_mel_filterbank, MAX_WAV_SAMPLES, FRAME_SAMPLES, HOP_SAMPLES, N_FFT, N_MELS,
)

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
MAX_RAW_FRAMES = 2900


class FrontendStageA(nn.Module):
    """wav -> power_no_dc [1, MAX_RAW_FRAMES, 256] (truoc mel filterbank & log)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav * float(1 << 15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        frames_pe[:, 0] = frames[:, 0] * (1.0 - 0.97)
        windowed = frames_pe * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2  # [n_frame, 256]
        actual = power_no_dc.shape[0]
        if actual >= MAX_RAW_FRAMES:
            power_no_dc = power_no_dc[:MAX_RAW_FRAMES, :]
        else:
            power_no_dc = F.pad(power_no_dc, (0, 0, 0, MAX_RAW_FRAMES - actual))
        return power_no_dc.unsqueeze(0)  # [1, MAX_RAW_FRAMES, 256]


class FrontendStageB(nn.Module):
    """wav -> mel_log [1, MAX_RAW_FRAMES, 80] (sau mel filterbank & log, TRUOC LFR/CMVN)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)
        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav * float(1 << 15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        frames_pe[:, 0] = frames[:, 0] * (1.0 - 0.97)
        windowed = frames_pe * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2
        mel = torch.matmul(power_no_dc, self.mel_fb.T)
        mel_log = torch.clamp(mel, min=1e-10).log()  # [n_frame, 80]
        actual = mel_log.shape[0]
        if actual >= MAX_RAW_FRAMES:
            mel_log = mel_log[:MAX_RAW_FRAMES, :]
        else:
            mel_log = F.pad(mel_log, (0, 0, 0, MAX_RAW_FRAMES - actual))
        return mel_log.unsqueeze(0)  # [1, MAX_RAW_FRAMES, 80]


def export_stage(module, name):
    module.eval()
    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)
    onnx_path = os.path.join(OUT_DIR, f"model_sv_frontend_{name}.onnx")
    print(f"Xuat {name} -> {onnx_path} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            module, (dummy_wav,), onnx_path,
            input_names=["wav"], output_names=[name],
            dynamic_axes={}, opset_version=17, do_constant_folding=True,
        )
    print(f"  xong sau {time.time()-t0:.1f}s, size={os.path.getsize(onnx_path)/1e6:.2f}MB")
    return onnx_path


def main():
    print("Xuat Stage A (power_no_dc, truoc mel+log)...")
    export_stage(FrontendStageA(), "stageA_power")
    print("\nXuat Stage B (mel_log, sau mel+log, truoc LFR/CMVN)...")
    export_stage(FrontendStageB(), "stageB_mellog")
    print("\nHoan tat. MAX_RAW_FRAMES =", MAX_RAW_FRAMES)


if __name__ == "__main__":
    main()
