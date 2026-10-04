# -*- coding: utf-8 -*-
"""Frontend v3: CHUAN HOA THEO TUNG FRAME (per-frame normalization) de giu moi gia tri trung
gian trong dai fp16, thay cho scale toan cuc (k=1 gay underflow 52% mel; k>=2^5 gay overflow).

  g_t  = P / max(peak_t, 1e-4)         (peak_t = max|frame_t| tho)
  frame_n = frame * g_t                (dinh moi frame ~ P)
  ... pre-emphasis / window / DFT / power / mel tren frame_n ...
  mel_log = log(clamp(mel_n, 6.2e-5)) - 2*log(g_t) + 2*log(32768)    (bu chinh xac, cong trong log)
  frame im lang tuyet doi (peak = 0) -> log(1e-10) dung nhu cong thuc goc.

Mo phong fp16 (sv_sim_perframe_norm.py): mean|dlog| = 0.0034 (global k=16 tot nhat: 0.30)."""
import os, sys, time, json, math
sys.stdout.reconfigure(encoding="utf-8")
import torch, torch.nn as nn, torch.nn.functional as F
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import (
    build_mel_filterbank, MAX_WAV_SAMPLES, MAX_LFR_FRAMES,
    FRAME_SAMPLES, HOP_SAMPLES, N_FFT, LFR_M, LFR_N,
)

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
P_TARGET = 2.0
LOG_SC2 = 2.0 * math.log(float(1 << 15))
LOG_FLOOR = math.log(1e-10)


class TraceableFrontendV3(nn.Module):
    def __init__(self, cmvn: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)
        self.register_buffer("mel_fb", build_mel_filterbank())
        self.register_buffer("cmvn_mean", cmvn[0:1, :])
        self.register_buffer("cmvn_scale", cmvn[1:2, :])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)                       # [T, 400]

        peak_raw = frames.abs().amax(dim=1, keepdim=True)                # [T, 1]
        silent = peak_raw < 3e-8
        peak = torch.clamp(peak_raw, min=1e-4)
        g = P_TARGET / peak                                              # [T, 1]
        frames = frames * g

        fpe = frames.clone()
        fpe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        fpe[:, 0] = frames[:, 0] * (1.0 - 0.97)
        padded = F.pad(fpe * self.window.unsqueeze(0), (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        power = real[:, :-1] ** 2 + imag[:, :-1] ** 2
        mel = torch.matmul(power, self.mel_fb.T)

        mel_log = torch.clamp(mel, min=6.2e-5).log() - 2.0 * torch.log(g) + LOG_SC2
        mel_log = torch.where(silent, torch.full_like(mel_log, LOG_FLOOR), mel_log)

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
        return mel_uf.unsqueeze(0)


class FrontendGraphV3(nn.Module):
    def __init__(self, fe):
        super().__init__()
        self.frontend = fe

    def forward(self, wav, wav_len):
        fbank = self.frontend(wav)
        wl = wav_len.to(torch.int64)
        fbank_len = torch.clamp((wl - FRAME_SAMPLES) // HOP_SAMPLES + 1, min=0)
        lfr_len = torch.clamp((fbank_len + (LFR_M // 2) - LFR_M) // LFR_N + 1, min=0, max=MAX_LFR_FRAMES)
        return fbank, lfr_len.to(torch.int32).reshape(1)


def main():
    from funasr import AutoModel
    try:
        am = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    fe = TraceableFrontendV3(am.kwargs.get("frontend").cmvn).eval()
    g = FrontendGraphV3(fe).eval()
    onnx_path = os.path.join(OUT_DIR, "model_sv_frontend_v3.onnx")
    with torch.no_grad():
        torch.onnx.export(g, (torch.randn(1, MAX_WAV_SAMPLES), torch.tensor([MAX_WAV_SAMPLES], dtype=torch.int32)),
                          onnx_path, input_names=["wav", "wav_len"], output_names=["fbank", "speech_lengths"],
                          dynamic_axes={}, opset_version=17, do_constant_folding=True)
    m = onnx.load(onnx_path, load_external_data=True)
    inline = onnx_path.replace(".onnx", "_inline.onnx")
    onnx.save_model(m, inline, save_as_external_data=False)
    print("Saved:", inline, f"{os.path.getsize(inline)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
