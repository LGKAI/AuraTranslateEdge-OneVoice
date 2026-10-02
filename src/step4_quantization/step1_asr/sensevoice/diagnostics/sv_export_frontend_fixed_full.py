# -*- coding: utf-8 -*-
"""Xuat lai FULL frontend graph (wav, wav_len -> fbank, speech_lengths) VOI FIX tran so fp16:
KHONG scale wav x32768 truoc FFT-matmul nua -- bu lai bang hang so CONG trong khong gian log
(log(k^2 * x) = log(k^2) + log(x), tuong duong toan hoc tuyet doi, an toan tuyet doi ve fp16
vi khong con phep nhan/tich luy gia tri lon nao trong FFT-matmul).

Da xac nhan bang Stage A test: bo scale truoc FFT -> cos_sim tu 0.24 len 1.000000 tren NPU that.
"""
import os, sys, time, json, math
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import (
    build_mel_filterbank, MAX_WAV_SAMPLES, MAX_LFR_FRAMES,
    FRAME_SAMPLES, HOP_SAMPLES, N_FFT, N_MELS, LFR_M, LFR_N,
)

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
SCALE = float(1 << 15)
LOG_SCALE_COMPENSATION = 2.0 * math.log(SCALE)  # bu cho viec bo "wav * SCALE" truoc FFT
SAFE_CLAMP_MIN_FP16 = 1e-10 / (SCALE ** 2)  # floor CHINH XAC tuong duong voi clamp(min=1e-10) o thang goc


class TraceableFrontendFixed(nn.Module):
    """Giong TraceableFrontend goc, nhung KHONG scale wav truoc FFT -- bu bang hang so cong
    sau log(), tranh tran so fp16 tren NPU that (da xac nhan qua Stage A bypass test)."""
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
        # FIX: KHONG con "wav = wav * SCALE" o day nua.
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
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2  # NHO HON 32768^2 lan -- an toan fp16

        mel = torch.matmul(power_no_dc, self.mel_fb.T)
        # FIX: cong bu LOG_SCALE_COMPENSATION thay vi nhan SCALE^2 truoc log -- tuong duong
        # toan hoc tuyet doi (log(k^2*x)=log(k^2)+log(x)) nhung an toan tuyet doi ve fp16.
        mel_log = torch.clamp(mel, min=SAFE_CLAMP_MIN_FP16).log() + LOG_SCALE_COMPENSATION

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


class FrontendGraphFixed(nn.Module):
    def __init__(self, frontend: TraceableFrontendFixed):
        super().__init__()
        self.frontend = frontend

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor):
        fbank = self.frontend(wav)
        wav_len_i = wav_len.to(torch.int64)
        fbank_len = (wav_len_i - FRAME_SAMPLES) // HOP_SAMPLES + 1
        fbank_len = torch.clamp(fbank_len, min=0)
        lfr_len = (fbank_len + (LFR_M // 2) - LFR_M) // LFR_N + 1
        lfr_len = torch.clamp(lfr_len, min=0, max=MAX_LFR_FRAMES)
        speech_lengths = lfr_len.to(torch.int32).reshape(1)
        return fbank, speech_lengths


def main():
    print("[1] Tai CMVN goc tu FunASR SenseVoiceSmall...")
    from funasr import AutoModel
    try:
        am = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    orig_fe = am.kwargs.get("frontend")
    frontend = TraceableFrontendFixed(orig_fe.cmvn)
    frontend.eval()

    graph1 = FrontendGraphFixed(frontend)
    graph1.eval()
    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)
    dummy_wav_len = torch.tensor([MAX_WAV_SAMPLES], dtype=torch.int32)

    onnx_path = os.path.join(OUT_DIR, "model_sv_frontend_fixed.onnx")
    print(f"\n[2] Xuat Frontend FIXED ONNX sang: {onnx_path} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            graph1, (dummy_wav, dummy_wav_len), onnx_path,
            input_names=["wav", "wav_len"], output_names=["fbank", "speech_lengths"],
            dynamic_axes={}, opset_version=17, do_constant_folding=True,
        )
    print(f"  Xuat thanh cong sau {time.time()-t0:.1f}s! Size: {os.path.getsize(onnx_path)/1e6:.1f} MB")

    print("\n[3] Resave inline (tranh loi external-data khi upload AI Hub)...")
    model = onnx.load(onnx_path, load_external_data=True)
    inline_path = onnx_path.replace(".onnx", "_inline.onnx")
    onnx.save_model(model, inline_path, save_as_external_data=False)
    print(f"  Saved: {inline_path}  size={os.path.getsize(inline_path)/1e6:.2f}MB")

    cfg_path = os.path.join(OUT_DIR, "frontend_fixed_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"frontend_fixed_onnx": onnx_path, "frontend_fixed_inline_onnx": inline_path}, f, indent=2)
    print("Saved config:", cfg_path)


if __name__ == "__main__":
    main()
