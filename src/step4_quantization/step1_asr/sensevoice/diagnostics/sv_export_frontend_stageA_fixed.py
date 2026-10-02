# -*- coding: utf-8 -*-
"""Fix thu nghiem: KHONG scale wav x32768 TRUOC FFT-matmul (gay tich luy vuot fp16 tren NPU
that -- da xac nhan bang stage-diag: avg cos_sim chi 0.24, max_abs_diff ~1e9-1e10, dung
bang binh phuong fp16_max ~4.29e9).

Thay vao do: giu wav o scale goc (~[-1,1]) khi tinh FFT-matmul (gia tri nho, an toan fp16),
roi BU LAI he so scale duoi dang HANG SO CONG trong khong gian log (tuong duong toan hoc
tuyet doi voi cong thuc cu, vi log(k^2 * x) = log(k^2) + log(x)):

  power_no_dc_moi = FFT-matmul tren wav KHONG scale  (nho hon 32768^2 lan, an toan)
  mel_log_moi = log(clamp(mel_moi, floor_an_toan)) + log(32768^2)   <-- CONG hang so, khong nhan

Xuat lai Stage A (power_no_dc, CHUA cong bu -- vi Stage A la truoc mel/log, se so sanh voi
tham chieu CPU cung o thang chua scale x32768^2, tuc la CPU cung phai bo scale tuong ung de
so sanh cong bang -- xem script test kem theo)."""
import os, sys, time, math
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import (
    build_mel_filterbank, MAX_WAV_SAMPLES, FRAME_SAMPLES, HOP_SAMPLES, N_FFT, N_MELS,
)

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
MAX_RAW_FRAMES = 2900
SCALE = float(1 << 15)


class FrontendStageAFixed(nn.Module):
    """wav -> power_no_dc [1, MAX_RAW_FRAMES, 256], KHONG scale x32768 truoc FFT.
    (So sanh voi tham chieu CPU cung KHONG scale, de kiem tra rieng phan FFT-matmul
    co con loi tren NPU hay khong sau khi bo scale lon.)"""
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # KHONG con "wav = wav * SCALE" o day -- day chinh la fix
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
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2  # [n_frame, 256], NHO HON 32768^2 lan
        actual = power_no_dc.shape[0]
        if actual >= MAX_RAW_FRAMES:
            power_no_dc = power_no_dc[:MAX_RAW_FRAMES, :]
        else:
            power_no_dc = F.pad(power_no_dc, (0, 0, 0, MAX_RAW_FRAMES - actual))
        return power_no_dc.unsqueeze(0)


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
    print("Xuat Stage A FIXED (power_no_dc, KHONG scale x32768 truoc FFT)...")
    export_stage(FrontendStageAFixed(), "stageA_power_fixed")
    print("\nHoan tat.")


if __name__ == "__main__":
    main()
