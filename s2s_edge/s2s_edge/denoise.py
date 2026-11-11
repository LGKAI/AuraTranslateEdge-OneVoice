# -*- coding: utf-8 -*-
"""Optional speech enhancement front end: GTCRN (48 k parameters, ~33 MMAC/s) on the device CPU through ONNX Runtime. 16 kHz mono in / out."""
import numpy as np

N_FFT, HOP = 512, 256


def _win(): return np.sqrt(np.hanning(N_FFT + 1)[:-1]).astype(np.float32)                    # sqrt of the periodic Hann window, as torch.hann_window(512).pow(0.5)


def stft(x):
    w = _win(); pad = N_FFT // 2; xp = np.pad(x, (pad, pad), mode="reflect"); n = 1 + (len(xp) - N_FFT) // HOP
    fr = np.lib.stride_tricks.as_strided(xp, (n, N_FFT), (xp.strides[0] * HOP, xp.strides[0])) * w
    s = np.fft.rfft(fr, axis=1).T                                                           # [257, T] complex
    return np.stack([s.real, s.imag], -1).astype(np.float32)[None]                          # [1,257,T,2]


def istft(spec, length):
    w = _win(); s = spec[0, :, :, 0] + 1j * spec[0, :, :, 1]; fr = np.fft.irfft(s.T, n=N_FFT, axis=1) * w
    n = fr.shape[0]; out = np.zeros(HOP * (n - 1) + N_FFT, np.float32); norm = np.zeros_like(out)
    for i in range(n): out[i * HOP: i * HOP + N_FFT] += fr[i]; norm[i * HOP: i * HOP + N_FFT] += w * w
    out = out / np.maximum(norm, 1e-8); pad = N_FFT // 2
    return out[pad: pad + length]


class Gtcrn:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3; self.s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    def __call__(self, wav):
        if len(wav) < N_FFT: return wav
        pk = float(np.abs(wav).max()) + 1e-9; x = wav / pk * 0.5 if pk < 0.1 else wav                    # keep the model in its usual level range
        y = istft(self.s.run(None, {"spec": stft(x.astype(np.float32))})[0], len(x)); return (y * (pk / 0.5) if pk < 0.1 else y).astype(np.float32)
