# -*- coding: utf-8 -*-
"""Buoc 1: xay fbank bang thuan matmul/gather (numpy), doi chieu CHINH XAC voi kaldi_native_fbank
TRUOC KHI dung ONNX/NPU. Dung config snip_edges=True (don gian, khong can dem bien) vi da do
thuc te: WER CTC voi config nay (6.22% clean) van chap nhan duoc, doi lay kha nang tai tao
CHINH XAC bang matmul (snip_edges=False can logic dem/phan chieu Kaldi phuc tap hon nhieu)."""
import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf

SR = 16000
FRAME_LEN = 400   # 25ms
FRAME_SHIFT = 160  # 10ms
NFFT = 512         # round_to_power_of_two(400)
NUM_BINS = 80
LOW_FREQ = 20.0
HIGH_FREQ = 0.0    # kaldi: <=0 nghia la nyquist + high_freq -> 8000 Hz
PREEMPH = 0.97


def povey_window(n=FRAME_LEN):
    i = np.arange(n)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1)), 0.85).astype(np.float64)


def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def build_mel_matrix(nfft=NFFT, sr=SR, num_bins=NUM_BINS, low_freq=LOW_FREQ, high_freq=HIGH_FREQ):
    nyquist = sr / 2.0
    hf = nyquist + high_freq if high_freq <= 0 else high_freq
    mel_low, mel_high = mel_scale(low_freq), mel_scale(hf)
    mel_pts = mel_low + (np.arange(num_bins + 2) / (num_bins + 1)) * (mel_high - mel_low)
    n_fft_bins = nfft // 2 + 1
    fft_freqs = np.arange(n_fft_bins) * sr / nfft
    fft_mel = mel_scale(fft_freqs)
    mat = np.zeros((n_fft_bins, num_bins), dtype=np.float64)
    for b in range(num_bins):
        left, center, right = mel_pts[b], mel_pts[b + 1], mel_pts[b + 2]
        for k in range(n_fft_bins):
            m = fft_mel[k]
            if left < m < right:
                w = (m - left) / (center - left) if m <= center else (right - m) / (right - center)
                mat[k, b] = max(w, 0.0)
    return mat.astype(np.float32)


def build_dft_matrices(nfft=NFFT):
    n_bins = nfft // 2 + 1
    n = np.arange(nfft)[:, None]
    k = np.arange(n_bins)[None, :]
    ang = -2 * np.pi * n * k / nfft
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


WINDOW = povey_window()
MEL_MAT = build_mel_matrix()
DFT_RE, DFT_IM = build_dft_matrices()


def fbank_matmul(wav):
    n = wav.shape[0]
    num_frames = 1 + (n - FRAME_LEN) // FRAME_SHIFT
    idx = (np.arange(num_frames)[:, None] * FRAME_SHIFT + np.arange(FRAME_LEN)[None, :])
    frames = wav[idx].astype(np.float64)                      # (T, 400)
    frames = frames - frames.mean(axis=1, keepdims=True)       # remove_dc_offset
    shifted = np.concatenate([frames[:, 0:1], frames[:, 0:-1]], axis=1)
    frames = frames - PREEMPH * shifted                        # preemphasis
    frames = frames * WINDOW[None, :]                          # povey window
    padded = np.zeros((num_frames, NFFT), dtype=np.float64)
    padded[:, :FRAME_LEN] = frames
    re = padded @ DFT_RE.astype(np.float64)
    im = padded @ DFT_IM.astype(np.float64)
    power = re * re + im * im
    mel = power @ MEL_MAT.astype(np.float64)
    return np.log(np.maximum(mel, 1.1920929e-07)).astype(np.float32)


def fbank_kaldi_ref(wav):
    o = knf.FbankOptions()
    o.frame_opts.samp_freq = SR
    o.mel_opts.num_bins = NUM_BINS
    o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = True
    o.mel_opts.low_freq = LOW_FREQ
    o.mel_opts.high_freq = HIGH_FREQ
    f = knf.OnlineFbank(o)
    f.accept_waveform(SR, wav.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


if __name__ == "__main__":
    wav, sr = sf.read("data/asr/vi/vi_1.wav", dtype="float32")
    assert sr == 16000
    # KHONG nhan 32768: da xac nhan bang thuc nghiem (dbg9) quy uoc THAT cua pipeline (build_zip150_ctc_static.py)
    # la dua wav [-1,1] THANG cho kaldi_native_fbank, khong scale ve int16. Nhan nham 32768 truoc day lam
    # dich hang so +20.8 trong khong gian log-mel -> khong pha "clean" (audio to, du du bien) nhung lam encoder
    # doan toan blank tren audio nhieu/nho tieng (snr5/snr0) -- xem bao cao muc bug that ngay 2026-09-22.
    mine = fbank_matmul(wav)
    ref = fbank_kaldi_ref(wav)
    print("shapes:", mine.shape, ref.shape)
    n = min(mine.shape[0], ref.shape[0])
    diff = np.abs(mine[:n] - ref[:n])
    print(f"max_abs_diff={diff.max():.6f}  mean_abs_diff={diff.mean():.6f}")
    print(f"ref std={ref[:n].std():.3f}  mine std={mine[:n].std():.3f}")
    rel = diff.max() / (np.abs(ref[:n]).max() + 1e-9)
    print(f"max_abs_diff / max|ref| = {rel:.5f}")
    print("mine[0,:8] =", mine[0, :8])
    print("ref [0,:8] =", ref[0, :8])
    print("mine[10,:8]=", mine[10, :8])
    print("ref [10,:8]=", ref[10, :8])
