"""Step 0 test -- classical microphone-array beamforming (Delay-and-Sum, MVDR).
Zero training, zero neural network -- implemented directly from the textbook
DSP formulas (numpy), matching the "MVDR / GSC / Delay-and-sum co dien"
choice in step0.md (train = 0% by construction). A 4-mic circular array is
simulated with pyroomacoustics (room + a speech source vs. a separate noise
source at another angle), matching the ReSpeaker-4-mic hardware plan.
Compares: 1-mic baseline vs Delay-and-Sum vs MVDR, via PESQ/STOI vs clean.
Maps to Technical Proposal SS4.2 / SS4.4 / SS5.2 (mic array row).
"""
import os
import sys
import csv
import time

import numpy as np
import librosa
import pyroomacoustics as pra
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, list_audio_files, load_wav, save_wav, rtf
from mix_noise import get_noise_segment, NOISE_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
OUT_DIR = os.path.join(ROOT, "outputs", "beamform")
RESULTS_CSV = os.path.join(ROOT, "outputs", "beamform_results.csv")

N_MICS = 4
ARRAY_RADIUS = 0.035        # m, ~ReSpeaker 4-mic circular array
SPEECH_DOA_DEG = 0          # speaker straight ahead of the array
NOISE_DOA_DEG = 100         # noise/interferer off to the side
SRC_DIST = 1.5               # m from array centre
ROOM_DIM = [5.0, 4.0, 3.0]
SOUND_SPEED = 343.0
N_FFT, HOP = 512, 256
TEST_SNR_DB = 5


def make_room(fs):
    """Handles both old (absorption=float) and new (materials=) pyroomacoustics APIs."""
    try:
        return pra.ShoeBox(ROOM_DIM, fs=fs, max_order=3, absorption=0.6)
    except TypeError:
        return pra.ShoeBox(ROOM_DIM, fs=fs, max_order=3,
                            materials=pra.Material(energy_absorption=0.6))


def circular_array(center, radius, n_mics):
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
    return np.stack([center[0] + radius * np.cos(angles),
                      center[1] + radius * np.sin(angles),
                      np.full(n_mics, center[2])])  # (3, n_mics)


def polar_to_xyz(center, dist, angle_deg, z):
    a = np.deg2rad(angle_deg)
    return np.array([center[0] + dist * np.cos(a), center[1] + dist * np.sin(a), z])


def simulate_multichannel(signal, mic_pts, src_pos, fs=SR):
    room = make_room(fs)
    room.add_microphone_array(pra.MicrophoneArray(mic_pts, fs))
    room.add_source(src_pos, signal=signal)
    room.simulate()
    return room.mic_array.signals  # (n_mics, n_samples)


def stft_multi(x, n_fft=N_FFT, hop=HOP):
    return np.stack([librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in x])  # (M, F, T)


def istft_single(X, hop=HOP, length=None):
    return librosa.istft(X, hop_length=hop, length=length)


def steering_vector(mic_pts, ref_idx, doa_deg, freqs, dist=SRC_DIST):
    """Ideal geometric steering vector for a known/assumed look direction."""
    center = mic_pts.mean(axis=1)
    src = polar_to_xyz(center, dist, doa_deg, z=center[2])
    tau = np.linalg.norm(mic_pts - src[:, None], axis=0) / SOUND_SPEED
    tau = tau - tau[ref_idx]
    return np.exp(-1j * 2 * np.pi * freqs[None, :] * tau[:, None])  # (M, F)


def delay_and_sum(X, mic_pts, doa_deg, fs=SR):
    M, F, T = X.shape
    freqs = np.fft.rfftfreq(N_FFT, d=1 / fs)
    d = steering_vector(mic_pts, ref_idx=0, doa_deg=doa_deg, freqs=freqs)  # (M, F)
    w = d.conj() / M
    return np.einsum("mf,mft->ft", w, X)


def mvdr(X_mix, X_noise_only, mic_pts, doa_deg, fs=SR, eps=1e-6):
    M, F, T = X_mix.shape
    freqs = np.fft.rfftfreq(N_FFT, d=1 / fs)
    d = steering_vector(mic_pts, ref_idx=0, doa_deg=doa_deg, freqs=freqs)  # (M, F)
    Y = np.zeros((F, T), dtype=complex)
    I = np.eye(M)
    for f in range(F):
        Xn = X_noise_only[:, f, :]
        Rn = (Xn @ Xn.conj().T) / max(Xn.shape[1], 1) + eps * I
        Rn_inv = np.linalg.inv(Rn)
        df = d[:, f]
        num = Rn_inv @ df
        den = (df.conj() @ num) + eps
        w = num / den
        Y[f, :] = w.conj() @ X_mix[:, f, :]
    return Y


def safe_pesq(clean, test, sr=SR):
    try:
        return pesq(sr, clean, test, "wb")
    except Exception:
        return float("nan")


def process_one(clean_path, lang, noise_files, snr_db=TEST_SNR_DB):
    clean = load_wav(clean_path)
    noise = get_noise_segment(len(clean), noise_files, seed=abs(hash(clean_path)) % (2**31))

    center = np.array([ROOM_DIM[0] / 2, ROOM_DIM[1] / 2, 1.5])
    mic_pts = circular_array(center, ARRAY_RADIUS, N_MICS)
    speech_pos = polar_to_xyz(center, SRC_DIST, SPEECH_DOA_DEG, z=center[2])
    noise_pos = polar_to_xyz(center, SRC_DIST, NOISE_DOA_DEG, z=center[2])

    speech_mc = simulate_multichannel(clean, mic_pts, speech_pos)
    noise_mc = simulate_multichannel(noise, mic_pts, noise_pos)
    n = min(speech_mc.shape[1], noise_mc.shape[1])
    speech_mc, noise_mc = speech_mc[:, :n], noise_mc[:, :n]

    cp = np.mean(speech_mc[0] ** 2) + 1e-12
    npow = np.mean(noise_mc[0] ** 2) + 1e-12
    scale = np.sqrt((cp / (10 ** (snr_db / 10))) / npow)
    noise_mc = noise_mc * scale
    mixed_mc = speech_mc + noise_mc

    ref = mixed_mc[0]  # single-mic baseline (no beamforming)

    X_mix = stft_multi(mixed_mc)
    X_noise = stft_multi(noise_mc)

    t0 = time.perf_counter()
    Y_das = delay_and_sum(X_mix, mic_pts, SPEECH_DOA_DEG)
    das_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_mvdr = mvdr(X_mix, X_noise, mic_pts, SPEECH_DOA_DEG)
    mvdr_time = time.perf_counter() - t0

    out_das = istft_single(Y_das, length=n)
    out_mvdr = istft_single(Y_mvdr, length=n)
    clean_ref = speech_mc[0][:n]

    m = min(len(clean_ref), len(ref), len(out_das), len(out_mvdr))
    clean_ref, ref, out_das, out_mvdr = clean_ref[:m], ref[:m], out_das[:m], out_mvdr[:m]
    audio_s = m / SR

    result = {
        "lang": lang, "file": os.path.basename(clean_path), "snr_db": snr_db,
        "pesq_1mic": round(safe_pesq(clean_ref, ref), 3),
        "pesq_das": round(safe_pesq(clean_ref, out_das), 3),
        "pesq_mvdr": round(safe_pesq(clean_ref, out_mvdr), 3),
        "stoi_1mic": round(stoi(clean_ref, ref, SR), 3),
        "stoi_das": round(stoi(clean_ref, out_das, SR), 3),
        "stoi_mvdr": round(stoi(clean_ref, out_mvdr, SR), 3),
        "rtf_das": round(rtf(das_time, audio_s), 5),
        "rtf_mvdr": round(rtf(mvdr_time, audio_s), 5),
    }

    tag = os.path.splitext(os.path.basename(clean_path))[0]
    save_wav(os.path.join(OUT_DIR, lang, f"{tag}_1mic.wav"), ref)
    save_wav(os.path.join(OUT_DIR, lang, f"{tag}_das.wav"), out_das)
    save_wav(os.path.join(OUT_DIR, lang, f"{tag}_mvdr.wav"), out_mvdr)
    return result


def main():
    noise_files = list_audio_files(NOISE_DIR)
    rows = []
    for lang in ["vi", "en", "zh", "ko"]:
        clean_files = list_audio_files(os.path.join(CLEAN_DIR, lang))
        if not clean_files:
            print(f"[test_beamform] SKIP '{lang}': no files in data/clean/{lang}/")
            continue
        for cpath in clean_files:
            r = process_one(cpath, lang, noise_files)
            rows.append(r)
            print(f"[test_beamform] {lang} {r['file']}  "
                  f"PESQ 1mic={r['pesq_1mic']:.2f} DAS={r['pesq_das']:.2f} MVDR={r['pesq_mvdr']:.2f}  "
                  f"STOI 1mic={r['stoi_1mic']:.2f} DAS={r['stoi_das']:.2f} MVDR={r['stoi_mvdr']:.2f}")

    if not rows:
        print("[test_beamform] No clean audio found under data/clean/<lang>/.")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_beamform] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
