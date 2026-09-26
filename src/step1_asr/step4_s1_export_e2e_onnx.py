"""step4_s1_export_e2e_onnx.py — Export SenseVoice-Small thành đồ thị ONNX end-to-end

Đầu vào: waveform [1, T_samples] (float32, 16kHz, tối đa 29 giây = 464,000 samples)
          language  [1] (int32: 3=en, 4=zh, 7=ko)
          textnorm  [1] (int32: 15=ITN)

Đầu ra:  token_ids [1, 500] (int64) — CTC argmax raw (chưa decode text)

Điểm khác so với bước cũ (step4_s1_export_sensevoice_onnx.py):
  - Frontend (STFT -> Mel -> LFR -> CMVN) được tích hợp vào ONNX thay vì chạy trên CPU
  - CTC argmax được gắn vào cuối đồ thị — output là token IDs chứ không phải logits
  - Chỉ còn CTC collapse + vocab lookup (~0.1ms) chạy trên CPU

Cách chạy:
  python src/step1_asr/step4_s1_export_e2e_onnx.py
"""
import os
import sys
import types
import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")

# SenseVoice WavFrontend params (đọc từ model)
FS = 16000
N_MELS = 80
FRAME_LENGTH_MS = 25         # 25ms -> 400 samples at 16kHz
FRAME_SHIFT_MS = 10          # 10ms -> 160 samples at 16kHz
LFR_M = 7                    # stack 7 fbank frames -> 7*80 = 560 dim
LFR_N = 6                    # subsample every 6 frames
MAX_WAV_SAMPLES = 464000     # ~29 giây tối đa (500 LFR frames * 6 * 160 samples/frame)
MAX_LFR_FRAMES = 500         # static shape cho NPU

FRAME_SAMPLES = int(FS * FRAME_LENGTH_MS / 1000)   # 400
HOP_SAMPLES = int(FS * FRAME_SHIFT_MS / 1000)      # 160
N_FFT = 512                   # next power of 2 >= FRAME_SAMPLES

MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
MODEL_ID_MS = "iic/SenseVoiceSmall"


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[e2e_export] Output dir: {OUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Mel filterbank matrix (pre-computed, baked thành constant trong ONNX)
# ─────────────────────────────────────────────────────────────────────────────

def build_mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sample_rate=FS):
    """Tạo kaldi mel filterbank matrix [n_mels, n_fft//2]."""
    import torchaudio.compliance.kaldi as K
    mel_banks, _ = K.get_mel_banks(n_mels, n_fft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)
    # mel_banks: [80, 256]
    return mel_banks


# ─────────────────────────────────────────────────────────────────────────────
# Frontend module (ONNX-traceable, không có Python loops / numpy)
# ─────────────────────────────────────────────────────────────────────────────

class TraceableFrontend(nn.Module):
    """
    Traceable frontend: raw waveform [1, T] -> fbank features [1, T_lfr, 560]

    Quy trình (match kaldi.fbank):
      1. Upsample biên độ (× 2^15, giống WavFrontend gốc)
      2. Frame (unfold) & Per-frame pre-emphasis
      3. Window (Hamming) & Pad
      4. STFT (rfft) -> power spectrum (skip DC bin)
      5. Mel filterbank (matmul với baked matrix từ kaldi)
      6. Log compression
      7. LFR: stack 7 liên tiếp, hop 6 -> [T_lfr, 560]
      8. CMVN: (x + mean) * scale
      9. Pad / crop về [1, 500, 560] static shape
    """

    def __init__(self, cmvn: torch.Tensor):
        super().__init__()

        # Window function (Hamming, 400 samples)
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))

        # ── DFT matrix thay cho torch.fft.rfft để fix lỗi ONNX export ──
        # dft_mat: [512, 257] (chỉ lấy phần positive frequencies)
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)  # [512, 257]
        self.register_buffer("dft_imag", dft_complex.imag)  # [512, 257]

        # Mel filterbank matrix [n_mels, n_fft//2] -> [80, 256]
        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

        # CMVN params [2, 560]
        self.register_buffer("cmvn_mean", cmvn[0:1, :])   # [1, 560]
        self.register_buffer("cmvn_scale", cmvn[1:2, :])  # [1, 560]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [1, T_samples] float32
        Returns:
            fbank: [1, 500, 560] float32
        """
        # ── 1. Scale amplitude ──
        wav = wav * float(1 << 15)                   # [1, T]

        # ── 2. Framing (F.unfold ONNX-friendly) ──
        # wav: [1, T] -> [1, 1, T, 1] cho F.unfold (như ảnh 1 channel, cao T, rộng 1)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)               # [1, 1, T, 1]
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1)) # [1, 400, T_frames]
        frames = frames.squeeze(0).transpose(0, 1)             # [T_frames, 400]

        # ── 3. Per-frame pre-emphasis (0.97) ──
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]

        # ── 4. Window & Pad ──
        windowed = frames_pe * self.window.unsqueeze(0)        # [T_frames, 400]
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))   # [T_frames, 512]

        # ── 5. FFT & Power spectrum (DFT via Matmul) ──
        # Tránh lỗi "aten::fft_rfft to ONNX opset version 17 is not supported"
        real = torch.matmul(padded, self.dft_real)             # [T_frames, 257]
        imag = torch.matmul(padded, self.dft_imag)             # [T_frames, 257]
        
        # kaldi skips DC bin (index 0)
        power_no_dc = real[:, 1:] ** 2 + imag[:, 1:] ** 2      # [T_frames, 256]

        # ── 6. Mel filterbank & Log ──
        mel = torch.matmul(power_no_dc, self.mel_fb.T)         # [T_frames, 80]
        mel_log = torch.clamp(mel, min=1e-10).log()            # [T_frames, 80]

        # ── 7. LFR: stack 7, hop 6 (F.unfold ONNX-friendly) ──
        left_pad = mel_log[0:1].expand(LFR_M // 2, -1)         # [3, 80]
        mel_padded = torch.cat([left_pad, mel_log], dim=0)     # [T_frames+3, 80]

        # F.unfold: reshape mel_padded -> [1, 80, T_frames+3, 1]
        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M, 1), stride=(LFR_N, 1))  # [1, 560, T_lfr_actual]
        
        # Sắp xếp lại từ (channel=80, M=7) -> (M=7, channel=80)
        mel_uf = mel_uf.squeeze(0).view(80, LFR_M, -1)         # [80, 7, T_lfr]
        mel_uf = mel_uf.permute(1, 0, 2).reshape(560, -1).transpose(0, 1)  # [T_lfr_actual, 560]

        # ── 8. CMVN: (x + mean) * scale ──
        mel_uf = (mel_uf + self.cmvn_mean) * self.cmvn_scale   # [T_lfr_actual, 560]

        # ── 9. Pad / crop to static shape [500, 560] ──
        T_lfr = MAX_LFR_FRAMES
        actual_len = mel_uf.shape[0]
        if actual_len >= T_lfr:
            mel_uf = mel_uf[:T_lfr, :]
        else:
            pad_amt = T_lfr - actual_len
            mel_uf = F.pad(mel_uf, (0, 0, 0, pad_amt))         # pad feature dim by 0, time dim by pad_amt

        return mel_uf.unsqueeze(0)                           # [1, 500, 560]


# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SenseVoiceE2EPipeline(nn.Module):
    """
    End-to-end pipeline:
      Input:  wav [1, T], language [1], textnorm [1]
      Output: token_ids [1, 500]  (int64, CTC argmax)
    """

    def __init__(self, frontend: TraceableFrontend, acoustic_model):
        super().__init__()
        self.frontend = frontend
        self.acoustic_model = acoustic_model

    def forward(
        self,
        wav: torch.Tensor,            # [1, T_samples]
        language: torch.Tensor,       # [1]
        textnorm: torch.Tensor,       # [1]
    ) -> torch.Tensor:
        # 1. Frontend: wav -> fbank [1, 500, 560]
        fbank = self.frontend(wav)
        speech_lengths = torch.tensor([MAX_LFR_FRAMES], dtype=torch.int32)

        # 2. Acoustic model: fbank -> logits [1, 500, 25055]
        logits, _ = self.acoustic_model(fbank, speech_lengths, language, textnorm)

        # 3. CTC argmax: [1, 500, 25055] -> [1, 500]
        token_ids = torch.argmax(logits, dim=-1)

        return token_ids


# ─────────────────────────────────────────────────────────────────────────────
# Numeric sanity check: compare traceable frontend vs original WavFrontend
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_frontend(am, frontend_new: TraceableFrontend, wav_path: str):
    """So sánh output của TraceableFrontend vs WavFrontend gốc."""
    import soundfile as sf

    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    wav_len = torch.tensor([wav.shape[0]], dtype=torch.int32)

    # Original frontend
    orig_fe = am.kwargs.get("frontend")
    with torch.no_grad():
        orig_out, _ = orig_fe(wav_t, wav_len)

    # New traceable frontend
    # Pad wav to MAX_WAV_SAMPLES for static shape
    T = wav_t.shape[1]
    if T > MAX_WAV_SAMPLES:
        wav_in = wav_t[:, :MAX_WAV_SAMPLES]
    else:
        wav_in = F.pad(wav_t, (0, MAX_WAV_SAMPLES - T))

    with torch.no_grad():
        new_out = frontend_new(wav_in)

    # Compare overlapping frames
    T_orig = orig_out.shape[1]
    T_new = new_out.shape[1]
    T_cmp = min(T_orig, T_new)

    orig_flat = orig_out[0, :T_cmp, :].numpy()
    new_flat = new_out[0, :T_cmp, :].numpy()

    cos_sim = (
        np.dot(orig_flat.flatten(), new_flat.flatten())
        / (np.linalg.norm(orig_flat) * np.linalg.norm(new_flat) + 1e-12)
    )
    max_err = np.abs(orig_flat - new_flat).max()
    print(f"[sanity] Frontend cosine_sim={cos_sim:.6f}  max_abs_err={max_err:.4f}")
    return cos_sim


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_e2e_onnx(pipeline: SenseVoiceE2EPipeline, onnx_path: str) -> bool:
    """Export SenseVoiceE2EPipeline sang ONNX."""
    if os.path.exists(onnx_path):
        print(f"[e2e_export] Already exists: {onnx_path} — skipping")
        return True

    # Dummy inputs
    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES)
    dummy_lang = torch.tensor([3], dtype=torch.int32)       # en
    dummy_tn = torch.tensor([15], dtype=torch.int32)

    print("[e2e_export] Testing forward pass...")
    with torch.no_grad():
        try:
            out = pipeline(dummy_wav, dummy_lang, dummy_tn)
            print(f"[e2e_export] Forward OK: token_ids.shape={out.shape}")
        except Exception as e:
            traceback.print_exc()
            return False

    print(f"[e2e_export] Exporting to {onnx_path} ...")
    t0 = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(
                pipeline,
                (dummy_wav, dummy_lang, dummy_tn),
                onnx_path,
                input_names=["wav", "language", "textnorm"],
                output_names=["token_ids"],
                dynamic_axes={},          # static shapes for NPU
                opset_version=17,
                do_constant_folding=True,
            )
        print(f"[e2e_export] ONNX export done in {time.time()-t0:.1f}s -> {onnx_path}")
        return True
    except Exception as e:
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Patch bias (giống bước cũ, giữ nguyên)
# ─────────────────────────────────────────────────────────────────────────────

def patch_conv_bias(onnx_path: str, patched_path: str) -> bool:
    """Bơm dummy zero-bias vào các Conv node thiếu bias (fix QAIRT crash)."""
    import onnx
    import onnx.numpy_helper as nph

    model = onnx.load(onnx_path)
    graph = model.graph

    existing_inputs = {init.name for init in graph.initializer}
    patched = 0

    for node in graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) >= 3 and node.input[2]:
            continue    # đã có bias

        # Tìm shape của weight để tạo bias đúng kích thước
        weight_name = node.input[1]
        weight_init = next(
            (i for i in graph.initializer if i.name == weight_name), None
        )
        if weight_init is None:
            continue
        out_channels = weight_init.dims[0]

        bias_name = f"{weight_name}_dummy_bias"
        if bias_name not in existing_inputs:
            bias_np = np.zeros(out_channels, dtype=np.float32)
            bias_tensor = nph.from_array(bias_np, name=bias_name)
            graph.initializer.append(bias_tensor)
            existing_inputs.add(bias_name)

        # Đảm bảo node có đủ 3 inputs
        while len(node.input) < 3:
            node.input.append("")
        node.input[2] = bias_name
        patched += 1

    print(f"[patch_bias] Patched {patched} Conv nodes")
    onnx.save(model, patched_path)
    print(f"[patch_bias] Saved patched ONNX -> {patched_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_out_dir()

    # ── 1. Load model ──
    print("\n[1/5] Loading SenseVoice-Small ...")
    from funasr import AutoModel
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()

    # ── 2. Apply export_meta (patch forward) ──
    print("\n[2/5] Patching acoustic model forward ...")
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv_exported = export_rebuild_model(sv, device="cpu", max_seq_len=512)

    # Disable dynamic axes (cần static shapes)
    sv_exported.export_dynamic_axes = types.MethodType(lambda self: {}, sv_exported)

    # ── 3. Build TraceableFrontend ──
    print("\n[3/5] Building TraceableFrontend ...")
    orig_fe = am.kwargs.get("frontend")
    cmvn = orig_fe.cmvn
    print(f"  CMVN shape: {cmvn.shape}")

    frontend = TraceableFrontend(cmvn)
    frontend.eval()

    # Sanity check: so sánh với frontend gốc
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        test_sample = next((r for r in manifest if r["lang"] == "en"), None)
        if test_sample:
            wav_path = os.path.join(ROOT, test_sample["path"])
            cos = sanity_check_frontend(am, frontend, wav_path)
            if cos < 0.99:
                print(f"[WARNING] Frontend sanity check cosine_sim={cos:.4f} < 0.99!")
                print("  -> TraceableFrontend output có thể lệch với gốc. Kiểm tra lại mel filterbank.")
            else:
                print(f"  ✅ Frontend sanity check PASSED (cos_sim={cos:.6f})")

    # ── 4. Build full pipeline & export ONNX ──
    print("\n[4/5] Building E2E pipeline ...")
    pipeline = SenseVoiceE2EPipeline(frontend, sv_exported)
    pipeline.eval()

    onnx_raw = os.path.join(OUT_DIR, "model_e2e.onnx")
    onnx_patched = os.path.join(OUT_DIR, "model_e2e_patched.onnx")

    ok = export_e2e_onnx(pipeline, onnx_raw)
    if not ok:
        print("[ERROR] ONNX export failed. Exiting.")
        return

    # ── 5. Patch Conv bias ──
    print("\n[5/5] Patching Conv bias ...")
    patch_conv_bias(onnx_raw, onnx_patched)

    # Save job config
    config = {
        "e2e_onnx_path": onnx_patched,
        "input": "waveform [1, 464000] float32",
        "output": "token_ids [1, 500] int64",
        "notes": "Frontend + Argmax baked in. Only CTC collapse + vocab lookup on CPU.",
    }
    cfg_path = os.path.join(OUT_DIR, "e2e_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Done. E2E ONNX ready at: {onnx_patched}")
    print(f"   Config: {cfg_path}")
    print("\nBước tiếp theo:")
    print("  python src/step1_asr/step4_s1_qai_hub_submit_e2e.py")


if __name__ == "__main__":
    main()
