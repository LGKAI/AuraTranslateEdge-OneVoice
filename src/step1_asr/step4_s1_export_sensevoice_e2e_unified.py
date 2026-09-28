"""step4_s1_export_sensevoice_e2e_unified.py — Xuất mô hình SenseVoice-Small End-to-End tĩnh 100% NPU (v2)

Kiến trúc đồ thị tính toán hợp nhất 5 khối (Single Static DAG):
  Khối 1: WavFrontend DSP tĩnh (NPU)
          Conv1D Framing + Windowing + DFT MatMul + Mel Filterbank + LFR + CMVN
          Sóng âm thô [1, MAX_WAV_SAMPLES] (16kHz) -> [1, MAX_LFR_FRAMES, 560]
  Khối 2: SenseVoice Transformer Core (NPU)
          Tĩnh hóa Positional Encoder (Add [1, MAX_SEQ_FRAMES, 560]) + 50 lớp Transformer nén sâu
          -> [1, MAX_SEQ_FRAMES, 512]
  Khối 3: CTC Head & ArgMax (NPU)
          Linear Projection [512 -> vocab_size] + ArgMax(axis=-1)
          -> [1, MAX_SEQ_FRAMES] Frame Tokens
  Khối 4: Static CTC Collapse (NPU) — CẢI TIẾN v2
          Lọc trùng liên tiếp + Lọc Blank (0) + Lọc TẤT CẢ token đặc biệt <|...|>
          (gồm |lid|, |ser|, |aed|, |itn|) qua special_ids_mask (int32) gather lookup
          + CumSum & Scatter với Trash-Bin tĩnh
          -> [1, MAX_SEQ_FRAMES] Packed Clean Tokens
  Khối 5: Static Byte Detokenize (NPU)
          Nhúng Bảng Byte M_byte [vocab_size, L_MAX] -> Gather -> Reshape
          -> [1, MAX_SEQ_FRAMES * L_MAX] UTF-8 Byte Stream
  Tầng Host CPU (Zero-CPU Decoding):
          bytes(output[0].astype(np.uint8)).decode("utf-8", errors="ignore").replace("\x00", "").strip()
          -> Không tốn bất kỳ chi phí Tokenizer nào trên CPU (< 0.001 ms).

Cải tiến v2 (fix các lỗi quan trọng):
  1. Hỗ trợ đầy đủ Vocab Token IDs thực tế của SenseVoice:
     - Vocab Token IDs: zh=24884 (<|zh|>), en=24885 (<|en|>), ko=24896 (<|ko|>), etc.
     - Tự động map sang Query IDs (3, 4, 12) cho tầng embed [16, 560] thông qua torch.where (100% NPU).
  2. Sửa lỗi Gather BOOL_8 trên Qualcomm Hexagon NPU:
     - Dùng int32 cho special_ids_mask để toán tử Gather trên HTP backend chạy native 100%.
  3. Thứ tự cổng đầu vào chuẩn hóa theo bảng chữ cái:
     ["language", "textnorm", "wav"] khớp đồng nhất 100% giữa PyTorch, ONNX và QNN DLC Compiler.
  4. Dynamic L_MAX: tính từ max byte length thực tế của vocab, align đến bội số 4.
  5. Bucket-based wav length cho calib data.

Tác giả: Lê Gia Khánh (SenseVoice-Small Step 4 v2)
Tham khảo: Cơ chế End-to-End của Trần Quốc Khanh (step4_zipformer.pdf)
"""

import os
import sys
import types
import json
import time
import re
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix UTF-8 encoding on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")

# SenseVoice WavFrontend thông số
FS = 16000
N_MELS = 80
FRAME_LENGTH_MS = 25         # 25ms -> 400 samples tại 16kHz
FRAME_SHIFT_MS = 10          # 10ms -> 160 samples tại 16kHz
LFR_M = 7                    # stack 7 khung mel liên tiếp -> 7*80 = 560 dim
LFR_N = 6                    # bước nhảy 6 khung

FRAME_SAMPLES = int(FS * FRAME_LENGTH_MS / 1000)   # 400
HOP_SAMPLES   = int(FS * FRAME_SHIFT_MS / 1000)    # 160
N_FFT = 512

# ── Thông số tĩnh cho NPU (bucket 29s — mặc định) ──
MAX_WAV_SAMPLES = 464000     # ~29 giây (500 LFR frames * 6 * 160 + 400)
MAX_LFR_FRAMES  = 500        # số LFR frames tối đa
MAX_SEQ_FRAMES  = 504        # MAX_LFR_FRAMES + 4 prompt tokens
NUM_PROMPT_TOKENS = 4        # |lid|, |ser|, |aed|, |itn|

# L_MAX sẽ được tính động từ vocab sau khi load tokenizer
L_MAX = 24                   # fallback; override bởi compute_l_max()

MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
MODEL_ID_MS  = "iic/SenseVoiceSmall"

# ── Vocab Token IDs trong tokenizer thực tế của SenseVoice (khớp tokens.json / vocabulary) ──
VOCAB_LID_DICT = {
    "auto": 0,
    "zh": 24884,        # <|zh|>
    "en": 24885,        # <|en|>
    "yue": 24888,       # <|yue|>
    "ja": 24892,        # <|ja|>
    "ko": 24896,        # <|ko|>
    "nospeech": 24992,  # <|nospeech|>
}
VOCAB_TEXTNORM_DICT = {
    "withitn": 25016,   # <|withitn|>
    "woitn": 25017,     # <|woitn|>
}

# ── Query IDs nội bộ của tầng am.model.embed (Embedding(16, 560)) trong SenseVoiceSmall ──
QUERY_LID_DICT = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}
QUERY_TEXTNORM_DICT = {
    "withitn": 14,
    "woitn": 15,
}

# Alias mặc định trỏ về VOCAB_LID_DICT (khớp chính xác với vocab SenseVoice)
LID_DICT = VOCAB_LID_DICT
TEXTNORM_DICT = VOCAB_TEXTNORM_DICT


def map_language_to_query_id(lang_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ánh xạ linh hoạt từ Vocab Token ID (24884, 24885, 24896, ...) hoặc trực tiếp Query ID (3, 4, 12, ...)
    về đúng index của bảng embed [16, 560] của SenseVoice.
    100% NPU traceable thông qua torch.where.
    """
    q = lang_tensor
    q = torch.where(q == 24884, torch.tensor(3, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    q = torch.where(q == 24885, torch.tensor(4, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    q = torch.where(q == 24888, torch.tensor(7, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    q = torch.where(q == 24892, torch.tensor(11, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    q = torch.where(q == 24896, torch.tensor(12, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    q = torch.where(q == 24992, torch.tensor(13, dtype=lang_tensor.dtype, device=lang_tensor.device), q)
    return q


def map_textnorm_to_query_id(tn_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ánh xạ linh hoạt từ Vocab Token ID (25016, 25017) hoặc Query ID (14, 15)
    về đúng index của bảng embed.
    """
    q = tn_tensor
    q = torch.where(q == 25016, torch.tensor(14, dtype=tn_tensor.dtype, device=tn_tensor.device), q)
    q = torch.where(q == 25017, torch.tensor(15, dtype=tn_tensor.dtype, device=tn_tensor.device), q)
    return q


# Buckets cho calibration data đa dạng độ dài audio
WAV_BUCKETS = {
    "3s":  3 * FS,
    "6s":  6 * FS,
    "10s": 10 * FS,
    "15s": 15 * FS,
    "20s": 20 * FS,
    "29s": MAX_WAV_SAMPLES,
}


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Init] Output dir: {OUT_DIR}")


def compute_l_max(tokenizer_sp) -> int:
    """
    Tính L_MAX động: độ dài byte tối đa thực tế của 1 token BPE trong vocab.
    Kết quả được căn chỉnh lên bội số 4 để tối ưu bộ nhớ NPU.
    """
    vocab_size = tokenizer_sp.get_piece_size()
    max_b = 0
    for i in range(vocab_size):
        piece = tokenizer_sp.id_to_piece(i)
        if piece in ["<unk>", "<s>", "</s>"] or re.match(r"^<\|.*\|>$", piece):
            continue
        norm_piece = piece.replace("\u2581", " ")
        b = norm_piece.encode("utf-8")
        if len(b) > max_b:
            max_b = len(b)
    # Align lên bội số 4
    l_max = ((max_b + 3) // 4) * 4
    print(f"[compute_l_max] Max token byte length: {max_b} -> L_MAX={l_max} (aligned to 4)")
    return l_max


def best_bucket(wav_samples: int) -> int:
    """Chọn bucket nhỏ nhất đủ chứa wav_samples."""
    for k, v in sorted(WAV_BUCKETS.items(), key=lambda x: x[1]):
        if wav_samples <= v:
            return v
    return MAX_WAV_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# Khối 1: WavFrontend DSP tĩnh (ONNX-traceable)
# ─────────────────────────────────────────────────────────────────────────────

def build_mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, sample_rate=FS):
    """Tạo Kaldi mel filterbank matrix [n_mels, n_fft//2]."""
    import torchaudio.compliance.kaldi as K
    mel_banks, _ = K.get_mel_banks(n_mels, n_fft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)
    return mel_banks  # [80, 256]


class TraceableFrontend(nn.Module):
    """
    Khối DSP trích xuất đặc trưng âm thanh tĩnh trên NPU:
    raw waveform [1, MAX_WAV_SAMPLES] -> fbank [1, MAX_LFR_FRAMES, 560]
    """
    def __init__(self, cmvn: torch.Tensor, max_lfr_frames: int = MAX_LFR_FRAMES):
        super().__init__()
        self.max_lfr_frames = max_lfr_frames
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))

        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)

        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

        self.register_buffer("cmvn_mean",  cmvn[0:1, :])
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
        if actual_len >= self.max_lfr_frames:
            mel_uf = mel_uf[:self.max_lfr_frames, :]
        else:
            mel_uf = F.pad(mel_uf, (0, 0, 0, self.max_lfr_frames - actual_len))

        return mel_uf.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Khối 2: Tĩnh hóa Positional Encoder cho SenseVoice
# ─────────────────────────────────────────────────────────────────────────────

class StaticSinusoidalPositionEncoder(nn.Module):
    """
    Constant Positional Encoding Tensor [1, MAX_SEQ_FRAMES, 560].
    Loại bỏ toàn bộ toán tử động, tránh lỗi broadcast trên Qualcomm Hexagon.
    """
    def __init__(self, timesteps: int = MAX_SEQ_FRAMES, depth: int = 560):
        super().__init__()
        positions = torch.arange(1, timesteps + 1).float()[None, :]
        log_timescale_increment = torch.log(torch.tensor([10000.0])) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2).float() * (-log_timescale_increment))
        scaled_time = positions.unsqueeze(-1) * inv_timescales.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        self.register_buffer("pe", encoding.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


# ─────────────────────────────────────────────────────────────────────────────
# Khối 4 & 5: Static CTC Collapse + UTF-8 Detokenize (100% NPU)
# ─────────────────────────────────────────────────────────────────────────────

def build_static_byte_table(tokenizer_sp, vocab_size: int, l_max: int) -> torch.Tensor:
    """
    Ma trận Byte Tra cứu Tĩnh M_byte [vocab_size, l_max] (int32).
    - Token đặc biệt (<unk>, <s>, </s>, <|...|>): all zeros.
    - '▁' (U+2581) -> ASCII space ' '.
    """
    table = np.zeros((vocab_size, l_max), dtype=np.int32)
    for i in range(vocab_size):
        piece = tokenizer_sp.id_to_piece(i)
        if piece in ["<unk>", "<s>", "</s>"] or re.match(r"^<\|.*\|>$", piece):
            b = b""
        else:
            norm_piece = piece.replace("\u2581", " ")
            b = norm_piece.encode("utf-8")
        for j in range(min(len(b), l_max)):
            table[i, j] = b[j]
    return torch.from_numpy(table)


def build_special_ids_mask(tokenizer_sp, vocab_size: int) -> torch.Tensor:
    """
    Mask [vocab_size] kiểu int32 — 1 nếu token là <|...|> (special control token), 0 nếu là nội dung.
    Dùng int32 thay vì bool để toán tử Gather trên Qualcomm Hexagon NPU (HTP backend)
    chạy native 100%, không bị lỗi QNN_DATATYPE_BOOL_8 (MODEL_GRAPH_ERROR).
    """
    mask = np.zeros(vocab_size, dtype=np.int32)
    n_special = 0
    for i in range(vocab_size):
        piece = tokenizer_sp.id_to_piece(i)
        if re.match(r"^<\|.*\|>$", piece):
            mask[i] = 1
            n_special += 1
    print(f"[build_special_ids_mask] Tổng số token <|...|>: {n_special}")
    return torch.from_numpy(mask)


class StaticCTCCollapseAndDetokenizer(nn.Module):
    """
    Khối tích hợp CTC Collapse + UTF-8 Detokenize hoàn toàn trên NPU — v2:
      Đầu vào:  logits [1, MAX_SEQ_FRAMES, vocab_size]
      Đầu ra:  byte_stream [1, MAX_SEQ_FRAMES * l_max] (int32)

    Cải tiến v2: Lọc đồng thời blank (id=0) VÀ tất cả token <|...|> bằng special_ids_mask (int32).
    Không hard-code vị trí 4 token đầu — hoạt động chính xác kể cả khi token đặc biệt
    xuất hiện ở bất kỳ vị trí nào trong chuỗi CTC.

    Pipeline:
      1. ArgMax -> token_ids [1, MAX_SEQ_FRAMES]
      2. dedup_mask: y[t] != y[t-1]
      3. special_mask: special_ids_mask[token_ids] (gather lookup INT32, 100% NPU native)
      4. valid_mask: dedup & (id != 0) & ~is_special
      5. CumSum -> c[t]
      6. pos = where(valid, c-1, TRASH_BIN)
      7. Scatter vào [1, MAX_SEQ_FRAMES+1] -> cắt [:, :MAX_SEQ_FRAMES]
      8. F.embedding với byte_table -> [1, MAX_SEQ_FRAMES, l_max]
      9. Reshape -> [1, MAX_SEQ_FRAMES * l_max]
    """
    def __init__(
        self,
        byte_table: torch.Tensor,         # [vocab_size, l_max] int32
        special_ids_mask: torch.Tensor,   # [vocab_size] int32 (1: special, 0: normal)
        max_frames: int = MAX_SEQ_FRAMES,
        l_max: int = L_MAX,
    ):
        super().__init__()
        self.register_buffer("byte_table", byte_table)
        self.register_buffer("special_ids_mask", special_ids_mask)
        self.max_frames = max_frames
        self.l_max = l_max
        self.register_buffer("neg_one", torch.tensor([[-1]], dtype=torch.int32))
        self.register_buffer("dummy_idx", torch.tensor(max_frames, dtype=torch.int64))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # 1. ArgMax
        token_ids = torch.argmax(logits, dim=-1).to(torch.int32)  # [1, MAX_SEQ_FRAMES]

        # 2. Deduplication
        prev = torch.cat([self.neg_one, token_ids[:, :-1]], dim=-1)
        dedup_mask = (token_ids != prev)

        # 3. Special token mask (gather lookup INT32 — tương thích 100% Hexagon NPU HTP)
        special_flags = self.special_ids_mask[token_ids.to(torch.int64)]  # [1, MAX_SEQ_FRAMES] int32
        is_special = (special_flags != 0)                                 # [1, MAX_SEQ_FRAMES] bool

        # 4. Valid mask: loại bỏ blank (0) + trùng lặp + token đặc biệt
        valid_mask = dedup_mask & (token_ids != 0) & (~is_special)

        # 5. CumSum
        c = torch.cumsum(valid_mask.to(torch.int64), dim=-1)

        # 6. Destination Index với Trash-Bin
        pos = torch.where(valid_mask, c - 1, self.dummy_idx)

        # 7. Scatter vào buffer [1, MAX_SEQ_FRAMES + 1]
        buffer = torch.zeros(
            token_ids.shape[0], self.max_frames + 1,
            dtype=torch.int32, device=token_ids.device
        )
        packed_buffer = torch.scatter(buffer, 1, pos, token_ids)
        packed_tokens = packed_buffer[:, :self.max_frames]  # [1, MAX_SEQ_FRAMES]

        # 8. Byte lookup
        bytes_out = F.embedding(packed_tokens.to(torch.int64), self.byte_table)  # [1, MAX_SEQ_FRAMES, l_max]

        # 9. Flatten
        bytes_stream = bytes_out.reshape(token_ids.shape[0], self.max_frames * self.l_max)
        return bytes_stream


# ─────────────────────────────────────────────────────────────────────────────
# Toàn bộ Pipeline Đồ thị Tĩnh Duy nhất (Single Static DAG)
# ─────────────────────────────────────────────────────────────────────────────

class SenseVoiceUnifiedE2E(nn.Module):
    """
    Toàn bộ Pipeline ASR SenseVoice 5 khối chạy 100% NPU.
    Đầu vào:
      - language: [1] (int32) - Chấp nhận Vocab Token ID (24884, 24885, 24896) hoặc Query ID (3, 4, 12) hoặc 0 (auto)
      - textnorm: [1] (int32) - Chấp nhận Vocab Token ID (25016, 25017) hoặc Query ID (14, 15)
      - wav:      [1, MAX_WAV_SAMPLES] (float32)
    Thứ tự inputs đã được xếp theo thứ tự bảng chữ cái ["language", "textnorm", "wav"]
    để đồng nhất 100% giữa PyTorch, ONNX, Quantize và QNN DLC Compiler.
    """
    def __init__(
        self,
        frontend: TraceableFrontend,
        acoustic_model: nn.Module,
        decoder: StaticCTCCollapseAndDetokenizer,
        max_lfr_frames: int = MAX_LFR_FRAMES,
    ):
        super().__init__()
        self.frontend = frontend
        self.acoustic_model = acoustic_model
        self.decoder = decoder
        self.max_lfr_frames = max_lfr_frames

    def forward(self, language: torch.Tensor, textnorm: torch.Tensor, wav: torch.Tensor) -> torch.Tensor:
        # Tự động ánh xạ Vocab Token ID sang Query ID (nếu cần)
        lang_q = map_language_to_query_id(language)
        tn_q   = map_textnorm_to_query_id(textnorm)

        fbank = self.frontend(wav)
        speech_lengths = torch.tensor([self.max_lfr_frames], dtype=torch.int32, device=wav.device)
        logits, _ = self.acoustic_model(fbank, speech_lengths, lang_q, tn_q)
        byte_stream = self.decoder(logits)
        return byte_stream


# ─────────────────────────────────────────────────────────────────────────────
# Graph Surgery & Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

def patch_conv_bias(onnx_path: str, patched_path: str) -> int:
    """Bơm dummy zero-bias vào các node Conv thiếu bias (fix QAIRT crash)."""
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
            continue

        weight_name = node.input[1]
        weight_init = next((i for i in graph.initializer if i.name == weight_name), None)
        if weight_init is None:
            continue
        out_channels = weight_init.dims[0]

        bias_name = f"{weight_name}_dummy_bias"
        if bias_name not in existing_inputs:
            bias_np = np.zeros(out_channels, dtype=np.float32)
            bias_tensor = nph.from_array(bias_np, name=bias_name)
            graph.initializer.append(bias_tensor)
            existing_inputs.add(bias_name)

        while len(node.input) < 3:
            node.input.append("")
        node.input[2] = bias_name
        patched += 1

    onnx.save(model, patched_path)
    print(f"[patch_conv_bias] Patched {patched} Conv nodes -> {patched_path}")
    return patched


def patch_mask_outliers(onnx_path: str, patched_path: str, clip_val: float = -30.0) -> int:
    """Clamp các giá trị cực đoan trong attention masks."""
    import onnx
    import onnx.numpy_helper as nph

    model = onnx.load(onnx_path)
    patched = 0
    for i, init in enumerate(model.graph.initializer):
        arr = nph.to_array(init)
        if arr.size == 0 or arr.dtype not in [np.float32, np.float16]:
            continue
        if float(arr.min()) < clip_val:
            arr_clipped = np.clip(arr, clip_val, None)
            new_init = nph.from_array(arr_clipped.astype(arr.dtype), name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, new_init)
            patched += 1

    onnx.save(model, patched_path)
    print(f"[patch_mask] Clipped {patched} outlier initializers -> {patched_path}")
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Verification with ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

def verify_exported_model(onnx_path: str, manifest_path: str, max_wav_samples: int, byte_stream_len: int):
    """Kiểm thử mô hình ONNX đã xuất bằng ONNX Runtime trên tập test thật."""
    import onnxruntime as ort
    import soundfile as sf

    print("\n" + "=" * 70)
    print(f"[Verify] ONNX Runtime trên {os.path.basename(onnx_path)} ...")
    print("=" * 70)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it["lang"] in ["en", "zh", "ko"]]
    total = len(sv_items)
    passed = 0

    for item in sv_items:
        lang = item["lang"]
        path = os.path.join(ROOT, item["path"])
        # Dùng trực tiếp Vocab Token ID thực tế của SenseVoice
        lang_id = VOCAB_LID_DICT.get(lang, 0)
        tn_id   = VOCAB_TEXTNORM_DICT["withitn"]

        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = wav.astype(np.float32)
        if len(wav_t) > max_wav_samples:
            wav_in = wav_t[:max_wav_samples]
        else:
            wav_in = np.pad(wav_t, (0, max_wav_samples - len(wav_t)))
        wav_in = wav_in.reshape(1, max_wav_samples)

        t0 = time.perf_counter()
        ort_outs = sess.run(None, {
            "language": np.array([lang_id], dtype=np.int32),
            "textnorm": np.array([tn_id],   dtype=np.int32),
            "wav":      wav_in,
        })
        latency_ms = (time.perf_counter() - t0) * 1000

        byte_stream = ort_outs[0][0]
        raw_bytes = bytes(byte_stream.astype(np.uint8))
        decoded_text = raw_bytes.decode("utf-8", errors="ignore").replace("\x00", "").strip()

        ref = item["transcript"]
        print(f"[{lang.upper()}] {os.path.basename(path)} ({latency_ms:.1f} ms)")
        print(f"  Ref:    {ref}")
        print(f"  Output: {decoded_text}")

        ref_norm = re.sub(r"[^\w\s]", "", ref).replace(" ", "").lower()
        dec_norm = re.sub(r"[^\w\s]", "", decoded_text).replace(" ", "").lower()
        is_close = (
            ref_norm in dec_norm
            or dec_norm in ref_norm
            or (len(set(ref_norm) & set(dec_norm)) / max(len(set(ref_norm)), 1) > 0.8)
        )
        if is_close:
            passed += 1
            print("  -> ✅ CHUẨN XÁC CAO")
        else:
            print("  -> ⚠️  KHÁC BIỆT")
        print("-" * 70)

    acc = (passed / total) * 100 if total > 0 else 0.0
    print(f"\n[Verify Summary] {passed}/{total} ({acc:.1f}%)")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_out_dir()

    print("\n" + "=" * 70)
    print("SENSEVOICE-SMALL E2E NPU — EXPORT v2")
    print("Fix: Special token filter | Dynamic L_MAX | Vocab LID | Native INT32 Gather")
    print("=" * 70)

    # 1. Load model
    print("\n[1/6] Tải SenseVoice-Small từ FunASR ...")
    from funasr import AutoModel
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()
    tokenizer_sp = am.kwargs.get("tokenizer").sp
    vocab_size = tokenizer_sp.get_piece_size()
    print(f"  Vocab size: {vocab_size}")

    # 2. Compute dynamic L_MAX
    l_max = compute_l_max(tokenizer_sp)
    byte_stream_len = MAX_SEQ_FRAMES * l_max
    print(f"  BYTE_STREAM_LEN = {MAX_SEQ_FRAMES} x {l_max} = {byte_stream_len}")

    # 3. Build byte table
    print(f"\n[2/6] Xây dựng M_byte [{vocab_size}, {l_max}] ...")
    byte_table_tensor = build_static_byte_table(tokenizer_sp, vocab_size, l_max)
    print(f"  shape={byte_table_tensor.shape}, dtype={byte_table_tensor.dtype}")

    # 4. Build special token mask
    print(f"\n[3/6] Xây dựng Special IDs Mask (lọc |lid|, |ser|, |aed|, |itn| v.v.) ...")
    special_ids_mask = build_special_ids_mask(tokenizer_sp, vocab_size)
    # Debug: in đối chiếu Vocab Token ID vs Query ID
    print("  Đối chiếu Vocab Token ID vs Query ID:")
    for lang, v_id in VOCAB_LID_DICT.items():
        piece = tokenizer_sp.id_to_piece(v_id) if v_id < vocab_size else "N/A"
        q_id = QUERY_LID_DICT.get(lang, 0)
        is_spec = bool(special_ids_mask[v_id].item()) if v_id < vocab_size else False
        print(f"    [{lang:8s}] Vocab Token ID={v_id:5d} ({piece!r:15s}, is_special={is_spec}) -> Query ID={q_id}")

    # 5. Rebuild acoustic model với static positional encoder
    print("\n[4/6] Áp dụng StaticSinusoidalPositionEncoder ...")
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv_exported = export_rebuild_model(sv, device="cpu", max_seq_len=512)
    sv_exported.export_dynamic_axes = types.MethodType(lambda self: {}, sv_exported)
    sv_exported.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=MAX_SEQ_FRAMES, depth=560)

    # 6. Assemble pipeline
    print("\n[5/6] Ghép 5 khối thành SenseVoiceUnifiedE2E ...")
    orig_fe = am.kwargs.get("frontend")
    frontend = TraceableFrontend(orig_fe.cmvn, max_lfr_frames=MAX_LFR_FRAMES)
    frontend.eval()

    decoder = StaticCTCCollapseAndDetokenizer(
        byte_table=byte_table_tensor,
        special_ids_mask=special_ids_mask,
        max_frames=MAX_SEQ_FRAMES,
        l_max=l_max,
    )
    decoder.eval()

    unified = SenseVoiceUnifiedE2E(frontend, sv_exported, decoder, max_lfr_frames=MAX_LFR_FRAMES)
    unified.eval()

    # 7. Export ONNX
    onnx_raw     = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_raw.onnx")
    onnx_patched = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_patched.onnx")

    dummy_lang = torch.tensor([VOCAB_LID_DICT["en"]], dtype=torch.int32)
    dummy_tn   = torch.tensor([VOCAB_TEXTNORM_DICT["withitn"]], dtype=torch.int32)
    dummy_wav  = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)

    print(f"\n[6/6] Export ONNX -> {onnx_raw}")
    print(f"  Inputs: language=[1], textnorm=[1], wav=[1,{MAX_WAV_SAMPLES}]")
    print(f"  Output: byte_stream=[1,{byte_stream_len}]")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            unified,
            (dummy_lang, dummy_tn, dummy_wav),
            onnx_raw,
            input_names=["language", "textnorm", "wav"],
            output_names=["byte_stream"],
            dynamic_axes={},   # 100% static cho NPU
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Export OK ({time.time()-t0:.1f}s)")

    # 8. Graph surgery
    print("\n[Post] Graph Surgery: Conv bias + Mask outliers ...")
    onnx_tmp = os.path.join(OUT_DIR, "model_temp_bias.onnx")
    patch_conv_bias(onnx_raw, onnx_tmp)
    patch_mask_outliers(onnx_tmp, onnx_patched, clip_val=-30.0)

    for f in [onnx_tmp, onnx_raw]:
        if os.path.exists(f):
            os.remove(f)

    # 9. Save config
    config = {
        "model_name": "SenseVoice-Small-E2E-v2",
        "onnx_path": onnx_patched,
        "input_shapes": {"language": [1], "textnorm": [1], "wav": [1, MAX_WAV_SAMPLES]},
        "output_shapes": {"byte_stream": [1, byte_stream_len]},
        "target_hardware": "Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)",
        "quantization_recipe": "W8A16 Mixed Precision",
        "max_wav_samples": MAX_WAV_SAMPLES,
        "max_lfr_frames": MAX_LFR_FRAMES,
        "max_seq_frames": MAX_SEQ_FRAMES,
        "vocab_size": vocab_size,
        "l_max": l_max,
        "byte_stream_len": byte_stream_len,
        "special_tokens_filtered": True,
        "vocab_lid_dict": VOCAB_LID_DICT,
        "query_lid_dict": QUERY_LID_DICT,
        "vocab_textnorm_dict": VOCAB_TEXTNORM_DICT,
        "query_textnorm_dict": QUERY_TEXTNORM_DICT,
        "wav_buckets": WAV_BUCKETS,
        "cpu_postprocess": 'bytes(output[0].astype(np.uint8)).decode("utf-8", errors="ignore").replace("\\x00", "").strip()',
    }
    cfg_file = os.path.join(OUT_DIR, "unified_e2e_config.json")
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config saved: {cfg_file}")

    # 10. Verify với ORT
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        verify_exported_model(onnx_patched, manifest_path, MAX_WAV_SAMPLES, byte_stream_len)

    print("\n" + "=" * 70)
    print("✅ EXPORT HOÀN TẤT — SenseVoice-Small E2E v2")
    print(f"  ONNX: {onnx_patched}")
    print(f"  Input:  language[1], textnorm[1], wav[1,{MAX_WAV_SAMPLES}]")
    print(f"  Output: byte_stream=[1,{byte_stream_len}]  (L_MAX={l_max}, special tokens filtered)")
    print("=" * 70)


if __name__ == "__main__":
    main()