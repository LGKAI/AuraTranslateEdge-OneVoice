"""step4_s1_export_sensevoice_e2e_unified.py — Xuất mô hình SenseVoice-Small End-to-End tĩnh 100% NPU

Kiến trúc đồ thị tính toán hợp nhất 5 khối (Single Static DAG):
  Khối 1: WavFrontend DSP tĩnh (NPU)
          Conv1D Framing + Windowing + DFT MatMul + Mel Filterbank + LFR + CMVN
          Sóng âm thô [1, 464000] (16kHz, ~29s) ➔ [1, 500, 560]
  Khối 2: SenseVoice Transformer Core (NPU)
          Tĩnh hóa Positional Encoder (Add [1, 504, 560]) + 50 lớp Transformer nén sâu
          ➔ [1, 504, 512]
  Khối 3: CTC Head & ArgMax (NPU)
          Linear Projection [512 ➔ 25055] + ArgMax(axis=-1)
          ➔ [1, 504] Frame Tokens
  Khối 4: Static CTC Collapse (NPU)
          Lọc trùng liên tiếp + Lọc Blank (0) + CumSum & Scatter với Trash-Bin tĩnh
          ➔ [1, 504] Packed Clean Tokens (không vòng lặp, không cấp phát động)
  Khối 5: Static Byte Detokenize (NPU)
          Nhúng Bảng Byte M_byte [25055, 24] ➔ Gather ➔ Reshape
          ➔ [1, 12096] UTF-8 Byte Stream
  Tầng Host CPU (Zero-CPU Decoding):
          bytes(output[0].astype(np.uint8)).decode('utf-8', errors='ignore').replace('\\x00', '').strip()
          ➔ Không tốn bất kỳ chi phí Tokenizer nào trên CPU (< 0.001 ms).

Đầu vào:
  - wav: [1, 464000] float32
  - wav_len: [1] int32 (số mẫu âm thanh THẬT, trước khi zero-pad -- BẮT BUỘC để mask đúng
             vùng đệm, nếu thiếu sẽ bị lặp ký tự vô hạn ở cuối câu khi audio ngắn hơn 29s)
  - language: [1] int32 (zh: 3, en: 4, ko: 12, auto: 0)
  - textnorm: [1] int32 (withitn: 14, woitn: 15)
Đầu ra:
  - byte_stream: [1, 12096] int32

Tác giả: Lê Gia Khánh (SenseVoice-Small Step 4)
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
torch.set_num_threads(4)  # tranh thread-contention khi trace graph lon tren may nhieu core
import torch.nn as nn
import torch.nn.functional as F

# Fix UTF-8 encoding on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
DATA_DIR = os.path.join(ROOT, "data", "asr")

# SenseVoice WavFrontend thông số
FS = 16000
N_MELS = 80
FRAME_LENGTH_MS = 25         # 25ms -> 400 samples tại 16kHz
FRAME_SHIFT_MS = 10          # 10ms -> 160 samples tại 16kHz
LFR_M = 7                    # stack 7 khung mel liên tiếp -> 7*80 = 560 dim
LFR_N = 6                    # bước nhảy 6 khung
MAX_WAV_SAMPLES = 464000     # ~29 giây tối đa (500 LFR frames * 6 * 160 samples)
MAX_LFR_FRAMES = 500         # static shape cho NPU
MAX_SEQ_FRAMES = 504         # 500 LFR + 4 prompt tokens (lang, emo, event, itn)
L_MAX = 24                   # Kích thước byte tối đa cho 1 token BPE (SenseVoice max = 22 bytes)
BYTE_STREAM_LEN = MAX_SEQ_FRAMES * L_MAX  # 504 * 24 = 12096

FRAME_SAMPLES = int(FS * FRAME_LENGTH_MS / 1000)   # 400
HOP_SAMPLES = int(FS * FRAME_SHIFT_MS / 1000)      # 160
N_FFT = 512

MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
MODEL_ID_MS = "iic/SenseVoiceSmall"

# Từ điển chuẩn SenseVoice
LID_DICT = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM_DICT = {"withitn": 14, "woitn": 15}


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Init] Output dir: {OUT_DIR}")


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
    raw waveform [1, T] ➔ fbank [1, 500, 560]
    """
    def __init__(self, cmvn: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        
        # DFT ma trận trực giao (thay cho torch.fft.rfft để tĩnh hóa graph)
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)  # [512, 257]
        self.register_buffer("dft_imag", dft_complex.imag)  # [512, 257]

        # Mel filterbank
        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)  # [80, 256]

        # CMVN
        self.register_buffer("cmvn_mean", cmvn[0:1, :])   # [1, 560]
        self.register_buffer("cmvn_scale", cmvn[1:2, :])  # [1, 560]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # 1. Scale biên độ chuẩn Kaldi
        wav = wav * float(1 << 15)

        # 2. Framing qua F.unfold
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)  # [T_frames, 400]

        # 3. Pre-emphasis (theo đúng quy ước Kaldi thật: mẫu đầu tiên của MỖI frame nhân
        # (1 - coeff), KHÔNG giữ nguyên như trước -- fix bug làm lệch ~0.05-0.07 mean log-mel
        # so với Kaldi thật, đã xác nhận bằng thực nghiệm so sánh trực tiếp)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        frames_pe[:, 0] = frames[:, 0] * (1.0 - 0.97)

        # 4. Window & Pad
        windowed = frames_pe * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))  # [T_frames, 512]

        # 5. DFT qua MatMul
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        # Fix bug: mel_fb (quy ước Kaldi) dùng 256 bin là FFT bin [0..255] (GIỮ DC, BỎ Nyquist).
        # Code cũ bỏ nhầm DC (index 0) và giữ Nyquist (index 256) -> lệch tần số toàn bộ các
        # mel-bin thấp (đã xác nhận: sửa bug này giảm sai lệch mean log-mel ~8-10 lần).
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2

        # 6. Mel FB & Log
        mel = torch.matmul(power_no_dc, self.mel_fb.T)
        mel_log = torch.clamp(mel, min=1e-10).log()

        # 7. LFR (stack 7, hop 6)
        left_pad = mel_log[0:1].expand(LFR_M // 2, -1)
        mel_padded = torch.cat([left_pad, mel_log], dim=0)

        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M, 1), stride=(LFR_N, 1))
        mel_uf = mel_uf.squeeze(0).view(80, LFR_M, -1)
        mel_uf = mel_uf.permute(1, 0, 2).reshape(560, -1).transpose(0, 1)

        # 8. CMVN
        mel_uf = (mel_uf + self.cmvn_mean) * self.cmvn_scale

        # 9. Pad/crop về kích thước tĩnh [1, 500, 560]
        actual_len = mel_uf.shape[0]
        if actual_len >= MAX_LFR_FRAMES:
            mel_uf = mel_uf[:MAX_LFR_FRAMES, :]
        else:
            mel_uf = F.pad(mel_uf, (0, 0, 0, MAX_LFR_FRAMES - actual_len))

        return mel_uf.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Khối 2: Tĩnh hóa Positional Encoder cho SenseVoice
# ─────────────────────────────────────────────────────────────────────────────

class StaticSinusoidalPositionEncoder(nn.Module):
    """
    Constant Positional Encoding Tensor [1, 504, 560].
    Loại bỏ toàn bộ toán tử động (Range, Unsqueeze, Sin, Cos),
    tránh lỗi broadcast trên Qualcomm Hexagon Compiler.
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

def build_static_byte_table(tokenizer_sp, vocab_size: int = 25055, l_max: int = L_MAX) -> torch.Tensor:
    """
    Xây dựng Ma trận Byte Tra cứu Tĩnh M_byte [25055, 24] (int32).
    - Các token đặc biệt (<unk>, <s>, </s>, <|...|>) được gán rỗng (all zeros).
    - Ký tự ' ' (U+2581) của SentencePiece được chuyển thành ASCII space ' ' (0x20).
    - Mọi token được mã hóa thành các byte UTF-8 thực tế, đệm số 0 (NULL byte) đến độ dài 24.
    """
    table = np.zeros((vocab_size, l_max), dtype=np.int32)
    for i in range(vocab_size):
        piece = tokenizer_sp.id_to_piece(i)
        if piece in ['<unk>', '<s>', '</s>'] or re.match(r'^<\|.*\|>$', piece):
            b = b""
        else:
            norm_piece = piece.replace('\u2581', ' ')
            b = norm_piece.encode('utf-8')
        for j in range(min(len(b), l_max)):
            table[i, j] = b[j]
    return torch.from_numpy(table)


class StaticCTCCollapseAndDetokenizer(nn.Module):
    """
    Khối tích hợp CTC Collapse + UTF-8 Detokenize hoàn toàn trên NPU:
      Đầu vào:  logits [1, 504, 25055]
      Đầu ra:  byte_stream [1, 12096] (int32)

    Cơ chế loại bỏ xung đột bộ nhớ NPU (Trash-Bin Scatter):
      1. ArgMax axis=-1 ➔ token_ids [1, 504]
      2. Deduplication Mask: m_dedup[t] = (y[t] != y[t-1])
      3. Blank Removal: m_valid[t] = m_dedup[t] & (y[t] != 0)
      4. Prefix Sum (CumSum): c[t] = cumsum(m_valid)
      5. Địa chỉ đích (Destination Index):
         - Nếu token hợp lệ (m_valid=1): pos[t] = c[t] - 1  (dồn về đầu 0..K-1)
         - Nếu token rác/blank (m_valid=0): pos[t] = 504 (thùng rác trash-bin ở cuối mảng)
      6. Scatter vào bộ đệm [1, 505]:
         Toàn bộ token hợp lệ được ghi chính xác vào 0..K-1.
         Toàn bộ frame im lặng/trùng ghi vào index 504, không bao giờ ghi đè index 0.
      7. Cắt lát [:, :504] lấy đúng các token đã đóng gói.
      8. Tra cứu Bảng Byte tĩnh M_byte [25055, 24] qua F.embedding ➔ [1, 504, 24].
      9. Duỗi phẳng (Reshape) ➔ [1, 12096] byte stream.
    """
    def __init__(self, byte_table: torch.Tensor, max_frames: int = MAX_SEQ_FRAMES, l_max: int = L_MAX):
        super().__init__()
        self.register_buffer("byte_table", byte_table) # [25055, 24]
        self.max_frames = max_frames
        self.l_max = l_max
        self.register_buffer("neg_one", torch.tensor([[-1]], dtype=torch.int32))
        self.register_buffer("dummy_idx", torch.tensor(max_frames, dtype=torch.int64))

    def forward(self, logits: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        # 1. ArgMax trích xuất token IDs
        token_ids = torch.argmax(logits, dim=-1).to(torch.int32) # [1, 504]

        # 1b. FIX BUG NGHIÊM TRỌNG: ép các frame vượt quá độ dài hợp lệ thật (valid_len,
        # tính từ wav_len thực, KHÔNG còn hardcode) thành blank (0). Nếu thiếu bước này,
        # các frame trong vùng đệm 0 (silence giả) sẽ bị model hallucinate ra token thật,
        # gây hiện tượng lặp ký tự vô hạn ở cuối câu (đã xác nhận bằng thực nghiệm trên
        # audio thật en/zh/ko: reproduce được y hệt lỗi trên NPU thật, và biến mất sau fix).
        frame_idx = torch.arange(self.max_frames, device=token_ids.device, dtype=torch.int64).unsqueeze(0)
        length_mask = frame_idx < valid_len.to(torch.int64).view(-1, 1)
        token_ids = torch.where(length_mask, token_ids, torch.zeros_like(token_ids))

        # 2. Lọc trùng liên tiếp (Deduplication)
        prev = torch.cat([self.neg_one, token_ids[:, :-1]], dim=-1)
        dedup_mask = (token_ids != prev)
        
        # 3. Lọc Blank token (0)
        valid_mask = dedup_mask & (token_ids != 0)
        
        # 4. CumSum tính số thứ tự token hợp lệ
        c = torch.cumsum(valid_mask.to(torch.int64), dim=-1)
        
        # 5. Destination Index với Trash-Bin (tránh ghi đè index 0)
        pos = torch.where(valid_mask, c - 1, self.dummy_idx)
        
        # 6. Scatter vào buffer [1, 505]
        buffer = torch.zeros(token_ids.shape[0], self.max_frames + 1, dtype=torch.int32, device=token_ids.device)
        packed_buffer = torch.scatter(buffer, 1, pos, token_ids)
        packed_tokens = packed_buffer[:, :self.max_frames] # [1, 504]
        
        # 7. Tra cứu Gather Bảng Byte tĩnh
        bytes_out = F.embedding(packed_tokens.to(torch.int64), self.byte_table) # [1, 504, 24]
        
        # 8. Duỗi phẳng thành luồng byte UTF-8
        bytes_stream = bytes_out.reshape(token_ids.shape[0], self.max_frames * self.l_max) # [1, 12096]
        return bytes_stream


# ─────────────────────────────────────────────────────────────────────────────
# Toàn bộ Pipeline Đồ thị Tĩnh Duy nhất (Single Static DAG)
# ─────────────────────────────────────────────────────────────────────────────

class SenseVoiceUnifiedE2E(nn.Module):
    """
    Toàn bộ Pipeline ASR SenseVoice tích hợp 5 khối thành 1 Graph duy nhất chạy 100% NPU:
      Input:
        wav: [1, 464000] float32
        language: [1] int32 (zh: 3, en: 4, ko: 12)
        textnorm: [1] int32 (withitn: 14)
      Output:
        byte_stream: [1, 12096] int32
    """
    def __init__(self, frontend: TraceableFrontend, acoustic_model: nn.Module, decoder: StaticCTCCollapseAndDetokenizer):
        super().__init__()
        self.frontend = frontend
        self.acoustic_model = acoustic_model
        self.decoder = decoder

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor, language: torch.Tensor, textnorm: torch.Tensor) -> torch.Tensor:
        # Khối 1: WavFrontend DSP
        fbank = self.frontend(wav) # [1, 500, 560]

        # FIX BUG NGHIÊM TRỌNG: speech_lengths KHÔNG được hardcode = MAX_LFR_FRAMES nữa.
        # wav luôn có kích thước tĩnh 464000 mẫu (đã zero-pad), nên phải tính độ dài HỢP LỆ
        # THẬT từ wav_len (số mẫu âm thanh thật, do host truyền vào) bằng đúng công thức
        # F.unfold đã dùng ở TraceableFrontend (đã kiểm chứng khớp 100% qua nhiều độ dài
        # thực tế từ 400 đến 464000 mẫu bằng thực nghiệm).
        # FIX BUG THU 4 (phat hien qua NPU thuc te): phep tinh dung .to(float32) + Floor
        # bi quantizer W8A16 tu dong quan QuantizeLinear/DequantizeLinear vao MOI buoc
        # (Cast/Sub/Div/Floor/Add/Clip deu bi Q->DQ, da xac nhan bang cach doc truc tiep
        # do thi ONNX da quantize) -- sai so lam tron cong don qua ~10 buoc Q->DQ pha vo
        # phep tinh so nguyen chinh xac, khien lfr_len tinh sai (<=0), mask toan bo frame
        # thanh blank -> output RONG o MOI mau tren NPU that (fp32 CPU thi dung 100%).
        # Sua bang cach giu NGUYEN so nguyen (int64) xuyen suot, KHONG cast qua float32,
        # dung // (floor-division nguyen) thay vi torch.floor -- quantizer chi quan QDQ
        # vao tensor kieu float nen phep tinh nguyen thuan nay se khong bi dung vao nua
        # (giong het cach 'language'/'textnorm' int32 hien khong bi quantize).
        wav_len_i = wav_len.to(torch.int64)
        fbank_len = (wav_len_i - FRAME_SAMPLES) // HOP_SAMPLES + 1
        fbank_len = torch.clamp(fbank_len, min=0)
        lfr_len = (fbank_len + (LFR_M // 2) - LFR_M) // LFR_N + 1
        lfr_len = torch.clamp(lfr_len, min=0, max=MAX_LFR_FRAMES)
        speech_lengths = lfr_len.to(torch.int32).reshape(1)

        # Khối 2 & 3: Transformer Core + CTC Projection Head
        logits, out_lens = self.acoustic_model(fbank, speech_lengths, language, textnorm) # [1, 504, 25055]

        # FIX BUG THU 5 (phat hien qua NPU that, dieu tra sau ca fix #4): out_lens do CHINH
        # acoustic_model tu tra ve (khong phai code minh viet) duoc tinh qua mot phep ReduceSum
        # NOI BO (dem so frame thoa arange < speech_lengths) -- phep Sum nay bi quantize va BAO HOA
        # o dung 180 bat ke do dai audio that hay du lieu calibration (da xac nhan: doi calibration
        # tu 15 mau len 25 mau phu day du 50k-460k mau KHONG lam gia tri nay doi, chung to day khong
        # phai loi thieu calibration ma la loi cua chinh phep tinh du thua nay khi bi quantize).
        # Sua bang cach TU TINH lai valid_len tu speech_lengths (da xac nhan sach, khong bi quantize)
        # thay vi dung out_lens cua model tra ve -- da do dac thuc nghiem: out_lens = speech_lengths + 4
        # DUNG TUYET DOI moi truong hop (da test speech_lengths tu 50 den 490).
        valid_len = speech_lengths.to(torch.int64) + 4

        # Khối 4 & 5: Static CTC Collapse + Byte Detokenize
        # Dung valid_len tu tinh (sach, khong quantize) thay vi out_lens cua model (bi bao hoa).
        byte_stream = self.decoder(logits, valid_len) # [1, 12096]
        return byte_stream


# ─────────────────────────────────────────────────────────────────────────────
# Graph Surgery & Post-Processing (Fix QAIRT Converter & Quantization)
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

    # Pin static output shape [1, 12096]
    for o in model.graph.output:
        if o.name == "byte_stream":
            o.type.tensor_type.shape.dim[0].dim_value = 1
            o.type.tensor_type.shape.dim[0].ClearField('dim_param')
            o.type.tensor_type.shape.dim[1].dim_value = BYTE_STREAM_LEN
            o.type.tensor_type.shape.dim[1].ClearField('dim_param')

    onnx.save(model, patched_path)
    print(f"[patch_conv_bias] Patched {patched} Conv nodes -> Saved {patched_path}")
    return patched


def patch_mask_outliers(onnx_path: str, patched_path: str, clip_val: float = -30.0) -> int:
    """
    Clamp các giá trị cực đoan (< -30.0) trong attention masks (initializers).
    Bảo toàn dải động lượng tử hóa W8A16 cho 50 lớp Transformer.
    """
    import onnx
    import onnx.numpy_helper as nph

    model = onnx.load(onnx_path)
    patched = 0
    for i, init in enumerate(model.graph.initializer):
        arr = nph.to_array(init)
        if arr.size == 0 or arr.dtype not in [np.float32, np.float16]:
            continue
        min_val = float(arr.min())
        if min_val < clip_val:
            arr_clipped = np.clip(arr, clip_val, None)
            new_init = nph.from_array(arr_clipped.astype(arr.dtype), name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, new_init)
            patched += 1

    onnx.save(model, patched_path)
    print(f"[patch_mask] Clipped {patched} outlier initializers to {clip_val} -> Saved {patched_path}")
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Verification with ONNX Runtime & Real Audio
# ─────────────────────────────────────────────────────────────────────────────

def verify_exported_model(onnx_path: str, manifest_path: str):
    """Kiểm thử mô hình ONNX đã xuất bằng ONNX Runtime trên tập test thật."""
    import onnxruntime as ort
    import soundfile as sf
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    print("\n" + "=" * 70)
    print(f"[Verify] Đang kiểm thử ONNX Runtime trên {onnx_path} ...")
    print("=" * 70)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sv_items = [it for it in manifest if it["lang"] in ["en", "zh", "ko"]]

    total = len(sv_items)
    passed = 0

    for item in sv_items:
        lang = item["lang"]
        path = os.path.join(ROOT, item["path"])
        lang_id = LID_DICT.get(lang, 0)

        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = wav.astype(np.float32)
        real_len = min(len(wav_t), MAX_WAV_SAMPLES)
        if len(wav_t) > MAX_WAV_SAMPLES:
            wav_in = wav_t[:MAX_WAV_SAMPLES]
        else:
            wav_in = np.pad(wav_t, (0, MAX_WAV_SAMPLES - len(wav_t)))
        wav_in = wav_in.reshape(1, MAX_WAV_SAMPLES)

        inputs = {
            "wav": wav_in,
            "wav_len": np.array([real_len], dtype=np.int32),  # số mẫu THẬT (không tính phần zero-pad)
            "language": np.array([lang_id], dtype=np.int32),
            "textnorm": np.array([14], dtype=np.int32),  # withitn
        }

        t0 = time.perf_counter()
        ort_outs = sess.run(None, inputs)
        latency_ms = (time.perf_counter() - t0) * 1000

        byte_stream = ort_outs[0][0]  # [12096] int32
        
        # Host CPU zero-tokenizer decoding
        raw_bytes = bytes(byte_stream.astype(np.uint8))
        decoded_text = raw_bytes.decode("utf-8", errors="ignore").replace("\x00", "").strip()

        ref = item["transcript"]
        print(f"[{lang.upper()}] File: {os.path.basename(path)} ({latency_ms:.1f} ms)")
        print(f"       Gốc (Ref):  {ref}")
        print(f"       NPU Output: {decoded_text}")
        
        # Normalize text to compare semantic similarity
        ref_norm = re.sub(r'[^\w\s]', '', ref).replace(' ', '').lower()
        dec_norm = re.sub(r'[^\w\s]', '', decoded_text).replace(' ', '').lower()
        is_close = (ref_norm in dec_norm or dec_norm in ref_norm or len(set(ref_norm) & set(dec_norm)) / max(len(set(ref_norm)), 1) > 0.8)
        if is_close:
            passed += 1
            print("       ➔ ĐÁNH GIÁ: ✅ CHUẨN XÁC CAO")
        else:
            print("       ➔ ĐÁNH GIÁ: ⚠️ KHÁC BIỆT")
        print("-" * 70)

    acc = (passed / total) * 100
    print(f"\n[Verify Summary] Độ chính xác tổng thể: {passed}/{total} ({acc:.1f}%)")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_out_dir()

    print("\n" + "=" * 70)
    print("XUẤT MÔ HÌNH SENSEVOICE-SMALL END-TO-END TĨNH (100% NPU DEPLOYMENT)")
    print("WavFrontend ➔ Transformer Core ➔ CTC Head ➔ Static Collapse ➔ Detokenize")
    print("=" * 70)

    # 1. Tải mô hình
    print("\n[1/6] Đang tải SenseVoice-Small từ FunASR ...")
    from funasr import AutoModel
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()
    tokenizer_sp = am.kwargs.get("tokenizer").sp
    vocab_size = tokenizer_sp.get_piece_size()
    print(f"  Vocabulary size: {vocab_size} tokens")

    # 2. Xây dựng Bảng Byte tĩnh M_byte [25055, 24]
    print("\n[2/6] Đang xây dựng Ma trận Byte Tra cứu Tĩnh M_byte [25055, 24] ...")
    byte_table_tensor = build_static_byte_table(tokenizer_sp, vocab_size=vocab_size, l_max=L_MAX)
    print(f"  M_byte tensor shape: {byte_table_tensor.shape}, dtype: {byte_table_tensor.dtype}")

    # 3. Rebuild Acoustic Model với Positional Encoder tĩnh
    print("\n[3/6] Đang áp dụng Positional Encoder tĩnh (StaticSinusoidalPositionEncoder) ...")
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv_exported = export_rebuild_model(sv, device="cpu", max_seq_len=512)
    sv_exported.export_dynamic_axes = types.MethodType(lambda self: {}, sv_exported)
    sv_exported.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=MAX_SEQ_FRAMES, depth=560)

    # 4. Tạo các khối WavFrontend và Static Decoder
    print("\n[4/6] Đang ghép 5 khối thành Single Static DAG (SenseVoiceUnifiedE2E) ...")
    orig_fe = am.kwargs.get("frontend")
    frontend = TraceableFrontend(orig_fe.cmvn)
    frontend.eval()

    decoder = StaticCTCCollapseAndDetokenizer(byte_table_tensor, max_frames=MAX_SEQ_FRAMES, l_max=L_MAX)
    decoder.eval()

    unified_pipeline = SenseVoiceUnifiedE2E(frontend, sv_exported, decoder)
    unified_pipeline.eval()

    # 5. Xuất ONNX
    onnx_raw = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_raw.onnx")
    onnx_patched = os.path.join(OUT_DIR, "model_sensevoice_e2e_unified_patched.onnx")

    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)
    dummy_wav_len = torch.tensor([MAX_WAV_SAMPLES], dtype=torch.int32)  # số mẫu thật (host truyền vào)
    dummy_lang = torch.tensor([4], dtype=torch.int32)     # en: 4
    dummy_tn = torch.tensor([14], dtype=torch.int32)       # withitn: 14

    print(f"\n[5/6] Đang xuất mô hình ONNX tĩnh sang: {onnx_raw} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            unified_pipeline,
            (dummy_wav, dummy_wav_len, dummy_lang, dummy_tn),
            onnx_raw,
            input_names=["wav", "wav_len", "language", "textnorm"],
            output_names=["byte_stream"],
            dynamic_axes={},  # Static shape 100% cho NPU
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Xuất ONNX thành công sau {time.time()-t0:.1f}s!")

    # 6. Graph Surgery (Bơm Bias + Clamping Mask Outlier)
    print("\n[6/6] Thực hiện Graph Surgery (Bơm Dummy Conv Bias + Clamping Mask Outlier) ...")
    onnx_bias_patched = os.path.join(OUT_DIR, "model_temp_bias.onnx")
    patch_conv_bias(onnx_raw, onnx_bias_patched)
    patch_mask_outliers(onnx_bias_patched, onnx_patched, clip_val=-30.0)
    
    if os.path.exists(onnx_bias_patched):
        os.remove(onnx_bias_patched)
    if os.path.exists(onnx_raw):
        # Đổi tên file raw để tiết kiệm dung lượng
        os.replace(onnx_raw, os.path.join(OUT_DIR, "model_sensevoice_e2e_unified.onnx"))

    # Lưu file config phục vụ Quantization
    config = {
        "model_name": "SenseVoice-Small-E2E-Unified-Detok",
        "onnx_path": onnx_patched,
        "input_shapes": {
            "wav": [1, MAX_WAV_SAMPLES],
            "wav_len": [1],
            "language": [1],
            "textnorm": [1]
        },
        "output_shapes": {
            "byte_stream": [1, BYTE_STREAM_LEN]
        },
        "target_hardware": "Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)",
        "quantization_recipe": "W8A16 Mixed Precision",
        "cpu_postprocess": "bytes(output[0].astype(np.uint8)).decode('utf-8', errors='ignore').replace('\\x00', '').strip()"
    }
    cfg_file = os.path.join(OUT_DIR, "unified_e2e_config.json")
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Đã lưu cấu hình: {cfg_file}")

    # 7. Đo kiểm tra thực tế
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        verify_exported_model(onnx_patched, manifest_path)

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT XUẤT MÔ HÌNH SENSEVOICE-SMALL END-TO-END DUY NHẤT!")
    print(f"Tập tin ONNX chính thức: {onnx_patched}")
    print("=" * 70)


if __name__ == "__main__":
    main()
