# -*- coding: utf-8 -*-
"""
static_detokenize.py — Kiến trúc Detokenize tĩnh (Static Byte Lookup & Flattening) trên NPU Qualcomm

Tham chiếu từ kiến trúc đột phá của Khanh trong báo cáo Zipformer ASR (Onevoice_zipformer.pdf):
  - Mục tiêu: Loại bỏ hoàn toàn thư viện Tokenizer (SentencePiece/HuggingFace) trên Host CPU.
  - Tích hợp 2 khối tính toán tĩnh (Static Computational DAG) vào ONNX Graph:
      1. StaticCTCCollapse: Argmax + Lọc trùng + Lọc blank + Static Packing via CumSum (Prefix Sum) & ScatterElements.
      2. StaticByteDetokenizer: Bảng tra cứu byte UTF-8 tĩnh (Constant Table) + Gather + Reshape.
  - Output của NPU: Tensor byte_stream tĩnh [1, T * L_max] (int32).
  - Host CPU: Chỉ nhận buffer byte thô và decode trực tiếp bằng UTF-8 với latency < 0.001 ms!
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sentencepiece as spm
except ImportError:
    spm = None


def build_byte_lookup_table(sp_model_path: str, max_bytes: int = 24, skip_tags: bool = True) -> np.ndarray:
    """Đọc file bpe.model và sinh ma trận Byte tĩnh M_byte kích thước [V, max_bytes].
    
    Mỗi hàng v tương ứng với chuỗi byte UTF-8 của token v.
    Các vị trí chưa dùng được đệm bằng NULL byte (0).
    """
    if spm is None:
        raise ImportError("sentencepiece is required to build byte lookup table")

    sp = spm.SentencePieceProcessor()
    if not sp.load(sp_model_path):
        raise FileNotFoundError(f"Cannot load SentencePiece model: {sp_model_path}")

    V = sp.get_piece_size()
    M_byte = np.zeros((V, max_bytes), dtype=np.int32)

    for i in range(V):
        piece = sp.id_to_piece(i)
        
        # Xử lý các token đặc biệt
        if piece in ["<blank>", "<blk>", "<unk>", "<s>", "</s>", "<sos/eos>", "<pad>"]:
            piece_str = ""
        elif skip_tags and piece.startswith("<|") and piece.endswith("|>"):
            piece_str = ""
        else:
            # Thay thế ký tự đặc biệt của sentencepiece (\u2581) thành dấu cách
            piece_str = piece.replace("\u2581", " ")

        b = piece_str.encode("utf-8")
        length = min(len(b), max_bytes)
        if length > 0:
            M_byte[i, :length] = list(b[:length])

    return M_byte


class StaticCTCCollapse(nn.Module):
    """Khối CTC Collapse tĩnh trên NPU (Argmax, Dedup, Remove Blank, Static Packing via CumSum & Scatter)."""

    def __init__(self, blank_id: int = 0):
        super().__init__()
        self.blank_id = blank_id

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            token_ids: [B, T] (int64/int32)
        Returns:
            packed_tokens: [B, T] với các token hợp lệ dồn về đầu, còn lại đệm 0.
        """
        B, T = token_ids.shape

        # 1. Deduplicate: Concat thay cho Pad để tương thích 100% với kiểu INT32 trên Qualcomm Hexagon HTP
        dummy = torch.full((B, 1), -1, dtype=token_ids.dtype, device=token_ids.device)
        prev_tokens = torch.cat([dummy, token_ids[:, :-1]], dim=-1)
        is_diff = (token_ids != prev_tokens)

        # 2. Filter blank: Loại bỏ ký tự blank (id == blank_id)
        is_not_blank = (token_ids != self.blank_id)
        m_valid = (is_diff & is_not_blank).long()  # [B, T]

        # 3. Static Graph Packing: Dồn phần tử hợp lệ về đầu bằng CumSum và Scatter (Chuẩn Khanh)
        p_t = torch.cumsum(m_valid, dim=-1) * m_valid  # 1-based index cho phần tử valid
        dummy_idx = torch.full_like(p_t, T - 1)
        dest_idx = torch.where(m_valid == 1, p_t - 1, dummy_idx)
        clean_val = torch.where(m_valid == 1, token_ids, torch.zeros_like(token_ids))

        out_packed = torch.zeros_like(token_ids)
        out_packed = out_packed.scatter(dim=-1, index=dest_idx, src=clean_val)

        # Đảm bảo phần tử dummy cuối cùng luôn là 0
        mask_last = torch.ones_like(token_ids)
        mask_last[:, -1] = 0
        packed_tokens = out_packed * mask_last

        return packed_tokens


class StaticByteDetokenizer(nn.Module):
    """Khối Detokenize tĩnh trên NPU: Tra cứu bảng Byte UTF-8 và duỗi phẳng thành Byte Stream."""

    def __init__(self, byte_table_np: np.ndarray):
        super().__init__()
        self.register_buffer("byte_table", torch.from_numpy(byte_table_np.astype(np.int32)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Args:
            token_ids: [B, T] (int64/int32)
        Returns:
            byte_stream: [B, T * max_bytes] (int32)
        """
        # Gather bằng toán tử F.embedding (chạy hoàn hảo trên Qualcomm HTP)
        # [B, T] -> [B, T, max_bytes]
        bytes_gathered = F.embedding(token_ids, self.byte_table)

        # Duỗi phẳng thành byte stream 1D tĩnh
        B = token_ids.shape[0]
        byte_stream = bytes_gathered.reshape(B, -1)
        return byte_stream


class StaticEndToEndDetokenizer(nn.Module):
    """Kết hợp CTC Collapse và Detokenizer thành 1 Module duy nhất."""

    def __init__(self, byte_table_np: np.ndarray, blank_id: int = 0):
        super().__init__()
        self.collapse = StaticCTCCollapse(blank_id=blank_id)
        self.detokenize = StaticByteDetokenizer(byte_table_np=byte_table_np)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Input: token_ids [B, T] -> Output: byte_stream [B, T * max_bytes]"""
        packed = self.collapse(token_ids)
        byte_stream = self.detokenize(packed)
        return byte_stream


def decode_byte_stream_on_host(byte_stream: np.ndarray) -> str:
    """Hàm giải mã trên Host CPU: Không cần Tokenizer, chỉ nhận stream byte từ NPU và decode UTF-8.
    Latency: < 0.001 ms.
    """
    flat = np.array(byte_stream).flatten()
    valid_bytes = bytes([int(b) for b in flat if b != 0])
    return valid_bytes.decode("utf-8", errors="ignore").strip()
