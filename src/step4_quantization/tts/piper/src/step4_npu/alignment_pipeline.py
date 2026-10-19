#!/usr/bin/env python3
"""Step 4 NPU -- Monotonic Time Alignment Pipeline (Problem 2).

Implements Direction 1: Pure Tensor Vectorized Monotonic Alignment for 100% NPU Execution.
Replaces CPU-side NumPy `generate_path_np()` with hardware-parallel vector operations:
1. Static coordinate grid broadcasting (Zero dynamic loops, zero if/else).
2. CumSum + Shifted Concat for cumulative duration tracking.
3. Hardware vector comparators (GreaterOrEqual, Less).
4. Pure static shapes: [1, 1, 512] -> [1, 1536, 512] and [1, 1, 1536].
5. 100% compatible with Qualcomm Hexagon HTP v73 NPU accelerator.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512
UPSAMPLED_MAX_SEQ_LEN = 1536


class MonotonicAligner(nn.Module):
    """QNN HTP-Native Monotonic Alignment Generator.
    
    Transforms token duration predictions into a 2D attention alignment matrix
    using 100% vector-parallel static tensor operations.
    """

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN, max_acoustic_len: int = UPSAMPLED_MAX_SEQ_LEN):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_acoustic_len = max_acoustic_len

        # Pre-allocated constant coordinate grids in NPU memory
        # y_grid: [1, max_acoustic_len, 1] -> [1, 1536, 1]
        self.register_buffer(
            "y_grid",
            torch.arange(max_acoustic_len, dtype=torch.float32).view(1, max_acoustic_len, 1)
        )
        # zero_pad: [1, 1, 1] for cumulative duration shift
        self.register_buffer("zero_pad", torch.zeros(1, 1, 1, dtype=torch.float32))

    def forward(self, w_ceil: torch.Tensor, x_mask: torch.Tensor, y_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate alignment matrix and y_mask natively on NPU.
        
        Args:
            w_ceil: Tensor [1, 1, 512] float32 (predicted integer frame durations).
            x_mask: Tensor [1, 1, 512] float32 (valid token mask).
            y_lengths: Tensor [1] int32/float32 (total valid acoustic frames).
            
        Returns:
            attn: Tensor [1, 1536, 512] float32 (binary alignment matrix).
            y_mask: Tensor [1, 1, 1536] float32 (valid acoustic frame mask).
        """
        # 1. Mask durations with input token mask
        w = w_ceil * x_mask  # [1, 1, 512]

        # 2. Cumulative summation along time axis Tx
        cum_duration = torch.cumsum(w, dim=-1)  # [1, 1, 512]

        # 3. Shifted cumulative boundary: [0, c_0, c_1, ..., c_510]
        cum_prev = torch.cat([self.zero_pad, cum_duration[:, :, :-1]], dim=-1)  # [1, 1, 512]

        # 4. Hardware Parallel Comparison (Broadcast [1, 1536, 1] against [1, 1, 512])
        # ge_mask: y_grid >= cum_prev -> [1, 1536, 512]
        ge_mask = (self.y_grid >= cum_prev).float()
        # lt_mask: y_grid < cum_duration -> [1, 1536, 512]
        lt_mask = (self.y_grid < cum_duration).float()

        # 5. Element-wise binary intersection
        attn_raw = ge_mask * lt_mask  # [1, 1536, 512]

        # 6. Generate y_mask natively on NPU without If/Else
        # y_lengths: [1] -> reshape to [1, 1, 1]
        y_len_reshaped = y_lengths.float().view(-1, 1, 1)
        y_mask_2d = (self.y_grid < y_len_reshaped).float()  # [1, 1536, 1]
        y_mask = y_mask_2d.transpose(1, 2)  # [1, 1, 1536]

        # 7. Apply 2D boundary masking
        attn = attn_raw * x_mask * y_mask_2d  # [1, 1536, 512]

        return attn, y_mask


def generate_path_np_reference(w_ceil: np.ndarray, x_mask: np.ndarray, y_lengths: int) -> tuple[np.ndarray, np.ndarray]:
    """Original CPU NumPy implementation for equivalence verification."""
    max_seq_len = w_ceil.shape[-1]
    y_mask = (np.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths)[None, None, :]
    attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]  # [1, 1, 1536, 512]

    cum_duration = np.cumsum(w_ceil, axis=-1)
    cum_prev = np.pad(cum_duration[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    y_pos = np.arange(UPSAMPLED_MAX_SEQ_LEN)[:, None]
    path = (y_pos >= cum_prev) & (y_pos < cum_duration)
    attn = path.astype(np.float32) * attn_mask
    attn_squeezed = attn[:, 0, :, :].astype(np.float32)
    return attn_squeezed, y_mask.astype(np.float32)


def verify_equivalence():
    """Verify that MonotonicAligner produces bit-exact match with CPU generate_path_np."""
    logger.info("=== VERIFYING NPU MONOTONIC ALIGNER EQUIVALENCE ===")
    
    aligner = MonotonicAligner()
    aligner.eval()

    # Generate realistic duration samples
    np.random.seed(42)
    dummy_text_len = 40
    durations = np.random.randint(1, 10, size=(1, 1, MAX_SEQ_LEN)).astype(np.float32)
    durations[:, :, dummy_text_len:] = 0.0  # Zero out padded tokens
    
    x_mask = np.zeros((1, 1, MAX_SEQ_LEN), dtype=np.float32)
    x_mask[:, :, :dummy_text_len] = 1.0
    
    y_lengths = int(durations.sum())
    logger.info("Test Setup: text_len=%d, total_acoustic_frames=%d / %d",
                dummy_text_len, y_lengths, UPSAMPLED_MAX_SEQ_LEN)

    # 1. CPU Reference Output
    ref_attn, ref_y_mask = generate_path_np_reference(durations, x_mask, y_lengths)

    # 2. NPU Tensor Output
    with torch.no_grad():
        w_t = torch.from_numpy(durations)
        xm_t = torch.from_numpy(x_mask)
        yl_t = torch.tensor([y_lengths], dtype=torch.int32)
        npu_attn_t, npu_y_mask_t = aligner(w_t, xm_t, yl_t)
        npu_attn = npu_attn_t.numpy()
        npu_y_mask = npu_y_mask_t.numpy()

    # 3. Check Differences
    attn_diff = np.max(np.abs(ref_attn - npu_attn))
    y_mask_diff = np.max(np.abs(ref_y_mask - npu_y_mask))

    logger.info("Max Attention Matrix Absolute Error: %.8f", attn_diff)
    logger.info("Max Y-Mask Absolute Error:           %.8f", y_mask_diff)

    assert attn_diff < 1e-6, f"Attention mismatch! Diff: {attn_diff}"
    assert y_mask_diff < 1e-6, f"Y-Mask mismatch! Diff: {y_mask_diff}"
    logger.info("✅ EQUIVALENCE VERIFIED: 100% EXACT BIT-LEVEL MATCH WITH CPU REFERENCE!")


def export_monotonic_aligner_onnx(output_path: str = "outputs/piper_vi_npu/components/monotonic_aligner.onnx"):
    """Export and verify the MonotonicAligner as a static-shape ONNX model for NPU."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    aligner = MonotonicAligner()
    aligner.eval()

    dummy_w = torch.ones(1, 1, MAX_SEQ_LEN, dtype=torch.float32) * 3.0
    dummy_xm = torch.ones(1, 1, MAX_SEQ_LEN, dtype=torch.float32)
    dummy_yl = torch.tensor([1536], dtype=torch.int32)

    logger.info("Exporting MonotonicAligner to ONNX: %s", out_file)
    torch.onnx.export(
        aligner,
        (dummy_w, dummy_xm, dummy_yl),
        str(out_file),
        input_names=["w_ceil", "x_mask", "y_lengths"],
        output_names=["attn_squeezed", "y_mask"],
        opset_version=15,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Successfully exported ONNX: %s", out_file)


if __name__ == "__main__":
    verify_equivalence()
    export_monotonic_aligner_onnx()
