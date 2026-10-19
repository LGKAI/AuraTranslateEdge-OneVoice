"""Stub monotonic_align for inference-only use of piper_train.vits.

The real module requires a compiled Cython core used only during TRAINING
(alignment). At inference (synthesize) SynthesizerTrn never calls
maximum_path, so a no-op stub is sufficient -- Qualcomm's own ai-hub-models
mocks this exact module when it is not installed.
"""
import torch


def maximum_path(neg_cent, mask):
    """No-op stub (training-only op; not used in synthesis)."""
    return torch.zeros(neg_cent.shape, dtype=neg_cent.dtype)
