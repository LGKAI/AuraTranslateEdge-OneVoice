# -*- coding: utf-8 -*-
"""Chuyen ONNX dang external-data (file .onnx + .onnx.data) thanh 1 file .onnx duy nhat (inline).

Tai sao can: torch.onnx.export (dynamo) tu dong tach weights ra file .data ke ca khi model nho.
qai_hub.upload_model() KHONG xu ly duoc dang nay, bao loi:
    "Data of TensorProto (tensor name: ...) should be stored in ...onnx.data, but it is not regular file."
Model < 2GB (gioi han protobuf) thi luu inline la on. SenseVoice enc+cls ~946MB.

Dung:
    python inline_onnx.py model_raw.onnx            # -> model_raw_inline.onnx
    python inline_onnx.py model_raw.onnx out.onnx
"""
import os
import sys

import onnx


def to_inline(src: str, dst: str = None) -> str:
    dst = dst or src.replace(".onnx", "_inline.onnx")
    model = onnx.load(src, load_external_data=True)
    onnx.save_model(model, dst, save_as_external_data=False)
    print(f"[inline_onnx] {src} -> {dst}  ({os.path.getsize(dst) / 1e6:.2f} MB)")
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    to_inline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
