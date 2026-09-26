"""Step 4 -- quantize Supertonic's 4 ONNX submodels to int8 (dynamic quantization).
Supertonic ships as 4 independent ONNX graphs (text_encoder, vector_estimator,
vocoder, duration_predictor) already cached locally from Step 3 testing --
no re-export needed, just quantize each in place to a new output dir.
"""
import os

from onnxruntime.quantization import quantize_dynamic, QuantType

SRC_DIR = os.path.expanduser("~/.cache/supertonic3/onnx")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "outputs", "supertonic-int8")

MODELS = ["text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx", "duration_predictor.onnx"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total_before, total_after = 0, 0
    for name in MODELS:
        src = os.path.join(SRC_DIR, name)
        dst = os.path.join(OUT_DIR, name)
        before = os.path.getsize(src)
        print(f"[quantize_supertonic] {name}: {before / 1e6:.1f}MB -> quantizing...")
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        after = os.path.getsize(dst)
        total_before += before
        total_after += after
        print(f"[quantize_supertonic] {name}: {before / 1e6:.1f}MB -> {after / 1e6:.1f}MB "
              f"({100 * (1 - after / before):.0f}% smaller)")
    print(f"\n[quantize_supertonic] TOTAL: {total_before / 1e6:.1f}MB -> {total_after / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
