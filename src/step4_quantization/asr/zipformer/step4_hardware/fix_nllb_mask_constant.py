"""Step 4 -- fix NLLB encoder's QNN quantization blocker: the standard
HuggingFace "extended attention mask" construction fills masked-out
(padding) positions with `torch.finfo(float32).min` = -3.4028235e+38 as
an effectively-minus-infinity additive bias before softmax. That constant
is real-valued but astronomically outside any sane int8 quantization
range, so AI Hub's quantizer rejects the whole float32 branch it lives on
("Tensor '/Cast_1_output_0_pre_quant' has a floating-point type which is
not supported by the targeted device").

Fix (a standard, well-precedented technique for deploying transformer
attention masks to fixed-point/NPU targets): replace -3.4e38 with a much
smaller-magnitude but still-effectively-blocking negative constant. Once
a masked position's logit is ~30000 below the unmasked ones, softmax
already drives its probability to exp(-30000) =~ 0 in float32 -- there is
no practical behavioral difference from using -3.4e38, but -30000 sits in
a numeric range a real quantizer can actually represent.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_MODEL = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model.onnx")
OUT_MODEL = os.path.join(ROOT, "outputs", "nllb-onnx", "encoder_model_fixedmask.onnx")

TARGET_CONSTANT = "/Constant_13"
ORIGINAL_VALUE = np.float32(-3.4028235e38)
NEW_VALUE = np.float32(-30000.0)


def apply_fix(model):
    by_name = {n.name: n for n in model.graph.node}
    node = by_name[TARGET_CONSTANT]
    for attr in node.attribute:
        if attr.name == "value":
            arr = numpy_helper.to_array(attr.t)
            assert np.isclose(float(arr), float(ORIGINAL_VALUE)), f"unexpected value {arr}"
            new_tensor = numpy_helper.from_array(
                np.array(NEW_VALUE, dtype=np.float32), name=attr.t.name)
            attr.t.CopyFrom(new_tensor)
    print(f"[fix_nllb_mask] replaced {TARGET_CONSTANT}: {ORIGINAL_VALUE} -> {NEW_VALUE}")


def verify(orig_path, new_path, seq_len=64, tol=1e-3):
    rng = np.random.default_rng(0)
    input_ids = rng.integers(3, 1000, size=(1, seq_len)).astype(np.int64)
    # real padding pattern: first 30 valid, rest padding (matches real usage)
    attention_mask = np.zeros((1, seq_len), dtype=np.int64)
    attention_mask[:, :30] = 1

    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    new = ort.InferenceSession(new_path, so, providers=["CPUExecutionProvider"])
    new_out = new.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    diff = float(np.abs(np.asarray(ref_out, dtype=np.float64)
                         - np.asarray(new_out, dtype=np.float64)).max())
    rel = diff / (float(np.abs(ref_out).max()) + 1e-8)
    print(f"  max_abs_diff={diff:.3e}  relative={rel:.3e}  "
          f"(output range ~[{ref_out.min():.2f}, {ref_out.max():.2f}])")
    return diff <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    print(f"[fix_nllb_mask] loading {SRC_MODEL} (with external data) ...")
    model = onnx.load(SRC_MODEL, load_external_data=True)
    apply_fix(model)

    onnx.checker.check_model(model, full_check=False)
    print("[fix_nllb_mask] onnx.checker passed")

    # the original encoder_model.onnx was already a single self-contained
    # file (~1.66GB, under the 2GB protobuf limit, no external data) --
    # forcing save_as_external_data=True here (first attempt) produced a
    # model AI Hub's uploader couldn't resolve ("should be stored in ...
    # but it is not regular file"). Plain single-file save instead.
    onnx.save(model, OUT_MODEL)
    print(f"[fix_nllb_mask] wrote {OUT_MODEL} ({os.path.getsize(OUT_MODEL)/1e6:.1f} MB)")

    print("[fix_nllb_mask] verifying vs original (real padding pattern) ...")
    if verify(SRC_MODEL, OUT_MODEL, args.seq_len):
        print("[fix_nllb_mask] OK -- outputs match within tolerance, safe to deploy")
    else:
        print("[fix_nllb_mask] MISMATCH -- do not deploy")


if __name__ == "__main__":
    main()
