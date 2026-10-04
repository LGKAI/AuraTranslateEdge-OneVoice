"""Step 4 -- local FULL int8 quantization simulation (weights + activations)
of the Zipformer encoder, to test whether int8-accumulation-through-depth
ALONE reproduces the hardware's ~0.21 cosine similarity.

If local-sim cos_sim ~= 0.21 (matches hardware): the hardware is FAITHFUL to
its int8 spec -- the int8 spec itself is too lossy for this deep network.
Fix path: mixed precision (keep sensitive layers fp16). AIMET QuantAnalyzer
would then pinpoint WHICH layers.

If local-sim cos_sim ~0.9 (>>hardware): hardware is doing something BEYOND
proper int8 simulation -- a real compiler/runtime bug; dig into graph surgery.

100% local, no AI Hub job. Uses real fbank features as calibration + eval.
"""
import os
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                       QuantFormat, CalibrationMethod, QuantType)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")
CALIB_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_calib_tmp.onnx")
QINT8_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_static_int8.onnx")
EVAL_FRAMES = 1500


def compute_fbank(wav, sr):
    assert sr == 16000
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav.tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)


def pad(feats, T=EVAL_FRAMES):
    if feats.shape[0] >= T:
        return feats[:T][None, :, :], np.array([T], dtype=np.int64)
    z = np.zeros((T - feats.shape[0], 80), dtype=np.float32)
    return np.concatenate([feats, z], axis=0)[None, :, :], np.array([feats.shape[0]], dtype=np.int64)


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FbankCalibReader(CalibrationDataReader):
    def __init__(self, items):
        self.items = items
        self.iter = iter(items)
        self.current = None

    def get_next(self):
        try:
            self.current = next(self.iter)
        except StopIteration:
            return None
        return {"x": self.current[0], "x_lens": self.current[1]}


def main():
    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi = [r for r in manifest if r["lang"] == "vi"]
    # calibration on 3 sentences; eval on the smallest (matches diag)
    calib_items = vi[:3]
    eval_item = min(vi, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    calib_inputs = []
    for it in calib_items:
        wav, sr = sf.read(os.path.join(ROOT, it["path"]), dtype="float32")
        f = compute_fbank(wav, sr)
        calib_inputs.append(pad(f))
    print(f"[sim] static int8 with {len(calib_inputs)} calibration sentences")

    # ---- precompute calibration model isn't needed; feed reader directly
    reader = FbankCalibReader(calib_inputs)
    try:
        quantize_static(ENCODER_ONNX, QINT8_ONNX, reader,
                       quant_format=QuantFormat.QDQ,
                       activation_type=QuantType.QUInt8,
                       weight_type=QuantType.QInt8,
                       calibrate_method=CalibrationMethod.MinMax,
                       per_channel=False)
        print(f"[sim] wrote {QINT8_ONNX}")
    except Exception as e:
        print(f"[sim] static quant FAILED: {type(e).__name__}: {str(e)[:300]}")
        return

    # ---- eval: run fp32 + int8-sim on the SAME eval input ----
    wav, sr = sf.read(os.path.join(ROOT, eval_item["path"]), dtype="float32")
    x, x_lens = pad(compute_fbank(wav, sr))

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess_fp = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(QINT8_ONNX, so, providers=["CPUExecutionProvider"])

    out_names = [o.name for o in sess_fp.get_outputs()]
    fp = sess_fp.run(out_names, {"x": x, "x_lens": x_lens})[0]
    q = sess_q.run(out_names, {"x": x, "x_lens": x_lens})[0]

    T = int(min(fp.shape[1], q.shape[1]))
    fp = fp[:, :T, :].astype(np.float64)
    q = q[:, :T, :].astype(np.float64)

    print(f"\n[sim] === RESULT ===")
    print(f"[sim] eval audio     : {eval_item['path']}  ({T} frames)")
    print(f"[sim] fp32  out: mean={fp.mean():+.4f} std={fp.std():.4f} min={fp.min():+.4f} max={fp.max():+.4f}")
    print(f"[sim] qsim  out: mean={q.mean():+.4f} std={q.std():.4f} min={q.min():+.4f} max={q.max():+.4f}")
    print(f"[sim] cos_sim(fp32, local_int8_sim) = {cos_sim(fp, q):.4f}")
    print(f"[sim] (hardware int8 cos_sim was 0.2121)")
    print()
    if abs(cos_sim(fp, q) - 0.21) < 0.15:
        print("[sim] LOCAL int8-sim REPRODUCES hardware corruption (~0.21).")
        print("[sim] => hardware is FAITHFUL to its int8 spec; the spec itself is")
        print("[sim]    too lossy for this depth. Fix = mixed precision (keep")
        print("[sim]    sensitive layers fp16). AIMET QuantAnalyzer pinpoints which.")
    elif cos_sim(fp, q) > 0.85:
        print("[sim] LOCAL int8-sim is NEAR-PERFECT (>>hardware's 0.21).")
        print("[sim] => hardware is doing something BEYOND proper int8 -- real")
        print("[sim]    compiler/runtime bug; dig into graph surgery / op lowering.")
    else:
        print(f"[sim] LOCAL int8-sim partial ({cos_sim(fp,q):.2f}); hardware differs.")


if __name__ == "__main__":
    main()