"""Zipformer int8 ablation probe.

This script builds a local full-int8 QDQ simulation of the Zipformer encoder,
measures its final cosine drift, then removes the QDQ pair for a small set of
candidate tensors one-by-one to estimate which tensor contributes most to the
corruption.

That gives a practical replacement for AIMET QuantAnalyzer on this Windows
setup:
  - large cosine improvement when one tensor is left fp32 => mixed precision
    candidate
  - large activation outlier ratio on the same tensor => SmoothQuant candidate
  - no single tensor helps much => corruption is more distributed, QAT/fine
    tuning is the fallback.
"""
import argparse
import copy
import json
import os
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    CalibrationMethod,
    QuantType,
)
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_no_bool_slice.onnx")
FIXED_FRAMES = 1500


def compute_fbank(wav, sr):
    assert sr == 16000
    opts = knf.FbankOptions()
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, wav.tolist())
    fbank.input_finished()
    return np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)


def pad(feats, frames=FIXED_FRAMES):
    real = feats.shape[0]
    if real >= frames:
        return feats[:frames][None, :, :], np.array([frames], dtype=np.int64), real
    z = np.zeros((frames - real, 80), dtype=np.float32)
    return np.concatenate([feats, z], axis=0)[None, :, :], np.array([real], dtype=np.int64), real


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def candidate_tensors():
    cands = ["/encoder_embed/out/Add_output_0"]
    for stage in range(6):
        if stage == 0:
            cands.extend([
                "/encoder/0/layers.0/bypass/Add_output_0",
                "/encoder/0/layers.1/bypass/Add_output_0",
            ])
        else:
            cands.extend([
                f"/encoder/{stage}/encoder/0/bypass/Add_output_0",
                f"/encoder/{stage}/encoder/1/bypass/Add_output_0",
                f"/encoder/{stage}/out_combiner/Add_output_0",
            ])
    return cands


class FbankCalibReader(CalibrationDataReader):
    def __init__(self, items):
        self.items = items
        self.iter = iter(items)

    def get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            return None


def activation_stats(x):
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1]) if x.ndim >= 3 else x.reshape(-1, 1)
    ch_max = np.max(np.abs(flat), axis=0)
    med = float(np.median(ch_max) + 1e-9)
    return {
        "channel_count": int(ch_max.shape[0]),
        "channel_max_median_ratio": float(ch_max.max() / med),
        "p95_over_median": float(np.percentile(ch_max, 95) / med),
        "p99_over_median": float(np.percentile(ch_max, 99) / med),
        "max_abs": float(ch_max.max()),
        "median_abs": med,
    }


def write_html(path, payload):
    rows = "\n".join(
        f"<tr><td>{r['tensor']}</td><td>{r['delta']:.4f}</td><td>{r['ablate_cos']:.4f}</td>"
        f"<td>{r['channel_max_median_ratio']:.1f}x</td><td>{r['suggestion']}</td></tr>"
        for r in payload["ranked"]
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Zipformer int8 ablation</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 22px; color: #1f2937; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.meta {{ color: #6b7280; margin-bottom: 14px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
<h1>Zipformer int8 ablation</h1>
<div class="meta">Eval <code>{payload['eval_path']}</code>, full int8 cosine <code>{payload['full_int8_cos']:.4f}</code></div>
<table>
<thead><tr><th>Tensor</th><th>Delta</th><th>Ablate cos</th><th>Max/median</th><th>Action</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def build_quantized_model(calib_inputs, out_path):
    reader = FbankCalibReader(calib_inputs)
    quantize_static(
        MODEL_PATH,
        out_path,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )


def remove_qdq_pair(model, tensor_name):
    q_name = tensor_name + "_QuantizeLinear"
    dq_name = tensor_name + "_DequantizeLinear"
    q_out = tensor_name + "_QuantizeLinear_Output"
    dq_out = tensor_name + "_DequantizeLinear_Output"

    node_idxs = [i for i, n in enumerate(model.graph.node) if n.name in {q_name, dq_name}]
    if len(node_idxs) != 2:
        return False

    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp == dq_out:
                node.input[idx] = tensor_name

    for idx in sorted(node_idxs, reverse=True):
        del model.graph.node[idx]
    return True


def tap_tensor(model, tensor_name):
    if any(o.name == tensor_name for o in model.graph.output):
        return
    vi = onnx.helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(vi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "outputs", "zipformer_int8_ablate"))
    ap.add_argument("--calib-count", type=int, default=3)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi = [r for r in manifest if r["lang"] == "vi"]
    calib_items = vi[:args.calib_count]
    eval_item = min(vi, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    calib_inputs = []
    for it in calib_items:
        wav, sr = sf.read(os.path.join(ROOT, it["path"]), dtype="float32")
        feats, lens, _ = pad(compute_fbank(wav, sr))
        calib_inputs.append({"x": feats.astype(np.float32), "x_lens": lens})

    wav, sr = sf.read(os.path.join(ROOT, eval_item["path"]), dtype="float32")
    eval_x, eval_lens, real_frames = pad(compute_fbank(wav, sr))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    fp_sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    fp_out = fp_sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})
    fp_enc = fp_out[0][:, : int(np.array(fp_out[1]).reshape(-1)[0]), :]

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        int8_path = tmp.name
    print(f"[ablate] quantizing full model -> {int8_path}")
    build_quantized_model(calib_inputs, int8_path)

    int8_sess = ort.InferenceSession(int8_path, so, providers=["CPUExecutionProvider"])
    int8_out = int8_sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})
    int8_enc = int8_out[0][:, : int(np.array(int8_out[1]).reshape(-1)[0]), :]
    full_cos = cos_sim(fp_enc, int8_enc)

    # activation stats come from fp32 taps; use the same eval sample for now
    ranked = []
    for tensor_name in candidate_tensors():
        tap_model = onnx.load(MODEL_PATH)
        tap_tensor(tap_model, tensor_name)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_tap:
            tap_path = tmp_tap.name
        onnx.save(tap_model, tap_path)
        tap_sess = ort.InferenceSession(tap_path, so, providers=["CPUExecutionProvider"])
        tap_out = tap_sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})
        tap_act = tap_out[-1]
        stats = activation_stats(tap_act)

        ablated = onnx.load(int8_path)
        removed = remove_qdq_pair(ablated, tensor_name)
        if not removed:
            continue
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_ablate:
            ablate_path = tmp_ablate.name
        onnx.save(ablated, ablate_path)
        try:
            ablate_sess = ort.InferenceSession(ablate_path, so, providers=["CPUExecutionProvider"])
            ablate_out = ablate_sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})
            ablate_enc = ablate_out[0][:, : int(np.array(ablate_out[1]).reshape(-1)[0]), :]
            score = cos_sim(fp_enc, ablate_enc)
        except Exception:
            score = float("nan")

        delta = score - full_cos if np.isfinite(score) else float("nan")
        suggestion = "keep fp16/int16" if np.isfinite(score) and delta > 0.02 else "QAT fallback"
        if stats["channel_max_median_ratio"] > 8.0:
            suggestion = "SmoothQuant candidate"
        ranked.append({
            "tensor": tensor_name,
            "full_int8_cos": float(full_cos),
            "ablate_cos": float(score),
            "delta": float(delta),
            **stats,
            "suggestion": suggestion,
        })

    ranked.sort(key=lambda r: (-(r["delta"] if np.isfinite(r["delta"]) else -1e9)))
    payload = {
        "eval_path": eval_item["path"],
        "real_input_frames": int(real_frames),
        "full_int8_cos": float(full_cos),
        "fp32_vs_fp32_cos": float(cos_sim(fp_enc, fp_enc)),
        "ranked": ranked,
    }

    json_path = args.out_prefix + ".json"
    html_path = args.out_prefix + ".html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_html(html_path, payload)
    print(json_path)
    print(html_path)
    print(json.dumps(ranked[:6], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
