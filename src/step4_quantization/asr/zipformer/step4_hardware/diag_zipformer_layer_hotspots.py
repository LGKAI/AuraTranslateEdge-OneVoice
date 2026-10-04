"""Zipformer layer hotspot probe.

This script approximates AIMET/QuantAnalyzer-style sensitivity analysis without
requiring AIMET on Windows:

1. Run fp32 ONNXRuntime on a real eval utterance.
2. For a small set of representative encoder tensors, insert a local QDQ pair
   after that tensor and measure how much the final encoder output drifts.
3. Compute simple activation outlier stats for the same tensors so the result
   can be mapped to SmoothQuant-style smoothing or mixed-precision overrides.

The output is a JSON file plus a tiny HTML report that makes the ranking easy to
scan.
"""
import argparse
import copy
import json
import os
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
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


def ensure_1d_f32(x):
    x = np.asarray(x, dtype=np.float32).reshape(1)
    return x


def make_candidate_tensors():
    cands = ["/encoder_embed/out/Add_output_0"]
    for stage in range(6):
        if stage == 0:
            cands.append(f"/encoder/{stage}/layers.1/bypass/Add_output_0")
        else:
            cands.append(f"/encoder/{stage}/encoder/1/bypass/Add_output_0")
            cands.append(f"/encoder/{stage}/out_combiner/Add_output_0")
    return cands


def get_node_by_output(model, tensor_name):
    for n in model.graph.node:
        if tensor_name in n.output:
            return n
    return None


def add_tensor_as_output(model, tensor_name):
    if any(o.name == tensor_name for o in model.graph.output):
        return
    node = get_node_by_output(model, tensor_name)
    if node is None:
        raise ValueError(f"tensor {tensor_name} not produced by graph")
    vi = helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(vi)


def insert_qdq_after_tensor(model, tensor_name, scale_value):
    """Insert symmetric int8 QDQ after tensor_name."""
    node = get_node_by_output(model, tensor_name)
    if node is None:
        raise ValueError(f"tensor {tensor_name} not produced by graph")

    consumers = []
    for other in model.graph.node:
        for idx, inp in enumerate(other.input):
            if inp == tensor_name:
                consumers.append((other, idx))

    q_name = tensor_name + "__q"
    dq_name = tensor_name + "__dq"
    scale_name = tensor_name.replace("/", "_") + "__scale"
    zp_name = tensor_name.replace("/", "_") + "__zp"

    scale_init = numpy_helper.from_array(ensure_1d_f32(scale_value), name=scale_name)
    zp_init = numpy_helper.from_array(np.array([0], dtype=np.int8), name=zp_name)
    q_node = helper.make_node(
        "QuantizeLinear", [tensor_name, scale_name, zp_name], [q_name],
        name=tensor_name + "__QuantizeLinear")
    dq_node = helper.make_node(
        "DequantizeLinear", [q_name, scale_name, zp_name], [dq_name],
        name=tensor_name + "__DequantizeLinear")

    for other, idx in consumers:
        other.input[idx] = dq_name

    producer_idx = list(model.graph.node).index(node)
    model.graph.node.insert(producer_idx + 1, q_node)
    model.graph.node.insert(producer_idx + 2, dq_node)
    model.graph.initializer.extend([scale_init, zp_init])


def activation_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 3:
        flat = x.reshape(-1, 1)
    else:
        flat = x.reshape(-1, x.shape[-1])
    ch_max = np.max(np.abs(flat), axis=0)
    med = float(np.median(ch_max) + 1e-9)
    p95 = float(np.percentile(ch_max, 95))
    p99 = float(np.percentile(ch_max, 99))
    mx = float(ch_max.max())
    return {
        "channel_count": int(ch_max.shape[0]),
        "channel_max_median_ratio": float(mx / med),
        "p95_over_median": float(p95 / med),
        "p99_over_median": float(p99 / med),
        "max_abs": mx,
        "median_abs": med,
    }, ch_max


def write_html(path, payload):
    rows = "\n".join(
        f"<tr><td>{r['tensor']}</td><td>{r['cos_after_qdq']:.4f}</td>"
        f"<td>{r['channel_max_median_ratio']:.1f}x</td>"
        f"<td>{r['p99_over_median']:.1f}x</td>"
        f"<td>{r['suggestion']}</td></tr>"
        for r in payload["ranked"]
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Zipformer layer hotspots</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 22px; color: #1f2937; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.meta {{ color: #6b7280; margin-bottom: 14px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.bad {{ color: #b91c1c; }}
.ok {{ color: #2563eb; }}
</style>
<h1>Zipformer layer hotspots</h1>
<div class="meta">Eval audio <code>{payload['eval_path']}</code>, fp32 cosine baseline <code>{payload['baseline_cos']:.4f}</code></div>
<table>
<thead><tr><th>Tensor</th><th>Cos after QDQ</th><th>Max/median</th><th>P99/median</th><th>Action</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "outputs", "zipformer_layer_hotspots"))
    args = ap.parse_args()

    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
    feats, x_lens, real_frames = pad(compute_fbank(wav, sr))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    base = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    base_out = base.run(None, {"x": feats.astype(np.float32), "x_lens": x_lens})
    base_enc = base_out[0]
    base_T = int(np.array(base_out[1]).reshape(-1)[0])
    base_enc = base_enc[:, :base_T, :]
    baseline_cos = cos_sim(base_enc, base_enc)

    candidates = make_candidate_tensors()
    ranked = []

    for tensor_name in candidates:
        model = onnx.load(MODEL_PATH)
        add_tensor_as_output(model, tensor_name)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tap_path = tmp.name
        onnx.save(model, tap_path)
        tap_sess = ort.InferenceSession(tap_path, so, providers=["CPUExecutionProvider"])
        tap_out = tap_sess.run(None, {"x": feats.astype(np.float32), "x_lens": x_lens})
        tap_act = tap_out[-1]
        tap_act = np.asarray(tap_act)

        stats, _ = activation_stats(tap_act)
        scale = max(stats["max_abs"] / 127.0, 1e-8)

        qmodel = onnx.load(MODEL_PATH)
        insert_qdq_after_tensor(qmodel, tensor_name, scale)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp2:
            qdq_path = tmp2.name
        onnx.save(qmodel, qdq_path)
        q_sess = ort.InferenceSession(qdq_path, so, providers=["CPUExecutionProvider"])
        q_out = q_sess.run(None, {"x": feats.astype(np.float32), "x_lens": x_lens})
        q_enc = q_out[0][:, :base_T, :]

        score = cos_sim(base_enc[:, :q_enc.shape[1], :], q_enc)
        suggestion = "keep fp16/int16" if score < 0.90 else "safe for int8"
        if stats["channel_max_median_ratio"] > 8.0:
            suggestion = "SmoothQuant candidate"
        ranked.append({
            "tensor": tensor_name,
            "cos_after_qdq": float(score),
            "channel_max_median_ratio": stats["channel_max_median_ratio"],
            "p99_over_median": stats["p99_over_median"],
            "max_abs": stats["max_abs"],
            "median_abs": stats["median_abs"],
            "suggestion": suggestion,
        })

    ranked.sort(key=lambda r: r["cos_after_qdq"])
    payload = {
        "eval_path": item["path"],
        "real_input_frames": int(real_frames),
        "baseline_cos": float(baseline_cos),
        "ranked": ranked,
    }

    json_path = args.out_prefix + ".json"
    html_path = args.out_prefix + ".html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_html(html_path, payload)
    print(json_path)
    print(html_path)
    print(json.dumps(ranked[:5], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
