"""Block-level mixed precision search for Zipformer.

Workflow:
1. Build a local full-int8 QDQ simulation.
2. Ablate whole blocks by removing all activation QDQ pairs whose tensor
   names live under that block prefix.
3. Rank blocks by how much cosine recovers relative to the full int8 model.
4. Optionally refine the best coarse block into sub-blocks.

This is the coarse-to-fine search you asked for: it avoids testing every
tensor, and instead tries meaningful encoder blocks first.
"""
import argparse
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
TMP_ROOT = os.path.join(ROOT, "outputs", "ort_tmp")


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


class FbankCalibReader(CalibrationDataReader):
    def __init__(self, items):
        self.items = items
        self.iter = iter(items)

    def get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            return None


def build_quantized_model(calib_inputs, out_path):
    os.makedirs(TMP_ROOT, exist_ok=True)
    tempfile.tempdir = TMP_ROOT
    os.environ["TEMP"] = TMP_ROOT
    os.environ["TMP"] = TMP_ROOT
    os.environ["TMPDIR"] = TMP_ROOT
    quantize_static(
        MODEL_PATH,
        out_path,
        FbankCalibReader(calib_inputs),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )


def block_specs():
    specs = [
        {"name": "encoder_embed", "prefix": "/encoder_embed"},
    ]
    for stage in range(6):
        specs.append({"name": f"stage_{stage}", "prefix": f"/encoder/{stage}"})
    return specs


def refine_specs(stage_prefix):
    if stage_prefix == "/encoder_embed":
        return [
            {"name": "embed_conv", "prefix": "/encoder_embed/conv"},
            {"name": "embed_convnext", "prefix": "/encoder_embed/convnext"},
            {"name": "embed_out", "prefix": "/encoder_embed/out"},
            {"name": "embed_out_norm", "prefix": "/encoder_embed/out_norm"},
        ]
    stage = int(stage_prefix.split("/")[2])
    if stage == 0:
        return [
            {"name": "s0_layer0", "prefix": "/encoder/0/layers.0"},
            {"name": "s0_layer1", "prefix": "/encoder/0/layers.1"},
        ]
    return [
        {"name": f"s{stage}_enc0", "prefix": f"/encoder/{stage}/encoder/0"},
        {"name": f"s{stage}_enc1", "prefix": f"/encoder/{stage}/encoder/1"},
        {"name": f"s{stage}_out_combiner", "prefix": f"/encoder/{stage}/out_combiner"},
        {"name": f"s{stage}_upsample", "prefix": f"/encoder/{stage}/upsample"},
    ]


def combo_specs():
    all_stages = [f"/encoder/{i}" for i in range(6)]
    return [
        {"name": "stage_1", "prefixes": ["/encoder/1"]},
        {"name": "stage_1+stage_0", "prefixes": ["/encoder/1", "/encoder/0"]},
        {"name": "stage_1+stage_0+stage_5", "prefixes": ["/encoder/1", "/encoder/0", "/encoder/5"]},
        {"name": "stage_1+stage_0+stage_5+stage_2", "prefixes": ["/encoder/1", "/encoder/0", "/encoder/5", "/encoder/2"]},
        {"name": "stage_1+stage_0+stage_5+stage_2+stage_3", "prefixes": ["/encoder/1", "/encoder/0", "/encoder/5", "/encoder/2", "/encoder/3"]},
        {"name": "stage_1+stage_0+stage_5+stage_2+stage_4", "prefixes": ["/encoder/1", "/encoder/0", "/encoder/5", "/encoder/2", "/encoder/4"]},
        {"name": "all_encoder_stages", "prefixes": all_stages},
        *[
            {"name": f"all_except_stage_{i}", "prefixes": [p for p in all_stages if p != f"/encoder/{i}"]}
            for i in range(6)
        ],
        {"name": "stage_1_internal_only", "prefixes": ["/encoder/1/encoder/0", "/encoder/1/encoder/1"]},
        {"name": "top_refined_3", "prefixes": ["/encoder/1/encoder/1", "/encoder/1/encoder/0", "/encoder/0/layers.1"]},
    ]


def remove_qdq_pairs_under_prefix(model, prefix):
    """Remove activation QDQ pairs whose tensor names live under a prefix."""
    q_by_out = {}
    q_base_by_out = {}
    dq_by_qout = {}
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if not (node.name.startswith(prefix) or (node.input and node.input[0].startswith(prefix))
                or (node.output and node.output[0].startswith(prefix))):
            continue
        if node.op_type == "QuantizeLinear" and node.output:
            q_by_out[node.output[0]] = idx
            q_base_by_out[node.output[0]] = node.input[0]
        elif node.op_type == "DequantizeLinear" and node.input:
            dq_by_qout[node.input[0]] = idx

    remove_indices = set()
    replacements = []
    for q_out, qidx in q_by_out.items():
        dqidx = dq_by_qout.get(q_out)
        if dqidx is None:
            continue
        base = q_base_by_out[q_out]
        dqnode = model.graph.node[dqidx]
        remove_indices.update([qidx, dqidx])
        replacements.append((dqnode.output[0], base))

    if not remove_indices:
        return 0

    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            for dq_out, base in replacements:
                if inp == dq_out:
                    node.input[i] = base

    for idx in sorted(remove_indices, reverse=True):
        del model.graph.node[idx]
    return len(remove_indices) // 2


def add_output_tap(model, tensor_name):
    if any(o.name == tensor_name for o in model.graph.output):
        return
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, None)
    )


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


def eval_model(sess, x, x_lens):
    out = sess.run(None, {"x": x.astype(np.float32), "x_lens": x_lens})
    enc = out[0]
    T = int(np.array(out[1]).reshape(-1)[0])
    return enc[:, :T, :]


def score_variant(base_fp, variant_model, x, x_lens, so):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        variant_path = tmp.name
    onnx.save(variant_model, variant_path)
    sess = ort.InferenceSession(variant_path, so, providers=["CPUExecutionProvider"])
    out = eval_model(sess, x, x_lens)
    return out, cos_sim(base_fp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default=os.path.join(ROOT, "outputs", "zipformer_block_mp"))
    ap.add_argument("--calib-count", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
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
    fp_base = eval_model(fp_sess, eval_x, eval_lens)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        int8_path = tmp.name
    build_quantized_model(calib_inputs, int8_path)
    int8_sess = ort.InferenceSession(int8_path, so, providers=["CPUExecutionProvider"])
    int8_base = eval_model(int8_sess, eval_x, eval_lens)
    full_cos = cos_sim(fp_base, int8_base)

    coarse = []
    for spec in block_specs():
        model = onnx.load(int8_path)
        removed = remove_qdq_pairs_under_prefix(model, spec["prefix"])
        if removed == 0:
            continue
        coarse.append({
            "name": spec["name"],
            "prefix": spec["prefix"],
            "removed_pairs": int(removed),
            "cos": float(score_variant(fp_base, model, eval_x, eval_lens, so)[1]),
            "delta": float(score_variant(fp_base, model, eval_x, eval_lens, so)[1] - full_cos),
        })

    coarse.sort(key=lambda r: r["delta"], reverse=True)
    refined = []
    for winner in coarse[:args.topk]:
        for spec in refine_specs(winner["prefix"]):
            model = onnx.load(int8_path)
            removed = remove_qdq_pairs_under_prefix(model, spec["prefix"])
            if removed == 0:
                continue
            _, score = score_variant(fp_base, model, eval_x, eval_lens, so)
            refined.append({
                "parent": winner["name"],
                "name": spec["name"],
                "prefix": spec["prefix"],
                "removed_pairs": int(removed),
                "cos": float(score),
                "delta": float(score - full_cos),
            })

    refined.sort(key=lambda r: r["delta"], reverse=True)
    combos = []
    for spec in combo_specs():
        model = onnx.load(int8_path)
        removed_total = 0
        for prefix in spec["prefixes"]:
            removed_total += remove_qdq_pairs_under_prefix(model, prefix)
        if removed_total == 0:
            continue
        _, score = score_variant(fp_base, model, eval_x, eval_lens, so)
        combos.append({
            "name": spec["name"],
            "prefixes": spec["prefixes"],
            "removed_pairs": int(removed_total),
            "cos": float(score),
            "delta": float(score - full_cos),
        })
    combos.sort(key=lambda r: r["delta"], reverse=True)

    payload = {
        "eval_path": eval_item["path"],
        "real_input_frames": int(real_frames),
        "full_int8_cos": float(full_cos),
        "coarse": coarse,
        "refined": refined,
        "combos": combos,
    }

    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json_path)
    print(json.dumps({
        "full_int8_cos": payload["full_int8_cos"],
        "top_coarse": coarse[:5],
        "top_refined": refined[:5],
        "top_combos": combos[:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
