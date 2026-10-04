"""Diagnose Zipformer w8a16 encoder corruption and write a small HTML report.

This reuses an already-completed AI Hub inference job by default, so it does
not submit new hardware work. It compares the hardware encoder output against
the local fp32 ONNX reference on the exact same eval utterance used by the
w8a16 experiment, then writes:

  outputs/zipformer_w8a16_diagnostics.json
  outputs/zipformer_w8a16_diagnostics.html
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import qai_hub as hub
import soundfile as sf
import kaldi_native_fbank as knf


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_ONNX = os.path.join(ROOT, "outputs", "zipformer-qnn", "encoder_static_1500.onnx")
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


def cos_sim(a, b, axis=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.sum(a * b, axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-9
    return num / den


def percentile_rows(values, n=12, reverse=False):
    order = np.argsort(values)
    if reverse:
        order = order[::-1]
    out = []
    for idx in order[:n]:
        out.append({"index": int(idx), "value": float(values[idx])})
    return out


def write_html(path, payload):
    data = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Zipformer w8a16 Diagnostics</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17202a; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  h2 {{ font-size: 16px; margin: 24px 0 8px; }}
  .meta {{ color: #59636e; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
  .card {{ border: 1px solid #d7dde4; border-radius: 6px; padding: 10px 12px; }}
  .label {{ color: #59636e; font-size: 12px; }}
  .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  svg {{ width: 100%; height: 280px; overflow: visible; }}
  .axis {{ stroke: #aab3bd; stroke-width: 1; }}
  .gridline {{ stroke: #e5e9ee; stroke-width: 1; }}
  .line {{ fill: none; stroke: #2563eb; stroke-width: 1.5; }}
  .bad {{ fill: #dc2626; }}
  .ok {{ fill: #2563eb; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e5e9ee; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }}
</style>
<h1>Zipformer w8a16 encoder diagnostics</h1>
<div class="meta">Hardware job <code>{payload["hw_job"]}</code>, eval audio <code>{payload["eval_path"]}</code></div>
<div class="grid">
  <div class="card"><div class="label">Full cosine</div><div class="value">{payload["summary"]["full_cos"]:.4f}</div></div>
  <div class="card"><div class="label">Per-frame mean</div><div class="value">{payload["summary"]["frame_cos_mean"]:.4f}</div></div>
  <div class="card"><div class="label">Worst frame</div><div class="value">{payload["summary"]["frame_cos_min"]:.4f}</div></div>
  <div class="card"><div class="label">Frames below 0.5</div><div class="value">{payload["summary"]["frames_lt_0_5"]}/{payload["summary"]["valid_T"]}</div></div>
</div>
<h2>Cosine by encoder frame</h2>
<svg id="frameChart" role="img" aria-label="Per-frame cosine similarity chart"></svg>
<h2>Worst frames</h2>
<table id="worstFrames"><thead><tr><th>Frame</th><th>Cosine</th><th>fp32 norm</th><th>hw norm</th></tr></thead><tbody></tbody></table>
<h2>Most distorted hidden channels</h2>
<table id="channels"><thead><tr><th>Channel</th><th>Mean abs diff</th><th>fp32 max abs</th><th>hw max abs</th></tr></thead><tbody></tbody></table>
<script>
const data = {data};
const svg = document.getElementById('frameChart');
const W = 960, H = 280, ml = 42, mr = 14, mt = 12, mb = 28;
svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
const xs = data.frames.map(d => d.frame);
const ys = data.frames.map(d => d.cos);
const maxX = Math.max(...xs), minY = Math.min(-0.1, ...ys), maxY = 1.0;
const x = v => ml + (v / maxX) * (W - ml - mr);
const y = v => mt + (maxY - v) / (maxY - minY) * (H - mt - mb);
function el(name, attrs) {{
  const n = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs || {{}})) n.setAttribute(k, v);
  return n;
}}
for (const t of [0, 0.5, 0.9, 1.0]) {{
  svg.appendChild(el('line', {{x1: ml, x2: W - mr, y1: y(t), y2: y(t), class: 'gridline'}}));
  const txt = el('text', {{x: 4, y: y(t) + 4, 'font-size': 11, fill: '#59636e'}});
  txt.textContent = t.toFixed(1);
  svg.appendChild(txt);
}}
svg.appendChild(el('line', {{x1: ml, x2: ml, y1: mt, y2: H - mb, class: 'axis'}}));
svg.appendChild(el('line', {{x1: ml, x2: W - mr, y1: H - mb, y2: H - mb, class: 'axis'}}));
const path = data.frames.map((d, i) => `${{i ? 'L' : 'M'}}${{x(d.frame).toFixed(1)}} ${{y(d.cos).toFixed(1)}}`).join(' ');
svg.appendChild(el('path', {{d: path, class: 'line'}}));
for (const d of data.frames) {{
  if (d.cos < 0.5) svg.appendChild(el('circle', {{cx: x(d.frame), cy: y(d.cos), r: 2, class: 'bad'}}));
}}
for (const d of data.worst_frames) {{
  svg.appendChild(el('circle', {{cx: x(d.frame), cy: y(d.cos), r: 4, class: 'bad'}}));
}}
for (const t of [0, Math.round(maxX / 2), maxX]) {{
  const txt = el('text', {{x: x(t), y: H - 8, 'font-size': 11, fill: '#59636e', 'text-anchor': 'middle'}});
  txt.textContent = t;
  svg.appendChild(txt);
}}
document.querySelector('#worstFrames tbody').innerHTML = data.worst_frames.map(d =>
  `<tr><td>${{d.frame}}</td><td>${{d.cos.toFixed(4)}}</td><td>${{d.fp_norm.toFixed(4)}}</td><td>${{d.hw_norm.toFixed(4)}}</td></tr>`).join('');
document.querySelector('#channels tbody').innerHTML = data.channels.map(d =>
  `<tr><td>${{d.channel}}</td><td>${{d.mean_abs_diff.toFixed(4)}}</td><td>${{d.fp_max_abs.toFixed(4)}}</td><td>${{d.hw_max_abs.toFixed(4)}}</td></tr>`).join('');
</script>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-job", default="jp2ezvyxp", help="Completed AI Hub inference job id")
    parser.add_argument("--out-prefix", default=os.path.join(ROOT, "outputs", "zipformer_w8a16_diagnostics"))
    args = parser.parse_args()

    with open(os.path.join(ROOT, "data", "asr", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    vi_items = [r for r in manifest if r["lang"] == "vi"]
    item = min(vi_items, key=lambda r: os.path.getsize(os.path.join(ROOT, r["path"])))

    wav, sr = sf.read(os.path.join(ROOT, item["path"]), dtype="float32")
    eval_x, eval_lens, real_frames = pad(compute_fbank(wav, sr))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ENCODER_ONNX, so, providers=["CPUExecutionProvider"])
    fp_out, fp_lens = sess.run(None, {"x": eval_x.astype(np.float32), "x_lens": eval_lens})
    fp_T = int(np.array(fp_lens).reshape(-1)[0])
    fp = fp_out[:, :fp_T, :].astype(np.float64)

    job = hub.get_job(args.hw_job)
    status = job.get_status()
    if status.code != "SUCCESS":
        raise RuntimeError(f"{args.hw_job} status is {status.code}: {status.message}")
    hw_data = job.download_output_data()
    hw_out = np.array(hw_data["output_0"][0])
    hw_T = int(np.array(hw_data["output_1"][0]).reshape(-1)[0])
    hw = hw_out[:, :hw_T, :].astype(np.float64)

    T = min(fp.shape[1], hw.shape[1])
    fp = fp[:, :T, :]
    hw = hw[:, :T, :]
    frame_cos = cos_sim(fp.reshape(T, -1), hw.reshape(T, -1), axis=1)
    fp_norm = np.linalg.norm(fp.reshape(T, -1), axis=1)
    hw_norm = np.linalg.norm(hw.reshape(T, -1), axis=1)
    diff = np.abs(fp - hw)[0]
    channel_diff = diff.mean(axis=0)
    fp_ch_max = np.max(np.abs(fp[0]), axis=0)
    hw_ch_max = np.max(np.abs(hw[0]), axis=0)
    worst_frame_idx = np.argsort(frame_cos)[:16]
    worst_channel_idx = np.argsort(channel_diff)[::-1][:24]

    payload = {
        "hw_job": args.hw_job,
        "eval_path": item["path"],
        "reference": item.get("transcript", ""),
        "summary": {
            "real_input_frames": int(real_frames),
            "valid_T": int(T),
            "full_cos": float(cos_sim(fp, hw)),
            "frame_cos_mean": float(frame_cos.mean()),
            "frame_cos_min": float(frame_cos.min()),
            "frame_cos_max": float(frame_cos.max()),
            "frames_lt_0_5": int((frame_cos < 0.5).sum()),
            "frames_lt_0_9": int((frame_cos < 0.9).sum()),
            "fp32_std": float(fp.std()),
            "hw_std": float(hw.std()),
        },
        "frames": [
            {"frame": int(i), "cos": float(frame_cos[i]), "fp_norm": float(fp_norm[i]), "hw_norm": float(hw_norm[i])}
            for i in range(T)
        ],
        "worst_frames": [
            {"frame": int(i), "cos": float(frame_cos[i]), "fp_norm": float(fp_norm[i]), "hw_norm": float(hw_norm[i])}
            for i in worst_frame_idx
        ],
        "channels": [
            {
                "channel": int(i),
                "mean_abs_diff": float(channel_diff[i]),
                "fp_max_abs": float(fp_ch_max[i]),
                "hw_max_abs": float(hw_ch_max[i]),
            }
            for i in worst_channel_idx
        ],
    }

    json_path = args.out_prefix + ".json"
    html_path = args.out_prefix + ".html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_html(html_path, payload)
    print(json_path)
    print(html_path)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
