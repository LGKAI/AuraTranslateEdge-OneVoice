# -*- coding: utf-8 -*-
"""Compile the ASR window graphs ("buckets") on Qualcomm AI Hub and LINK all windows of one model into ONE context binary (one graph per window).
AI Hub link jobs store weights that are identical (same name and content) once, so N windows cost about one model of memory; s2s_edge selects a
window with QNNContext.Inference(graphIndex=i) (models.json: "ctx" = the shared .bin, "graph_index" = i, see s2s_edge/backends.py).
Mirrors the team's job options (src/step4_quantization/.../submit_full_sensevoice_pipeline.py, tools/relink_device.py).

  # 1. ONNX windows (CPU):
  python ../src/step4_quantization/asr/zipformer/step4_hardware/build_zip150_buckets.py --model-dir <hynt model> --out-dir out/zip --frames 300,500,800,1200,1500 --with-encoder-out
  python ../src/step4_quantization/asr/sensevoice/export/export_sensevoice_buckets.py --out-dir out/sv --frames 50,100,200,500
  # 2. this script (needs `qai-hub` configured with the team's API token):
  python tools/build_asr_buckets_aihub.py --model zip --onnx-dir out/zip --frames 300,500,800,1200,1500 --out models_qcs8550 [--dry-run]
  python tools/build_asr_buckets_aihub.py --model sv  --onnx-dir out/sv  --frames 50,100,200,500 --precision fp16 --out models_qcs8550
It writes <out>/<base>_buckets_<precision>.bin and prints the models.json entries (graph_index = link order, VERIFY on the box). Check on the box with `python tools/golden.py check`.
Precision: Zipformer stays fp16 (w8a16 was measured bad on chip). SenseVoice: fp16 keeps the weights byte-identical across windows (shared in the
link) and avoids the quantisation loss; w8a16 (the current 500-frame graph) needs --calib (npz with fbank (N,1,500,560) + speech_lengths ...) and is
quantised per window, so its weights are not shared. A/B both on AI Hub (profile + golden) before choosing.
NOT run by the author of this script (no AI Hub access here): read the printed plan with --dry-run first."""
import argparse, json, os, sys, time


def onnx_path(model, d, w):
    return os.path.join(d, f"zip150_ctc_rnnt_static_{w}.onnx" if model == "zip" else f"sv_enc_cls_{w}.onnx")


def input_specs(model, w):
    if model == "zip": return {"x": ((1, w, 80), "float32"), "x_lens": ((1,), "int64")}
    return {"fbank": ((1, w, 560), "float32"), "speech_lengths": ((1,), "int32"), "language": ((1,), "int32"), "textnorm": ((1,), "int32")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["zip", "sv"]); ap.add_argument("--onnx-dir", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--device", default="QCS8550 (Proxy)"); ap.add_argument("--qairt", default="2.51"); ap.add_argument("--out", default="models_qcs8550")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "w8a16"]); ap.add_argument("--calib", default="", help="w8a16 only: npz of 500-frame calibration fbank")
    ap.add_argument("--fp16-options", default="--target_runtime qnn_dlc --truncate_64bit_io --quantize_full_type float16",
                    help="compile options for fp16; COPY them from the shipped fp16 Zipformer job first: hub.get_job('jpv226qjg').options")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
    ws = sorted(int(x) for x in a.frames.split(",")); paths = {w: onnx_path(a.model, a.onnx_dir, w) for w in ws}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing: sys.exit(f"missing ONNX windows: {missing}")
    base = "asr_vi" if a.model == "zip" else "asr_xx_encoder"; big = 1500 if a.model == "zip" else 500
    bin_name = f"{base}_buckets_{a.precision}.bin"
    entries = {(base if w == ws[-1] else f"{base}@{w}"): {"ctx": bin_name, "graph_index": i, **({"window": w} if w == ws[-1] and w != big else {})} for i, w in enumerate(ws)}
    # graph_index = position in the link job is an ASSUMPTION (AI Hub does not document the order): after linking, check the graph names on the
    # box (QNNContext.getGraphName / golden.py check: a wrong window fails on its input size) and fix graph_index if needed.
    print("plan:", json.dumps(dict(model=a.model, windows=ws, precision=a.precision, device=a.device, qairt=a.qairt, out=os.path.join(a.out, bin_name)), indent=1))
    print("models.json entries (graph_index = position in the link job):\n" + json.dumps(entries, indent=1))
    if a.dry_run: return
    import numpy as np
    import qai_hub as hub
    dev = hub.Device(a.device); os.makedirs(a.out, exist_ok=True); compiled = []
    calib = np.load(a.calib) if (a.precision == "w8a16" and a.calib) else None
    if a.precision == "w8a16" and calib is None: sys.exit("--precision w8a16 needs --calib")
    if a.model == "zip" and a.precision == "w8a16": sys.exit("Zipformer: fp16 only (w8a16 measured bad on chip; the calibration builder here is SenseVoice's)")
    for w in ws:
        m = hub.upload_model(paths[w], name=f"{base}_{w}")
        if a.precision == "w8a16":
            n = len(calib["fbank"]); sl = calib["speech_lengths"].reshape(n, -1)[:, 0]
            data = {"fbank": [calib["fbank"][i].reshape(1, -1, 560)[:, :w].astype(np.float32) for i in range(n)],
                    "speech_lengths": [np.array([min(int(sl[i]), w)], np.int32) for i in range(n)],
                    "language": [calib["language"][i].reshape(1).astype(np.int32) for i in range(n)], "textnorm": [calib["textnorm"][i].reshape(1).astype(np.int32) for i in range(n)]}
            q = hub.submit_quantize_job(model=m, calibration_data=hub.upload_dataset(data, name=f"{base}_calib_{w}"), weights_dtype=hub.QuantizeDtype.INT8,
                                        activations_dtype=hub.QuantizeDtype.INT16, name=f"{base}_{w}_w8a16")
            if not q.wait().success: sys.exit(f"quantize {w} failed: {q.get_status().message}")
            m = q.get_target_model(); opts = "--target_runtime qnn_dlc --truncate_64bit_io"
        else:
            opts = a.fp16_options
        gname = f"{base.replace('@', '_')}_w{w}"                          # explicit graph name per window (AI Hub link example; names must be unique)
        c = hub.submit_compile_job(model=m, device=dev, input_specs=input_specs(a.model, w), options=f"{opts} --qairt_version {a.qairt} --qnn_options context_enable_graphs={gname}",
                                   name=f"{base}_{w}_{a.precision}")
        print(f"window {w}: compile job {c.job_id}", flush=True); compiled.append((w, c))
    models = []
    for w, c in compiled:
        st = c.wait()
        if not st.success: sys.exit(f"compile {w} failed: {st.message}")
        models.append(c.get_target_model())
    link = hub.submit_link_job(models=models, device=dev, options=f"--qairt_version {a.qairt}", name=f"{base}_buckets_{a.qairt}")
    print("link job", link.job_id, flush=True)
    if not link.wait().success: sys.exit(f"link failed: {link.get_status().message}")
    dst = os.path.join(a.out, bin_name); link.get_target_model().download(dst)
    json.dump(dict(link_job=link.job_id, compile_jobs={w: c.job_id for w, c in compiled}, entries=entries, time=time.strftime("%Y-%m-%d %H:%M")),
              open(os.path.join(a.out, f"{base}_buckets_{a.precision}.json"), "w"), indent=1)
    print("saved", dst, "-> add the entries above to models.json (paths relative to it), then run tools/golden.py check on the box")


if __name__ == "__main__":
    main()
