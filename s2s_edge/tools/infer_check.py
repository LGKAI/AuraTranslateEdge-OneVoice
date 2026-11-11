# -*- coding: utf-8 -*-
"""Correctness of the QCS8550 context binaries BEFORE we have the box: run each linked graph on AI Hub's hosted "QCS8550 (Proxy)" with the
golden inputs (golden/<key>.npz) and compare the outputs with the CPU reference, using the same rule as tools/golden.py
(float outputs cosine >= 0.99, integer outputs >= 95 % equal). A profile job only proves the graph loads; this proves it computes the right thing.
  python tools/infer_check.py --links ../s2s_pkg/models_qcs8550/link_jobs.json --device "QCS8550 (Proxy)" --out logs/infer_qcs8550.json
Outputs are matched by position (the graph's output order); if that fails, the best match by shape and value is also reported, because a
permuted output order is fixable in models.json ("out": [...]) while wrong values are not."""
import argparse, json, os, sys
import numpy as np
import qai_hub as hub
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from golden import judge

KEY = {"asr_vi_zip150_ctc_fp16": "asr_vi", "asr_xx_sensevoice_frontend_v3": "asr_xx_frontend", "asr_xx_sensevoice_encoder_q": "asr_xx_encoder",
       "mt_enc_nllb_s64": "mt_enc", "mt_dec_nllb_s64": "mt_dec", "mt_decB_nllb_s64": "mt_decB",
       "tts_vi_piper_fullg8": "tts_vi", "tts_zh_melo_one": "tts_zh", "tts_enko_supertonic_full_v3": "tts_enko"}
def key_of(n):
    """golden key of a graph file name: the 600M names above, or any mt_enc*/mt_dec*/mt_decB* (e.g. the NLLB-1.3B graphs)."""
    if n in KEY: return KEY[n]
    for p in ("mt_decB", "mt_dec", "mt_enc"):
        if n.startswith(p): return p
    raise KeyError(n)


GOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden")


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size != b.size: return "size", float("nan")
    if np.issubdtype(b.dtype, np.integer): return "int", float((a.reshape(-1).astype(np.int64) == b.reshape(-1).astype(np.int64)).mean())
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64); return "cos", float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def good(kind, v): return (kind == "cos" and v >= 0.99) or (kind == "int" and v >= 0.95)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--links", required=True); ap.add_argument("--device", default="QCS8550 (Proxy)"); ap.add_argument("--out", default="logs/infer_qcs8550.json")
    ap.add_argument("--only", default=""); ap.add_argument("--golden", default="golden", help="golden directory of this model set (golden_13b for NLLB-1.3B)")
    a = ap.parse_args()
    G = lambda k: next((p for p in (os.path.join(os.path.dirname(GOLD), a.golden, k + ".npz"), os.path.join(GOLD, k + ".npz")) if os.path.exists(p)), os.path.join(GOLD, k + ".npz")); sys.stdout.reconfigure(encoding="utf-8")
    dev = hub.Device(a.device); links = json.load(open(a.links)); jobs = {}
    for n, jid in links.items():
        if a.only and n not in a.only.split(","): continue
        z = np.load(G(key_of(n)))
        # HTP graphs take int32 where the ONNX reference used int64 (e.g. Zipformer x_lens); QnnGraph casts on the box, AI Hub needs it done here
        ins = {k[3:]: [np.ascontiguousarray(z[k].astype(np.int32) if z[k].dtype == np.int64 else z[k])] for k in z.files if k.startswith("in_")}
        jobs[n] = hub.submit_inference_job(model=hub.get_job(jid).get_target_model(), device=dev, inputs=ins, name=f"s2s_infer_{n}")
        print(n, "inference job", jobs[n].job_id, {k: (v[0].shape, str(v[0].dtype)) for k, v in ins.items()}, flush=True)
    res = {}
    for n, j in jobs.items():
        st = j.wait()
        if not st.success: res[n] = dict(job=j.job_id, ok=False, message=st.message); print("FAILED", n, st.message, flush=True); continue
        out = j.download_output_data(); names = list(out.keys()); z = np.load(G(key_of(n))); ref = [z[f"out_{i}"] for i in range(len([k for k in z.files if k.startswith("out_")]))]
        got = [np.asarray(out[k][0]) for k in names]; ok, status, note = judge(key_of(n), got, ref)          # same rule as the on-device golden check
        res[n] = dict(job=j.job_id, ok=ok, output_names=names, outputs=status, note=note)
        print(("PASS " if ok else "FAIL ") + n, status, note, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print("all pass" if res and all(r.get("ok") for r in res.values()) else "SOME NOT PASSING", "->", a.out)


if __name__ == "__main__": main()
