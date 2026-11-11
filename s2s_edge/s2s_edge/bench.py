# -*- coding: utf-8 -*-
"""Benchmark harness: runs the translator over a list of utterances, records per-stage and end-to-end timing, graph (NPU) time,
peak RSS, model sizes and (optionally) translation quality. Writes JSON + CSV + a human readable log into --logs (raw evidence).

  python -m s2s_edge.bench --src vi --tgt en --models models/models.json --manifest data/bench_vi.json --runs 3 --logs logs/

manifest: JSON list of {"path": "x.wav", "ref": "optional reference translation", "src_text": "optional reference transcript"}"""
import argparse, csv, json, os, platform, statistics, sys, time
import numpy as np
from .audio import load_audio, save_wav
from .registry import Registry
from .pipeline import Translator


def peak_rss_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmHWM"): return int(l.split()[1]) / 1024.0
    except Exception: pass
    try:
        import psutil; return psutil.Process().memory_info().peak_wset / 1e6 if hasattr(psutil.Process().memory_info(), "peak_wset") else psutil.Process().memory_info().rss / 1e6
    except Exception: return float("nan")


def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--models", default="models/models.json"); ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "npu"))
    ap.add_argument("--manifest", required=True); ap.add_argument("--runs", type=int, default=3); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--logs", default="logs"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--save-wav", action="store_true")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(a.logs, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S"); tag = f"{a.src}-{a.tgt}_{a.backend}_{stamp}"
    reg = Registry(a.models, a.backend); t0 = time.perf_counter(); tr = Translator(reg, a.src, a.tgt); load_s = time.perf_counter() - t0
    files = reg.files(reg.components_for(a.src, a.tgt)); size_mb = sum(os.path.getsize(f) for f in files if os.path.exists(f)) / 1e6
    items = json.load(open(a.manifest, encoding="utf-8")); items = items[: a.limit] if a.limit else items; base = os.path.dirname(os.path.abspath(a.manifest))
    rows = []; hyps = []; refs = []
    for it in items:
        p = it["path"] if os.path.isabs(it["path"]) else os.path.join(base, it["path"]); wav = load_audio(p)
        for k in range(a.warmup + a.runs):
            res = tr.translate(wav)
            if k < a.warmup: continue
            tot = sum(r["timing"]["total_s"] for r in res); gr = sum(r["timing"]["graph_s"] for r in res)
            rows.append(dict(file=os.path.basename(p), run=k - a.warmup, audio_s=len(wav) / 16000, segments=len(res), e2e_s=tot, asr_s=sum(r["timing"]["asr_s"] for r in res), mt_s=sum(r["timing"]["mt_s"] for r in res),
                             tts_s=sum(r["timing"]["tts_s"] for r in res), graph_s=gr, host_s=tot - gr, rtf=tot / (len(wav) / 16000), asr=" ".join(r["asr"] for r in res), mt=" ".join(r["mt"] for r in res),
                             first_seg_s=res[0]["timing"]["total_s"] if res else 0.0))
        hyps.append(rows[-1]["mt"]); refs.append(it.get("ref"))
        if a.save_wav and res: save_wav(os.path.join(a.logs, f"{tag}_{os.path.basename(p)}.wav"), np.concatenate([r["wav"] for r in res]), res[0]["sr"])
        print(f"{os.path.basename(p)}: {rows[-1]['asr']!r} -> {rows[-1]['mt']!r}  e2e {rows[-1]['e2e_s']:.2f}s (graph {rows[-1]['graph_s']:.2f}s) rtf {rows[-1]['rtf']:.2f}", flush=True)
    quality = {}
    if any(refs):
        try:
            import sacrebleu
            pairs = [(h, r) for h, r in zip(hyps, refs) if r]; quality["chrF"] = sacrebleu.corpus_chrf([h for h, _ in pairs], [[r for _, r in pairs]]).score; quality["BLEU"] = sacrebleu.corpus_bleu([h for h, _ in pairs], [[r for _, r in pairs]]).score; quality["n"] = len(pairs)
        except Exception as e: quality["error"] = str(e)
    e2e = [r["e2e_s"] for r in rows]
    summary = dict(tag=tag, direction=f"{a.src}->{a.tgt}", backend=a.backend, device=platform.platform(), python=platform.python_version(), n_utterances=len(items), runs=a.runs, load_models_s=load_s,
                   model_files=[os.path.basename(f) for f in files], model_size_mb=size_mb, peak_rss_mb=peak_rss_mb(),
                   e2e_s=dict(mean=statistics.mean(e2e), median=statistics.median(e2e), p95=pct(e2e, 95), max=max(e2e)),
                   graph_s_mean=statistics.mean(r["graph_s"] for r in rows), host_s_mean=statistics.mean(r["host_s"] for r in rows), rtf_mean=statistics.mean(r["rtf"] for r in rows),
                   stage_mean_s=dict(asr=statistics.mean(r["asr_s"] for r in rows), mt=statistics.mean(r["mt_s"] for r in rows), tts=statistics.mean(r["tts_s"] for r in rows)),
                   mt_calls=tr.mt.calls, mt_truncations=tr.mt.truncations, quality=quality)
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(a.logs, f"bench_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(a.logs, f"bench_{tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__": main()
