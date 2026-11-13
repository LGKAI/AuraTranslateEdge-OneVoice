# -*- coding: utf-8 -*-
"""ASR-only evaluation of the shipped ASR path (same objects as the pipeline: Registry -> Engine.prep -> AsrVi / AsrXx), so speed and accuracy
changes of the ASR stage can be measured without MT / TTS.
  * accuracy : corpus WER (vi, en) or CER (zh, ko) after the same normalisation as tools/eval_btc.py, plus "tail_err" = share of clips whose
               last reference word is deleted or substituted (the symptom of a segmenter that cuts the end of speech)
  * speed    : wall time of every ASR call and graph (NPU / ORT) time; with bucketed graphs the window chosen for each call is counted
Data: a FLEURS split (test.tsv + extracted wav folder) or a JSON manifest [{"path": ..., "text": ...}].
  python tools/eval_asr.py --models models/models.json --backend cpu --lang vi --fleurs /data/fleurs/vi_vn --n 200
  python tools/eval_asr.py --models models/models.json --lang en --manifest eval/manifest_en.json --seg vad --gain-db -30
Options that change the input (to stress the front end): --seg vad (pipeline segmenter, one ASR call per segment), --gain-db (scale, then
re-quantise to 16 bit like a quiet microphone), --pad-ms (-70 dBFS noise around the clip, like a push-to-talk recording)."""
import argparse, csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.audio import SR, load_audio, segment
from s2s_edge.registry import Registry
from s2s_edge.conversation import Engine
from s2s_edge.textnorm import normalize


def norm_text(s, lang):
    s = normalize(s, lang) if lang in ("vi", "en", "zh", "ko") else s
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def edits(ref, hyp, lang):
    """(edit distance, reference length) at word level (vi/en) or character level (zh/ko)."""
    import jiwer
    r, h = norm_text(ref, lang), norm_text(hyp, lang)
    if lang in ("zh", "ko"): r, h = list(r.replace(" ", "")), list(h.replace(" ", ""))
    else: r, h = r.split(), h.split()
    if not r: return 0, 0
    if not h: return len(r), len(r)
    o = jiwer.process_words(" ".join(r), " ".join(h))
    return o.substitutions + o.deletions + o.insertions, len(r)


def tail_deleted(ref, hyp, lang):
    """The last reference word (zh/ko: character) is not recognised: in the alignment it is deleted or substituted."""
    import jiwer
    r, h = norm_text(ref, lang), norm_text(hyp, lang)
    if lang in ("zh", "ko"): r, h = " ".join(r.replace(" ", "")), " ".join(h.replace(" ", ""))
    if not r: return False
    if not h: return True
    o = jiwer.process_words(r, h); last = len(r.split()) - 1
    return any(c.type != "equal" and c.ref_start_idx <= last < c.ref_end_idx for c in o.alignments[0])


def normalize_level_legacy(x, target_peak=0.5, below=0.2):
    """s2s_edge.audio.normalize_level before the active-RMS change (git 53d4f87), for A/B runs (--norm legacy)."""
    pk = float(np.abs(x).max()) if len(x) else 0.0
    return (x * (target_peak / pk)).astype(np.float32) if 1e-5 < pk < below else x


def segment_legacy(x, sr=SR, max_s=12.0, min_s=0.3, frame_ms=20, pad_ms=150, min_sil_ms=500, merge_gap_s=1.5, short_s=3.0):
    """s2s_edge.audio.segment before the hysteresis / join change (git 53d4f87), kept here for A/B runs (--seg vad_legacy)."""
    n = int(sr * frame_ms / 1000); nf = len(x) // n
    if nf == 0: return []
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    e = np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1) + 1e-12)
    if np.percentile(e, 95) < 3e-5: return []
    es = uniform_filter1d(e, 3)
    ceil = maximum_filter1d(es, int(3.0 * 1000 / frame_ms)); floor = np.maximum(minimum_filter1d(es, int(6.0 * 1000 / frame_ms)), 1e-6)
    thr = np.maximum.reduce([floor * 2.5, floor + 0.12 * (ceil - floor), np.full_like(e, 2e-5)])
    voiced = e > thr; k = max(1, int(min_sil_ms / frame_ms)); out = []; i = 0
    while i < nf:
        if not voiced[i]: i += 1; continue
        j = i; last = i
        while j < nf and j - last <= k:
            if voiced[j]: last = j
            j += 1
        out.append((i, last + 1)); i = j
    pad = int(sr * pad_ms / 1000); segs = []
    for a, b in out:
        a, b = max(0, a * n - pad), min(len(x), b * n + pad)
        while (b - a) / sr > max_s:
            lo = a + int(0.6 * max_s * sr); hi = a + int(max_s * sr); w = e[lo // n: hi // n]; cut = (lo // n + int(np.argmin(w))) * n
            segs.append((a, cut)); a = cut
        if (b - a) / sr >= min_s: segs.append((a, b))
    merged = []
    for a, b in segs:
        if merged:
            pa, pb = merged[-1]
            if (a - pb) / sr < merge_gap_s and (b - pa) / sr <= max_s and ((pb - pa) / sr < short_s or (b - a) / sr < short_s):
                merged[-1] = (pa, b); continue
        merged.append((a, b))
    return merged


def fleurs_items(root, n=0, max_s=0.0):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(root, "test.tsv"), encoding="utf-8")]
    wavdir = next((d for d in (os.path.join(root, "wav", "test"), os.path.join(root, "audio", "test"), os.path.join(root, "test"), os.path.join(root, "wav")) if os.path.isdir(d)), root)
    out = []
    for r in rows:
        p = os.path.join(wavdir, r[1])
        if not os.path.exists(p): continue
        if max_s and int(r[5]) / SR > max_s: continue
        out.append(dict(id=r[1], path=p, text=r[2]))
    return out[:n] if n else out


def requantise(w, gain_db):
    x = np.clip(w * (10 ** (gain_db / 20.0)), -1, 1)
    return (np.round(x * 32767) / 32768.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--backend", default="cpu"); ap.add_argument("--lang", required=True, choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--fleurs", default="", help="FLEURS language folder with test.tsv (+ extracted wavs)"); ap.add_argument("--manifest", default="")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--max-s", type=float, default=0.0, help="skip clips longer than this (s)")
    ap.add_argument("--seg", default="off", choices=["off", "vad", "vad_legacy"], help="off: whole clip -> one ASR call; vad: pipeline segmenter (s2s_edge.audio.segment), one call per segment; vad_legacy: the segmenter before the hysteresis change")
    ap.add_argument("--gain-db", type=float, default=0.0); ap.add_argument("--pad-ms", type=int, default=0)
    ap.add_argument("--seg-max-s", type=float, default=0.0, help="longest segment (default: s2s_edge.audio.SEG_MAX_S; legacy segmenter: 12)")
    ap.add_argument("--norm", default="pipeline", choices=["pipeline", "legacy", "none"], help="level normalisation before ASR: pipeline = Engine.prep as shipped")
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--out", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
    if a.fleurs: items = fleurs_items(a.fleurs, a.n, a.max_s)
    elif a.manifest:
        base = os.path.dirname(os.path.abspath(a.manifest)); items = json.load(open(a.manifest, encoding="utf-8"))
        items = [dict(id=it.get("id", os.path.basename(it["path"])), path=it["path"] if os.path.isabs(it["path"]) else os.path.join(base, it["path"]), text=it.get("text") or it.get("transcript") or it.get("src_text", "")) for it in items]
        items = items[: a.n] if a.n else items
    else: ap.error("--fleurs or --manifest")
    if not items: sys.exit("no audio found (FLEURS: extract audio/test.tar.gz next to test.tsv)")
    reg = Registry(a.models, a.backend); eng = Engine(reg); asr = eng.asr(a.lang)
    from s2s_edge.audio import SEG_MAX_S
    seg_max = a.seg_max_s or min(getattr(asr, "max_s", SEG_MAX_S), SEG_MAX_S)          # as pipeline.Translator
    prep = eng.prep if a.norm == "pipeline" else normalize_level_legacy if a.norm == "legacy" else (lambda v: v)
    graphs = lambda: [g for g in reg._cache.values()]
    if a.warmup and items:                                                             # first call of every window pays for allocation / HTP setup
        for _ in range(a.warmup):
            if hasattr(asr, "warmup"): asr.warmup()
            else: asr(prep(load_audio(items[0]["path"])[: int(3 * SR)]))
    rows = []; E = N = 0; rng = np.random.default_rng(0)
    for it in items:
        w = load_audio(it["path"])
        if a.gain_db: w = requantise(w, a.gain_db)
        if a.pad_ms:                                                    # low-level noise (-70 dBFS), not digital zeros: zero frames are far out of the
            z = lambda: (rng.standard_normal(int(SR * a.pad_ms / 1000)) * 10 ** (-70 / 20)).astype(np.float32)   # training distribution of both models
            w = np.concatenate([z(), w, z()])
        spans = segment(w, max_s=seg_max) if a.seg == "vad" else segment_legacy(w, max_s=a.seg_max_s or 12.0) if a.seg == "vad_legacy" else [(0, len(w))]
        texts = []; t_asr = 0.0; g0 = sum(g.total_s for g in graphs()); calls = []
        for s, e in spans:
            x = prep(w[s:e]); t = time.perf_counter(); texts.append(asr(x)); t_asr += time.perf_counter() - t
            calls.append(dict(asr.last) if getattr(asr, "last", None) else {})
        hyp = " ".join(x for x in texts if x).strip(); e_, n_ = edits(it["text"], hyp, a.lang); E += e_; N += n_
        rows.append(dict(id=it["id"], audio_s=round(len(w) / SR, 2), n_seg=len(spans), seg_s=round(sum(e - s for s, e in spans) / SR, 2), ref=it["text"], hyp=hyp,
                         err=round(e_ / max(n_, 1), 4), tail_err=tail_deleted(it["text"], hyp, a.lang), asr_s=round(t_asr, 4),
                         graph_s=round(sum(g.total_s for g in graphs()) - g0, 4), calls=calls))
        print(f"{it['id']} err={rows[-1]['err']:.2f} {rows[-1]['asr_s'] * 1000:.0f} ms | {hyp[:90]}", flush=True)
    t = [r["asr_s"] for r in rows]; aud = [r["audio_s"] for r in rows]
    summ = dict(tag=a.tag, lang=a.lang, n=len(rows), metric="CER" if a.lang in ("zh", "ko") else "WER", err=round(E / max(N, 1), 4),
                tail_err=round(float(np.mean([r["tail_err"] for r in rows])), 4) if rows else None, empty=sum(1 for r in rows if not r["hyp"]),
                asr_ms=dict(mean=round(1000 * float(np.mean(t)), 1), p50=round(1000 * float(np.median(t)), 1), p90=round(1000 * float(np.percentile(t, 90)), 1)) if t else None,
                rtf=round(float(np.sum(t) / max(np.sum(aud), 1e-9)), 4) if t else None, seg=a.seg, seg_max_s=a.seg_max_s, norm=a.norm, gain_db=a.gain_db, pad_ms=a.pad_ms, models=a.models, backend=a.backend)
    buckets = {}
    for r in rows:
        for c in r["calls"]:
            if "window" in c: buckets[c["window"]] = buckets.get(c["window"], 0) + 1
    if buckets: summ["windows_used"] = dict(sorted(buckets.items()))
    print(json.dumps(summ, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(summary=summ, rows=rows), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(os.path.splitext(a.out)[0] + ".csv", "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "calls"] if rows else ["id"], extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
