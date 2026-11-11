# -*- coding: utf-8 -*-
"""Is method B really better than method A on the sample set? Paired bootstrap over the clips both runs share.
With 30 to 180 clips per direction a +1 chrF change is often noise; this prints the delta, a 95 % interval and how often B wins over resamples.
  python tools/compare_runs.py logs/A/btc_vi-ko.json logs/B/btc_vi-ko.json [--metric heard|mt_e2e|mt_gold|src_err] [--clean]
Each file is an eval_btc.py output (one or more directions). Decision rule used in the plan: adopt B only if the 95 % interval is above 0
(or below 0 for src_err) on the dev split, then confirm once on holdout.
Text normalisation and Korean character-level scoring mirror tools/eval_btc.py (kept in sync by hand)."""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sacrebleu, jiwer
from s2s_edge.textnorm import normalize


def norm_text(s, lang):
    s = normalize(s, lang) if lang in ("vi", "en", "zh", "ko") else s
    s = re.sub(r"[^\w\s]", " ", s.lower()); return re.sub(r"\s+", " ", s).strip()


def prep(h, r, lang):
    H, R = [norm_text(x, lang) for x in h], [norm_text(x, lang) for x in r]
    if lang == "ko": H, R = [" ".join(x.replace(" ", "")) for x in H], [" ".join(x.replace(" ", "")) for x in R]
    return H, R


def chrf(H, R): return sacrebleu.corpus_chrf(H, [R]).score


def err(H, R, lang):
    keep = [i for i, r in enumerate(R) if r]; H, R = [H[i] or "_" for i in keep], [R[i] for i in keep]
    if lang in ("zh", "ko"): return 100 * jiwer.cer([r.replace(" ", "") for r in R], [h.replace(" ", "") or "_" for h in H])
    return 100 * jiwer.wer(R, H)


def flagged(x): return x.get("flag") or (x.get("src_err") is not None and x["src_err"] > 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--metric", default="heard", choices=["heard", "mt_e2e", "mt_gold", "src_err"])
    ap.add_argument("--clean", action="store_true", help="drop rows flagged as mislabelled (as eval_btc's 'clean' summary)"); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); A, B = json.load(open(a.a, encoding="utf-8")), json.load(open(a.b, encoding="utf-8")); rng = np.random.default_rng(a.seed)
    for d in [k for k in A if k in B]:
        src, tgt = d.split("-"); ra = {str(x["id"]): x for x in A[d]["rows"]}; rb = {str(x["id"]): x for x in B[d]["rows"]}
        ids = [i for i in ra if i in rb and not (a.clean and (flagged(ra[i]) or flagged(rb[i])))]
        if not ids: print(d, "no shared rows"); continue
        if a.metric == "src_err":
            R, _ = prep([ra[i]["src_text"] for i in ids], [ra[i]["src_text"] for i in ids], src)
            Ha, _ = prep([ra[i]["asr"] for i in ids], [ra[i]["asr"] for i in ids], src); Hb, _ = prep([rb[i]["asr"] for i in ids], [rb[i]["asr"] for i in ids], src)
            f = lambda idx, H: err([H[k] for k in idx], [R[k] for k in idx], src); better = "lower"
        else:
            if not any(ra[i].get(a.metric) for i in ids) or not any(rb[i].get(a.metric) for i in ids): print(d, f"metric {a.metric} missing in one run"); continue
            Ha, R = prep([ra[i].get(a.metric, "") for i in ids], [ra[i]["ref"] for i in ids], tgt); Hb, _ = prep([rb[i].get(a.metric, "") for i in ids], [rb[i]["ref"] for i in ids], tgt)
            f = lambda idx, H: chrf([H[k] for k in idx], [R[k] for k in idx]); better = "higher"
        full = list(range(len(ids))); sa, sb = f(full, Ha), f(full, Hb); deltas = []
        for _ in range(a.n):
            idx = rng.integers(0, len(ids), len(ids)); deltas.append(f(idx, Hb) - f(idx, Ha))
        deltas = np.array(deltas); lo, hi = np.percentile(deltas, [2.5, 97.5]); win = float((deltas > 0).mean() if better == "higher" else (deltas < 0).mean())
        verdict = "B better" if (lo > 0 if better == "higher" else hi < 0) else ("B worse" if (hi < 0 if better == "higher" else lo > 0) else "no clear difference")
        print(f"{d}: n={len(ids)} {a.metric} A={sa:.2f} B={sb:.2f} delta={sb - sa:+.2f} 95% [{lo:+.2f}, {hi:+.2f}] B wins {100 * win:.0f}% of resamples -> {verdict}")


if __name__ == "__main__": main()
