# -*- coding: utf-8 -*-
"""MT-only evaluation on FLORES-200 devtest sentences (gold source text): chrF / BLEU, truncation count, calls. Compares NLLB modes.
 python tools/eval_mt.py --models models.dev.json --eval ../../s2s_npu/eval --n 25 --out logs/eval_mt.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sacrebleu
from s2s_edge.registry import Registry, DIRECTIONS

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--n", type=int, default=25); ap.add_argument("--out", default="logs/eval_mt.json"); ap.add_argument("--backend", default="cpu"); ap.add_argument("--dirs", default="")
a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); reg = Registry(a.models, a.backend); mt = reg.mt(); res = {}
man = {l: json.load(open(os.path.join(a.eval, f"manifest_{l}.json"), encoding="utf-8"))[: a.n] for l in ("vi", "en", "zh", "ko")}
for src, tgt in ([tuple(d.split("-")) for d in a.dirs.split(",")] if a.dirs else DIRECTIONS):
    hyps, refs, t0 = [], [], time.time(); c0, tr0 = mt.calls, mt.truncations
    for it in man[src]:
        hyps.append(mt(it["refs"][src], src, tgt)); refs.append(it["refs"][tgt])
    d = dict(n=len(hyps), chrF=round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2), BLEU=round(sacrebleu.corpus_bleu(hyps, [refs], tokenize={"zh": "zh", "ko": "ko-mecab"}.get(tgt, "13a")).score, 2) if tgt != "ko" else None,
             calls=mt.calls - c0, truncations=mt.truncations - tr0, seconds=round(time.time() - t0, 1), example=dict(src=man[src][0]["refs"][src], hyp=hyps[0], ref=refs[0]))
    res[f"{src}->{tgt}"] = d; print(src, tgt, {k: v for k, v in d.items() if k != "example"}, flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
