# -*- coding: utf-8 -*-
"""Can re-ranking the beam's candidates beat the beam's own choice? Dev split, 6 directions, NLLB-600M, the exact NllbBeam loop (tools/tune_decoding.Beam).
  1) oracle    : pick, per sentence, the candidate with the best sentence-chrF against the reference -> upper bound of any re-ranker
  2) channel   : noisy-channel re-ranking (Yee et al. 2019, arXiv 1908.05731 Sec. 2): score = forward + lam * log p(source | candidate) / |source|,
                 the channel model being NLLB itself in the reverse direction (no extra model on the device)
  3) roundtrip : back-translate each candidate greedily and keep the one whose back-translation has the best chrF to the source
Writes logs/rerank/{nbest_dev.json,result.json}. Resumable (n-best and scores are cached).
  python tools/rerank_experiments.py [--threads 4]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, sacrebleu
import tune_decoding as td

OUT = os.path.join("logs", "rerank"); os.makedirs(OUT, exist_ok=True)
L = {"vi": "vie_Latn", "en": "eng_Latn", "ko": "kor_Hang", "zh": "zho_Hans"}


class NBest(td.Beam):
    """Same search; returns every finished hypothesis (best first) with its length-normalised forward score."""

    def nbest(self, text, src, tgt, lp=1.0, nr=3, cap=2.2):
        best, fin = self.translate_all(text, src, tgt, lp=lp, nr=nr, cap=cap); seen, out = set(), []
        for sc, s in fin:
            if s not in seen: seen.add(s); out.append((sc, s))
        return out or [(0.0, best)], best


def sent_chrf(h, r, t):
    H, R = td.prep([h], [r], t); return sacrebleu.sentence_chrf(H[0], [R[0]]).score


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--split", default="dev"); ap.add_argument("--no_roundtrip", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ds = td.data(a.split); nb = NBest(); path = os.path.join(OUT, f"nbest_{a.split}.json"); cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for d, (src, ref) in ds.items():                                            # 1) n-best lists (B=4, shipped parameters)
        if d in cache: continue
        s, t = d.split("-"); t0 = time.time(); cache[d] = []
        for x in src: c, best = nb.nbest(x, s, t); cache[d].append(dict(cands=c, best=best))
        json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False); print(f"n-best {d}: {len(src)} sentences, {np.mean([len(e['cands']) for e in cache[d]]):.1f} candidates each ({time.time() - t0:.0f} s)", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M"); m = nb.m
    sc_path = os.path.join(OUT, f"scores_{a.split}.json"); sc = json.load(open(sc_path, encoding="utf-8")) if os.path.exists(sc_path) else {}

    @torch.no_grad()
    def channel(cand, source, s, t):                                            # log p(source | cand) / |source| with NLLB t -> s
        tok.src_lang, tok.tgt_lang = L[t], L[s]; b = tok(cand, text_target=source, return_tensors="pt")
        lab = b["labels"]; out = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=lab)
        return -float(out.loss)                                                 # mean token log-prob of the source (incl. lang code / eos)

    for d, (src, ref) in ds.items():                                            # 2) + 3) per-candidate scores
        if d in sc: continue
        s, t = d.split("-"); t0 = time.time(); sc[d] = []
        for x, e in zip(src, cache[d]):
            ch = [channel(c, x, s, t) for _, c in e["cands"]]
            rt = [0.0] * len(e["cands"]) if a.no_roundtrip else [sent_chrf(nb.translate(c, t, s, lp=1.0, nr=3, cap=2.2) if len(e["cands"]) > 1 else x, x, s) for _, c in e["cands"]]
            sc[d].append(dict(channel=ch, roundtrip=rt))
        json.dump(sc, open(sc_path, "w", encoding="utf-8"), ensure_ascii=False); print(f"scores {d} ({time.time() - t0:.0f} s)", flush=True)

    res = {}; lams = [0.0, 0.25, 0.5, 1.0, 2.0]
    print("\n| direction | beam (shipped) | oracle of n-best | channel lam=0.5 | channel lam=1 | round-trip | best lam on dev |"); print("|---|---|---|---|---|---|---|")
    for d, (src, ref) in ds.items():
        s, t = d.split("-"); E = cache[d]; S = sc[d]
        pick = lambda f: [e["cands"][int(np.argmax(f(e, z)))][1] for e, z in zip(E, S)]
        base = [e["best"] for e in E]
        oracle = [max(e["cands"], key=lambda c: sent_chrf(c[1], r, t))[1] for e, r in zip(E, ref)]
        chan = {lam: pick(lambda e, z, lam=lam: [c[0] + lam * ch for c, ch in zip(e["cands"], z["channel"])]) for lam in lams}
        rt = pick(lambda e, z: z["roundtrip"])
        row = dict(beam=td.chrf(base, ref, t), oracle=td.chrf(oracle, ref, t), roundtrip=td.chrf(rt, ref, t), **{f"channel_{lam}": td.chrf(chan[lam], ref, t) for lam in lams})
        bl = max(lams, key=lambda lam: row[f"channel_{lam}"]); res[d] = dict(row, best_lam=bl)
        print(f"| {d} | {row['beam']:.1f} | {row['oracle']:.1f} | {row['channel_0.5']:.1f} | {row['channel_1.0']:.1f} | {row['roundtrip']:.1f} | {bl} ({row[f'channel_{bl}']:.1f}) |", flush=True)
    mean = lambda k: float(np.mean([res[d][k] for d in res]))
    print(f"\nmean: beam {mean('beam'):.2f} | oracle {mean('oracle'):.2f} | " + " | ".join(f"channel lam={lam} {mean(f'channel_{lam}'):.2f}" for lam in lams) + f" | round-trip {mean('roundtrip'):.2f}")
    json.dump(res, open(os.path.join(OUT, f"result_{a.split}.json"), "w"), indent=1)


if __name__ == "__main__": main()
