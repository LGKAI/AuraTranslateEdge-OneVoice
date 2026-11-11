# -*- coding: utf-8 -*-
"""Grid search of the MT beam-search parameters (length_penalty, no_repeat_ngram, length cap) on the sample-set DEV split, then one check on HOLDOUT.
The decoder loop is a line-by-line copy of s2s_edge/mt.py NllbBeam._run (pruned 139k vocabulary, banned n-grams, length-normalised finished
hypotheses, the same early stop and runaway guard), but driven by the Hugging Face NLLB-600M model with a KV cache so a config takes minutes on the
laptop CPU instead of hours with the ONNX graphs. `--check` first verifies that it reproduces the shipped ONNX beam decoder on a few sentences.
  python tools/tune_decoding.py --check           (parity with the ONNX graphs)
  python tools/tune_decoding.py --grid            (dev split, results in logs/tune_decoding/, resumable)
  python tools/tune_decoding.py --holdout lp=1.0,nr=3,cap=2.2 lp=0.8,nr=3,cap=2.2
Only the gold source text is translated (MT in isolation); refs and chrF as in tools/eval_btc.py (Korean at character level)."""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, openpyxl, sacrebleu
from s2s_edge.mt import Nllb, CODES
from s2s_edge.registry import ASSETS

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(8)
OUT = os.path.join("logs", "tune_decoding"); os.makedirs(OUT, exist_ok=True)
R = "C:/Users/Admin/Downloads/onevoice_btc_data"
FILES = {"vi-en": ("Vie-Eng dataset/Vie-Eng/OneVoice_V_to_E_samples_100.xlsx", 1, 2), "en-vi": ("Vie-Eng dataset/Eng-Vie/OneVoice_E_to_V_samples_100.xlsx", 1, 2),
         "vi-ko": ("Vie-Kor dataset/OneVoice_V_to_K_samples_180.xlsx", 1, 2), "ko-vi": ("Vie-Kor dataset/OneVoice_V_to_K_samples_180.xlsx", 2, 1),
         "vi-zh": ("Vie-Man dataset/OneVoice_V_to_M_samples_180.xlsx", 1, 2), "zh-vi": ("Vie-Man dataset/OneVoice_V_to_M_samples_180.xlsx", 2, 1)}


def rows(p): r = list(openpyxl.load_workbook(p, read_only=True).active.values); return [x for x in r[1:] if x and x[0]]
strip = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def prep(h, r, t):
    H, Rr = [strip(x) for x in h], [strip(x) for x in r]
    if t in ("ko", "zh"): H, Rr = [" ".join(x.replace(" ", "")) for x in H], [" ".join(x.replace(" ", "")) for x in Rr]
    return H, Rr


def chrf(h, r, t): H, Rr = prep(h, r, t); return sacrebleu.corpus_chrf(H, [Rr]).score


def data(split):
    sp = json.load(open("eval/btc_split.json", encoding="utf-8"))["by_dir"]; out = {}
    for d, (f, cs, cr) in FILES.items():
        keep = set(sp[d][split]); rs = [r for r in rows(os.path.join(R, f)) if str(r[0]) in keep]
        out[d] = ([str(r[cs]).strip() for r in rs], [str(r[cr]).strip() for r in rs])
    return out


class Beam:
    """NllbBeam._run semantics on the HF model (S=64, Tm=96, B=4 by default)."""

    def __init__(self, S=64, Tm=96, B=4):
        from transformers import AutoModelForSeq2SeqLM
        self.m = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").eval(); self.nl = Nllb(None, ASSETS, S=S)
        self.S, self.Tm, self.B = S, Tm, B; self.keep = np.load(os.path.join(ASSETS, "nllb_keep.npy")).astype(np.int64); self.keep_t = torch.from_numpy(self.keep)
        self.inv = np.full(int(self.keep.max()) + 1, -1, np.int64); self.inv[self.keep] = np.arange(len(self.keep))

    @staticmethod
    def banned(toks, n):
        if n < 2 or len(toks) < n - 1: return []
        pre = tuple(toks[-(n - 1):]); return [toks[i + n - 1] for i in range(len(toks) - n + 1) if tuple(toks[i: i + n - 1]) == pre]

    @torch.no_grad()
    def run(self, toks, src, tgt, lp=1.0, nr=3, cap=2.2, add=14, return_all=False):
        B, Tm = self.B, self.Tm; seq = [self.nl.lang[CODES[src]]] + toks + [2]; lang = self.nl.lang[CODES[tgt]]
        ids = torch.tensor([seq]); enc = self.m.model.encoder(input_ids=ids).last_hidden_state.expand(B, -1, -1)
        alive = [[] for _ in range(B)]; ascore = np.array([0.0] + [-1e9] * (B - 1)); fin = []; cur = torch.full((B, 1), 2); past = None
        tmax = min(Tm - 1, int(cap * len(toks)) + add)
        for t in range(tmax):
            o = self.m.model.decoder(input_ids=cur, encoder_hidden_states=enc, past_key_values=past, use_cache=True); past = o.past_key_values
            lg = self.m.lm_head(o.last_hidden_state[:, -1])[:, self.keep_t].double().numpy()
            if t == 0: cur = torch.full((B, 1), lang); continue
            lpv = lg - (lg.max(-1, keepdims=True) + np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)))
            for bi in range(B):
                for tid in self.banned(alive[bi], nr):
                    if tid < len(self.inv) and self.inv[tid] >= 0: lpv[bi, self.inv[tid]] = -1e9
            cand = (ascore[:, None] + lpv).reshape(-1); top = np.argpartition(-cand, 2 * B)[: 2 * B]; top = top[np.argsort(-cand[top])]
            new_alive, new_score, parents = [], [], []
            for rank, c in enumerate(top):
                bi, vi = divmod(int(c), lpv.shape[1]); tokid = int(self.keep[vi])
                if tokid == 2:
                    if rank < B and ascore[bi] > -1e8: fin.append((float(cand[c]) / ((len(alive[bi]) + 1) ** lp), alive[bi] + [2]))
                else:
                    if len(new_alive) < B and ascore[bi] > -1e8: new_alive.append(alive[bi] + [tokid]); new_score.append(float(cand[c])); parents.append(bi)
            if not new_alive: break
            while len(new_alive) < B: new_alive.append(new_alive[-1]); new_score.append(-1e9); parents.append(parents[-1])
            alive = new_alive; ascore = np.array(new_score); idx = torch.tensor(parents)
            past = self._reorder(past, idx)
            cur = torch.tensor([[x[-1]] for x in alive])
            if len(fin) >= B:
                best_alive = ascore.max() / ((len(alive[0]) + 1) ** lp)
                if best_alive <= min(f[0] for f in sorted(fin, reverse=True)[:B]): break
        body = max(fin, key=lambda f: f[0])[1] if fin else alive[int(ascore.argmax())]
        best = self.nl.decode(body[: body.index(2)] if 2 in body else body)
        if return_all:                                                          # n-best for re-ranking experiments: finished hypotheses, best first
            dec = lambda b: self.nl.decode(b[: b.index(2)] if 2 in b else b)
            return best, [(float(sc), dec(b)) for sc, b in sorted(fin, key=lambda f: -f[0])]
        return best

    def translate_all(self, text, src, tgt, **kw):
        toks = self.nl.encode(text)[: self.S - 2]; return self.run(toks, src, tgt, return_all=True, **kw)

    @staticmethod
    def _reorder(past, idx):
        if hasattr(past, "self_attention_cache"):                               # EncoderDecoderCache
            for c in (past.self_attention_cache, past.cross_attention_cache):
                for i in range(len(c.key_cache)): c.key_cache[i] = c.key_cache[i].index_select(0, idx); c.value_cache[i] = c.value_cache[i].index_select(0, idx)
            return past
        return tuple(tuple(x.index_select(0, idx) for x in layer) for layer in past)

    def translate(self, text, src, tgt, **kw):
        toks = self.nl.encode(text)[: self.S - 2]; return self.run(toks, src, tgt, **kw)


def parse(s): kv = dict(x.split("=") for x in s.split(",")); return dict(lp=float(kv.get("lp", 1.0)), nr=int(kv.get("nr", 3)), cap=float(kv.get("cap", 2.2)))
def name(c): return f"lp={c['lp']},nr={c['nr']},cap={c['cap']}"


def evaluate(bm, ds, cfg):
    res = {}
    for d, (src, ref) in ds.items():
        s, t = d.split("-"); hyp = [bm.translate(x, s, t, **cfg) for x in src]; res[d] = dict(chrF=round(chrf(hyp, ref, t), 2), hyp=hyp)
    res["mean"] = round(float(np.mean([res[d]["chrF"] for d in ds])), 2); return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--grid", action="store_true"); ap.add_argument("--holdout", nargs="*")
    a = ap.parse_args(); bm = Beam()
    if a.check:
        from s2s_edge.registry import Registry
        ref = Registry("models.dev_beam.json", "cpu").mt(); ok = 0; tests = [("vi", "en", "Anh cần thêm dây không?"), ("vi", "ko", "Không có phòng khách hướng Nam à?"),
                                                                              ("ko", "vi", "회의는 세 시에 시작합니다."), ("vi", "zh", "Tôi đã sống ở Bắc Kinh hai năm."), ("en", "vi", "Please wear your helmet before entering the factory.")]
        for s, t, x in tests:
            g, h = ref(x, s, t), bm.translate(x, s, t); ok += g == h; print(("same " if g == h else "DIFF ") + f"{s}->{t}: onnx={g} | torch={h}", flush=True)
        print(f"parity {ok}/{len(tests)}"); return
    if a.grid:
        ds = data("dev"); path = os.path.join(OUT, "grid_dev.json"); res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        grid = [dict(lp=lp, nr=3, cap=2.2) for lp in (1.0, 0.6, 0.8, 1.2, 1.4)] + [dict(lp=1.0, nr=nr, cap=2.2) for nr in (0, 4)] + [dict(lp=1.0, nr=3, cap=c) for c in (1.6, 3.0)]
        for cfg in grid:
            if name(cfg) in res: continue
            t0 = time.time(); r = evaluate(bm, ds, cfg); res[name(cfg)] = r; json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{name(cfg):22s} mean {r['mean']:.2f} | " + " ".join(f"{d} {r[d]['chrF']:.1f}" for d in ds) + f"  ({time.time() - t0:.0f} s)", flush=True)
        base = res[name(dict(lp=1.0, nr=3, cap=2.2))]["mean"]; print("\nDEV, sorted (delta vs default lp=1.0,nr=3,cap=2.2):")
        for k, r in sorted(res.items(), key=lambda kv: -kv[1]["mean"]): print(f"  {k:22s} {r['mean']:.2f} ({r['mean'] - base:+.2f})")
        return
    if a.holdout:
        ds = data("holdout"); cfgs = [parse(x) for x in a.holdout]; out = {name(c): evaluate(bm, ds, c) for c in cfgs}; json.dump(out, open(os.path.join(OUT, "holdout.json"), "w", encoding="utf-8"), ensure_ascii=False)
        b, n = cfgs[0], cfgs[1:]
        for c in cfgs: print(f"HOLDOUT {name(c):22s} mean {out[name(c)]['mean']:.2f} | " + " ".join(f"{d} {out[name(c)][d]['chrF']:.1f}" for d in ds), flush=True)
        rng = np.random.default_rng(0)
        for c in n:                                                             # paired bootstrap per direction vs the first config
            for d, (src, ref) in ds.items():
                t = d.split("-")[1]; A, Rr = prep(out[name(b)][d]["hyp"], ref, t); Bh, _ = prep(out[name(c)][d]["hyp"], ref, t); dl = []
                for _ in range(1000):
                    i = rng.integers(0, len(Rr), len(Rr)); dl.append(sacrebleu.corpus_chrf([Bh[k] for k in i], [[Rr[k] for k in i]]).score - sacrebleu.corpus_chrf([A[k] for k in i], [[Rr[k] for k in i]]).score)
                lo, hi = np.percentile(dl, [2.5, 97.5]); print(f"  {name(c)} vs {name(b)} {d}: {np.mean(dl):+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)


if __name__ == "__main__": main()
