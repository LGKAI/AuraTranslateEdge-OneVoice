# -*- coding: utf-8 -*-
"""NLLB-200-distilled-600M, one static graph (src_ids[1,32] i32, tgt_lang[1] i32 -> out_ids[1,16] i32 = [lang, y1..y15]).
Host: SentencePiece tokenize / detokenize and ADAPTIVE CHUNKING: a source piece whose translation is cut by the 15-token limit
(no </s> in output) is split in two and translated again, so long sentences are complete (costs extra graph calls)."""
import json, os, re, numpy as np

S, T = 32, 16
CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
SENT_END = re.compile(r"(?<=[\.\!\?;。！？；\n])\s*")
CLAUSE = re.compile(r"(?<=[,，、:：])\s*")


class Nllb:
    def __init__(self, graph, assets_dir, S=S):
        import sentencepiece as spm
        self.S = S; self.g = graph; self.sp = spm.SentencePieceProcessor(model_file=os.path.join(assets_dir, "nllb_spm.model"))
        self.ids = json.load(open(os.path.join(assets_dir, "nllb_ids.json"), encoding="utf-8")); self.lang = self.ids["lang"]
        self.calls = 0; self.truncations = 0

    def encode(self, text):
        return [3 if p == 0 else p + 1 for p in self.sp.encode(text)]            # fairseq offset: spm id + 1, <unk> -> 3

    def decode(self, ids):
        keep = [i - 1 for i in ids if i > 3 and i < 256001]
        return self.sp.decode(keep)

    def _run(self, toks, src, tgt):
        a = np.ones((1, self.S), np.int32); seq = [self.lang[CODES[src]]] + toks + [2]; a[0, :len(seq)] = seq
        out = np.asarray(self.g(src_ids=a, tgt_lang=np.array([self.lang[CODES[tgt]]], np.int32))[0]).reshape(-1)
        self.calls += 1; body = [int(v) for v in out[1:]]
        return body, (2 in body)

    def _split(self, text, src):
        for rx in (SENT_END, CLAUSE):
            parts = [p for p in rx.split(text) if p.strip()]
            if len(parts) > 1: return parts
        joiner = "" if src in ("zh",) else " "; units = list(text) if src in ("zh", "ko") and " " not in text.strip() else text.split()
        if len(units) < 2: return [text]
        h = len(units) // 2; return [joiner.join(units[:h]), joiner.join(units[h:])]

    def _tr(self, text, src, tgt, depth=0):
        text = text.strip()
        if not text: return ""
        toks = self.encode(text)
        if len(toks) <= self.S - 2:
            body, done = self._run(toks, src, tgt)
            if done or depth > 6:
                if not done: self.truncations += 1
                return self.decode(body[: body.index(2)] if 2 in body else body)
        parts = self._split(text, src)
        if len(parts) == 1:                                        # cannot split further: accept the truncated output
            self.truncations += 1; body, _ = self._run(toks[: self.S - 2], src, tgt); return self.decode(body[: body.index(2)] if 2 in body else body)
        sep = "" if tgt == "zh" else " "
        return sep.join(x for x in (self._tr(p, src, tgt, depth + 1) for p in parts) if x)

    def __call__(self, text, src, tgt):
        sents = [p for p in SENT_END.split(text.strip()) if p.strip()]
        sep = "" if tgt == "zh" else " "
        return sep.join(x for x in (self._tr(s, src, tgt) for s in sents) if x)

    def translate_atomic(self, text, src, tgt):
        """Translate an already segmented span without splitting its decimal points."""
        if len(self.encode(text)) > self.S - 2:
            raise ValueError('Prepared MT span exceeds graph input length')
        return self._tr(text, src, tgt)


class NllbSplit(Nllb):
    """Same host logic, but the model is two static graphs (encoder, one decoder step with static KV cache) and the greedy decode loop runs on the host.
    No 15-token limit (up to Tm-1 = 47 target tokens), small files (no unrolled copies of the decoder weights)."""
    def __init__(self, enc, dec, assets_dir, S=32, Tm=48, L=12, H=16, dk=64):
        # L/H/dk = decoder layers / heads / head size of the KV cache: 12/16/64 for NLLB-600M, 24/16/64 for NLLB-1.3B (models.json "L")
        Nllb.__init__(self, None, assets_dir, S=S); self.enc = enc; self.dec = dec; self.L, self.H, self.dk = L, H, dk
        self.Tm = Tm; self.steps = 0

    def _run(self, toks, src, tgt):
        Tm, L, H, dk = self.Tm, self.L, self.H, self.dk
        a = np.ones((1, self.S), np.int32); seq = [self.lang[CODES[src]]] + toks + [2]; a[0, :len(seq)] = seq; lang = self.lang[CODES[tgt]]
        ck, cv, bias = [np.asarray(x) for x in self.enc(src_ids=a)]
        sk = np.zeros((L, H, Tm, dk), np.float32); sv = sk.copy(); tok = 2; out = []
        for t in range(Tm - 1):
            oh = np.zeros(Tm, np.float32); oh[t] = 1.0; mask = np.where(np.arange(Tm) <= t, 0.0, -1e4).astype(np.float32)
            nxt, sk, sv = self.dec(tok=np.array([tok], np.int32), oh=oh, mask=mask, sk=sk, sv=sv, ck=ck, cv=cv, bias=bias)
            if np.asarray(nxt).size != 1 or np.asarray(sk).size != L * H * Tm * dk:
                raise RuntimeError(f"decoder output layout unexpected: token {np.asarray(nxt).shape}, kv {np.asarray(sk).shape} (check \"out\" in models.json)")
            nxt = int(np.asarray(nxt).reshape(-1)[0]); sk = np.asarray(sk); sv = np.asarray(sv); self.steps += 1
            tok = lang if t == 0 else nxt; out.append(tok)
            if t > 0 and nxt == 2: break
        self.calls += 1; body = out[1:]
        return body, (2 in body)


class NllbBeam(NllbSplit):
    """Beam search (default 4 beams) over the batched decoder-step graph. The graph returns logits for the pruned vocabulary; log-softmax, top-k, beam
    bookkeeping and the reordering of the per-beam KV caches happen on the host (numpy). Same length-normalised scoring idea as Hugging Face generate()."""

    def __init__(self, enc, decB, assets_dir, S=64, Tm=96, B=4, length_penalty=1.0, no_repeat_ngram=3, L=12, H=16, dk=64):
        NllbSplit.__init__(self, enc, decB, assets_dir, S=S, Tm=Tm, L=L, H=H, dk=dk); self.B = B; self.lp = length_penalty; self.nr = no_repeat_ngram
        self.keep = np.load(os.path.join(assets_dir, "nllb_keep.npy")).astype(np.int64)                  # pruned id -> original id
        self.inv = np.full(int(self.keep.max()) + 1, -1, np.int64); self.inv[self.keep] = np.arange(len(self.keep))

    def _banned(self, toks):
        """token ids that would repeat an n-gram already generated (stops degenerate loops such as 'the number of RSBs, the number of RSBs, ...')."""
        n = self.nr
        if n < 2 or len(toks) < n - 1: return []
        pre = tuple(toks[-(n - 1):]); return [toks[i + n - 1] for i in range(len(toks) - n + 1) if tuple(toks[i: i + n - 1]) == pre]

    def _run(self, toks, src, tgt):
        Tm, L, H, dk, B = self.Tm, self.L, self.H, self.dk, self.B
        a = np.ones((1, self.S), np.int32); seq = [self.lang[CODES[src]]] + toks + [2]; a[0, :len(seq)] = seq; lang = self.lang[CODES[tgt]]
        ck, cv, bias = [np.asarray(x) for x in self.enc(src_ids=a)]
        sk = np.zeros((L, B, H, Tm, dk), np.float32); sv = sk.copy(); alive = [[] for _ in range(B)]; ascore = np.array([0.0] + [-1e9] * (B - 1)); fin = []         # fin: (norm_score, tokens)
        cur = np.full(B, 2, np.int32); tmax = min(Tm - 1, int(2.2 * len(toks)) + 14)                  # runaway guard: target <= 2.2 x source + 14 tokens
        for t in range(tmax):
            oh = np.zeros(Tm, np.float32); oh[t] = 1.0; mask = np.where(np.arange(Tm) <= t, 0.0, -1e4).astype(np.float32)
            lg, sk, sv = self.dec(tok=cur, oh=oh, mask=mask, sk=sk, sv=sv, ck=ck, cv=cv, bias=bias); self.steps += 1
            sk = np.asarray(sk); sv = np.asarray(sv); lg = np.asarray(lg)
            if lg.size != B * len(self.keep) or sk.size != L * B * H * Tm * dk:                      # wrong output order / wrong graph: fail loudly -> MtFallback
                raise RuntimeError(f"beam decoder output layout unexpected: logits {lg.shape}, kv {sk.shape} (check \"out\" in models.json)")
            lg = lg.reshape(B, -1).astype(np.float64)
            if t == 0: cur = np.full(B, lang, np.int32); continue                                    # decoder start; next input is the forced target-language token
            lp = lg - (lg.max(-1, keepdims=True) + np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)))
            for bi in range(B):
                for tid in self._banned(alive[bi]):
                    if tid < len(self.inv) and self.inv[tid] >= 0: lp[bi, self.inv[tid]] = -1e9
            cand = (ascore[:, None] + lp).reshape(-1); top = np.argpartition(-cand, 2 * B)[: 2 * B]; top = top[np.argsort(-cand[top])]
            new_alive, new_score, parents = [], [], []
            for rank, c in enumerate(top):
                bi, vi = divmod(int(c), lp.shape[1]); tokid = int(self.keep[vi])
                if tokid == 2:
                    if rank < B and ascore[bi] > -1e8: fin.append((float(cand[c]) / ((len(alive[bi]) + 1) ** self.lp), alive[bi] + [2]))
                else:
                    if len(new_alive) < B and ascore[bi] > -1e8: new_alive.append(alive[bi] + [tokid]); new_score.append(float(cand[c])); parents.append(bi)
            if not new_alive: break
            while len(new_alive) < B: new_alive.append(new_alive[-1]); new_score.append(-1e9); parents.append(parents[-1])
            alive = new_alive; ascore = np.array(new_score); sk = sk[:, parents]; sv = sv[:, parents]; cur = np.array([x[-1] for x in alive], np.int32)
            if len(fin) >= B:
                best_alive = ascore.max() / ((len(alive[0]) + 1) ** self.lp)
                if best_alive <= min(f[0] for f in sorted(fin, reverse=True)[:B]): break
        self.calls += 1
        if fin: body = max(fin, key=lambda f: f[0])[1]; return body, True
        return alive[int(ascore.argmax())], False


class MtFallback:
    """MT with automatic degradation: decoder candidates are tried from the largest beam down to greedy, e.g.
    mt_decB4 (4 beams) -> mt_decB (2 beams) -> mt_dec (greedy). A candidate is dropped (its graph released to free HTP memory) when building it,
    the probe translation, or any later translation raises (QNN_COMMON_ERROR_MEM_ALLOC, a context the runtime rejects, a wrong output layout ...);
    the failing sentence is then translated again with the next candidate, so the caller never sees the error unless every candidate fails.
    Everything else (calls, truncations, steps ...) is delegated to the active decoder. `active` names it; `failures` lists what was dropped and why."""

    def __init__(self, registry, candidates, build, log=None, probe=True):
        import sys
        self.reg = registry; self.cands = list(candidates); self.build = build; self.impl = None; self.active = None; self.failures = []
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True)); self._next(probe)

    def _drop(self, key, err):
        self.failures.append((key, repr(err)[:300])); self.log(f"[s2s] MT decoder {key} failed: {repr(err)[:200]} -> released, falling back")
        self.impl = None; self.reg.release(key)

    def _next(self, probe):
        while self.cands:
            k = self.cands.pop(0)
            try:
                impl = self.build(k)
                if probe: impl("xin chào các bạn", "vi", "en")                    # allocates and runs every graph once: OOM shows up here, not mid-demo
                self.impl, self.active = impl, k; self.reg.mt_active = k; self.log(f"[s2s] MT decoder in use: {k}"); return
            except Exception as e:
                self._drop(k, e)
        raise RuntimeError(f"no MT decoder candidate works: {self.failures}")

    def __call__(self, text, src, tgt):
        while True:
            try: return self.impl(text, src, tgt)
            except Exception as e:
                self._drop(self.active, e); self._next(probe=False)            # raises when nothing is left

    def translate_atomic(self, text, src, tgt):
        while True:
            try: return self.impl.translate_atomic(text, src, tgt)
            except ValueError:
                raise
            except Exception as e:
                self._drop(self.active, e); self._next(probe=False)

    def __getattr__(self, name):
        impl = self.__dict__.get("impl")
        if impl is None: raise AttributeError(name)
        return getattr(impl, name)
