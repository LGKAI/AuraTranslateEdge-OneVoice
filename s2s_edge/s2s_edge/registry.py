# -*- coding: utf-8 -*-
"""Model registry: models.json maps component names to files. Paths are relative to the json file (or absolute for development).
  {"asr_vi": {"onnx": ..., "ctx": ..., "name": ...}, "asr_xx": ..., "mt": ..., "tts_vi": ..., "tts_zh": ..., "tts_enko": ...}"""
import json, os
from .backends import make_graph
from .asr import AsrVi, AsrXx
from .mt import Nllb
from .tts import TtsVi, TtsZh, TtsEnKo

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DIRECTIONS = [("vi", "en"), ("vi", "zh"), ("vi", "ko"), ("en", "vi"), ("zh", "vi"), ("ko", "vi")]


class Registry:
    def __init__(self, config_path, backend="npu", glossary_path=None):
        self.cfg = json.load(open(config_path, encoding="utf-8")); self.dir = os.path.dirname(os.path.abspath(config_path)); self.backend = backend; self._cache = {}
        gp = glossary_path or os.environ.get("S2S_GLOSSARY"); self.glossary = json.load(open(gp, encoding="utf-8")) if gp and os.path.exists(gp) else {}

    def graph(self, key):
        if key not in self._cache:
            spec = dict(self.cfg[key]); spec.setdefault("name", key)
            self._cache[key] = make_graph(self.dir, spec, self.backend)
        return self._cache[key]

    def files(self, keys):
        ext = "onnx" if self.backend == "cpu" else "ctx"
        return [os.path.join(self.dir, self.cfg[k][ext]) for k in keys]

    def asr(self, lang):
        if lang == "vi":
            known = set(open(os.path.join(ASSETS, "piper_vi_words.txt"), encoding="utf-8").read().splitlines())
            return AsrVi(self.graph("asr_vi"), os.path.join(ASSETS, "zipformer_bpe.model"), known)
        if "asr_xx" in self.cfg: return AsrXx(self.graph("asr_xx"), None, lang)
        return AsrXx(self.graph("asr_xx_frontend"), self.graph("asr_xx_encoder"), lang)

    def release(self, key):
        """Drop a graph from the cache and free its runtime memory (MT fallback after an OOM / runtime error)."""
        import gc
        g = self._cache.pop(key, None)
        if g is not None: g.release()
        gc.collect()

    def mt_candidates(self):
        """Decoder graphs from the most beams to greedy: every mt_decB* key (sorted by "B", descending), then mt_dec.
        S2S_MT_MAXB=<n> skips beam graphs with more than n beams (S2S_MT_MAXB=1 -> greedy only)."""
        maxb = int(os.environ.get("S2S_MT_MAXB", "99"))
        beams = sorted([k for k in self.cfg if k.startswith("mt_decB") and self.cfg[k].get("B", 4) <= maxb], key=lambda k: -self.cfg[k].get("B", 4))
        return beams + (["mt_dec"] if "mt_dec" in self.cfg else [])

    def _build_mt(self, key):
        from .mt import NllbSplit, NllbBeam
        e = self.cfg["mt_enc"]; d = self.cfg[key]; kw = dict(L=d.get("L", 12), H=d.get("H", 16), dk=d.get("dk", 64))
        if key == "mt_dec": return NllbSplit(self.graph("mt_enc"), self.graph("mt_dec"), ASSETS, S=e.get("S", 32), Tm=d.get("Tm", 48), **kw)
        return NllbBeam(self.graph("mt_enc"), self.graph(key), ASSETS, S=e.get("S", 64), Tm=d.get("Tm", 96), B=d.get("B", 4), **kw)

    def mt(self):
        """One shared MT object per registry. With split graphs it is an MtFallback over mt_candidates() (largest beam first, released and
        degraded on OOM / runtime errors)."""
        if getattr(self, "_mt_obj", None) is not None: return self._mt_obj
        if "mt" in self.cfg: self._mt_obj = Nllb(self.graph("mt"), ASSETS); return self._mt_obj
        from .mt import MtFallback
        self._mt_obj = MtFallback(self, self.mt_candidates(), self._build_mt, probe=os.environ.get("S2S_MT_PROBE", "1") != "0")
        return self._mt_obj

    def tts(self, lang):
        if lang == "vi": return TtsVi(self.graph("tts_vi"), ASSETS, self.glossary)
        if lang == "zh": return TtsZh(self.graph("tts_zh"))
        return TtsEnKo(self.graph("tts_enko"), lang, ASSETS)

    def gold_path(self, key):
        """Golden vectors of a graph: the model set's own directory first (models.json "_golden", e.g. golden_13b for the NLLB-1.3B set), then golden/."""
        root = os.path.dirname(ASSETS)
        for d in ([self.cfg["_golden"]] if "_golden" in self.cfg else []) + ["golden"]:
            p = os.path.join(root, d, f"{key}.npz")
            if os.path.exists(p): return p
        return os.path.join(root, self.cfg.get("_golden", "golden"), f"{key}.npz")

    def warmup(self, directions, log=print):
        """Load every graph the given directions need and run each once on its golden input (golden/<key>.npz), so the first live
        utterance does not pay for loading ~2 GB of context binaries and the first-call HTP setup. Returns {key: (load_s, first_run_s)}."""
        import time, numpy as np
        out = {}; t_all = time.perf_counter()
        t0 = time.perf_counter(); m = self.mt()                                # MT first: MtFallback probes the biggest beam and degrades if it does not fit
        log(f"  warmup MT: decoder {getattr(m, 'active', 'mt')} ready in {time.perf_counter() - t0:.1f} s" + (f" (dropped: {[k for k, _ in m.failures]})" if getattr(m, "failures", None) else ""))
        keys = []
        for s, t in directions:
            for k in self.components_for(s, t):
                if k not in keys: keys.append(k)
        for k in keys:
            try:
                t0 = time.perf_counter(); g = self.graph(k); t1 = time.perf_counter(); run_s = None; gp = self.gold_path(k)
                if os.path.exists(gp):
                    z = np.load(gp); g(**{n[3:]: z[n] for n in z.files if n.startswith("in_")}); run_s = time.perf_counter() - t1
                out[k] = (round(t1 - t0, 2), None if run_s is None else round(run_s, 3))
                log(f"  warmup {k}: load {t1 - t0:.1f} s" + ("" if run_s is None else f", first run {run_s * 1000:.0f} ms"))
            except Exception as e:
                log(f"  warmup {k}: FAILED {repr(e)[:200]}")
        if any(s == "vi" for s, _ in directions):                         # Vietnamese punctuation/casing model (CPU), see conversation.restore_vi
            from .conversation import VI_PUNCT_MODE
            if VI_PUNCT_MODE == "model":
                from .punct_vi import get
                t0 = time.perf_counter(); m = get(ASSETS)
                if m is not None: m("xin chào hôm nay chúng ta họp lúc ba giờ"); log(f"  warmup vi punctuation model: {time.perf_counter() - t0:.1f} s")
                else: log("  warmup vi punctuation model: assets/vi_capu missing -> rule-based vi_punct")
        log(f"  warmup done: {len(keys)} graphs in {time.perf_counter() - t_all:.1f} s")
        return out

    def components_for(self, src, tgt):
        asr = ["asr_vi"] if src == "vi" else (["asr_xx"] if "asr_xx" in self.cfg else ["asr_xx_frontend", "asr_xx_encoder"])
        if "mt" in self.cfg: mt = ["mt"]
        else:
            if getattr(self, "mt_active", None) is None: self.mt()            # build (probe + fallback) before anyone loads the biggest beam graph directly
            mt = ["mt_enc", self.mt_active]                                   # the decoder MtFallback settled on
        return asr + mt + [{"vi": "tts_vi", "zh": "tts_zh"}.get(tgt, "tts_enko")]
