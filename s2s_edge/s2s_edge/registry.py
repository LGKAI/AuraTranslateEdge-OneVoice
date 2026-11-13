# -*- coding: utf-8 -*-
"""Model registry: models.json maps component names to files. Paths are relative to the json file (or absolute for development).
  {"asr_vi": {"onnx": ..., "ctx": ..., "name": ...}, "asr_xx": ..., "mt": ..., "tts_vi": ..., "tts_zh": ..., "tts_enko": ...}
ASR windows: extra static windows of the same ASR graph are keys "<base>@<window>" (Zipformer: fbank frames, 100/s; SenseVoice encoder: LFR
frames, 60 ms each), e.g. "asr_vi@500", "asr_xx_encoder@100"; the base key is the largest window (1500 / 500, or its "window" field).
SenseVoice features: "_sv_frontend": "host" (exact FunASR features on the CPU, no frontend graph) or "graph" (the fp16 NPU frontend graph,
default when "asr_xx_frontend" is in the file); environment variable S2S_SV_FRONTEND overrides it."""
import hashlib, json, os, subprocess
import numpy as np
from .audio import resolve_vad_options
from .backends import make_graph, resolve_ort_policy
from .asr import AsrVi, AsrXx, T_FRAMES, SV_FRAMES, load_cmvn
from .mt import Nllb
from .tts import TtsVi, TtsZh, TtsEnKo

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DIRECTIONS = [("vi", "en"), ("vi", "zh"), ("vi", "ko"), ("en", "vi"), ("zh", "vi"), ("ko", "vi")]
_VALID_SOURCE_MODES = ("0", "1")
_VALID_TTS_FORMATS = ("hybrid", "legacy", "neural")


def resolve_runtime_profile(vad_profile=None, ort_policy=None, source_mode=None, tts_format=None):
    """Single shared profile resolver for CLI, web, batch, stream, and evaluation harnesses."""
    if vad_profile is not None:
        vad_origin = "arg"
    elif os.environ.get("S2S_VAD_PROFILE") not in (None, ""):
        vad_origin = "env"
    else:
        vad_origin = "default"
    eff_vad, vad_opts = resolve_vad_options(vad_profile=vad_profile)

    if ort_policy is not None:
        ort_origin = "arg"
    elif os.environ.get("S2S_ORT_POLICY") not in (None, ""):
        ort_origin = "env"
    else:
        ort_origin = "default"
    eff_ort = resolve_ort_policy(ort_policy=ort_policy)

    if source_mode is not None:
        eff_src = str(source_mode)
        src_origin = "arg"
    elif os.environ.get("S2S_NEURAL_TN") not in (None, ""):
        eff_src = str(os.environ.get("S2S_NEURAL_TN"))
        src_origin = "env"
    else:
        eff_src = "0"
        src_origin = "default"
    if eff_src not in _VALID_SOURCE_MODES:
        raise ValueError(f"Unsupported source_mode / S2S_NEURAL_TN {eff_src!r}; expected one of {_VALID_SOURCE_MODES}")

    if tts_format is not None:
        eff_tts = str(tts_format)
        tts_origin = "arg"
    elif os.environ.get("S2S_TTS_FORMAT") not in (None, ""):
        eff_tts = str(os.environ.get("S2S_TTS_FORMAT"))
        tts_origin = "env"
    else:
        eff_tts = "hybrid"
        tts_origin = "default"
    if eff_tts not in _VALID_TTS_FORMATS:
        raise ValueError(f"Unsupported tts_format / S2S_TTS_FORMAT {eff_tts!r}; expected one of {_VALID_TTS_FORMATS}")

    return {
        "vad_profile": eff_vad,
        "vad_profile_origin": vad_origin,
        "vad_options": vad_opts,
        "ort_policy": eff_ort,
        "ort_policy_origin": ort_origin,
        "source_mode": eff_src,
        "source_mode_origin": src_origin,
        "tts_format": eff_tts,
        "tts_format_origin": tts_origin,
    }


def runtime_code_fingerprint(extra_files=None):
    """Compute lightweight SHA256 fingerprint over active s2s_edge modules, optional caller files, and git HEAD."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(pkg_dir) if f.endswith(".py")):
        p = os.path.join(pkg_dir, name)
        with open(p, "rb") as f:
            h.update(name.encode("utf-8") + b":" + hashlib.sha256(f.read()).hexdigest().encode("utf-8") + b"\n")
    head = None
    try:
        head = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    out = {"git_head": head, "s2s_edge_sha256": h.hexdigest()}
    if extra_files:
        eh = hashlib.sha256()
        for ep in sorted(os.path.abspath(str(p)) for p in extra_files):
            if os.path.isfile(ep):
                with open(ep, "rb") as f:
                    eh.update(os.path.relpath(ep, repo_root).encode("utf-8") + b":" + hashlib.sha256(f.read()).hexdigest().encode("utf-8") + b"\n")
        out["server_files_sha256"] = eh.hexdigest()
    return out


class Registry:
    def __init__(self, config_path, backend="npu", glossary_path=None, ort_policy=None, vad_profile=None, source_mode=None, tts_format=None):
        with open(config_path, encoding="utf-8") as _f:
            self.cfg = json.load(_f)
        self.dir = os.path.dirname(os.path.abspath(config_path)); self.backend = backend; self._cache = {}
        gp = glossary_path or os.environ.get("S2S_GLOSSARY")
        if gp and os.path.exists(gp):
            with open(gp, encoding="utf-8") as _gf:
                self.glossary = json.load(_gf)
        else:
            self.glossary = {}
        self.profile = resolve_runtime_profile(vad_profile=vad_profile, ort_policy=ort_policy, source_mode=source_mode, tts_format=tts_format)
        self.ort_policy = self.profile["ort_policy"]
        self.vad_profile = self.profile["vad_profile"]
        self.source_mode = self.profile["source_mode"]
        self.tts_format = self.profile["tts_format"]
        self.source_mode_explicit = source_mode is not None
        self.tts_format_explicit = tts_format is not None

    def graph(self, key):
        if key not in self._cache:
            spec = dict(self.cfg[key]); spec.setdefault("name", key)
            g = make_graph(self.dir, spec, self.backend, ort_policy=self.ort_policy); self._cache[key] = g
            if hasattr(g, "set_out_dtypes"):                                   # NPU: tell the graph its true integer output types (see backends.fix_int_output)
                try:
                    gp = self.gold_path(key)
                    if os.path.exists(gp):
                        z = np.load(gp); outs = sorted((n for n in z.files if n.startswith("out_")), key=lambda n: int(n[4:])); g.set_out_dtypes([(z[n].dtype, z[n].size) for n in outs])
                except Exception as e: print(f"[s2s] could not read output dtypes of {key} from the golden file: {e!r}", flush=True)
        return self._cache[key]

    def graph_telemetry(self):
        return {k: g.ort_info() for k, g in self._cache.items() if hasattr(g, "ort_info")}

    def files(self, keys):
        """Model files of the given keys, each once (ASR windows linked into one context binary share the file)."""
        ext = "onnx" if self.backend == "cpu" else "ctx"
        return list(dict.fromkeys(os.path.join(self.dir, self.cfg[k][ext]) for k in keys))

    def windows(self, base, default):
        """[(window, key)] ascending: the base key (its "window" field, else `default`) and every "<base>@<window>" key."""
        out = []
        for k, spec in self.cfg.items():
            if k == base: out.append((int(spec.get("window", default)), k))
            elif k.startswith(base + "@"): out.append((int(k.split("@", 1)[1]), k))
        return sorted(out)

    def sv_frontend(self):
        """'graph' (NPU frontend graph) or 'host' (s2s_edge.asr.sv_features)."""
        want = (os.environ.get("S2S_SV_FRONTEND") or self.cfg.get("_sv_frontend") or ("graph" if "asr_xx_frontend" in self.cfg else "host")).lower()
        return "graph" if want == "graph" and "asr_xx_frontend" in self.cfg else "host"

    def asr_keys(self, lang):
        if lang == "vi": return [k for _, k in self.windows("asr_vi", T_FRAMES)]
        if "asr_xx" in self.cfg: return ["asr_xx"]
        return (["asr_xx_frontend"] if self.sv_frontend() == "graph" else []) + [k for _, k in self.windows("asr_xx_encoder", SV_FRAMES)]

    def transducer(self):
        """RNN-T head of the Vietnamese Zipformer on the host CPU (asr.Transducer), when models.json has "asr_vi_decoder" and "asr_vi_joiner"
        (their "onnx" files are used with every backend: 5 + 4 MB, tiny host work). S2S_VI_DECODE=ctc turns it off."""
        if os.environ.get("S2S_VI_DECODE", "rnnt").lower() == "ctc" or not ("asr_vi_decoder" in self.cfg and "asr_vi_joiner" in self.cfg): return None
        if getattr(self, "_rnnt", None) is None:
            from .asr import Transducer
            p = lambda k: os.path.join(self.dir, self.cfg[k]["onnx"])
            try: self._rnnt = Transducer(p("asr_vi_decoder"), p("asr_vi_joiner"))
            except Exception as e:                                        # never take the Vietnamese ASR down: fall back to the CTC head
                import sys; print(f"[s2s] RNN-T decoder/joiner unusable ({e!r}) -> CTC greedy", file=sys.stderr); self._rnnt = False
        return self._rnnt or None

    def asr(self, lang):
        if lang == "vi":
            known = set(open(os.path.join(ASSETS, "piper_vi_words.txt"), encoding="utf-8").read().splitlines())
            return AsrVi([(w, self.graph(k)) for w, k in self.windows("asr_vi", T_FRAMES)], os.path.join(ASSETS, "zipformer_bpe.model"), known, self.transducer())
        if "asr_xx" in self.cfg: return AsrXx(self.graph("asr_xx"), None, lang)
        enc = [(w, self.graph(k)) for w, k in self.windows("asr_xx_encoder", SV_FRAMES)]
        if self.sv_frontend() == "graph": return AsrXx(self.graph("asr_xx_frontend"), enc, lang)
        return AsrXx(None, enc, lang, load_cmvn(os.path.join(ASSETS, "sensevoice_am.mvn")))

    def release(self, key):
        """Drop a graph from the cache and free its runtime memory (MT fallback after an OOM / runtime error)."""
        import gc
        for k in (["mt33_encA", "mt33_encB", "mt33_decA", "mt33_decB"] if key == "mt33" else [key]):
            g = self._cache.pop(k, None)
            if g is not None: g.release()
        gc.collect()

    def mt_candidates(self):
        """Decoder graphs from the most beams to greedy: every mt_decB* key (sorted by "B", descending), then mt_dec.
        S2S_MT_MAXB=<n> skips beam graphs with more than n beams (S2S_MT_MAXB=1 -> greedy only)."""
        maxb = int(os.environ.get("S2S_MT_MAXB", "99"))
        beams = sorted([k for k in self.cfg if k.startswith("mt_decB") and self.cfg[k].get("B", 4) <= maxb], key=lambda k: -self.cfg[k].get("B", 4))
        return beams + (["mt_dec"] if "mt_dec" in self.cfg else [])          # the 600M chain; NLLB-3.3B is built separately (mt()) and routed per direction

    def _build_mt(self, key):
        from .mt import NllbSplit, NllbBeam
        if key == "mt33":
            from .mt33 import Nllb33Beam
            c = self.cfg["mt33_decB"]
            return Nllb33Beam(self.graph("mt33_encA"), self.graph("mt33_encB"), self.graph("mt33_decA"), self.graph("mt33_decB"), ASSETS,
                              S=c.get("S", 64), Tm=c.get("Tm", 96), B=c.get("B", 4), L=c.get("L", 24), H=c.get("H", 16), dk=c.get("dk", 128))
        e = self.cfg["mt_enc"]; d = self.cfg[key]; kw = dict(L=d.get("L", 12), H=d.get("H", 16), dk=d.get("dk", 64))
        if key == "mt_dec": return NllbSplit(self.graph("mt_enc"), self.graph("mt_dec"), ASSETS, S=e.get("S", 32), Tm=d.get("Tm", 48), **kw)
        return NllbBeam(self.graph("mt_enc"), self.graph(key), ASSETS, S=e.get("S", 64), Tm=d.get("Tm", 96), B=d.get("B", 4), **kw)

    def mt(self):
        """One shared MT object per registry. With split graphs it is an MtFallback over mt_candidates() (largest beam first, released and
        degraded on OOM / runtime errors)."""
        if getattr(self, "_mt_obj", None) is not None: return self._mt_obj
        if "mt" in self.cfg: self._mt_obj = Nllb(self.graph("mt"), ASSETS); return self._mt_obj
        from .mt import MtFallback, MtRouter, DEFAULT_ROUTE
        small = MtFallback(self, self.mt_candidates(), self._build_mt, probe=os.environ.get("S2S_MT_PROBE", "1") != "0")
        if "mt33_decB" not in self.cfg or os.environ.get("S2S_MT", "").lower() == "600m":
            self._mt_obj = small; return small
        big, err = None, None
        try:
            big = self._build_mt("mt33"); big("xin chào các bạn", "vi", "zh")                                  # allocate + run every 3.3B graph once: OOM shows up here
        except Exception as e:
            err = repr(e)[:300]; big = None; self.release("mt33")
        self._mt_obj = MtRouter(small, big, self.cfg.get("mt_route") or DEFAULT_ROUTE, strict=os.environ.get("S2S_MT_STRICT", "0") == "1", big_error=err,
                                release_big=lambda: self.release("mt33"))
        return self._mt_obj

    def tts(self, lang):
        if lang == "vi": return TtsVi(self.graph("tts_vi"), ASSETS, self.glossary, tts_format=self.tts_format)
        if lang == "zh": return TtsZh(self.graph("tts_zh"), tts_format=self.tts_format)
        return TtsEnKo(self.graph("tts_enko"), lang, ASSETS, tts_format=self.tts_format)

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
        langs = list(dict.fromkeys("vi" if s == "vi" else "xx" for s, _ in directions))   # en/zh/ko share the SenseVoice graphs: warm them once
        for lang in langs:                                                # every ASR window once (a golden run covers only one input length)
            lang = "en" if lang == "xx" else lang
            try:
                t0 = time.perf_counter(); a = self.asr(lang); a.warmup()
                log(f"  warmup ASR {lang}: {len(getattr(a, 'ws', []) or [1])} window(s) in {time.perf_counter() - t0:.1f} s")
            except Exception as e:
                log(f"  warmup ASR {lang}: FAILED {repr(e)[:200]}")
        if any(s == "vi" for s, _ in directions):                         # Vietnamese punctuation/casing model (CPU), see conversation.restore_vi
            try:
                from .conversation import VI_PUNCT_MODE
                if VI_PUNCT_MODE == "model":
                    from .punct_vi import get
                    t0 = time.perf_counter(); m = get(ASSETS)
                    if m is not None: m("xin chào hôm nay chúng ta họp lúc ba giờ"); log(f"  warmup vi punctuation model: {time.perf_counter() - t0:.1f} s")
                    else: log("  warmup vi punctuation model: assets/vi_capu missing -> rule-based vi_punct")
            except Exception as e:
                import traceback; traceback.print_exc(); log(f"  warmup vi punctuation model: FAILED {repr(e)[:300]}")
        log(f"  warmup done: {len(keys)} graphs in {time.perf_counter() - t_all:.1f} s")
        return out

    def components_for(self, src, tgt):
        asr = self.asr_keys(src)
        if "mt" in self.cfg: mt = ["mt"]
        else:
            if getattr(self, "mt_active", None) is None: self.mt()            # build (probe + fallback) before anyone loads the biggest beam graph directly
            router = self.mt()
            mt = ["mt33_encA", "mt33_encB", "mt33_decA", "mt33_decB"] if hasattr(router, "uses_big") and router.uses_big(src, tgt) else ["mt_enc", self.mt_active]
        return asr + mt + [{"vi": "tts_vi", "zh": "tts_zh"}.get(tgt, "tts_enko")]
