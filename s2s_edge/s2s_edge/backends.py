# -*- coding: utf-8 -*-
"""Graph backends. A Graph wraps ONE compiled model and is called with named numpy inputs, returning outputs in ONNX output order.
OrtGraph : onnxruntime CPU (development / reference, runs everywhere).
QnnGraph : Qualcomm NPU on the device (context binary produced by AI Hub). Loader chosen at run time, see `QnnGraph.load` (decided after tools/recon.sh)."""
import os, sys, time, json
import numpy as np


def _ort_session(path, so=None, providers=None):
    """The REAL onnxruntime.InferenceSession. qai_appbuilder replaces the onnxruntime module with its QNN wrapper (breaks plain CPU models: "backend library does not exist"); s2s_edge/__init__ keeps the real one."""
    from . import _ort_real
    return _ort_real.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])


class Graph:
    name = "graph"

    def __init__(self):
        self.calls = 0; self.total_s = 0.0

    def run(self, inputs):
        raise NotImplementedError

    def __call__(self, **inputs):
        t = time.perf_counter(); out = self.run({k: np.ascontiguousarray(v) for k, v in inputs.items()})
        self.calls += 1; self.total_s += time.perf_counter() - t; return out

    def release(self):
        """Free the runtime resources (used by the MT fallback after an out-of-memory / runtime error)."""
        pass


ORT_POLICIES = ("auto", "basic", "disable")
DEFAULT_ORT_POLICY = "auto"


def resolve_ort_policy(ort_policy=None) -> str:
    """Resolve explicit ORT session optimization policy ('auto', 'basic', 'disable')."""
    if ort_policy is None:
        env_pol = os.environ.get("S2S_ORT_POLICY")
        pol = env_pol if (env_pol is not None and env_pol != "") else DEFAULT_ORT_POLICY
    else:
        pol = ort_policy
    if not isinstance(pol, str) or not pol.strip():
        raise ValueError(f"Unsupported ort_policy {pol!r}; expected one of {ORT_POLICIES}")
    key = pol.strip().lower()
    if key not in ORT_POLICIES:
        raise ValueError(f"Unsupported ort_policy {pol!r}; expected one of {ORT_POLICIES}")
    return key


class OrtGraph(Graph):
    def __init__(self, path, name=None, ort_policy=None):
        super().__init__(); from . import cpu_ort; ort = cpu_ort()            # the REAL onnxruntime even after qai_appbuilder replaced the module on the box
        self.path = path
        self.ort_policy = resolve_ort_policy(ort_policy)
        self.external_data = os.path.exists(path + ".data")
        self.size_bytes = os.path.getsize(path) if os.path.exists(path) else None
        self.large_or_ext = self.external_data or (self.size_bytes is not None and self.size_bytes > 200_000_000)

        if self.ort_policy == "disable":
            plan = [("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]
        elif self.ort_policy == "basic":
            plan = [
                ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
            ]
        else:
            if self.large_or_ext:
                plan = [("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]
            else:
                plan = [
                    ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                    ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                ]

        self.ort_attempts = []
        self.effective_ort_level = None
        self.ort_fallback = False
        last_exc = None
        for idx, (level_name, level_enum) in enumerate(plan, start=1):
            so = ort.SessionOptions(); so.log_severity_level = 3; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
            so.graph_optimization_level = level_enum
            try:
                self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                self.ort_attempts.append({"attempt": idx, "level": level_name, "ok": True})
                self.effective_ort_level = level_name
                self.ort_fallback = idx > 1
                last_exc = None
                break
            except Exception as exc:
                self.ort_attempts.append({"attempt": idx, "level": level_name, "ok": False, "error": repr(exc)[:300]})
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        self.name = name or os.path.basename(path)
        self.in_names = [i.name for i in self.s.get_inputs()]
        self.in_types = {i.name: {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32, "tensor(float16)": np.float16}.get(i.type, np.float32) for i in self.s.get_inputs()}

    def run(self, inputs):
        return self.s.run(None, {k: inputs[k].astype(self.in_types[k], copy=False) for k in self.in_names})

    def release(self):
        self.s = None

    def ort_info(self):
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "external_data": self.external_data,
            "large_or_ext": self.large_or_ext,
            "ort_policy": self.ort_policy,
            "effective_ort_level": self.effective_ort_level,
            "ort_fallback": self.ort_fallback,
            "attempts": list(self.ort_attempts),
        }


_NP = {"float": np.float32, "float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16": np.int16, "uint32": np.uint32}
_qnn_ready = False


def _qnn_init():
    """QAI AppBuilder (python wheel qai_appbuilder-2.51.40 for linux_aarch64 bundles the QAIRT 2.51 runtime incl. the Hexagon v73 skel; AI Hub jobs are pinned with `--qairt_version 2.51`)."""
    global _qnn_ready
    if _qnn_ready: return
    import qai_appbuilder
    from qai_appbuilder import QNNConfig, Runtime, LogLevel, ProfilingLevel, PerfProfile
    # The wheel sets ADSP_LIBRARY_PATH only when it is empty; if the box already exports one (HSPTek installs QAIRT 2.41 skels in
    # /usr/lib/rfsa/adsp), the 2.51 stub would meet a 2.41 skel. Put the bundled libs first (same rule as tools/env.sh; S2S_SYSTEM_QNN=1 opts out).
    libs = os.path.join(os.path.dirname(qai_appbuilder.__file__), "libs")
    if os.environ.get("S2S_SYSTEM_QNN", "0") != "1" and not sys.platform.startswith("win") and not os.environ.get("ADSP_LIBRARY_PATH", "").startswith(libs):
        os.environ["ADSP_LIBRARY_PATH"] = f"{libs};/usr/lib/rfsa/adsp;/dsp"
    print(f"[s2s] ADSP_LIBRARY_PATH={os.environ.get('ADSP_LIBRARY_PATH', '')}", file=sys.stderr)
    QNNConfig.Config(Runtime.HTP, LogLevel.WARN, ProfilingLevel.OFF)
    _qnn_ready = True


_perf_done = False


def _apply_perf():
    """S2S_PERF picks the HTP perf profile (default BURST = keep the HTP at burst clocks while the app runs); NONE leaves the system default.
    Used by tools/runbook.sh stage A3 to compare profiles on the real box. Names follow qai_appbuilder's PerfProfile enum.
    Called once, AFTER the first context is loaded: qai_appbuilder docs/guide_en.md section 3.9 says to call SetPerfProfileGlobal() after the
    first model is loaded, and builds up to the first 2.51.40 release silently ignore a call made before it. This package pins 2.51.40 and used
    to call it in _qnn_init (before any load), so the HTP may have run at default clocks: wall times 3-5x the AI Hub profile on the box."""
    global _perf_done
    if _perf_done: return
    _perf_done = True
    want = os.environ.get("S2S_PERF", "BURST").upper()
    if want == "NONE": return
    from qai_appbuilder import PerfProfile
    try:
        PerfProfile.SetPerfProfileGlobal(getattr(PerfProfile, want))
        print(f"[s2s] HTP perf profile: {want} (set after the first context load)", file=sys.stderr)
    except Exception as e:
        print(f"[s2s] could not set perf profile {want}: {e!r}", file=sys.stderr)


_contexts = {}                                                       # path -> QNNContext: several graphs (ASR windows) linked in one binary


def fix_int_output(o, dt):
    """On the AIBOX 8550 (qai_appbuilder 2.51.40) small integer outputs (lengths, token ids: tts_vi / tts_zh length, asr lengths, greedy MT token) come back as a
    float32 array that holds the INT32 BIT PATTERN (216 -> 3.03e-43), which breaks every consumer (empty ASR text, zero-length audio). `dt` = the dtype the graph
    really has (taken from the golden reference). Bit patterns of small ints are denormal floats, so: all values tiny -> reinterpret bits; otherwise cast the values."""
    o = np.ascontiguousarray(np.asarray(o))
    if dt is None or o.dtype == dt or not np.issubdtype(dt, np.integer) or o.dtype.kind != "f": return o
    if o.dtype.itemsize == 4 and np.all(np.abs(o) < 1e-30): return o.view(np.int32).astype(dt)
    return np.rint(o).astype(dt)


class QnnGraph(Graph):
    """One compiled graph on the Hexagon NPU through QAI AppBuilder. `path` is a context binary (.bin from an AI Hub link job, must match the wheel's
    QAIRT version) or a portable .dlc (needs QAIRT >= 2.41; the first load writes a cached <name>.dlc.bin next to it, later loads are fast).
    Inputs / outputs use the graph's native dtypes (int32 ids, float32 tensors). Output order is the graph's output order; `out_order` can
    remap it if golden check shows a permutation (models.json: "out": [index, ...])."""

    def __init__(self, path, name=None, input_order=None, output_order=None, graph_index=None, in_quant=None):
        super().__init__(); _qnn_init(); self.in_quant = in_quant or {}
        from qai_appbuilder import QNNContext
        self.path = path; self.name = name or os.path.basename(path); self.out_order = output_order; self.gi = graph_index
        # keyword arguments: the positional order of QNNContext is (name, path, backend_lib, system_lib, is_async, input_dtype, output_dtype, ...)
        # graph_index (models.json "graph_index"): one graph of a multi-graph context binary (AI Hub link job of several static windows, weights
        # shared); keys with the same "ctx" share one loaded context and pick their graph with Inference(graphIndex=...).
        key = os.path.abspath(path) if graph_index is not None else None
        if key and key in _contexts: self.ctx = _contexts[key]
        else:
            self.ctx = QNNContext(model_name=self.name, model_path=path, input_data_type="native", output_data_type="native")
            if key: _contexts[key] = self.ctx
        _apply_perf()
        def _try(f, default=None):
            try: return f(graph_index) if graph_index is not None else f()
            except TypeError:
                try: return f()
                except Exception: return default
            except Exception: return default
        self.in_names = _try(self.ctx.getInputName, None); self.in_shapes = _try(self.ctx.getInputShapes, None); self.in_types = _try(self.ctx.getInputDataType, None)
        self.out_shapes = _try(self.ctx.getOutputShapes, None); self.out_types = _try(self.ctx.getOutputDataType, None)
        if graph_index:                                                   # per-graph shape queries may not exist: shapes of graph 0 would be wrong here
            self.in_shapes = self.out_shapes = None
        self.in_order = input_order or self.in_names; self.out_dtypes = None

    def set_out_dtypes(self, specs):
        """[(dtype, size), ...] of the graph outputs as in golden/<graph>.npz; used by fix_int_output. Matched to the real outputs by position when the sizes
        agree, otherwise by size (the order of some chip outputs is permuted, e.g. MeloTTS returns (length, audio): never apply an integer dtype to the audio)."""
        self.out_dtypes = [(np.dtype(d), int(n)) for d, n in specs]

    def _dtype_for(self, i, o, used):
        specs = self.out_dtypes
        if i < len(specs) and specs[i][1] == o.size and i not in used: used.add(i); return specs[i][0]
        cand = [j for j, (_, n) in enumerate(specs) if n == o.size and j not in used]
        if len(cand) == 1: used.add(cand[0]); return specs[cand[0]][0]
        return None                                                            # ambiguous or unknown: leave the array as it is

    def run(self, inputs):
        order = self.in_order or list(inputs); arrs = []
        for i, k in enumerate(order):
            a = inputs[k]
            if k in self.in_quant:
                # Quantized input (the w8a16 SenseVoice encoder: fbank is UFIXED_POINT_16). With native I/O the bytes go to the HTP as they are, so float features
                # must be quantized here: q = round(x / scale) - offset (QNN convention real = (q + offset) * scale). Without this the encoder hears noise.
                sc, off = self.in_quant[k]; arrs.append(np.ascontiguousarray(np.clip(np.rint(np.asarray(a, np.float64) / sc) - off, 0, 65535).astype(np.uint16)).reshape(-1)); continue
            if self.in_types and i < len(self.in_types):
                dt = _NP.get(str(self.in_types[i]).lower().replace("np.", "").replace("numpy.", ""), None)
                if dt is not None: a = a.astype(dt, copy=False)
            arrs.append(np.ascontiguousarray(a).reshape(-1))
        outs = self.ctx.Inference(arrs) if self.gi is None else self.ctx.Inference(arrs, graphIndex=int(self.gi)); res = []
        for i, o in enumerate(outs):
            o = np.asarray(o)
            if self.out_shapes and i < len(self.out_shapes):
                try: o = o.reshape(tuple(self.out_shapes[i]))
                except Exception: pass
            res.append(o)
        res = [res[j] for j in self.out_order] if self.out_order else res
        if self.out_dtypes:
            used = set(); res = [fix_int_output(o, self._dtype_for(i, np.asarray(o), used)) for i, o in enumerate(res)]
        return res

    def release(self):
        """QNNContext.release() frees the context (and its HTP memory) deterministically (qai_appbuilder >= 2.51). A context shared by several
        graphs (graph_index) is released by the first of them; the others must not be used afterwards (only MT graphs are ever released)."""
        try: self.ctx.release()
        except Exception: pass
        if self.gi is not None: _contexts.pop(os.path.abspath(self.path), None)
        self.ctx = None


def make_graph(model_dir, spec, backend, ort_policy=None):
    """spec: {'onnx': relative path (cpu ref), 'ctx': relative path (npu), 'in': [...], 'out': [...], 'graph_index': i (multi-graph ctx only),
    'in_quant': {input: [scale, offset]} (quantized native inputs, e.g. the w8a16 SenseVoice encoder)}"""
    if backend == "cpu": return OrtGraph(os.path.join(model_dir, spec["onnx"]), spec.get("name"), ort_policy=ort_policy)
    if backend == "npu": return QnnGraph(os.path.join(model_dir, spec["ctx"]), spec.get("name"), spec.get("in"), spec.get("out"), spec.get("graph_index"), spec.get("in_quant"))
    raise ValueError(backend)
