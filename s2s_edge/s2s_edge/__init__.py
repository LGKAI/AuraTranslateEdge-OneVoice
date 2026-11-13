"""s2s_edge: offline on-device speech-to-speech translation (vi <-> en / zh / ko) for HSPTEK AIBOX 8550 (Qualcomm QCS8550)."""
__version__ = "0.1.0"

# qai_appbuilder (imported when the first NPU graph is created) REPLACES sys.modules["onnxruntime"] with its own QNN wrapper module, which breaks plain CPU ONNX
# models (punctuation model, GTCRN). Keep the real module here, imported before anything can touch it, and use `from . import _ort_real as ort` for CPU sessions.
try:
    import onnxruntime as _ort_real
except Exception:                                   # CPU onnxruntime missing: only the NPU graphs work
    _ort_real = None


def cpu_ort():
    """The onnxruntime module for CPU sessions: once qai_appbuilder is loaded (it replaces sys.modules["onnxruntime"]) the real module kept above,
    otherwise whatever `import onnxruntime` gives (so tests can inject a fake module)."""
    import sys
    if "qai_appbuilder" in sys.modules and _ort_real is not None: return _ort_real
    import onnxruntime
    return onnxruntime
