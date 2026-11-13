# -*- coding: utf-8 -*-
"""web_demo/engine.py S2SEngine against the real s2s_edge code with fake graphs (no models needed):
VAD segments -> ASR -> clean_asr -> MT path (split, clean_mt, ko politeness) -> TTS WAV; and the server's /api/text + /api/translate streams.
Run:  python web_demo/test_engine.py   (from the repo root or the package folder)"""
import io, json, os, sys, threading, urllib.request, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import engine as E
root = E._find_s2s_root(); assert root, "s2s_edge package not found"; sys.path.insert(0, root)
from s2s_edge import registry as R, conversation as C

SEEN = []
class FakeASR:
    max_s = 14.0; last = {}
    def __init__(self, lang): self.lang = lang
    def __call__(self, wav): return {"vi": "Q. xin chào các bạn", "ko": "음. 안녕하세요.", "en": "Um. Hello. How are you?", "zh": "嗯。你好。"}[self.lang]
class FakeMT:
    S = 64; last = "mt600"
    def wanted(self, s, t): return "mt600"
    def status(self): return {"big_in_use": False, "degraded": False}
    def __call__(self, text, s, t): SEEN.append(text); return {"ko": "알았어", "zh": "你好。"}.get(t, "T(" + text + ")")
class FakeTTS:
    sr = 22050
    def synthesize(self, text): return np.zeros(2205, np.float32) + 0.1, 22050
class FakeReg:
    glossary = {}; vad_profile = "hybrid"; profile = {}; source_mode = "0"
    def __init__(self, *a, **k): self._mt = FakeMT()
    def warmup(self, dirs, log=print): log(f"fake warmup {len(dirs)} directions")
    def asr(self, lang): return FakeASR(lang)
    def tts(self, lang): return FakeTTS()
    def mt(self): return self._mt
    def graph(self, k): return type("G", (), dict(total_s=0.0, calls=0))()
    def components_for(self, s, t): return []
R.Registry = FakeReg; C.VI_PUNCT = False

def wav_bytes(sec=2.0, sr=16000):
    t = np.arange(int(sec * sr)) / sr; x = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((x * 32767).astype("<i2").tobytes())
    return b.getvalue()

def test_engine():
    e = E.S2SEngine(models="fake.json", backend="npu"); assert "NPU" in e.name and "600M" in e.name, e.name
    t = e.asr(wav_bytes(), "vi"); assert t and not t.startswith("Q."), t                 # junk label removed by clean_asr
    assert e.mt("Xin chào. Tôi là Suji.", "vi", "en") == "T(Xin chào.) T(Tôi là Suji.)", SEEN[-3:]             # sentence by sentence
    assert e.mt("Tôi rất vui.", "vi", "ko") == "알았어요"                                    # ko politeness on the web path too
    assert e.mt("   ", "vi", "en") == ""
    w = e.tts("hello", "en"); assert w[:4] == b"RIFF"
    with wave.open(io.BytesIO(w)) as r: assert r.getframerate() == 22050 and r.getnframes() == 2205
    assert e.tts("", "en")[:4] == b"RIFF"
    x = E._wav_to_f32(wav_bytes(1.0, 48000)); assert abs(len(x) - 16000) <= 2, len(x)       # browser uploads at 48 kHz are resampled

def test_server():
    os.environ["S2S_ENGINE"] = "real"; os.environ["S2S_MODELS"] = "fake.json"
    import importlib, server; importlib.reload(server)
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); port = srv.server_address[1]; threading.Thread(target=srv.serve_forever, daemon=True).start()
    def post(path, body, ctype):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, headers={"Content-Type": ctype}, method="POST")
        return [json.loads(l) for l in urllib.request.urlopen(req, timeout=30).read().decode().splitlines() if l.strip()]
    ev = post("/api/translate?src=vi&tgt=ko", wav_bytes(), "audio/wav"); st = [e["stage"] for e in ev]
    assert st == ["asr", "mt", "tts", "done"], ev
    assert ev[1]["text"] == "알았어요"
    ev = post("/api/text?src=vi&tgt=en", json.dumps({"text": "Xin chào các bạn."}).encode(), "application/json")
    assert [e["stage"] for e in ev] == ["mt", "tts", "done"] and ev[0]["text"].startswith("T("), ev
    h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health").read()); assert "NPU" in h["engine"] and h["info"]["backend"] == "npu", h
    srv.shutdown()

if __name__ == "__main__":
    os.environ["S2S_WARMUP"] = "1"
    test_engine(); print("ok test_engine"); test_server(); print("ok test_server")
