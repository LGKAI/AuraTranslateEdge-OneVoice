"""Tests for T2 independent VAD/ORT profiles, caller overrides, call-site parity, and real EN ASR integrity."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(ROOT / "s2s_edge"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from s2s_edge.audio import SR, get_vad_options, load_audio, segment
from s2s_edge.backends import OrtGraph, resolve_ort_policy
from s2s_edge.conversation import Conversation, Engine
from s2s_edge.pipeline import Translator
from s2s_edge.registry import Registry, resolve_runtime_profile
import evaluate_asr_integration
import s2s_web
import start_s2s_web


class RuntimeProfileUnitTests(unittest.TestCase):
    def test_get_vad_options_and_invalid_profiles(self):
        legacy = get_vad_options("legacy")
        self.assertEqual(legacy, {"pad_ms": 150, "min_sil_ms": 500, "merge_gap_s": 1.5, "short_s": 3.0})
        upstream = get_vad_options("upstream")
        self.assertEqual(upstream, {"pad_ms": 120, "min_sil_ms": 320, "merge_gap_s": 0.32, "short_s": 1.1})

        # Returned dict is defensive copy
        legacy["min_sil_ms"] = 999
        self.assertEqual(get_vad_options("legacy")["min_sil_ms"], 500)

        for bad in ("unknown", "", "   ", None):
            with self.assertRaises(ValueError):
                get_vad_options(bad)  # type: ignore[arg-type]

        with patch.dict(os.environ, {"S2S_VAD_PROFILE": "bogus"}):
            with self.assertRaises(ValueError):
                resolve_runtime_profile()

        for bad_kwargs in (
            {"ort_policy": "fast"},
            {"source_mode": "2"},
            {"tts_format": "unknown"},
        ):
            with self.assertRaises(ValueError):
                resolve_runtime_profile(**bad_kwargs)  # type: ignore[arg-type]

    def test_caller_override_not_mutated_by_vad_profile(self):
        t = np.arange(SR) / SR
        voice = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        # Two 1.0s voiced tones separated by 0.6s silence
        wav = np.concatenate([np.zeros(3200), voice, np.zeros(9600), voice, np.zeros(3200)]).astype(np.float32)

        # Default legacy merges 0.6s gap (1 seg); caller explicitly overriding min_sil_ms=260 & merge_gap_s=0.2 forces split (2 segs)
        self.assertEqual(len(segment(wav, vad_profile="legacy")), 1)
        self.assertEqual(len(segment(wav, min_sil_ms=260, merge_gap_s=0.2, vad_profile="legacy")), 2)
        # hybrid (default): an in-sentence 0.6 s pause is joined (join_gap_s 0.8); join_gap_s=0 keeps caller overrides effective
        self.assertEqual(len(segment(wav, vad_profile="hybrid")), 1)
        self.assertEqual(len(segment(wav, min_sil_ms=260, merge_gap_s=0.2, join_gap_s=0.0, vad_profile="hybrid")), 2)

        # Even with upstream profile, caller overriding min_sil_ms=700 & merge_gap_s=1.0 preserves 1 merged segment
        with patch.dict(os.environ, {"S2S_VAD_PROFILE": "upstream"}):
            self.assertEqual(len(segment(wav)), 2)
            self.assertEqual(len(segment(wav, min_sil_ms=700, merge_gap_s=1.0)), 1)

    def test_ort_policy_fallback_keeps_chosen_policy_and_records_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            small = Path(folder) / "small.onnx"
            small.write_bytes(b"small-onnx")
            large = Path(folder) / "large.onnx"
            large.write_bytes(b"large-placeholder")
            ext = Path(folder) / "ext.onnx"
            ext.write_bytes(b"ext-main")
            Path(str(ext) + ".data").write_bytes(b"ext-weights")

            def run_mock(path, ort_policy, fail_first=False, fake_size=None):
                seen_levels = []

                def session(_p, opts, providers):
                    seen_levels.append(opts.graph_optimization_level)
                    if fail_first and len(seen_levels) == 1:
                        raise RuntimeError("optimizer failure")
                    return types.SimpleNamespace(get_inputs=lambda: [])

                fake_ort = types.SimpleNamespace(
                    SessionOptions=types.SimpleNamespace,
                    GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_BASIC=1, ORT_DISABLE_ALL=0),
                    InferenceSession=session,
                )
                orig_getsize = os.path.getsize
                with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                    if fake_size is not None:
                        with patch("os.path.getsize", side_effect=lambda p: fake_size if str(p) == str(path) else orig_getsize(p)):
                            g = OrtGraph(str(path), ort_policy=ort_policy)
                    else:
                        g = OrtGraph(str(path), ort_policy=ort_policy)
                return g, seen_levels

            # basic tries ORT_ENABLE_BASIC even on >200MB and retries ORT_DISABLE_ALL on error, without rewriting ort_policy
            g_basic_ok, lv_basic_ok = run_mock(large, "basic", fail_first=False, fake_size=250_000_000)
            self.assertEqual(lv_basic_ok, [1])
            self.assertEqual(g_basic_ok.ort_policy, "basic")
            self.assertEqual(g_basic_ok.effective_ort_level, "ORT_ENABLE_BASIC")
            self.assertFalse(g_basic_ok.ort_fallback)

            g_basic_fb, lv_basic_fb = run_mock(large, "basic", fail_first=True, fake_size=250_000_000)
            self.assertEqual(lv_basic_fb, [1, 0])
            self.assertEqual(g_basic_fb.ort_policy, "basic")
            self.assertEqual(g_basic_fb.effective_ort_level, "ORT_DISABLE_ALL")
            self.assertTrue(g_basic_fb.ort_fallback)
            self.assertEqual(len(g_basic_fb.ort_attempts), 2)
            self.assertFalse(g_basic_fb.ort_attempts[0]["ok"])
            self.assertTrue(g_basic_fb.ort_attempts[1]["ok"])

            # disable always uses ORT_DISABLE_ALL directly even on small graph
            g_dis, lv_dis = run_mock(small, "disable")
            self.assertEqual(lv_dis, [0])
            self.assertEqual(g_dis.ort_policy, "disable")
            self.assertEqual(g_dis.effective_ort_level, "ORT_DISABLE_ALL")
            self.assertFalse(g_dis.ort_fallback)

            # auto uses ORT_DISABLE_ALL for >200MB or external .data, ORT_ENABLE_BASIC for small
            g_auto_ext, lv_auto_ext = run_mock(ext, "auto")
            self.assertEqual(lv_auto_ext, [0])
            self.assertEqual(g_auto_ext.ort_policy, "auto")
            self.assertEqual(g_auto_ext.effective_ort_level, "ORT_DISABLE_ALL")

            g_auto_lg, lv_auto_lg = run_mock(large, "auto", fake_size=250_000_000)
            self.assertEqual(lv_auto_lg, [0])
            self.assertEqual(g_auto_lg.effective_ort_level, "ORT_DISABLE_ALL")

            g_auto_sm, lv_auto_sm = run_mock(small, "auto")
            self.assertEqual(lv_auto_sm, [1])
            self.assertEqual(g_auto_sm.effective_ort_level, "ORT_ENABLE_BASIC")

            with self.assertRaises(ValueError):
                resolve_ort_policy("invalid")

    def test_launcher_respects_caller_source_mode_zero_and_web_health_contract(self):
        env_zero = start_s2s_web.build_launch_env(
            {"S2S_NEURAL_TN": "0", "S2S_VAD_PROFILE": "upstream", "S2S_ORT_POLICY": "basic"}
        )
        self.assertEqual(env_zero["S2S_NEURAL_TN"], "0")
        self.assertEqual(env_zero["S2S_VAD_PROFILE"], "upstream")
        self.assertEqual(env_zero["S2S_ORT_POLICY"], "basic")

        env_default = start_s2s_web.build_launch_env({})
        self.assertEqual(env_default["S2S_NEURAL_TN"], "1")
        self.assertEqual(env_default["S2S_VAD_PROFILE"], "legacy")
        self.assertEqual(env_default["S2S_ORT_POLICY"], "auto")

        # Verify /api/health schema and fields via build_handler without requiring host TCP port binding
        import io
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"S2S_NEURAL_TN": "0", "S2S_VAD_PROFILE": "legacy", "S2S_ORT_POLICY": "disable"}):
                srv_obj = s2s_web.Service(Path(td))
                handler_cls = s2s_web.build_handler(srv_obj, 8767)
                h = handler_cls.__new__(handler_cls)
                h.path = "/api/health"
                h.headers = {}
                h.rfile = io.BytesIO(b"")
                out_buf = io.BytesIO()
                h.wfile = out_buf
                h.send_response = lambda code: setattr(h, "_status", code)
                h.send_header = lambda k, v: None
                h.end_headers = lambda: None
                h.do_GET()
                self.assertEqual(h._status, 200)
                data = json.loads(out_buf.getvalue().decode("utf-8"))

            self.assertFalse(data["normalization"])
            self.assertEqual(data["source_mode"], "0")
            self.assertEqual(data["vad_profile"], "legacy")
            self.assertEqual(data["ort_policy"], "disable")
            self.assertIn("s2s_edge_sha256", data["code_fingerprint"])

            # Verify Service.run records honest backend label, source_mode, ort_sessions when source_mode='0'
            import soundfile as sf
            wav_path = Path(td) / "sample.wav"
            t_arr = np.arange(SR) / SR
            voice_tone = (0.15 * np.sin(2 * np.pi * 220 * t_arr)).astype(np.float32)
            tone = np.concatenate([np.zeros(3200), voice_tone, np.zeros(3200)]).astype(np.float32)
            sf.write(str(wav_path), tone, 16000)

            class FakeReg:
                def __init__(self, *a, **kw):
                    self._cache = {}
                def graph_telemetry(self):
                    return {"asr_xx_encoder": {"ort_policy": "disable", "effective_ort_level": "ORT_DISABLE_ALL", "ort_fallback": False}}
                def release(self, k):
                    pass

            class FakeAsr:
                max_s = 12.0
                def __call__(self, _w):
                    return "hello"

            class FakeMt:
                def __call__(self, *a):
                    return "xin chao"

            class FakeTts:
                def synthesize(self, txt):
                    return np.ones(800, np.float32) * 0.05, 16000

            class FakeTranslator:
                def __init__(self, reg, src, tgt, vad_profile=None):
                    self.asr = FakeAsr()
                    self.tts = FakeTts()
                    self.e = types.SimpleNamespace(_asr={src: self.asr}, _mt=FakeMt(), _tts={tgt: self.tts}, neural_enabled=lambda: False)
                def translate_segment(self, seg):
                    self.e._asr["en"](seg)
                    return {"src": "en", "tgt": "vi", "asr": "hello", "mt": "xin chao", "tts_text": "xin chao", "wav": np.ones(800, np.float32) * 0.05, "sr": 16000, "timing": {}}

            with patch.dict(os.environ, {"S2S_NEURAL_TN": "0", "S2S_VAD_PROFILE": "legacy", "S2S_ORT_POLICY": "disable"}):
                srv2 = s2s_web.Service(Path(td) / "jobs", source_mode="0", vad_profile="legacy", ort_policy="disable")
                job_id = "0" * 32
                jdir = srv2.out / job_id
                jdir.mkdir(parents=True)
                (jdir / "input.wav").write_bytes(wav_path.read_bytes())
                srv2.jobs[job_id] = {"id": job_id, "status": "queued", "src": "en", "tgt": "vi"}
                with patch("s2s_edge.registry.Registry", FakeReg), patch("s2s_edge.pipeline.Translator", FakeTranslator):
                    srv2.run(job_id)
                res_json = json.loads((jdir / "result.json").read_text())
                self.assertEqual(res_json["status"], "done")
                self.assertEqual(res_json["source_mode"], "0")
                self.assertNotIn("Apple Metal", res_json["backend"])
                self.assertEqual(res_json["effective_profile"]["ort_policy"], "disable")
                self.assertIn("asr_xx_encoder", res_json["ort_sessions"])

    def test_batch_and_stream_call_sites_share_resolved_profile(self):
        seen_profiles = []

        class StubReg:
            def __init__(self, ort_policy="auto", vad_profile="legacy", source_mode="0"):
                self.glossary = {}
                self.profile = resolve_runtime_profile(vad_profile=vad_profile, ort_policy=ort_policy, source_mode=source_mode)
                self.ort_policy = self.profile["ort_policy"]
                self.vad_profile = self.profile["vad_profile"]
                self.source_mode = self.profile["source_mode"]
                self.source_mode_explicit = False

            def asr(self, lang):
                return types.SimpleNamespace(max_s=12.0, stats={"avg_logprob": -0.1}, __call__=lambda self2, w: "ok")

            def mt(self):
                return object()

            def tts(self, lang):
                return object()

            def components_for(self, src, tgt):
                return []

            def graph(self, key):
                raise AssertionError(key)

        with patch.dict(os.environ, {"S2S_VAD_PROFILE": "upstream", "S2S_ORT_POLICY": "basic", "S2S_NEURAL_TN": "0"}):
            reg = StubReg(ort_policy=None, vad_profile=None, source_mode=None)
            self.assertEqual(reg.vad_profile, "upstream")
            self.assertEqual(reg.ort_policy, "basic")
            self.assertEqual(reg.source_mode, "0")

            tr = Translator(reg, "en", "vi")
            tr.e.translate_segment = lambda wav, src, tgt, text=None, prepped=False, breaks=None: {
                "src": src,
                "tgt": tgt,
                "asr": "hi",
                "mt": "chao",
                "wav": np.zeros(100, np.float32),
                "sr": 16000,
                "timing": {},
            }
            tr.e.translate_segment_stream = lambda wav, src, tgt, text=None, prepped=False, breaks=None: iter(
                [{"src": src, "tgt": tgt, "asr": "hi", "mt": "chao", "wav": np.zeros(100, np.float32), "sr": 16000, "timing": {}}]
            )

            conv = Conversation(reg, pair=("vi", "en"))
            conv.e = tr.e
            conv.route = lambda seg: ("en", "vi", "hi", {})

            def fake_segment(wav, max_s=12.0, vad_profile=None, **kw):
                seen_profiles.append(resolve_runtime_profile(vad_profile=vad_profile)["vad_profile"])
                return [(0, len(wav))]

            dummy = np.ones(1600, np.float32) * 0.1
            with patch("s2s_edge.audio.segment", side_effect=fake_segment):
                self.assertEqual(len(tr.translate(dummy)), 1)
                self.assertEqual(len(list(tr.translate_stream(dummy))), 1)
                self.assertEqual(len(conv.translate(dummy)), 1)
                self.assertEqual(len(list(conv.translate_stream(dummy))), 1)

            self.assertEqual(seen_profiles, ["upstream", "upstream", "upstream", "upstream"])
            ev_opts = evaluate_asr_integration.resolve_effective_options(source_mode=None, vad_profile=None, ort_policy=None)
            self.assertEqual(ev_opts["vad_profile"], "upstream")
            self.assertEqual(ev_opts["ort_policy"], "basic")
            self.assertEqual(ev_opts["source_mode"], "0")


class RealAsrVadRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = Registry(str(ROOT / "models/models.cpu.json"), "cpu", ort_policy="auto")
        cls.asr_en = cls.reg.asr("en")
        cls.en_special = load_audio(str(ROOT / "results/e2e/audio/en_to_vi_special_input.wav"))
        cls.ev0358_path = ROOT.parent / "en-vi/BTC/EV-0358_in.wav"
        cls.ev5301_path = ROOT.parent / "en-vi/BTC/EV-5301_in.wav"
        cls.ev3775_path = ROOT.parent / "en-vi/BTC/EV-3775_in.wav"

    def test_en_regression_wav_preserves_zip_under_legacy_and_breaks_under_upstream(self):
        legacy_spans = segment(self.en_special, max_s=12.0, vad_profile="legacy")
        legacy_texts = [self.asr_en(self.en_special[a:b]) for a, b in legacy_spans]
        self.assertTrue(any("67297" in t for t in legacy_texts), legacy_texts)

        upstream_spans = segment(self.en_special, max_s=12.0, vad_profile="upstream")
        upstream_texts = [self.asr_en(self.en_special[a:b]) for a, b in upstream_spans]
        self.assertFalse(any("67297" in t for t in upstream_texts), upstream_texts)

    def test_native_human_zip_clip_preserved_under_legacy(self):
        self.assertTrue(self.ev0358_path.exists(), f"Missing native clip: {self.ev0358_path}")
        ev0358 = load_audio(str(self.ev0358_path))
        legacy_spans = segment(ev0358, max_s=12.0, vad_profile="legacy")
        legacy_texts = [self.asr_en(ev0358[a:b]) for a, b in legacy_spans]
        joined_legacy = " ".join(legacy_texts)
        self.assertIn("734", joined_legacy)
        self.assertIn("67297", joined_legacy)

        upstream_spans = segment(ev0358, max_s=12.0, vad_profile="upstream")
        upstream_texts = [self.asr_en(ev0358[a:b]) for a, b in upstream_spans]
        self.assertFalse(any("67297" in t for t in upstream_texts), upstream_texts)

    def test_pause_in_digits_boundary_adjacent_and_long_utterance_real_asr(self):
        self.assertTrue(self.ev0358_path.exists() and self.ev5301_path.exists() and self.ev3775_path.exists())
        ev0358 = load_audio(str(self.ev0358_path))
        ev5301 = load_audio(str(self.ev5301_path))
        ev3775 = load_audio(str(self.ev3775_path))

        # 1. Pause inside digits (0.35s silence inserted between "6" and "7297")
        cut = int(18.54 * SR)
        pause_digits = np.concatenate([self.en_special[int(14.0 * SR) : cut], np.zeros(int(0.35 * SR), np.float32), self.en_special[cut:]])
        leg_pause_texts = [self.asr_en(pause_digits[a:b]) for a, b in segment(pause_digits, max_s=12.0, vad_profile="legacy")]
        self.assertTrue(any("67297" in t for t in leg_pause_texts), leg_pause_texts)
        up_pause_texts = [self.asr_en(pause_digits[a:b]) for a, b in segment(pause_digits, max_s=12.0, vad_profile="upstream")]
        self.assertFalse(any("67297" in t for t in up_pause_texts), up_pause_texts)

        # 2. Boundary-adjacent clauses (0.45s gap between EV-0358 and EV-5301)
        boundary_wav = np.concatenate([ev0358, np.zeros(int(0.45 * SR), np.float32), ev5301])
        leg_bound_texts = " ".join(
            self.asr_en(boundary_wav[a:b]) for a, b in segment(boundary_wav, max_s=12.0, vad_profile="legacy")
        )
        self.assertIn("67297", leg_bound_texts)
        self.assertIn("800", leg_bound_texts)

        # 3. Long utterance (>12s max_s) across multiple clauses
        long_wav = np.concatenate(
            [
                self.en_special[int(14.0 * SR) :],
                np.zeros(int(0.4 * SR), np.float32),
                ev0358,
                np.zeros(int(0.4 * SR), np.float32),
                ev5301,
                np.zeros(int(0.4 * SR), np.float32),
                ev3775,
            ]
        )
        self.assertGreater(len(long_wav) / SR, 12.0)
        long_spans = segment(long_wav, max_s=12.0, vad_profile="legacy")
        self.assertTrue(all((b - a) / SR <= 12.0 + 1e-6 for a, b in long_spans), long_spans)
        long_joined = " ".join(self.asr_en(long_wav[a:b]) for a, b in long_spans)
        for token in ("67297", "734", "800", "530"):
            self.assertIn(token, long_joined, long_joined)

    def test_tts_format_and_source_mode_bound_to_instances_despite_env_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            models_cfg = base / "models.json"
            models_cfg.write_text(
                json.dumps(
                    {
                        "asr_vi": {"onnx": "asr_vi.onnx", "name": "asr_vi"},
                        "asr_xx_frontend": {"onnx": "asr_f.onnx", "name": "asr_f"},
                        "asr_xx_encoder": {"onnx": "asr_e.onnx", "name": "asr_e"},
                        "mt": {"onnx": "mt.onnx", "name": "mt"},
                        "tts_vi": {"onnx": "tts_vi.onnx", "name": "tts_vi"},
                        "tts_zh": {"onnx": "tts_zh.onnx", "name": "tts_zh"},
                        "tts_enko": {"onnx": "tts_enko.onnx", "name": "tts_enko"},
                    }
                )
            )
            captured_calls = {"vi_legacy": [], "vi_hybrid": [], "en_legacy": [], "en_hybrid": [], "zh_legacy": [], "zh_hybrid": []}

            def fake_make_graph(mdir, spec, backend, ort_policy="auto"):
                name = spec.get("name")

                def _run(**kwargs):
                    if name == "tts_vi":
                        b = bytes(int(v) for v in kwargs["text_bytes"].reshape(-1) if int(v) != 0)
                        tag = _active_tag[0]
                        captured_calls[tag].append(b.decode("utf-8", errors="replace"))
                        return [np.ones((1, 512), dtype=np.float32), np.array([2], dtype=np.int32), np.array([0.0], dtype=np.float32)]
                    if name == "tts_zh":
                        b = bytes(int(v) for v in kwargs["text_bytes"].reshape(-1) if int(v) != 0)
                        tag = _active_tag[0]
                        captured_calls[tag].append(b.decode("utf-8", errors="replace"))
                        return [np.ones((1, 512), dtype=np.float32), np.array([2], dtype=np.int32)]
                    if name == "tts_enko":
                        cps = "".join(chr(int(c)) for c in kwargs["cp"].reshape(-1) if int(c) > 0)
                        tag = _active_tag[0]
                        captured_calls[tag].append(cps)
                        return [np.ones((1, 460800), dtype=np.float32), np.array([2.0], dtype=np.float32)]
                    raise AssertionError(f"Unexpected graph {name}")

                return _run

            _active_tag = ["vi_legacy"]
            with patch("s2s_edge.registry.make_graph", fake_make_graph), patch.dict(
                os.environ, {"S2S_TTS_FORMAT": "hybrid", "S2S_NEURAL_TN": "1"}, clear=False
            ):
                reg_legacy = Registry(str(models_cfg), "cpu", ort_policy="basic", vad_profile="legacy", source_mode="0", tts_format="legacy")
                reg_hybrid = Registry(str(models_cfg), "cpu", ort_policy="basic", vad_profile="legacy", source_mode="1", tts_format="hybrid")
                tts_vi_leg = reg_legacy.tts("vi")
                tts_vi_hyb = reg_hybrid.tts("vi")
                tts_en_leg = reg_legacy.tts("en")
                tts_en_hyb = reg_hybrid.tts("en")
                tts_zh_leg = reg_legacy.tts("zh")
                tts_zh_hyb = reg_hybrid.tts("zh")
                eng_zero = Engine(reg_legacy)
                eng_one = Engine(reg_hybrid)

            # Flip environment AFTER instances were constructed; instances must stay bound to init modes
            sample_vi = "Cuộc họp lúc 3:30 PM giá $15.50."
            sample_en = "Meeting at 3:30 PM; ticket $15.50, temperature 37.5°C."
            sample_zh = "温度37.5°C。"
            with patch.dict(os.environ, {"S2S_TTS_FORMAT": "hybrid", "S2S_NEURAL_TN": "1"}, clear=False):
                _active_tag[0] = "vi_legacy"
                tts_vi_leg.synthesize(sample_vi)
                _active_tag[0] = "en_legacy"
                tts_en_leg.synthesize(sample_en)
                _active_tag[0] = "zh_legacy"
                tts_zh_leg.synthesize(sample_zh)
                self.assertFalse(eng_zero.neural_enabled())

            with patch.dict(os.environ, {"S2S_TTS_FORMAT": "legacy", "S2S_NEURAL_TN": "0"}, clear=False):
                _active_tag[0] = "vi_hybrid"
                tts_vi_hyb.synthesize(sample_vi)
                _active_tag[0] = "en_hybrid"
                tts_en_hyb.synthesize(sample_en)
                _active_tag[0] = "zh_hybrid"
                tts_zh_hyb.synthesize(sample_zh)
                self.assertTrue(eng_one.neural_enabled())

            self.assertEqual(tts_vi_leg.format_trace["stage"], "tts_legacy")
            self.assertEqual(tts_vi_hyb.format_trace["stage"], "tts_hybrid")
            self.assertNotEqual(tts_vi_leg.tts_input_text, tts_vi_hyb.tts_input_text)
            self.assertNotEqual(captured_calls["vi_legacy"], captured_calls["vi_hybrid"])

            self.assertEqual(tts_zh_leg.format_trace["stage"], "tts_legacy")
            self.assertEqual(tts_zh_hyb.format_trace["stage"], "tts_hybrid")
            self.assertEqual(tts_en_leg.format_trace["stage"], "tts_legacy")
            self.assertEqual(tts_en_hyb.format_trace["stage"], "tts_hybrid")
            self.assertIn("three : thirty", tts_en_leg.tts_input_text)
            self.assertNotEqual(tts_en_leg.tts_input_text, tts_en_hyb.tts_input_text)
            self.assertNotEqual(captured_calls["en_legacy"], captured_calls["en_hybrid"])

    def test_start_s2s_web_reuse_mismatch_stale_code_busy_reload_and_invalid_env(self):
        disk_fp = start_s2s_web.disk_code_fingerprint()
        matching_profile = {
            "source_mode": "0",
            "vad_profile": "legacy",
            "ort_policy": "basic",
            "tts_format": "hybrid",
        }
        matching_health = {
            "ok": True,
            "pid": 43210,
            "pending": 0,
            **matching_profile,
            "effective_profile": dict(matching_profile),
            "loaded_code_fingerprint": dict(disk_fp),
        }

        popen_calls = []
        kill_calls = []

        class FakeProc:
            pid = 99991
            def poll(self):
                return None

        def fake_popen(cmd, **kw):
            popen_calls.append((cmd, kw))
            return FakeProc()

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        with tempfile.TemporaryDirectory() as td:
            log_file = Path(td) / "web.log"
            out_dir = Path(td) / "web_out"
            args_reuse = start_s2s_web.parse_args(
                ["--port", "8791", "--out-dir", str(out_dir), "--log-file", str(log_file),
                 "--source-mode", "0", "--vad-profile", "legacy", "--ort-policy", "basic", "--tts-format", "hybrid"]
            )
            res = start_s2s_web.launch_or_reuse(
                args_reuse,
                base_env={},
                health_fn=lambda _u: dict(matching_health),
                fingerprint_fn=lambda: dict(disk_fp),
                popen_fn=fake_popen,
                kill_fn=fake_kill,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(res["action"], "reused")
            self.assertEqual(popen_calls, [])
            self.assertEqual(kill_calls, [])

            # Profile mismatch without --reload must fail non-zero without spawning or killing
            args_mismatch = start_s2s_web.parse_args(
                ["--port", "8791", "--out-dir", str(out_dir), "--log-file", str(log_file),
                 "--source-mode", "1", "--vad-profile", "legacy", "--ort-policy", "basic", "--tts-format", "hybrid"]
            )
            with self.assertRaises(start_s2s_web.LauncherError) as ctx_prof:
                start_s2s_web.launch_or_reuse(
                    args_mismatch,
                    base_env={},
                    health_fn=lambda _u: dict(matching_health),
                    fingerprint_fn=lambda: dict(disk_fp),
                    popen_fn=fake_popen,
                    kill_fn=fake_kill,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(ctx_prof.exception.code, 2)
            self.assertEqual(kill_calls, [])

            # Stale loaded code fingerprint on server must fail without --reload
            stale_fp = {**disk_fp, "server_files_sha256": "0" * 64}
            stale_health = {**matching_health, "loaded_code_fingerprint": stale_fp}
            with self.assertRaises(start_s2s_web.LauncherError) as ctx_stale:
                start_s2s_web.launch_or_reuse(
                    args_reuse,
                    base_env={},
                    health_fn=lambda _u: dict(stale_health),
                    fingerprint_fn=lambda: dict(disk_fp),
                    popen_fn=fake_popen,
                    kill_fn=fake_kill,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(ctx_stale.exception.code, 2)
            self.assertIn("stale_code", str(ctx_stale.exception))
            self.assertEqual(kill_calls, [])

            # --reload refuses when pending > 0 (never kills busy server)
            args_reload = start_s2s_web.parse_args(
                ["--port", "8791", "--out-dir", str(out_dir), "--log-file", str(log_file),
                 "--source-mode", "1", "--vad-profile", "legacy", "--ort-policy", "basic", "--tts-format", "hybrid", "--reload"]
            )
            busy_health = {**matching_health, "pending": 2}
            with self.assertRaises(start_s2s_web.LauncherError) as ctx_busy:
                start_s2s_web.launch_or_reuse(
                    args_reload,
                    base_env={},
                    health_fn=lambda _u: dict(busy_health),
                    fingerprint_fn=lambda: dict(disk_fp),
                    popen_fn=fake_popen,
                    kill_fn=fake_kill,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(ctx_busy.exception.code, 3)
            self.assertEqual(kill_calls, [])

            # --reload refuses when PID missing
            no_pid_health = {k: v for k, v in matching_health.items() if k != "pid"}
            with self.assertRaises(start_s2s_web.LauncherError) as ctx_nopid:
                start_s2s_web.launch_or_reuse(
                    args_reload,
                    base_env={},
                    health_fn=lambda _u: dict(no_pid_health),
                    fingerprint_fn=lambda: dict(disk_fp),
                    popen_fn=fake_popen,
                    kill_fn=fake_kill,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(ctx_nopid.exception.code, 3)
            self.assertEqual(kill_calls, [])

            # --reload succeeds when idle (pending == 0) and PID present
            # Sequence: initial healthy -> None after SIGTERM -> new healthy after spawn
            seq = [dict(matching_health), None, {**matching_health, "source_mode": "1", "pid": 99991}]
            def seq_health(_u):
                return seq.pop(0) if seq else {**matching_health, "source_mode": "1", "pid": 99991}

            res_rel = start_s2s_web.launch_or_reuse(
                args_reload,
                base_env={},
                health_fn=seq_health,
                fingerprint_fn=lambda: dict(disk_fp),
                popen_fn=fake_popen,
                kill_fn=fake_kill,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(res_rel["action"], "reloaded")
            self.assertEqual(len(kill_calls), 1)
            self.assertEqual(kill_calls[0][0], 43210)
            self.assertEqual(len(popen_calls), 1)
            self.assertEqual((out_dir / "server.pid").read_text(), "99991")

            # Invalid env rejected BEFORE /api/health check even if server would be healthy
            health_called = {"n": 0}
            def spy_health(_u):
                health_called["n"] += 1
                return dict(matching_health)

            args_default = start_s2s_web.parse_args(["--port", "8791"])
            with self.assertRaises(ValueError):
                start_s2s_web.launch_or_reuse(
                    args_default,
                    base_env={"S2S_VAD_PROFILE": "bogus_profile"},
                    health_fn=spy_health,
                )
            self.assertEqual(health_called["n"], 0)


if __name__ == "__main__":
    unittest.main()
