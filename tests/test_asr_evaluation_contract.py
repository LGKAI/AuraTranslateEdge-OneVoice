"""Contract tests for T1 evaluation harness, cache invalidation, partial retention, and verified rollback."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import evaluate_asr_integration as harness
import rollback_baseline


class AsrEvaluationContractTests(unittest.TestCase):
    def test_rejects_reserved_before_after_and_existing_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_root = td_path / "runs"
            out_root.mkdir()
            wav = td_path / "clip.wav"
            wav.write_bytes(b"RIFF0000WAVE")
            manifest = td_path / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav)}]}))

            for reserved in ("before", "after", "BEFORE"):
                with self.assertRaises(ValueError):
                    harness.validate_run_id(reserved)
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "tools/evaluate_asr_integration.py"), "--run-id", reserved, "--manifest", str(manifest), "--out-root", str(out_root)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)

            existing = out_root / "run_001"
            existing.mkdir()
            (existing / "summary.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                harness.run_evaluation(run_id="run_001", manifest_path=manifest, out_root=out_root)

    def test_weight_cache_reuse_and_invalidation_on_change(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            w = base / "weights.bin"
            w.write_bytes(b"initial-weights-v1")
            cache = base / "cache.json"

            first = harness.hash_files_with_cache([w], base, cache_path=cache, persist_cache=True)
            self.assertFalse(first["weights.bin"]["cache_hit"])
            h1 = first["weights.bin"]["sha256"]

            second = harness.hash_files_with_cache([w], base, cache_path=cache, persist_cache=True)
            self.assertTrue(second["weights.bin"]["cache_hit"])
            self.assertEqual(second["weights.bin"]["sha256"], h1)

            w.write_bytes(b"modified-weights-v2!")
            third = harness.hash_files_with_cache([w], base, cache_path=cache, persist_cache=True)
            self.assertFalse(third["weights.bin"]["cache_hit"])
            self.assertNotEqual(third["weights.bin"]["sha256"], h1)

    def test_write_run_manifest_separates_baseline_from_candidate_and_records_effective_options(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_dir = base / "out"
            wav = base / "in.wav"
            wav.write_bytes(b"wavbytes")
            mf = base / "cases.json"
            mf.write_text("[]")
            src = base / "mod.py"
            src.write_text("x = 1\n")
            mdl = base / "m.onnx"
            mdl.write_bytes(b"onnx")

            manifest = harness.write_run_manifest(
                out_dir=out_dir,
                run_id="cand_01",
                manifest_path=mf,
                cases=[{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav)}],
                effective_options={"source_mode": "0", "vad_profile": "legacy", "ort_policy": "auto"},
                root=base,
                cache_path=base / "cache.json",
                baseline_freeze_path=ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json",
                source_paths=[src],
                model_paths=[mdl],
            )
            self.assertEqual(manifest["run_id"], "cand_01")
            self.assertIn("baseline_reference", manifest)
            self.assertIn("candidate_fingerprint", manifest)
            self.assertEqual(manifest["effective_options"]["vad_profile"], "legacy")
            self.assertTrue((out_dir / "run_manifest.json").exists())

    def test_case_error_exits_nonzero_and_retains_partial_results(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "runs"
            # Generate two real 16kHz wav files via scipy/wave so load_audio succeeds cleanly
            import numpy as np
            import soundfile as sf

            wav1 = base / "ok.wav"
            wav2 = base / "fail.wav"
            tone = (0.1 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32)
            sf.write(str(wav1), tone, 16000)
            sf.write(str(wav2), tone, 16000)

            manifest = base / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"id": "case_ok", "src": "en", "tgt": "vi", "input_wav": str(wav1), "input_kind": "synthetic"},
                            {"id": "case_err", "src": "ko", "tgt": "vi", "input_wav": str(wav2), "input_kind": "synthetic"},
                        ]
                    }
                )
            )

            class FakeAsr:
                max_s = 12.0

            class FakeReg:
                def warmup(self, dirs, log=None):
                    return {"asr_xx": (0.01, 0.002)}

            class FakeEngine:
                def asr(self, lang):
                    return FakeAsr()

                def translate_segment(self, wav, src, tgt):
                    if src == "ko":
                        raise RuntimeError("injected ko failure")
                    return {
                        "src": src,
                        "tgt": tgt,
                        "asr": "hello",
                        "mt": "xin chao",
                        "tts_text": "xin chao",
                        "tts_format": "hybrid",
                        "wav": np.ones(8000, dtype=np.float32) * 0.05,
                        "sr": 16000,
                        "timing": {"asr_s": 0.1, "mt_s": 0.2, "tts_s": 0.3, "total_s": 0.6, "graph_s": 0.5, "host_s": 0.1, "graph_calls": 4},
                    }

            old_env = os.environ.get("S2S_NEURAL_TN")
            os.environ["S2S_NEURAL_TN"] = "0"
            try:
                code, summary = harness.run_evaluation(
                    run_id="partial_fail_test",
                    manifest_path=manifest,
                    out_root=out_root,
                    source_mode=None,
                    vad_profile="legacy",
                    ort_policy="auto",
                    cache_path=base / "cache.json",
                    baseline_freeze_path=ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json",
                    engine_factory=lambda *_: (FakeReg(), FakeEngine()),
                    segment_fn=lambda wav, max_s, prof: [(0, len(wav))],
                )
            finally:
                if old_env is None:
                    os.environ.pop("S2S_NEURAL_TN", None)
                else:
                    os.environ["S2S_NEURAL_TN"] = old_env

            self.assertNotEqual(code, 0)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["completed_count"], 2)
            self.assertEqual(summary["cases"][0]["status"], "ok")
            self.assertEqual(summary["cases"][1]["status"], "error")
            self.assertIn("run_load_s", summary["cases"][0])
            self.assertIn("warmup_s", summary["cases"][0])
            self.assertEqual(summary["cases"][0]["segment_count"], 1)
            self.assertTrue((out_root / "partial_fail_test/summary.json").exists())
            self.assertTrue((out_root / "partial_fail_test/case_ok.wav").exists())
            saved = json.loads((out_root / "partial_fail_test/summary.json").read_text())
            self.assertEqual(saved["failure_count"], 1)

    def test_caller_env_preserved_when_source_mode_omitted(self):
        old = os.environ.get("S2S_NEURAL_TN")
        os.environ["S2S_NEURAL_TN"] = "1"
        try:
            eff = harness.resolve_effective_options(source_mode=None, vad_profile="legacy", ort_policy="auto")
            self.assertEqual(eff["source_mode"], "1")
            self.assertEqual(eff["source_mode_origin"], "caller_env")
        finally:
            if old is None:
                os.environ.pop("S2S_NEURAL_TN", None)
            else:
                os.environ["S2S_NEURAL_TN"] = old

    def test_rollback_checks_expected_hashes_including_config_and_docs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            freeze_dir = root / "freeze"
            snap = freeze_dir / "source_snapshot"
            (root / "docs").mkdir()
            (root / "models").mkdir()
            (snap / "docs").mkdir(parents=True)
            (snap / "models").mkdir(parents=True)

            # Baseline files
            (snap / "docs/spec.md").write_bytes(b"v1 spec")
            (snap / "models/cfg.json").write_bytes(b'{"a":1}')
            freeze_manifest = {
                "freeze_id": "test",
                "source_files": {
                    "docs/spec.md": {"sha256": hashlib.sha256(b"v1 spec").hexdigest()},
                    "models/cfg.json": {"sha256": hashlib.sha256(b'{"a":1}').hexdigest()},
                },
            }
            (freeze_dir / "freeze_manifest.json").write_text(json.dumps(freeze_manifest))

            # Current candidate modified docs + config + added new file
            (root / "docs/spec.md").write_bytes(b"v2 spec")
            (root / "models/cfg.json").write_bytes(b'{"a":2}')
            (root / "new_test.py").write_bytes(b"print(1)")
            expected = {
                "docs/spec.md": hashlib.sha256(b"v2 spec").hexdigest(),
                "models/cfg.json": hashlib.sha256(b'{"a":2}').hexdigest(),
                "new_test.py": hashlib.sha256(b"print(1)").hexdigest(),
            }

            # Unexpected edit to docs triggers abort
            (root / "docs/spec.md").write_bytes(b"untracked edit")
            ok, rep = rollback_baseline.verify_and_rollback(freeze_dir=freeze_dir, expected_current=expected, root=root, apply=True)
            self.assertFalse(ok)
            self.assertEqual(rep["status"], "aborted_precheck")
            self.assertEqual((root / "models/cfg.json").read_bytes(), b'{"a":2}')

            # Restore matching expected state -> rollback succeeds and verifies post hashes
            (root / "docs/spec.md").write_bytes(b"v2 spec")
            ok, rep = rollback_baseline.verify_and_rollback(freeze_dir=freeze_dir, expected_current=expected, root=root, apply=True)
            self.assertTrue(ok)
            self.assertEqual(rep["status"], "applied")
            self.assertEqual((root / "docs/spec.md").read_bytes(), b"v1 spec")
            self.assertEqual((root / "models/cfg.json").read_bytes(), b'{"a":1}')
            self.assertFalse((root / "new_test.py").exists())

    def test_expected_candidate_revision_mismatch_and_invalid_profile_fail(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "runs"
            wav = base / "clip.wav"
            wav.write_bytes(b"RIFF0000WAVE")
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav)}]}))

            with self.assertRaises(ValueError):
                harness.resolve_effective_options(source_mode=None, vad_profile="bad_profile", ort_policy="auto")

            code, summary = harness.run_evaluation(
                run_id="rev_mismatch",
                manifest_path=manifest,
                out_root=out_root,
                vad_profile="legacy",
                ort_policy="auto",
                cache_path=base / "cache.json",
                baseline_freeze_path=ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json",
                expected_candidate_rev="0" * 64,
            )
            self.assertNotEqual(code, 0)
            self.assertEqual(summary["status"], "failed_revision_mismatch")
            saved_manifest = json.loads((out_root / "rev_mismatch/run_manifest.json").read_text())
            self.assertEqual(saved_manifest["run_status"], "failed_revision_mismatch")

    def test_mid_run_source_drift_causes_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "runs"
            import numpy as np
            import soundfile as sf

            wav = base / "clip.wav"
            sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32), 16000)
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav)}]}))

            # Point _collect_source_files at a tiny synthetic temp source file that mutates mid-run
            temp_src = base / "dummy_src.py"
            temp_src.write_text("v = 1\n")
            orig_collect = harness._collect_source_files
            harness._collect_source_files = lambda _root: [temp_src]
            try:
                class FakeAsr:
                    max_s = 12.0

                class FakeReg:
                    def warmup(self, dirs, log=None):
                        return {}

                class MutatingEngine:
                    def asr(self, lang):
                        return FakeAsr()

                    def translate_segment(self, audio, src, tgt):
                        temp_src.write_text("v = 2  # mid-run drift\n")
                        return {
                            "src": src,
                            "tgt": tgt,
                            "asr": "hi",
                            "mt": "xin chao",
                            "tts_text": "xin chao",
                            "tts_format": "hybrid",
                            "wav": np.ones(4000, dtype=np.float32) * 0.05,
                            "sr": 16000,
                            "timing": {"asr_s": 0.1, "mt_s": 0.1, "tts_s": 0.1, "total_s": 0.3, "graph_s": 0.2, "host_s": 0.1, "graph_calls": 1},
                        }

                code, summary = harness.run_evaluation(
                    run_id="drift_test",
                    manifest_path=manifest,
                    out_root=out_root,
                    cache_path=base / "cache.json",
                    baseline_freeze_path=ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json",
                    engine_factory=lambda *_: (FakeReg(), MutatingEngine()),
                    segment_fn=lambda w, m, p: [(0, len(w))],
                )
            finally:
                harness._collect_source_files = orig_collect

            self.assertNotEqual(code, 0)
            self.assertTrue(summary["drift_detected"])
            self.assertEqual(summary["run_status"], "failed_drift")

    def test_comparative_run_fails_before_engine_when_baseline_missing_empty_or_tampered(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "runs"
            wav = base / "clip.wav"
            sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32), 16000)
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav)}]}))

            engine_calls = {"count": 0}

            class FakeAsr:
                max_s = 12.0

            class FakeReg:
                def warmup(self, dirs, log=None):
                    return {}
                def graph_telemetry(self):
                    return {}

            class FakeEngine:
                def asr(self, lang):
                    return FakeAsr()
                def translate_segment(self, audio, src, tgt):
                    return {
                        "src": src, "tgt": tgt, "asr": "hi", "mt": "xin chao",
                        "tts_text": "xin chao", "tts_format": "hybrid",
                        "wav": np.ones(4000, dtype=np.float32) * 0.05, "sr": 16000,
                        "timing": {"asr_s": 0.1, "mt_s": 0.1, "tts_s": 0.1, "total_s": 0.3, "graph_s": 0.2, "host_s": 0.1, "graph_calls": 1},
                    }

            def counting_factory(*_a, **_kw):
                engine_calls["count"] += 1
                return FakeReg(), FakeEngine()

            # 1. Missing baseline freeze -> non-zero exit & engine NEVER called
            code_miss, sum_miss = harness.run_evaluation(
                run_id="r4_missing_freeze",
                manifest_path=manifest,
                out_root=out_root,
                cache_path=base / "cache.json",
                baseline_freeze_path=base / "nonexistent_freeze.json",
                engine_factory=counting_factory,
                segment_fn=lambda w, m, p: [(0, len(w))],
            )
            self.assertNotEqual(code_miss, 0)
            self.assertEqual(engine_calls["count"], 0)
            self.assertEqual(sum_miss["run_status"], "failed_baseline_verification")
            self.assertFalse(sum_miss["baseline_verified"])

            # 2. Empty evidence_files -> non-zero exit & engine NEVER called
            empty_freeze = base / "empty_freeze.json"
            empty_freeze.write_text(json.dumps({"freeze_id": "empty", "evidence_files": {}}))
            code_emp, _ = harness.run_evaluation(
                run_id="r4_empty_freeze",
                manifest_path=manifest,
                out_root=out_root,
                cache_path=base / "cache.json",
                baseline_freeze_path=empty_freeze,
                engine_factory=counting_factory,
                segment_fn=lambda w, m, p: [(0, len(w))],
            )
            self.assertNotEqual(code_emp, 0)
            self.assertEqual(engine_calls["count"], 0)

            # 3. Tampered baseline freeze -> non-zero exit & engine NEVER called
            bad_freeze = base / "tampered_freeze.json"
            bad_freeze.write_text(json.dumps({"freeze_id": "bad", "evidence_files": {"results/asr_integration/before/summary.json": {"sha256": "deadbeef"}}}))
            code_bad, sum_bad = harness.run_evaluation(
                run_id="r4_tampered_freeze",
                manifest_path=manifest,
                out_root=out_root,
                cache_path=base / "cache.json",
                baseline_freeze_path=bad_freeze,
                engine_factory=counting_factory,
                segment_fn=lambda w, m, p: [(0, len(w))],
            )
            self.assertNotEqual(code_bad, 0)
            self.assertEqual(engine_calls["count"], 0)
            self.assertEqual(sum_bad["run_status"], "failed_baseline_verification")

            # 4. Standalone mode succeeds without valid freeze and clearly labels standalone_non_comparative
            code_std, sum_std = harness.run_evaluation(
                run_id="r4_standalone",
                manifest_path=manifest,
                out_root=out_root,
                cache_path=base / "cache.json",
                baseline_freeze_path=base / "nonexistent_freeze.json",
                standalone=True,
                engine_factory=counting_factory,
                segment_fn=lambda w, m, p: [(0, len(w))],
            )
            self.assertEqual(code_std, 0)
            self.assertEqual(engine_calls["count"], 1)
            self.assertEqual(sum_std["comparison_mode"], "standalone_non_comparative")
            self.assertFalse(sum_std["baseline_verified"])

    def test_ablate_profiles_manifest_reserved_overwrite_baseline_and_drift(self):
        import ablate_profiles
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_root = base / "abl_runs"
            out_root.mkdir()
            wav = base / "clip.wav"
            sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 20, 16000))).astype(np.float32), 16000)
            split_path = base / "split.json"
            split_obj = {
                "native_en_dev": [
                    {
                        "id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav),
                        "input_kind": "native_human_speech",
                        "written": "Age 32 years old.", "expected_entities": ["32"],
                    }
                ],
                "native_en_holdout": [],
                "synthetic_dev": [],
                "paired_vad_core_ids": ["c1"],
                "paired_ort_core_cases": [{"id": "c1", "src": "en", "tgt": "vi", "input_wav": str(wav), "input_kind": "native_human_speech"}],
            }
            split_path.write_text(json.dumps(split_obj))

            # Reject reserved run ID including t3_ablation & existing run directory
            for bad_id in ("before", "after", "t3_ablation"):
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "tools/ablate_profiles.py"), "vad", "--run-id", bad_id, "--split", str(split_path), "--out-root", str(out_root)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(proc.returncode, 0)

            existing = out_root / "already_exists"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                ablate_profiles.run_ablation(step="vad", run_id="already_exists", split_path=split_path, out_root=out_root)

            # Comparative mode blocks BEFORE calling engine_factory on missing baseline freeze
            calls = {"n": 0}
            def fake_factory(*_a, **_kw):
                calls["n"] += 1
                class R:
                    _cache = {}
                    def warmup(self, d, log=None): return {}
                    def graph_telemetry(self): return {}
                    def release(self, k): pass
                class E:
                    def asr(self, l): return type("A", (), {"max_s": 12.0})()
                    def translate_segment(self, audio, src, tgt):
                        return {
                            "src": src, "tgt": tgt, "asr": "Age 32", "asr_before_normalization": "Age 32",
                            "mt": "32 tuoi", "tts_text": "32 tuoi", "wav": np.ones(1600, np.float32) * 0.05,
                            "sr": 16000, "timing": {"asr_s": 0.1, "mt_s": 0.1, "tts_s": 0.1, "total_s": 0.3, "graph_s": 0.2, "host_s": 0.1, "graph_calls": 1},
                        }
                return R(), E()

            code_fail, _ = ablate_profiles.run_ablation(
                step="vad", run_id="r3_bad_base", split_path=split_path, out_root=out_root,
                repeats=1, cache_path=base / "cache.json", baseline_freeze_path=base / "no_freeze.json",
                engine_factory=fake_factory, segment_fn=lambda w, m, p: [(0, len(w))],
            )
            self.assertNotEqual(code_fail, 0)
            self.assertEqual(calls["n"], 0)

            # Valid baseline freeze -> succeeds and writes full run_manifest with seeded_orders and hashes
            src_dummy = base / "monitored.py"
            src_dummy.write_text("x = 1\n")
            mdl_dummy = base / "fake_model.onnx"
            mdl_dummy.write_bytes(b"ONNX")
            code_ok, sum_ok = ablate_profiles.run_ablation(
                step="vad", run_id="r3_valid_run", split_path=split_path, out_root=out_root,
                repeats=1, seed=42, cache_path=base / "cache.json",
                baseline_freeze_path=ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json",
                engine_factory=fake_factory, segment_fn=lambda w, m, p: [(0, len(w))],
                source_paths=[src_dummy], model_paths=[mdl_dummy],
            )
            self.assertEqual(code_ok, 0)
            self.assertTrue(sum_ok["baseline_verified"])
            rm = json.loads((out_root / "r3_valid_run/run_manifest.json").read_text())
            self.assertIn("seeded_orders", rm)
            self.assertIn("input_hashes", rm)
            self.assertIn("model_hashes", rm)
            self.assertIn("source_hashes", rm)
            self.assertFalse(rm["drift_check"]["drift_detected"])

            # Mid-run source drift causes non-zero exit & failed_drift
            def drifting_factory(*_a, **_kw):
                reg, eng = fake_factory()
                orig_ts = eng.translate_segment
                def drifted(audio, src, tgt):
                    src_dummy.write_text("x = 2  # drift\n")
                    return orig_ts(audio, src, tgt)
                eng.translate_segment = drifted
                return reg, eng

            code_drift, sum_drift = ablate_profiles.run_ablation(
                step="vad", run_id="r3_drift_run", split_path=split_path, out_root=out_root,
                repeats=1, seed=42, standalone=True, cache_path=base / "cache.json",
                engine_factory=drifting_factory, segment_fn=lambda w, m, p: [(0, len(w))],
                source_paths=[src_dummy], model_paths=[mdl_dummy],
            )
            self.assertNotEqual(code_drift, 0)
            self.assertEqual(sum_drift["run_status"], "failed_drift")
            self.assertTrue(sum_drift["drift_detected"])


if __name__ == "__main__":
    unittest.main()
