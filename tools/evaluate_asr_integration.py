"""Reproducible ASR/S2S integration evaluation harness with immutable run IDs and verified manifests."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(ROOT / "s2s_edge"))

RESERVED_RUN_IDS = {"before", "after", "baseline_freeze", "manifests", "t3_ablation"}
DEFAULT_MANIFEST = ROOT / "results/next_steps/manifests/asr_integration_6dir.json"
DEFAULT_OUT_ROOT = ROOT / "results/next_steps"
DEFAULT_CACHE_PATH = ROOT / "results/next_steps/baseline_freeze/model_weights_cache.json"
DEFAULT_BASELINE_FREEZE = ROOT / "results/next_steps/baseline_freeze/freeze_manifest.json"
DEFAULT_MODELS = ROOT / "models/models.cpu.json"

TRACKED_SOURCE_DIRS = ("s2s_edge/s2s_edge", "tools", "tests", "models")


def validate_run_id(run_id: str) -> str:
    """Validate run ID and reject reserved baseline labels or path traversal."""
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    norm = run_id.strip()
    if norm.lower() in RESERVED_RUN_IDS:
        raise ValueError(
            f"Refusing reserved run_id {run_id!r}: baseline labels ('before'/'after') cannot be impersonated or overwritten."
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", norm) or norm in (".", ".."):
        raise ValueError(f"Invalid run_id {run_id!r}: allowed pattern is [A-Za-z0-9._-]+")
    return norm


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files_with_cache(
    paths: list[Path],
    base_dir: Path,
    cache_path: Path | None = None,
    persist_cache: bool = True,
) -> dict[str, dict]:
    """Hash files using a size+mtime validated cache, invalidating whenever a file changes."""
    cache_data: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_data = loaded.get("entries", loaded) if isinstance(loaded, dict) else {}
        except Exception:
            cache_data = {}

    results: dict[str, dict] = {}
    dirty = False
    for p in sorted({Path(x).resolve() for x in paths}):
        try:
            rel = str(p.relative_to(base_dir.resolve()))
        except ValueError:
            rel = str(p)
        if not p.exists():
            results[rel] = {"exists": False, "sha256": None, "cache_hit": False}
            continue
        st = p.stat()
        cached = cache_data.get(rel)
        if (
            isinstance(cached, dict)
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
            and len(cached["sha256"]) == 64
        ):
            results[rel] = {
                "exists": True,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha256": cached["sha256"],
                "cache_hit": True,
            }
        else:
            digest = _sha256_file(p)
            record = {
                "path": rel,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha256": digest,
            }
            cache_data[rel] = record
            dirty = True
            results[rel] = {
                "exists": True,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha256": digest,
                "cache_hit": False,
            }

    if persist_cache and cache_path is not None and dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "entries": cache_data}
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_path)

    return results


def _collect_source_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for rel_dir in TRACKED_SOURCE_DIRS:
        base = root / rel_dir
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or ".cache" in p.parts or "source" in p.parts:
                continue
            if p.suffix in (".py", ".json", ".crfsuite") or p.name in ("models.cpu.json", "manifest.json"):
                out.append(p)
    return out


def _collect_model_files(root: Path, models_cfg_path: Path) -> list[Path]:
    files: list[Path] = []
    if models_cfg_path.exists():
        cfg = json.loads(models_cfg_path.read_text(encoding="utf-8"))
        mdir = models_cfg_path.parent
        for spec in cfg.values():
            if isinstance(spec, dict) and "onnx" in spec:
                p = (mdir / spec["onnx"]).resolve()
                files.append(p)
                ext = p.with_name(p.name + ".data")
                if ext.exists():
                    files.append(ext)
    assets_dir = root / "s2s_edge/assets"
    if assets_dir.exists():
        files.extend(p for p in sorted(assets_dir.rglob("*")) if p.is_file())
    qwen_weights = root / "models/source/qwen3-4b-instruct-mlx4/model.safetensors"
    if qwen_weights.exists():
        files.append(qwen_weights)
    return files


def compute_code_fingerprint(root: Path, source_hashes: dict[str, dict]) -> dict:
    """Compute candidate source fingerprint plus git HEAD and unified diff hash."""
    head = None
    dirty = None
    diff_sha = None
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        diff_txt = subprocess.check_output(["git", "-C", str(root), "diff"], stderr=subprocess.DEVNULL)
        diff_sha = hashlib.sha256(diff_txt).hexdigest()
    except Exception:
        pass
    h = hashlib.sha256()
    for rel in sorted(source_hashes):
        h.update(f"{rel}:{source_hashes[rel].get('sha256')}\n".encode("utf-8"))
    return {
        "git_head": head,
        "git_diff_sha256": diff_sha,
        "git_dirty_entries": len(dirty.splitlines()) if dirty else 0,
        "source_tree_sha256": h.hexdigest(),
    }


def collect_environment_info(env: dict[str, str] | None = None) -> dict:
    """Record Python, library versions, CPU metadata, and relevant S2S environment variables."""
    env_map = env if env is not None else os.environ
    libs = {}
    for pkg in ("numpy", "scipy", "onnxruntime", "soundfile", "sentencepiece", "mlx", "mlx-lm"):
        try:
            libs[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            libs[pkg] = None
    cpu_brand = None
    try:
        cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        cpu_brand = platform.processor() or None
    keys = [
        "S2S_NEURAL_TN",
        "S2S_TTS_FORMAT",
        "S2S_VAD_PROFILE",
        "S2S_ORT_POLICY",
        "S2S_MT_MAXB",
        "S2S_MT_PROBE",
        "S2S_BACKEND",
        "S2S_MODELS",
        "OMP_NUM_THREADS",
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_brand,
        "cpu_count": os.cpu_count(),
        "libraries": libs,
        "env": {k: env_map.get(k) for k in keys if k in env_map},
    }


def load_cases_manifest(manifest_path: Path, root: Path = ROOT) -> tuple[dict, list[dict]]:
    """Load and validate evaluation fixture manifest JSON."""
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest {manifest_path} contains no cases")
    norm_cases = []
    for idx, item in enumerate(items):
        src = item["src"]
        tgt = item["tgt"]
        wav_raw = item.get("input_wav") or item.get("wav") or item.get("path")
        if not wav_raw:
            raise ValueError(f"Case {idx} missing input_wav path")
        wav_path = Path(wav_raw)
        if not wav_path.is_absolute():
            cand_root = (root / wav_path).resolve()
            cand_ws = (root.parent / wav_path).resolve()
            wav_path = cand_root if cand_root.exists() else cand_ws
        case_id = item.get("id") or f"{src}_to_{tgt}_{idx}"
        norm_cases.append(
            {
                "id": case_id,
                "src": src,
                "tgt": tgt,
                "input_wav": str(wav_path),
                "input_kind": item.get("input_kind", "unknown"),
                "expected": item.get("expected"),
                "reference_text": item.get("written") or item.get("text"),
            }
        )
    meta = raw if isinstance(raw, dict) else {"cases": norm_cases}
    return meta, norm_cases


def _verify_immutable_baseline(baseline_freeze_path: Path | None, root: Path) -> dict:
    if baseline_freeze_path is None or not baseline_freeze_path.exists():
        return {"verified": False, "reason": "baseline_freeze_manifest_missing"}
    try:
        freeze = json.loads(baseline_freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "reason": f"baseline_freeze_unreadable: {exc}"}
    if not isinstance(freeze, dict):
        return {"verified": False, "reason": "baseline_freeze_invalid_type"}
    evidence = freeze.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        return {
            "verified": False,
            "reason": "baseline_freeze_empty_evidence",
            "freeze_id": freeze.get("freeze_id"),
            "freeze_git_head": freeze.get("git_head"),
            "evidence_checked": 0,
            "evidence_mismatches": [],
        }
    mismatches = []
    for rel, info in evidence.items():
        p = root / rel
        if not p.exists():
            mismatches.append({"file": rel, "issue": "missing"})
            continue
        got = _sha256_file(p)
        expected = info.get("sha256") if isinstance(info, dict) else None
        if not expected or got != expected:
            mismatches.append({"file": rel, "expected": expected, "actual": got})
    return {
        "verified": len(mismatches) == 0,
        "reason": "ok" if not mismatches else "baseline_evidence_mismatch",
        "freeze_id": freeze.get("freeze_id"),
        "freeze_git_head": freeze.get("git_head"),
        "evidence_checked": len(evidence),
        "evidence_mismatches": mismatches,
    }


def write_run_manifest(
    *,
    out_dir: Path,
    run_id: str,
    manifest_path: Path,
    cases: list[dict],
    effective_options: dict,
    models_cfg_path: Path = DEFAULT_MODELS,
    root: Path = ROOT,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    baseline_freeze_path: Path | None = DEFAULT_BASELINE_FREEZE,
    env: dict[str, str] | None = None,
    source_paths: list[Path] | None = None,
    model_paths: list[Path] | None = None,
    standalone: bool = False,
) -> dict:
    """Build and write immutable run_manifest.json separating baseline reference from candidate fingerprint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(c["input_wav"]) for c in cases] + [Path(manifest_path)]
    src_paths = source_paths if source_paths is not None else _collect_source_files(root)
    mdl_paths = model_paths if model_paths is not None else _collect_model_files(root, models_cfg_path)

    input_hashes = hash_files_with_cache(input_paths, root, cache_path=cache_path, persist_cache=True)
    model_hashes = hash_files_with_cache(mdl_paths, root, cache_path=cache_path, persist_cache=True)
    source_hashes = hash_files_with_cache(src_paths, root, cache_path=cache_path, persist_cache=True)
    code_fp = compute_code_fingerprint(root, source_hashes)
    comparison_mode = "standalone_non_comparative" if standalone else "comparative"
    if standalone:
        baseline_check = {
            "verified": False,
            "reason": "standalone_non_comparative",
            "note": "Standalone non-comparative run; baseline freeze comparison disabled.",
            "evidence_checked": 0,
            "evidence_mismatches": [],
        }
    else:
        baseline_check = _verify_immutable_baseline(baseline_freeze_path, root)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "comparison_mode": comparison_mode,
        "baseline_verified": bool(baseline_check.get("verified")) and not standalone,
        "manifest_path": str(manifest_path),
        "baseline_reference": baseline_check,
        "candidate_fingerprint": code_fp,
        "effective_options": effective_options,
        "environment": collect_environment_info(env),
        "input_hashes": input_hashes,
        "model_hashes": model_hashes,
        "source_hashes": source_hashes,
        "drift_check": {"before": code_fp["source_tree_sha256"], "after": None, "drift_detected": False},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def resolve_effective_options(
    *,
    source_mode: str | None,
    vad_profile: str | None,
    ort_policy: str | None,
    tts_format: str | None = None,
    backend: str = "cpu",
    models_path: Path = DEFAULT_MODELS,
    warmup: int = 1,
) -> dict:
    """Resolve explicit CLI args vs caller environment without silently overwriting caller settings."""
    caller_env = {
        "S2S_NEURAL_TN": os.environ.get("S2S_NEURAL_TN"),
        "S2S_VAD_PROFILE": os.environ.get("S2S_VAD_PROFILE"),
        "S2S_ORT_POLICY": os.environ.get("S2S_ORT_POLICY"),
        "S2S_TTS_FORMAT": os.environ.get("S2S_TTS_FORMAT"),
    }
    from s2s_edge.audio import get_vad_options
    from s2s_edge.registry import resolve_runtime_profile

    eff = resolve_runtime_profile(
        vad_profile=vad_profile,
        ort_policy=ort_policy,
        source_mode=source_mode,
        tts_format=tts_format,
    )

    origin_map = {"arg": "cli", "env": "caller_env", "default": "default"}
    return {
        "source_mode": eff["source_mode"],
        "source_mode_origin": "cli" if source_mode is not None else origin_map[eff["source_mode_origin"]],
        "vad_profile": eff["vad_profile"],
        "vad_profile_origin": "cli" if vad_profile is not None else origin_map[eff["vad_profile_origin"]],
        "vad_options": get_vad_options(eff["vad_profile"]),
        "ort_policy": eff["ort_policy"],
        "ort_policy_origin": "cli" if ort_policy is not None else origin_map[eff["ort_policy_origin"]],
        "tts_format": eff["tts_format"],
        "tts_format_origin": "cli" if tts_format is not None else origin_map[eff["tts_format_origin"]],
        "backend": backend,
        "models_config": str(models_path),
        "warmup": int(warmup),
        "caller_env": caller_env,
    }


def _aggregate_stages(segments: list[dict]) -> dict[str, float]:
    keys = ("asr_s", "source_normalize_s", "mt_s", "tts_s", "total_s", "graph_s", "host_s", "graph_calls")
    agg = {k: 0.0 for k in keys}
    for seg in segments:
        t = seg.get("timing") or {}
        for k in keys:
            agg[k] += float(t.get(k, 0.0))
    return agg


def _call_segment(wav: np.ndarray, max_s: float, vad_profile: str):
    from s2s_edge.audio import segment
    return segment(wav, max_s=max_s, vad_profile=vad_profile)


def _create_engine(models_cfg: Path, backend: str, ort_policy: str, vad_profile: str | None = None, source_mode: str | None = None, tts_format: str | None = None):
    from s2s_edge.registry import Registry
    from s2s_edge.conversation import Engine
    reg = Registry(str(models_cfg), backend, ort_policy=ort_policy, vad_profile=vad_profile, source_mode=source_mode, tts_format=tts_format)
    return reg, Engine(reg)


def run_evaluation(
    *,
    run_id: str,
    manifest_path: Path,
    out_root: Path = DEFAULT_OUT_ROOT,
    source_mode: str | None = None,
    vad_profile: str | None = None,
    ort_policy: str | None = None,
    tts_format: str | None = None,
    models_path: Path = DEFAULT_MODELS,
    backend: str = "cpu",
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    baseline_freeze_path: Path | None = DEFAULT_BASELINE_FREEZE,
    expected_candidate_rev: str | None = None,
    warmup: int = 1,
    standalone: bool = False,
    engine_factory=None,
    segment_fn=None,
    source_paths: list[Path] | None = None,
    model_paths: list[Path] | None = None,
) -> tuple[int, dict]:
    """Execute evaluation contract with immutable output dir, manifest, stage timings, and non-zero exit on error."""
    norm_id = validate_run_id(run_id)
    out_dir = Path(out_root) / norm_id
    try:
        out_resolved = out_dir.resolve()
        for protected_rel in ("results/asr_integration", "results/next_steps/t3_ablation"):
            prot = (ROOT / protected_rel).resolve()
            if prot in out_resolved.parents or out_resolved == prot:
                raise ValueError(f"Refusing to write inside protected baseline directory {prot}")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
    if out_dir.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {out_dir}")

    effective = resolve_effective_options(
        source_mode=source_mode,
        vad_profile=vad_profile,
        ort_policy=ort_policy,
        tts_format=tts_format,
        backend=backend,
        models_path=models_path,
        warmup=warmup,
    )
    _, cases = load_cases_manifest(manifest_path, root=ROOT)

    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = write_run_manifest(
        out_dir=out_dir,
        run_id=norm_id,
        manifest_path=manifest_path,
        cases=cases,
        effective_options=effective,
        models_cfg_path=models_path,
        root=ROOT,
        cache_path=cache_path,
        baseline_freeze_path=baseline_freeze_path,
        source_paths=source_paths,
        model_paths=model_paths,
        standalone=standalone,
    )

    if not standalone and not manifest["baseline_reference"].get("verified"):
        manifest["run_status"] = "failed_baseline_verification"
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        fail_summary = {
            "run_id": norm_id,
            "status": "failed_baseline_verification",
            "run_status": "failed_baseline_verification",
            "comparison_mode": manifest["comparison_mode"],
            "baseline_verified": False,
            "baseline_reference": manifest["baseline_reference"],
            "failure_count": 1,
            "completed_count": 0,
            "total_cases": len(cases),
            "cases": [],
        }
        (out_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2, fail_summary

    if expected_candidate_rev and manifest["candidate_fingerprint"]["source_tree_sha256"] != expected_candidate_rev:
        manifest["expected_candidate_rev"] = expected_candidate_rev
        manifest["run_status"] = "failed_revision_mismatch"
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        fail_summary = {
            "run_id": norm_id,
            "status": "failed_revision_mismatch",
            "run_status": "failed_revision_mismatch",
            "comparison_mode": manifest["comparison_mode"],
            "baseline_verified": manifest["baseline_verified"],
            "failure_count": 1,
            "completed_count": 0,
            "total_cases": len(cases),
            "cases": [],
        }
        (out_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2, fail_summary

    from s2s_edge.audio import load_audio, save_wav, concat_results, SR

    t_load0 = time.perf_counter()
    if engine_factory is not None:
        reg, engine = engine_factory(models_path, backend, effective["ort_policy"])
    else:
        reg, engine = _create_engine(
            models_path,
            backend,
            effective["ort_policy"],
            vad_profile=effective["vad_profile"],
            source_mode=effective["source_mode"],
            tts_format=effective["tts_format"],
        )
    load_s = time.perf_counter() - t_load0

    warmup_s = 0.0
    warmup_details = {}
    if effective["warmup"] > 0 and hasattr(reg, "warmup"):
        t_warm0 = time.perf_counter()
        dirs = sorted({(c["src"], c["tgt"]) for c in cases})
        try:
            warmup_details = reg.warmup(dirs, log=lambda *_: None) or {}
        except TypeError:
            warmup_details = reg.warmup(dirs) or {}
        warmup_s = time.perf_counter() - t_warm0

    seg_fn = segment_fn or _call_segment
    rows = []
    failure_count = 0
    for idx, case in enumerate(cases):
        src, tgt = case["src"], case["tgt"]
        wav_path = Path(case["input_wav"])
        row_base = {
            "case_id": case["id"],
            "src": src,
            "tgt": tgt,
            "input": str(wav_path),
            "input_kind": case["input_kind"],
            "input_sha256": _sha256_file(wav_path) if wav_path.exists() else None,
            "cold_load_s": round(load_s, 6) if idx == 0 else 0.0,
            "run_load_s": round(load_s, 6),
            "warmup_s": round(warmup_s, 6),
            "effective_options": {
                "source_mode": effective["source_mode"],
                "vad_profile": effective["vad_profile"],
                "ort_policy": effective["ort_policy"],
                "tts_format": effective["tts_format"],
            },
        }
        start = time.perf_counter()
        try:
            wav = load_audio(str(wav_path))
            results = []
            max_seg_s = min(float(engine.asr(src).max_s), 12.0)
            spans = seg_fn(wav, max_seg_s, effective["vad_profile"])
            for a, b in spans:
                z = engine.translate_segment(wav[a:b], src, tgt)
                z["span"] = [a / SR, b / SR]
                results.append(z)
            audio, sr = concat_results(results)
            if not len(audio) or not np.isfinite(audio).all():
                raise RuntimeError("Synthesized audio empty or non-finite")
            target = out_dir / f"{case['id']}.wav"
            save_wav(str(target), audio, sr)
            row = {
                **row_base,
                "status": "ok",
                "output": str(target),
                "output_sha256": _sha256_file(target),
                "output_seconds": len(audio) / sr,
                "seconds": time.perf_counter() - start,
                "segment_count": len(results),
                "stages": _aggregate_stages(results),
                "ort_sessions": reg.graph_telemetry() if hasattr(reg, "graph_telemetry") else {},
                "segments": [{k: v for k, v in z.items() if k != "wav"} for z in results],
            }
        except Exception as exc:
            failure_count += 1
            traceback.print_exc()
            row = {
                **row_base,
                "status": "error",
                "error": str(exc),
                "seconds": time.perf_counter() - start,
                "segment_count": 0,
                "stages": {},
                "segments": [],
            }
        rows.append(row)
        partial_payload = {
            "run_id": norm_id,
            "effective_options": effective,
            "run_load_s": round(load_s, 6),
            "warmup_s": round(warmup_s, 6),
            "failure_count": failure_count,
            "completed_count": len(rows),
            "total_cases": len(cases),
            "cases": rows,
        }
        (out_dir / "summary.json").write_text(json.dumps(partial_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(norm_id, case["id"], src, tgt, row["status"], round(row.get("seconds", 0.0), 3), flush=True)

    # Check source/config drift during run
    drift_paths = source_paths if source_paths is not None else _collect_source_files(ROOT)
    after_src = hash_files_with_cache(drift_paths, ROOT, cache_path=None, persist_cache=False)
    after_fp = compute_code_fingerprint(ROOT, after_src)["source_tree_sha256"]
    drift = after_fp != manifest["candidate_fingerprint"]["source_tree_sha256"]
    manifest["drift_check"] = {
        "before": manifest["candidate_fingerprint"]["source_tree_sha256"],
        "after": after_fp,
        "drift_detected": drift,
    }
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["run_load_s"] = round(load_s, 6)
    manifest["warmup_s"] = round(warmup_s, 6)
    manifest["warmup_details"] = warmup_details
    manifest["failure_count"] = failure_count
    manifest["completed_count"] = len(rows)
    manifest["ort_sessions"] = reg.graph_telemetry() if hasattr(reg, "graph_telemetry") else {}
    manifest["run_status"] = "failed_drift" if drift else ("failed_cases" if failure_count > 0 else "ok")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": norm_id,
        "run_status": manifest["run_status"],
        "comparison_mode": manifest["comparison_mode"],
        "baseline_verified": manifest["baseline_verified"],
        "baseline_reference": manifest["baseline_reference"],
        "drift_detected": drift,
        "failure_count": failure_count,
        "completed_count": len(rows),
        "total_cases": len(cases),
        "run_load_s": round(load_s, 6),
        "warmup_s": round(warmup_s, 6),
        "effective_options": effective,
        "cases": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    exit_code = 0 if (failure_count == 0 and not drift and len(rows) > 0) else 1
    return exit_code, summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run reproducible ASR/S2S integration evaluation.")
    ap.add_argument("--run-id", required=True, help="Unique immutable run identifier (cannot be 'before'/'after' or already exist).")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Fixture manifest JSON path.")
    ap.add_argument("--source-mode", choices=["0", "1"], default=None, help="S2S_NEURAL_TN value; preserves caller env when omitted.")
    ap.add_argument("--vad-profile", choices=["hybrid", "legacy", "upstream"], default=None, help="VAD profile; defaults to safe 'legacy'.")
    ap.add_argument("--ort-policy", choices=["basic", "disable", "auto"], default=None, help="ONNX Runtime session optimization policy.")
    ap.add_argument("--tts-format", choices=["hybrid", "legacy", "neural"], default=None, help="TTS format mode.")
    ap.add_argument("--models", default=str(DEFAULT_MODELS), help="models.json path.")
    ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "cpu"), choices=["cpu", "npu"])
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Directory for new run artifacts.")
    ap.add_argument("--cache-manifest", default=str(DEFAULT_CACHE_PATH), help="Weight hash cache JSON path.")
    ap.add_argument("--baseline-freeze", default=str(DEFAULT_BASELINE_FREEZE), help="Immutable baseline freeze manifest.")
    ap.add_argument("--expected-candidate-rev", default=None, help="Expected candidate source tree SHA256.")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup passes (0 or 1).")
    ap.add_argument(
        "--standalone",
        action="store_true",
        help="Run standalone non-comparative evaluation without baseline freeze comparison.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, summary = run_evaluation(
            run_id=args.run_id,
            manifest_path=Path(args.manifest),
            out_root=Path(args.out_root),
            source_mode=args.source_mode,
            vad_profile=args.vad_profile,
            ort_policy=args.ort_policy,
            tts_format=args.tts_format,
            models_path=Path(args.models),
            backend=args.backend,
            cache_path=Path(args.cache_manifest) if args.cache_manifest else None,
            baseline_freeze_path=Path(args.baseline_freeze) if args.baseline_freeze else None,
            expected_candidate_rev=args.expected_candidate_rev,
            warmup=args.warmup,
            standalone=args.standalone,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"RUN_STATUS {summary['run_status']} failures={summary['failure_count']} cases={summary['completed_count']}/{summary['total_cases']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
