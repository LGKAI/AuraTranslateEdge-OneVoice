"""Controlled single-variable ablation runner for Plan 01 T3 with immutable manifests, drift detection, and typed entity scoring."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(ROOT / "s2s_edge"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from s2s_edge.audio import SR, concat_results, load_audio, save_wav
import entity_scorer
import evaluate_asr_integration as harness


def _p95_descriptive(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = (len(s) - 1) * 0.95
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[lo]
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None, "p95_descriptive": None}
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "p95_descriptive": round(_p95_descriptive(vals), 4),
    }


def _resolve_wav(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    cand1 = (ROOT / p).resolve()
    if cand1.exists():
        return cand1
    return (ROOT.parent / p).resolve()


def _re_sub_compact(text: str) -> str:
    return "".join(text.lower().split())


def check_lexical_proxy(asr_joined: str, expected: list[str] | None) -> dict:
    """Explicitly labeled non-semantic lexical presence proxy retained only for comparison against legacy logs."""
    exp = expected or []
    if not exp:
        return {
            "metric_role": "lexical_presence_proxy_non_semantic",
            "is_semantic_gate": False,
            "checked": False,
            "unknown_or_unlabeled": True,
            "matched": [],
            "missing": [],
        }
    norm = _re_sub_compact(asr_joined)
    matched = []
    missing = []
    for tok in exp:
        tok_compact = tok.replace(" ", "").lower()
        if tok.lower() in asr_joined.lower() or tok_compact in norm:
            matched.append(tok)
        else:
            missing.append(tok)
    return {
        "metric_role": "lexical_presence_proxy_non_semantic",
        "is_semantic_gate": False,
        "checked": True,
        "unknown_or_unlabeled": False,
        "matched": matched,
        "missing": missing,
        "all_preserved": len(missing) == 0,
    }


def summarize_entity_rows(rows: list[dict]) -> dict:
    """Aggregate tri-state entity scores and coverage across a slice of cases."""
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok" and "entity_eval" in r]
    covs = [r["entity_eval"]["coverage"] for r in ok_rows if r["entity_eval"].get("coverage") is not None]
    return {
        "scorer": "typed_boundary_entity_v1",
        "cases_total": len(rows),
        "cases_ok": len(ok_rows),
        "cases_all_labeled_correct": sum(1 for r in ok_rows if r["entity_eval"].get("all_labeled_correct")),
        "entities_correct": sum(r["entity_eval"].get("correct", 0) for r in ok_rows),
        "entities_wrong": sum(r["entity_eval"].get("wrong", 0) for r in ok_rows),
        "entities_unknown": sum(r["entity_eval"].get("unknown", 0) for r in ok_rows),
        "written_entities_detected": sum(r["entity_eval"].get("total_detected", 0) for r in ok_rows),
        "written_entities_labeled": sum(r["entity_eval"].get("labeled_detected", 0) for r in ok_rows),
        "mean_coverage": round(statistics.mean(covs), 4) if covs else None,
    }


def _run_case(
    engine,
    case: dict,
    vad_profile: str,
    save_wav_path: Path | None = None,
    segment_fn=None,
) -> dict:
    wav_path = _resolve_wav(case["input_wav"])
    seg_fn = segment_fn or harness._call_segment
    t0 = time.perf_counter()
    try:
        wav = load_audio(str(wav_path))
        max_s = min(float(engine.asr(case["src"]).max_s), 12.0)
        spans = seg_fn(wav, max_s, vad_profile)
        vad_s = time.perf_counter() - t0
        if not spans:
            raise RuntimeError(f"VAD returned 0 speech spans for {case['id']} ({vad_profile})")

        segs = []
        for a, b in spans:
            z = engine.translate_segment(wav[a:b], case["src"], case["tgt"])
            z["span"] = [round(a / SR, 4), round(b / SR, 4)]
            segs.append(z)
        audio, sr = concat_results(segs)
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid audio output for {case['id']}")
        if save_wav_path is not None:
            save_wav_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav(str(save_wav_path), audio, sr)
            out_sha = hashlib.sha256(save_wav_path.read_bytes()).hexdigest()
        else:
            out_sha = None
        total_s = time.perf_counter() - t0
        agg = harness._aggregate_stages(segs)
        asr_joined = " | ".join(s.get("asr", "") for s in segs)
        asr_raw_joined = " | ".join(s.get("asr_before_normalization", s.get("asr", "")) for s in segs)
        mt_joined = " | ".join(s.get("mt", "") for s in segs)
        lex = check_lexical_proxy(asr_raw_joined, case.get("expected_entities"))
        ent_eval = entity_scorer.score_text_entities(
            case.get("written"),
            asr_raw_joined,
            case.get("expected_entities"),
            lang=case["src"],
        )
        return {
            "id": case["id"],
            "src": case["src"],
            "tgt": case["tgt"],
            "input_kind": case["input_kind"],
            "input_wav": str(wav_path),
            "input_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            "output_wav": str(save_wav_path) if save_wav_path else None,
            "output_sha256": out_sha,
            "status": "ok",
            "vad_profile": vad_profile,
            "segment_count": len(segs),
            "spans": [s["span"] for s in segs],
            "vad_s": round(vad_s, 6),
            "total_s": round(total_s, 4),
            "asr_s": round(agg["asr_s"], 4),
            "source_normalize_s": round(agg["source_normalize_s"], 4),
            "mt_s": round(agg["mt_s"], 4),
            "tts_s": round(agg["tts_s"], 4),
            "graph_s": round(agg["graph_s"], 4),
            "host_s": round(agg["host_s"], 4),
            "asr_raw": asr_raw_joined,
            "asr_prepared": asr_joined,
            "mt": mt_joined,
            "tts_texts": [s.get("tts_text", "") for s in segs],
            "lexical_proxy": lex,
            "entity_eval": ent_eval,
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "id": case["id"],
            "src": case["src"],
            "tgt": case["tgt"],
            "input_kind": case.get("input_kind"),
            "input_wav": str(wav_path),
            "status": "error",
            "error": str(exc),
            "vad_profile": vad_profile,
            "segment_count": 0,
            "total_s": round(time.perf_counter() - t0, 4),
            "lexical_proxy": check_lexical_proxy("", case.get("expected_entities")),
            "entity_eval": entity_scorer.score_text_entities(
                case.get("written"), "", case.get("expected_entities"), lang=case.get("src", "en")
            ),
        }


def _make_engine(models_cfg: Path, backend: str, ort_policy: str, vad_profile: str, source_mode: str, tts_format: str, engine_factory=None):
    if engine_factory is not None:
        return engine_factory(models_cfg, backend, ort_policy, vad_profile, source_mode, tts_format)
    from s2s_edge.registry import Registry
    from s2s_edge.conversation import Engine

    reg = Registry(
        str(models_cfg),
        backend,
        ort_policy=ort_policy,
        vad_profile=vad_profile,
        source_mode=source_mode,
        tts_format=tts_format,
    )
    return reg, Engine(reg)


def _warmup_dirs(reg, dirs):
    if not hasattr(reg, "warmup"):
        return {}
    try:
        return reg.warmup(dirs, log=lambda *_: None) or {}
    except TypeError:
        return reg.warmup(dirs) or {}


def _release_reg(reg):
    if hasattr(reg, "_cache") and hasattr(reg, "release"):
        for k in list(reg._cache):
            reg.release(k)
    gc.collect()


def _precompute_orders(step: str, seed: int, repeats: int) -> dict:
    out = {}
    if step in ("vad", "all"):
        rng = random.Random(seed)
        vad_orders = []
        for rep in range(repeats):
            pair = ["legacy", "upstream"]
            rng.shuffle(pair)
            vad_orders.append({"rep": rep, "order": list(pair)})
        out["vad"] = vad_orders
    if step in ("ort", "all"):
        rng = random.Random(seed + 1)
        ort_orders = []
        for rep in range(repeats):
            policies = ["auto", "basic", "disable"]
            rng.shuffle(policies)
            ort_orders.append({"rep": rep, "order": list(policies)})
        out["ort"] = ort_orders
    return out


def _collect_ablation_cases(split: dict, step: str) -> list[dict]:
    cases = []
    if step in ("vad", "all"):
        cases.extend(split.get("native_en_dev", []))
        cases.extend(split.get("native_en_holdout", []))
        cases.extend(split.get("synthetic_dev", []))
    if step in ("ort", "all"):
        cases.extend(split.get("paired_ort_core_cases", []))
    return cases


def run_vad_ablation(
    split: dict,
    out_dir: Path,
    *,
    orders: list[dict],
    seed: int = 20261113,
    repeats: int = 3,
    models_cfg: Path = harness.DEFAULT_MODELS,
    backend: str = "cpu",
    engine_factory=None,
    segment_fn=None,
) -> tuple[int, dict]:
    """Compare VAD legacy vs upstream with ORT fixed at 'auto' and source_mode='0' (instance-bound, no env mutation)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cases = split.get("native_en_dev", []) + split.get("native_en_holdout", []) + split.get("synthetic_dev", [])
    paired_ids = set(split["paired_vad_core_ids"])
    paired_cases = [c for c in all_cases if c["id"] in paired_ids]

    t0 = time.perf_counter()
    reg, engine = _make_engine(models_cfg, backend, "auto", "legacy", "0", "hybrid", engine_factory=engine_factory)
    init_s = time.perf_counter() - t0

    t_w0 = time.perf_counter()
    dirs = sorted({(c["src"], c["tgt"]) for c in all_cases})
    golden_warmup = _warmup_dirs(reg, dirs)
    golden_warmup_s = time.perf_counter() - t_w0

    failure_count = 0
    print("=== Step 3a Full Audit (legacy/upstream) ===", flush=True)
    audit = {"legacy": [], "upstream": []}
    for prof in ("legacy", "upstream"):
        for c in all_cases:
            wav_out = out_dir / "audio" / prof / f"{c['id']}.wav"
            row = _run_case(engine, c, prof, save_wav_path=wav_out, segment_fn=segment_fn)
            if row["status"] != "ok":
                failure_count += 1
            audit[prof].append(row)
            ent = row["entity_eval"]
            print(
                f"  audit {prof:8s} {c['id']:26s} status={row['status']} segs={row['segment_count']} "
                f"total={row['total_s']:.2f}s ent=({ent['correct']}c/{ent['wrong']}w/{ent['unknown']}u cov={ent['coverage']})",
                flush=True,
            )

    print("=== Step 3a Explicit Paired Warmup + Seeded Paired Turns ===", flush=True)
    for prof in ("legacy", "upstream"):
        for c in paired_cases[:2]:
            _run_case(engine, c, prof, segment_fn=segment_fn)

    paired_rows: dict[str, list[dict]] = {"legacy": [], "upstream": []}
    for entry in orders:
        rep = entry["rep"]
        for prof in entry["order"]:
            for c in paired_cases:
                r = _run_case(engine, c, prof, segment_fn=segment_fn)
                if r["status"] != "ok":
                    failure_count += 1
                r["rep"] = rep
                paired_rows[prof].append(r)
                print(
                    f"  paired rep={rep} {prof:8s} {c['id']:26s} status={r['status']} segs={r['segment_count']} total={r['total_s']:.2f}s",
                    flush=True,
                )

    summary_stats = {}
    for prof in ("legacy", "upstream"):
        by_case = {}
        for c in paired_cases:
            cid = c["id"]
            sub = [r for r in paired_rows[prof] if r["id"] == cid and r["status"] == "ok"]
            by_case[cid] = {
                "total_s": _stats([x["total_s"] for x in sub]),
                "asr_s": _stats([x["asr_s"] for x in sub]),
                "mt_s": _stats([x["mt_s"] for x in sub]),
                "tts_s": _stats([x["tts_s"] for x in sub]),
                "segments": sub[0]["segment_count"] if sub else 0,
            }
        rep_totals = [
            sum(r["total_s"] for r in paired_rows[prof] if r["rep"] == rep and r["status"] == "ok")
            for rep in range(repeats)
        ]
        summary_stats[prof] = {
            "core_sum_total_s": _stats(rep_totals),
            "per_case": by_case,
            "all_clips_total_segments": sum(r["segment_count"] for r in audit[prof]),
            "lexical_proxy_passes": sum(1 for r in audit[prof] if r["lexical_proxy"]["checked"] and r["lexical_proxy"]["all_preserved"]),
            "lexical_proxy_total": sum(1 for r in audit[prof] if r["lexical_proxy"]["checked"]),
            "lexical_proxy_role": "non_semantic_lexical_presence_proxy",
            "entity_eval_summary": summarize_entity_rows(audit[prof]),
        }

    result = {
        "experiment": "3a_vad_ablation",
        "fixed": {"ort_policy": "auto", "source_mode": "0", "tts_format": "hybrid", "backend": backend},
        "seed": seed,
        "repeats": repeats,
        "seeded_orders": orders,
        "failure_count": failure_count,
        "registry_init_s": round(init_s, 4),
        "golden_warmup_s": round(golden_warmup_s, 4),
        "golden_warmup": golden_warmup,
        "ort_sessions": reg.graph_telemetry() if hasattr(reg, "graph_telemetry") else {},
        "summary": summary_stats,
        "full_audit": audit,
        "paired_runs": paired_rows,
    }
    (out_dir / "vad_ablation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _release_reg(reg)
    return failure_count, result


def run_ort_ablation(
    split: dict,
    out_dir: Path,
    *,
    orders: list[dict],
    fixed_vad: str = "legacy",
    seed: int = 20261113,
    repeats: int = 3,
    models_cfg: Path = harness.DEFAULT_MODELS,
    backend: str = "cpu",
    engine_factory=None,
    segment_fn=None,
) -> tuple[int, dict]:
    """Compare ORT policies ('auto', 'basic', 'disable') across active graphs with VAD fixed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = split["paired_ort_core_cases"]
    dirs = sorted({(c["src"], c["tgt"]) for c in cases})

    cold_loads: dict[str, list[dict]] = {p: [] for p in ("auto", "basic", "disable")}
    warm_runs: dict[str, list[dict]] = {p: [] for p in ("auto", "basic", "disable")}
    telemetry_sample = {}
    failure_count = 0

    print(f"=== Step 3b ORT Policy Ablation (fixed_vad={fixed_vad}, {repeats} seeded triplets) ===", flush=True)
    for entry in orders:
        rep = entry["rep"]
        for pol in entry["order"]:
            t0 = time.perf_counter()
            reg, engine = _make_engine(models_cfg, backend, pol, fixed_vad, "0", "hybrid", engine_factory=engine_factory)
            t_warm0 = time.perf_counter()
            golden_details = _warmup_dirs(reg, dirs)
            cold_warmup_s = time.perf_counter() - t_warm0
            total_cold_s = time.perf_counter() - t0
            telemetry = reg.graph_telemetry() if hasattr(reg, "graph_telemetry") else {}
            if pol not in telemetry_sample:
                telemetry_sample[pol] = telemetry
            cold_loads[pol].append(
                {
                    "rep": rep,
                    "total_cold_load_and_golden_s": round(total_cold_s, 4),
                    "golden_warmup_s": round(cold_warmup_s, 4),
                    "per_graph": golden_details,
                }
            )
            _run_case(engine, cases[0], fixed_vad, segment_fn=segment_fn)
            for c in cases:
                wav_out = out_dir / "audio" / pol / f"{c['id']}.wav" if rep == 0 else None
                row = _run_case(engine, c, fixed_vad, save_wav_path=wav_out, segment_fn=segment_fn)
                if row["status"] != "ok":
                    failure_count += 1
                row["rep"] = rep
                row["ort_policy"] = pol
                warm_runs[pol].append(row)
                print(f"  ort rep={rep} {pol:7s} {c['id']:24s} status={row['status']} total={row['total_s']:.2f}s", flush=True)
            _release_reg(reg)

    summary = {}
    for pol in ("auto", "basic", "disable"):
        rep_totals = [
            sum(r["total_s"] for r in warm_runs[pol] if r["rep"] == rep and r["status"] == "ok")
            for rep in range(repeats)
        ]
        by_case = {}
        for c in cases:
            cid = c["id"]
            sub = [r for r in warm_runs[pol] if r["id"] == cid and r["status"] == "ok"]
            by_case[cid] = {
                "total_s": _stats([x["total_s"] for x in sub]),
                "asr_s": _stats([x["asr_s"] for x in sub]),
                "mt_s": _stats([x["mt_s"] for x in sub]),
                "tts_s": _stats([x["tts_s"] for x in sub]),
            }
        summary[pol] = {
            "cold_load_and_golden_s": _stats([x["total_cold_load_and_golden_s"] for x in cold_loads[pol]]),
            "core_sum_total_s": _stats(rep_totals),
            "per_case": by_case,
            "telemetry": telemetry_sample.get(pol, {}),
            "entity_eval_summary": summarize_entity_rows([r for r in warm_runs[pol] if r["rep"] == 0]),
        }

    result = {
        "experiment": "3b_ort_ablation",
        "fixed": {"vad_profile": fixed_vad, "source_mode": "0", "tts_format": "hybrid", "backend": backend},
        "seed": seed + 1,
        "repeats": repeats,
        "seeded_orders": orders,
        "failure_count": failure_count,
        "summary": summary,
        "cold_loads": cold_loads,
        "warm_runs": warm_runs,
    }
    (out_dir / "ort_ablation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return failure_count, result


def run_ablation(
    *,
    step: str,
    run_id: str,
    split_path: Path,
    out_root: Path = harness.DEFAULT_OUT_ROOT,
    fixed_vad: str = "legacy",
    seed: int = 20261113,
    repeats: int = 3,
    models_path: Path = harness.DEFAULT_MODELS,
    backend: str = "cpu",
    cache_path: Path | None = harness.DEFAULT_CACHE_PATH,
    baseline_freeze_path: Path | None = harness.DEFAULT_BASELINE_FREEZE,
    expected_candidate_rev: str | None = None,
    standalone: bool = False,
    engine_factory=None,
    segment_fn=None,
    source_paths: list[Path] | None = None,
    model_paths: list[Path] | None = None,
) -> tuple[int, dict]:
    """Run controlled ablation with immutable run directory, pre-registered manifest/orders, baseline check, and drift detection."""
    norm_id = harness.validate_run_id(run_id)
    out_dir = Path(out_root) / norm_id
    out_resolved = out_dir.resolve()
    for protected_rel in ("results/asr_integration", "results/next_steps/t3_ablation"):
        prot = (ROOT / protected_rel).resolve()
        if prot in out_resolved.parents or out_resolved == prot:
            raise ValueError(f"Refusing to write inside protected directory {prot}")
    if out_dir.exists():
        raise FileExistsError(f"Ablation run directory already exists and will not be overwritten: {out_dir}")

    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    seeded_orders = _precompute_orders(step, seed, repeats)
    all_cases = _collect_ablation_cases(split, step)

    input_paths = [_resolve_wav(c["input_wav"]) for c in all_cases] + [Path(split_path)]
    src_paths = list(source_paths) if source_paths is not None else harness._collect_source_files(ROOT)
    # Include split config & models config in monitored config/source drift set
    cfg_paths = [Path(split_path).resolve()]
    if models_path.exists():
        cfg_paths.append(models_path.resolve())
    monitored_paths = src_paths + [p for p in cfg_paths if p not in src_paths]
    mdl_paths = model_paths if model_paths is not None else harness._collect_model_files(ROOT, models_path)

    out_dir.mkdir(parents=True, exist_ok=False)
    input_hashes = harness.hash_files_with_cache(input_paths, ROOT, cache_path=cache_path, persist_cache=True)
    model_hashes = harness.hash_files_with_cache(mdl_paths, ROOT, cache_path=cache_path, persist_cache=True)
    source_hashes = harness.hash_files_with_cache(monitored_paths, ROOT, cache_path=cache_path, persist_cache=True)
    code_fp = harness.compute_code_fingerprint(ROOT, source_hashes)

    comparison_mode = "standalone_non_comparative" if standalone else "comparative"
    if standalone:
        baseline_check = {
            "verified": False,
            "reason": "standalone_non_comparative",
            "note": "Standalone non-comparative ablation; baseline freeze comparison disabled.",
            "evidence_checked": 0,
            "evidence_mismatches": [],
        }
    else:
        baseline_check = harness._verify_immutable_baseline(baseline_freeze_path, ROOT)
    baseline_verified = bool(baseline_check.get("verified")) and not standalone

    manifest = {
        "schema_version": 1,
        "run_id": norm_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": f"t3_ablation_{step}",
        "step": step,
        "comparison_mode": comparison_mode,
        "baseline_verified": baseline_verified,
        "baseline_reference": baseline_check,
        "candidate_fingerprint": code_fp,
        "split_path": str(split_path),
        "split_sha256": harness._sha256_file(Path(split_path)),
        "effective_options": {
            "step": step,
            "fixed_vad": fixed_vad,
            "seed": seed,
            "repeats": repeats,
            "source_mode": "0",
            "tts_format": "hybrid",
            "backend": backend,
            "models_config": str(models_path),
        },
        "seeded_orders": seeded_orders,
        "environment": harness.collect_environment_info(),
        "input_hashes": input_hashes,
        "model_hashes": model_hashes,
        "source_hashes": source_hashes,
        "drift_check": {"before": code_fp["source_tree_sha256"], "after": None, "drift_detected": False},
        "run_status": "initialized",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not standalone and not baseline_verified:
        manifest["run_status"] = "failed_baseline_verification"
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        fail_summary = {
            "run_id": norm_id,
            "run_status": "failed_baseline_verification",
            "comparison_mode": comparison_mode,
            "baseline_verified": False,
            "baseline_reference": baseline_check,
            "failure_count": 1,
        }
        (out_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2, fail_summary

    if expected_candidate_rev and manifest["candidate_fingerprint"]["source_tree_sha256"] != expected_candidate_rev:
        manifest["expected_candidate_rev"] = expected_candidate_rev
        manifest["run_status"] = "failed_revision_mismatch"
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        fail_summary = {
            "run_id": norm_id,
            "run_status": "failed_revision_mismatch",
            "comparison_mode": comparison_mode,
            "baseline_verified": baseline_verified,
            "failure_count": 1,
        }
        (out_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return 2, fail_summary

    total_failures = 0
    step_outputs = {}
    if step in ("vad", "all"):
        f_vad, res_vad = run_vad_ablation(
            split,
            out_dir / "vad",
            orders=seeded_orders["vad"],
            seed=seed,
            repeats=repeats,
            models_cfg=models_path,
            backend=backend,
            engine_factory=engine_factory,
            segment_fn=segment_fn,
        )
        total_failures += f_vad
        step_outputs["vad"] = {"failure_count": f_vad, "summary": res_vad["summary"]}
    if step in ("ort", "all"):
        f_ort, res_ort = run_ort_ablation(
            split,
            out_dir / "ort",
            orders=seeded_orders["ort"],
            fixed_vad=fixed_vad,
            seed=seed,
            repeats=repeats,
            models_cfg=models_path,
            backend=backend,
            engine_factory=engine_factory,
            segment_fn=segment_fn,
        )
        total_failures += f_ort
        step_outputs["ort"] = {"failure_count": f_ort, "summary": res_ort["summary"]}

    after_src = harness.hash_files_with_cache(monitored_paths, ROOT, cache_path=None, persist_cache=False)
    after_fp = harness.compute_code_fingerprint(ROOT, after_src)["source_tree_sha256"]
    drift = after_fp != manifest["candidate_fingerprint"]["source_tree_sha256"]
    manifest["drift_check"] = {
        "before": manifest["candidate_fingerprint"]["source_tree_sha256"],
        "after": after_fp,
        "drift_detected": drift,
    }
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["failure_count"] = total_failures
    manifest["run_status"] = "failed_drift" if drift else ("failed_cases" if total_failures > 0 else "ok")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": norm_id,
        "run_status": manifest["run_status"],
        "comparison_mode": comparison_mode,
        "baseline_verified": baseline_verified,
        "baseline_reference": baseline_check,
        "drift_detected": drift,
        "failure_count": total_failures,
        "seeded_orders": seeded_orders,
        "steps": step_outputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    exit_code = 0 if (total_failures == 0 and not drift) else 1
    return exit_code, summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Step 3 controlled VAD and ORT ablations with immutable run manifests.")
    ap.add_argument("step", choices=["vad", "ort", "all"])
    ap.add_argument("--run-id", required=True, help="Unique immutable run ID (cannot overwrite existing dirs or protected baseline/t3_ablation).")
    ap.add_argument("--split", default=str(ROOT / "results/next_steps/manifests/t3_ablation_split.json"))
    ap.add_argument("--out-root", default=str(harness.DEFAULT_OUT_ROOT))
    ap.add_argument("--fixed-vad", default="legacy", choices=["legacy", "upstream"])
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--models", default=str(harness.DEFAULT_MODELS))
    ap.add_argument("--backend", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--cache-manifest", default=str(harness.DEFAULT_CACHE_PATH))
    ap.add_argument("--baseline-freeze", default=str(harness.DEFAULT_BASELINE_FREEZE))
    ap.add_argument("--expected-candidate-rev", default=None)
    ap.add_argument("--standalone", action="store_true", help="Run standalone non-comparative ablation without baseline freeze verification.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, summary = run_ablation(
            step=args.step,
            run_id=args.run_id,
            split_path=Path(args.split),
            out_root=Path(args.out_root),
            fixed_vad=args.fixed_vad,
            seed=args.seed,
            repeats=args.repeats,
            models_path=Path(args.models),
            backend=args.backend,
            cache_path=Path(args.cache_manifest) if args.cache_manifest else None,
            baseline_freeze_path=Path(args.baseline_freeze) if args.baseline_freeze else None,
            expected_candidate_rev=args.expected_candidate_rev,
            standalone=args.standalone,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"ABLATION_COMPLETE status={summary['run_status']} step={args.step} run_id={args.run_id} failures={summary['failure_count']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
