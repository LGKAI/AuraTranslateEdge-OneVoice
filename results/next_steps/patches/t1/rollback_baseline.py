"""Hash-verified rollback tool to restore pre-Plan-01 state safely (source, tools, tests, config, docs)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_DIR = ROOT / "results/next_steps/baseline_freeze"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_rollback(
    *,
    freeze_dir: Path,
    expected_current: dict[str, str | None],
    root: Path = ROOT,
    apply: bool = False,
) -> tuple[bool, dict]:
    """Check expected current SHA256s (incl. config/docs) and freeze snapshot integrity before restoring."""
    freeze_manifest = json.loads((freeze_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    snap_root = freeze_dir / "source_snapshot"
    frozen_files: dict[str, dict] = freeze_manifest["source_files"]

    # 1. Verify freeze snapshot integrity first
    snap_errors = []
    for rel, info in frozen_files.items():
        sp = snap_root / rel
        if not sp.exists():
            snap_errors.append({"file": rel, "issue": "snapshot_missing"})
        elif _sha256(sp) != info["sha256"]:
            snap_errors.append({"file": rel, "issue": "snapshot_corrupt"})

    # 2. Verify every target file matches expected current hash (no silent overwrite of untracked/newer edits)
    precheck_mismatches = []
    for rel, exp_hash in sorted(expected_current.items()):
        cur = root / rel
        if exp_hash is None:
            if cur.exists():
                precheck_mismatches.append({"file": rel, "expected": None, "actual": _sha256(cur)})
        else:
            if not cur.exists():
                precheck_mismatches.append({"file": rel, "expected": exp_hash, "actual": None})
            else:
                got = _sha256(cur)
                if got != exp_hash:
                    precheck_mismatches.append({"file": rel, "expected": exp_hash, "actual": got})

    # Ensure every modified/added file in scope is accounted for in freeze or marked added (not in frozen_files -> delete)
    if snap_errors or precheck_mismatches:
        return False, {
            "status": "aborted_precheck",
            "snapshot_errors": snap_errors,
            "precheck_mismatches": precheck_mismatches,
        }

    restored = []
    removed = []
    if apply:
        for rel in sorted(expected_current):
            target = root / rel
            if rel in frozen_files:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap_root / rel, target)
                post_sha = _sha256(target)
                assert post_sha == frozen_files[rel]["sha256"], f"Post-rollback hash mismatch on {rel}"
                restored.append({"file": rel, "restored_sha256": post_sha})
            elif target.exists():
                target.unlink()
                removed.append(rel)

    return True, {
        "status": "applied" if apply else "dry_run_verified",
        "checked_files": len(expected_current),
        "restored": restored,
        "removed": removed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify expected hashes (including config/docs) before rolling back.")
    ap.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    ap.add_argument("--expected-manifest", required=True, help="JSON mapping relpath -> expected current SHA256 (or JSON with 'expected_current').")
    ap.add_argument("--apply", action="store_true", help="Apply restore after hash verification succeeds.")
    args = ap.parse_args(argv)

    raw = json.loads(Path(args.expected_manifest).read_text(encoding="utf-8"))
    exp = raw.get("expected_current", raw)
    ok, report = verify_and_rollback(
        freeze_dir=Path(args.freeze_dir),
        expected_current=exp,
        root=ROOT,
        apply=args.apply,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
