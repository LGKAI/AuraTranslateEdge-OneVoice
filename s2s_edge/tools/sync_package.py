# -*- coding: utf-8 -*-
"""Refresh the CODE of an existing offline package (which already holds the NPU context binaries, wheelhouses, golden vectors and eval data)
from this repo, without touching the big files. Use it instead of build_package.py when the model files are only in the package.
  python s2s_edge/tools/sync_package.py <package dir, e.g. ../onevoice_deploy/s2s_edge_pkg> [--dry-run]
Copies: s2s_edge/ (python package), tools/, tests/, docs/, small assets, eval/*.json that the repo tracks, models_qcs8550/*.json, start.sh,
install.sh, run_demo.sh, README.md, requirements.txt, glossary*.json, and the repo's web_demo/ -> <package>/web_demo/.
Python files of the package that no longer exist in the repo are removed (old modules must not shadow new ones). Writes VERSION.txt."""
import argparse, os, shutil, subprocess, sys, time

S2S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repo/s2s_edge
REPO = os.path.dirname(S2S)
SKIP = ("__pycache__", ".pyc", ".onnx", ".bin", ".npz", ".dlc")


def copy_tree(src, dst, dry, prune_py=False, log=None):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(root, src); out = os.path.normpath(os.path.join(dst, rel))
        for f in files:
            if f.endswith(SKIP): continue
            a, b = os.path.join(root, f), os.path.join(out, f)
            if os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read(): continue
            n += 1; log.append(f"  {os.path.relpath(b, dst_root)}")
            if not dry: os.makedirs(out, exist_ok=True); shutil.copy2(a, b)
    if prune_py and os.path.isdir(dst):                                      # stale python modules
        for root, dirs, files in os.walk(dst):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and not os.path.exists(os.path.join(src, os.path.relpath(os.path.join(root, f), dst))):
                    log.append(f"  removed stale {os.path.relpath(os.path.join(root, f), dst_root)}"); n += 1
                    if not dry: os.remove(os.path.join(root, f))
    return n


def main():
    global dst_root
    ap = argparse.ArgumentParser(); ap.add_argument("package"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    dst_root = os.path.abspath(a.package)
    if not os.path.isdir(os.path.join(dst_root, "models_qcs8550")): sys.exit(f"{dst_root} does not look like the package (no models_qcs8550/)")
    log = []; n = 0
    n += copy_tree(os.path.join(S2S, "s2s_edge"), os.path.join(dst_root, "s2s_edge"), a.dry_run, prune_py=True, log=log)
    for d in ("tools", "tests", "docs"):
        n += copy_tree(os.path.join(S2S, d), os.path.join(dst_root, d), a.dry_run, log=log)
    n += copy_tree(os.path.join(REPO, "web_demo"), os.path.join(dst_root, "web_demo"), a.dry_run, log=log)
    for d in ("assets", "eval", "models_qcs8550"):                              # small tracked files only; big binaries stay as they are
        n += copy_tree(os.path.join(S2S, d), os.path.join(dst_root, d), a.dry_run, log=log)
    for f in ("start.sh", "install.sh", "run_demo.sh", "README.md", "HUONG_DAN_CHAY_BOX.md", "requirements.txt", "glossary_enterprise.json", "glossary.example.json"):
        s, t = os.path.join(S2S, f), os.path.join(dst_root, f)
        if os.path.isfile(s) and not (os.path.exists(t) and open(s, "rb").read() == open(t, "rb").read()):
            n += 1; log.append(f"  {f}")
            if not a.dry_run:
                data = open(s, "rb").read()
                if f.endswith(".sh"): data = data.replace(b"\r\n", b"\n")
                open(t, "wb").write(data)
    try: rev = subprocess.check_output(["git", "-C", REPO, "describe", "--always", "--dirty"], text=True).strip(); br = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception: rev, br = "unknown", "unknown"
    if not a.dry_run:
        old = open(os.path.join(dst_root, "VERSION.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(dst_root, "VERSION.txt")) else ""
        open(os.path.join(dst_root, "VERSION.txt"), "w", encoding="utf-8").write(
            f"s2s_edge package, code synced {time.strftime('%Y-%m-%d %H:%M')} from git {rev} (branch {br}) with tools/sync_package.py\n"
            "code: integrate/aibox8550 (ASR speed/accuracy + TTS hybrid format + Plan-01) merged with the AIBOX 8550 box work (web UI, NLLB-3.3B route, junk filter, web_demo)\n"
            "start: bash start.sh demo | box | check | live   (guide: HUONG_DAN_CHAY_BOX.md)\n\n--- previous VERSION.txt ---\n" + old)
    print("\n".join(log)); print(f"{'would update' if a.dry_run else 'updated'} {n} files in {dst_root} (git {rev})")


if __name__ == "__main__": main()
