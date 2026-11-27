#!/usr/bin/env bash
# OVAC Phase 3 CLI Eval Mode entry point:
#   ./run_eval.sh "/path/to/input" "/path/to/output"
# Runs 100% offline on HSPTEK AIBOX 8550 (QCS8550, Ubuntu 22.04 aarch64) using the submitted Phase 3 demo pipeline.
set -u

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input_dir> <output_dir>" >&2
  exit 2
fi

# Resolve input and output paths to absolute paths BEFORE changing directory (supports relative paths & spaces).
IN_ABS="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1")"
OUT_ABS="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$2")"
PKG_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

if [ ! -f "$IN_ABS/manifest.csv" ]; then
  echo "ERROR: manifest.csv not found in input directory: $IN_ABS" >&2
  exit 2
fi

mkdir -p "$OUT_ABS" || { echo "ERROR: cannot create output directory: $OUT_ABS" >&2; exit 2; }

cd "$PKG_DIR"

# On HSPTEK AIBOX 8550, root (CapEff 0xc0) cannot open /dev/adsprpc-smd -> re-run as user 'system'.
if [ "$(id -u)" = "0" ] && [ "$(id -un)" != "system" ] && ! { [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ]; } && id system >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  if ! runuser -u system -- test -r "$PKG_DIR/s2s_edge/__init__.py"; then
    echo "ERROR: user 'system' cannot read $PKG_DIR (move package out of /root, e.g. /opt/onevoice or /tmp/onevoice)" >&2
    exit 2
  fi
  for d in logs .venv; do
    [ -e "$d" ] && [ "$(stat -c %U "$d")" != "system" ] && chown -R system:system "$d" 2>/dev/null || true
  done
  mkdir -p logs && chown system:system logs 2>/dev/null || true
  chmod a+rwx logs 2>/dev/null || true

  # If judges placed input/ or output/ under a private directory like /root where user 'system' cannot traverse,
  # transparently stage input (read-only copy, never modifying IN_ABS) and mirror output back to OUT_ABS
  # (keeping OUT_ABS directory itself intact, never deleting it).
  STAGE_ROOT=""
  RUN_IN="$IN_ABS"
  RUN_OUT="$OUT_ABS"
  if ! runuser -u system -- test -w "$OUT_ABS"; then
    chown -R system:system "$OUT_ABS" 2>/dev/null || chmod -R a+rwx "$OUT_ABS" 2>/dev/null || true
  fi
  if ! runuser -u system -- test -r "$IN_ABS/manifest.csv" || ! runuser -u system -- test -w "$OUT_ABS"; then
    STAGE_ROOT="/tmp/ovac_eval_stage_$$"
    RUN_IN="$STAGE_ROOT/input"
    RUN_OUT="$STAGE_ROOT/output"
    mkdir -p "$RUN_IN" "$RUN_OUT"
    cp -a "$IN_ABS/." "$RUN_IN/"
    cp -a "$OUT_ABS/." "$RUN_OUT/" 2>/dev/null || true
    chown -R system:system "$STAGE_ROOT"
    chmod -R a+rX "$RUN_IN"
    chmod -R a+rwx "$RUN_OUT"

    sync_out() {
      python3 - "$RUN_OUT" "$OUT_ABS" <<'PY'
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
if not os.path.isdir(src) or not os.path.isdir(dst):
    sys.exit(0)
# Remove files in dst that were cleared in src (without ever deleting dst itself)
src_entries = set(os.listdir(src))
for name in os.listdir(dst):
    if name not in src_entries:
        p = os.path.join(dst, name)
        shutil.rmtree(p) if (os.path.isdir(p) and not os.path.islink(p)) else os.unlink(p)
# Copy new/updated files from src to dst
for root_d, dirs, files in os.walk(src):
    rel = os.path.relpath(root_d, src)
    target_d = dst if rel == "." else os.path.join(dst, rel)
    os.makedirs(target_d, exist_ok=True)
    for fn in files:
        s = os.path.join(root_d, fn)
        t = os.path.join(target_d, fn)
        if not os.path.exists(t) or os.stat(s).st_mtime_ns != os.stat(t).st_mtime_ns or os.stat(s).st_size != os.stat(t).st_size:
            shutil.copy2(s, t)
PY
    }

    ( while [ -d "$STAGE_ROOT" ]; do sleep 2; sync_out >/dev/null 2>&1 || true; done ) &
    SYNC_PID=$!
    cleanup_stage() {
      kill "$SYNC_PID" >/dev/null 2>&1 || true
      wait "$SYNC_PID" 2>/dev/null || true
      sync_out || true
      rm -rf "$STAGE_ROOT"
    }
    trap cleanup_stage EXIT INT TERM

    runuser -u system -- env HOME=/tmp/ov_home_system PATH="$PATH" \
      S2S_MODELS="${S2S_MODELS:-}" S2S_BACKEND="${S2S_BACKEND:-npu}" \
      OVAC_TEAM="${OVAC_TEAM:-GiaSuDinhCao}" OVAC_TIMEOUT_S="${OVAC_TIMEOUT_S:-60}" \
      bash "$PKG_DIR/run_eval.sh" "$RUN_IN" "$RUN_OUT"
    RC=$?
    exit "$RC"
  fi

  exec runuser -u system -- env HOME=/tmp/ov_home_system PATH="$PATH" \
    S2S_MODELS="${S2S_MODELS:-}" S2S_BACKEND="${S2S_BACKEND:-npu}" \
    OVAC_TEAM="${OVAC_TEAM:-GiaSuDinhCao}" OVAC_TIMEOUT_S="${OVAC_TIMEOUT_S:-60}" \
    bash "$PKG_DIR/run_eval.sh" "$IN_ABS" "$OUT_ABS"
fi

mkdir -p "${HOME:-/tmp}" "$PKG_DIR/logs" 2>/dev/null || true

# Offline setup if .venv is not installed yet
if [ ! -x "$PKG_DIR/.venv/bin/python" ]; then
  echo "== First run: offline setup into ./.venv ..."
  bash "$PKG_DIR/install.sh" > "$PKG_DIR/logs/eval_install.log" 2>&1 || {
    echo "ERROR: offline install failed (see $PKG_DIR/logs/eval_install.log)" >&2
    tail -20 "$PKG_DIR/logs/eval_install.log" >&2
    exit 3
  }
fi

# shellcheck disable=SC1091
. "$PKG_DIR/.venv/bin/activate"
# shellcheck disable=SC1091
. "$PKG_DIR/tools/env.sh" 2>/dev/null || true

M="${S2S_MODELS:-}"
if [ -z "$M" ] && [ -f "$PKG_DIR/logs/models_ok.txt" ]; then
  CAND="$(tr -d '\r\n' < "$PKG_DIR/logs/models_ok.txt")"
  [ -n "$CAND" ] && [ -f "$CAND" ] && M="$CAND"
fi
if [ -z "$M" ]; then
  if [ -f "$PKG_DIR/models_qcs8550/models.json" ]; then
    M="$PKG_DIR/models_qcs8550/models.json"
  else
    M="$PKG_DIR/models/models.json"
  fi
fi

export PYTHONUNBUFFERED=1
exec python -u -m s2s_edge eval \
  --input "$IN_ABS" \
  --output "$OUT_ABS" \
  --models "$M" \
  --backend "${S2S_BACKEND:-npu}" \
  --team "${OVAC_TEAM:-GiaSuDinhCao}" \
  --timeout "${OVAC_TIMEOUT_S:-60}"
