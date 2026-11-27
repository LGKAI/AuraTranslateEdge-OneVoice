#!/usr/bin/env bash
# Offline setup script for OVAC Phase 3 CLI package (HSPTEK AIBOX 8550 / QCS8550, Ubuntu 22.04 aarch64).
# Creates ./.venv offline from wheelhouse_cp310/ (or wheelhouse/) with zero network access.
set -eu
PKG_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$PKG_DIR"

# On HSPTEK AIBOX 8550, /dev/adsprpc-smd belongs to user 'system'. If invoked as root, ensure 'system'
# can read the package and own ./.venv and ./logs so run_eval.sh works without permission errors.
if [ "$(id -u)" = "0" ] && [ "$(id -un)" != "system" ] && ! { [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ]; } && id system >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  if ! runuser -u system -- test -r "$PKG_DIR/s2s_edge/__init__.py"; then
    echo "!! user 'system' cannot read $PKG_DIR (check parent folder permissions, e.g. do not place under /root)" >&2
    exit 2
  fi
  mkdir -p "$PKG_DIR/logs"
  chown -R system:system "$PKG_DIR/logs" 2>/dev/null || true
  chmod a+rwx "$PKG_DIR/logs" 2>/dev/null || true
  [ -e "$PKG_DIR/.venv" ] && chown -R system:system "$PKG_DIR/.venv" 2>/dev/null || true
  runuser -u system -- test -w "$PKG_DIR" || chmod a+rwx "$PKG_DIR" 2>/dev/null || true
  exec runuser -u system -- env HOME=/tmp/ov_home_system PATH="$PATH" bash "$PKG_DIR/setup.sh" "$@"
fi

bash "$PKG_DIR/install.sh"
echo "Offline setup complete. Run: ./run_eval.sh \"<input_dir>\" \"<output_dir>\""
