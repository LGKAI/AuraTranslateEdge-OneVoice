#!/usr/bin/env bash
# ONE entry point on the AIBOX 8550 (run from anywhere, as root or as system):
#   bash start.sh demo            web demo (web_demo/: mic / upload / edit the recognised text and translate again), port 8000
#   bash start.sh box             box web UI (batch BTC / ADV runs, zip export), port 8770  (= tools/start_box.sh)
#   bash start.sh check           install if needed + check every graph on the NPU + smoke test (tools/session0.sh)
#   bash start.sh live vi ko      microphone conversation in the terminal (needs a USB mic/speaker on the box)
# Env: PORT, HOST (default 127.0.0.1: open it from the laptop through `adb forward` / `ssh -L`), MODELS (default: the set session0 validated, else
#      models_qcs8550/models.json = NLLB-3.3B route + 600M), SKIP_CHECK=1, S2S_MT=600m (no 3.3B), S2S_ASR_CLEAN=0, S2S_MT_SPLIT=0.
# Logs: logs/start_<mode>_<time>.log. Nothing outside this folder is changed.
set -u
cd "$(dirname "$(readlink -f "$0")")"
MODE=${1:-demo}; shift 2>/dev/null || true
case "$MODE" in demo|box|check|live) ;; -h|--help|help) sed -n 2,10p "$0"; exit 0 ;; *) echo "unknown mode '$MODE' (demo | box | check | live)"; exit 2 ;; esac

# --- the NPU device belongs to user "system": a root shell on this image (CapEff 0xc0) cannot open /dev/adsprpc-smd -> re-run as system ---
if [ "$(id -u)" = "0" ] && [ "$(id -un)" != "system" ] && ! { [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ]; } && id system >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  echo "== root cannot open the NPU (/dev/adsprpc-smd): re-running as user 'system'"
  if ! runuser -u system -- test -r "$PWD/s2s_edge/__init__.py"; then
    echo "!! user 'system' cannot read $PWD (a parent folder is private, e.g. /root)."
    echo "   Fix (as root): mkdir -p /opt/onevoice && mv \"$PWD\" /opt/onevoice/ && chmod -R a+rX /opt/onevoice   then run it from there"
    echo "   (or put the package under /tmp/onevoice as on day 1)."; exit 2
  fi
  for d in logs .venv; do                                                   # folders system must write: make them writable (inside this package only)
    [ -e "$d" ] && [ "$(stat -c %U "$d")" != "system" ] && chown -R system:system "$d" 2>/dev/null
  done
  mkdir -p logs && chown system:system logs 2>/dev/null; chmod a+rwx logs 2>/dev/null
  runuser -u system -- test -w logs || { echo "!! user system cannot write $PWD/logs: chmod a+rwx $PWD/logs"; exit 2; }
  runuser -u system -- test -w . || chmod a+rwx . 2>/dev/null || true       # install.sh creates .venv here
  exec runuser -u system -- env HOME=/tmp/ov_home_system PATH="$PATH" bash "$PWD/start.sh" "$MODE" "$@"
fi
mkdir -p "${HOME:-/tmp}" logs 2>/dev/null
LOG=logs/start_${MODE}_$(date +%Y%m%d_%H%M%S).log
echo "== $(date '+%H:%M:%S') $(hostname) | user $(id -un) | $(uname -m) | python3 $(python3 -V 2>&1 | cut -d' ' -f2) | free RAM $(free -h | awk '/Mem:/{print $7}') | disk $(df -h . | awk 'NR==2{print $4}')"
[ -e /dev/adsprpc-smd ] && { [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ] && echo "== NPU device: OK" || echo "!! NPU device /dev/adsprpc-smd not accessible for $(id -un)"; } || echo "!! /dev/adsprpc-smd missing: not the AIBOX 8550? (NPU graphs will fail)"

if [ "$MODE" = "box" ]; then exec bash tools/start_box.sh "$@"; fi

if [ ! -x .venv/bin/python ]; then
  echo "== first run: offline install into ./.venv (1-3 min), log logs_install.txt"
  bash install.sh > logs_install.txt 2>&1 || { echo "!! install failed: tail logs_install.txt"; tail -15 logs_install.txt; exit 2; }
  tail -3 logs_install.txt
fi
. .venv/bin/activate; . tools/env.sh 2>/dev/null

if [ "$MODE" = "check" ] || { [ "${SKIP_CHECK:-0}" != "1" ] && [ ! -f logs/models_ok.txt ]; }; then
  echo "== check every graph on the NPU + smoke test (2-5 min, details logs/session0_*)"
  bash tools/session0.sh 2>&1 | tee -a "$LOG" | grep -aE "^=== |^exit|model set|no model set|packed"
  [ "$MODE" = "check" ] && exit 0
fi
M=${MODELS:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}
export S2S_MODELS="$M" S2S_BACKEND=npu

if [ "$MODE" = "live" ]; then
  A=${1:-vi}; B=${2:-ko}; shift 2 2>/dev/null || true
  exec python -m s2s_edge live --pair "$A,$B" --models "$M" --backend npu --continuous "$@"
fi

# demo
PORT=${PORT:-8000}; HOST=${HOST:-127.0.0.1}
echo "== web demo with $M on $HOST:$PORT. Loading + warming up every graph takes 1-2 min; wait for 'OneVoice S2S demo ->'."
echo "   Laptop:  adb forward tcp:$PORT tcp:$PORT   then open http://localhost:$PORT   (mic works only on localhost/https)"
echo "   Log: $LOG"
S2S_ENGINE=real python -u web_demo/server.py --host "$HOST" --port "$PORT" 2>&1 | tee -a "$LOG"
