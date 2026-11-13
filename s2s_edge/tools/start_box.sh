#!/usr/bin/env bash
# ON THE BOX, one command after the files are pushed: install (first time) -> check every model graph on the NPU -> start the web UI.
# Usage (package root):  bash tools/start_box.sh            (second time: SKIP_CHECK=1 bash tools/start_box.sh skips the check)
# Then on the laptop:    python tools/laptop_web.py
# Logs: logs/session0_*/ (check), logs/serve_<date>.log (web). Nothing outside this folder is changed.
set -u
cd "$(dirname "$0")/.."
# On the AIBOX 8550 image the shell you get from `adb shell` + `su` is "root" with almost no capabilities (CapEff 0xc0: only setuid/setgid), so it cannot
# open /dev/adsprpc-smd (owned by user "system", the FastRPC daemons' user). That user CAN open it: re-run this script as "system".
if [ "$(id -un)" != "system" ] && [ "$(id -u)" = "0" ] && ! { [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ]; } && id system >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
  echo "== this shell cannot open /dev/adsprpc-smd (NPU): re-running as user 'system' (the owner of the NPU device)"
  [ -d logs ] && [ "$(stat -c %U logs)" != "system" ] && mv logs "logs_root_$(date +%H%M%S)"; mkdir -p logs 2>/dev/null; chmod a+rwx logs 2>/dev/null
  runuser -u system -- test -w logs || { echo "!! user system cannot write to $(pwd)/logs: run  chmod a+rwx $(pwd)/logs  (as the owner) or ask BTC"; exit 2; }
  exec runuser -u system -- env HOME=/tmp/ov_home_system bash "$0" "$@"
fi
mkdir -p "${HOME:-/tmp}" 2>/dev/null
echo "== $(date '+%H:%M:%S') box: $(hostname) | $(uname -m) | $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME") | python3 $(python3 -V 2>&1 | cut -d' ' -f2) | free $(free -h | awk '/Mem:/{print $7}') | disk $(df -h . | awk 'NR==2{print $4}') free"
ls /dev | grep -iE "fastrpc|adsprpc|cdsp" | tr '\n' ' ' | sed 's/^/== FastRPC devices: /'; echo
if [ ! -x .venv/bin/python ]; then
  echo "== install (offline, about 1-3 min)"; bash install.sh > logs_install.txt 2>&1; rc=$?; tail -4 logs_install.txt
  [ $rc -eq 0 ] || { echo "!! install failed (exit $rc): full log in logs_install.txt; fixes in docs/HUONG_DAN_NGAY1.md section 6"; exit 2; }
fi
if [ "${SKIP_CHECK:-0}" != "1" ]; then
  echo "== check every graph on the NPU + smoke test (2-5 min); details in logs/session0_*/"
  bash tools/session0.sh 2>&1 | grep -E "^=== |^exit|model set|no model set|packed"
fi
M=${MODELS:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}   # default models.json = NLLB-3.3B routed (mt_route); MODELS=models_qcs8550/models_600m.json -> old all-600M set
echo "== start the web UI with $M (models load once, about 1 min). Keep this window open. Laptop: python tools/laptop_web.py"
S2S_DATA=${S2S_DATA:-$(cd .. && pwd)} exec bash tools/box_serve.sh "${PAIR:-vi,ko}" "$M"
