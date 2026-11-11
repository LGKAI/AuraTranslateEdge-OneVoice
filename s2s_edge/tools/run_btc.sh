#!/usr/bin/env bash
# Run the organisers' sample set on the device NPU and pack the raw logs (per-clip CSV + JSON summary + console log).
# Usage (package root, after tools/session0.sh):  bash tools/run_btc.sh /path/to/onevoice_btc_data [n_per_direction] [models.json]
#   n_per_direction 0 = all clips (100 en pairs, 180 ko / zh pairs: about 920 clips)
set -u
cd "$(dirname "$0")/.."
DATA=${1:?path to the sample data folder (contains "Vie-Eng dataset", "Vie-Kor dataset", "Vie-Man dataset")}; N=${2:-0}
M=${3:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}      # set chosen by tools/session0.sh (AIBOX 8550 default)
. .venv/bin/activate 2>/dev/null && . tools/env.sh
D=logs/btc_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
{ date -Is; uname -a; cat /sys/devices/soc0/soc_id 2>/dev/null; free -h; } > "$D/env.txt"
for d in vi-en en-vi vi-ko ko-vi vi-zh zh-vi; do
  echo "=== $d"
  python tools/eval_btc.py --data "$DATA" --models "$M" --backend npu --dirs "$d" --n "$N" --out "$D/btc_$d.json" 2>&1 | tee "$D/btc_$d.log" | grep -E "^vi-|^en-|^ko-|^zh-|Error|Traceback"
done
tar czf "$D.tgz" "$D" && echo "packed $D.tgz  -> copy it back"
