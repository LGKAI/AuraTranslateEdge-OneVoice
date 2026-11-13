#!/usr/bin/env bash
# Noise-robustness run on the box (ADV set: 25 sentences x 4 noise levels x 6 directions = 600 clips).
# Usage (package root):  bash tools/run_adv.sh /root/onevoice/onevoice_btc_adv [n_sentences=0 (all 25)] [models.json]
set -u
cd "$(dirname "$0")/.."
DATA=${1:?path to the ADV data folder}; N=${2:-0}
M=${3:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}
. .venv/bin/activate 2>/dev/null && . tools/env.sh
D=logs/adv_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
python tools/eval_adv.py --data "$DATA" --models "$M" --backend npu --n "$N" --out "$D/adv.json" 2>&1 | tee "$D/adv.log" | grep -E "^===|^\||Error|Traceback"
tar czf "$D.tgz" "$D" && echo "packed $D.tgz  -> copy it back"
