#!/usr/bin/env bash
# Start the web UI + translation server ON THE BOX (models stay loaded; inference on the NPU).
# Usage (package root):  bash tools/box_serve.sh [default pair for the microphone test = vi,ko] [models.json]
# Then on the laptop:     python tools/laptop_web.py      (= adb forward tcp:8770 tcp:8770 + opens http://127.0.0.1:8770)
#   data sets read from $S2S_DATA (default: the parent folder of this package): onevoice_btc_data/ and onevoice_btc_adv/
#   S2S_NO_WARMUP=1: start the page at once, models load on the first request (default: load + run every graph once first)
#   results: logs/webui/<run>.jsonl, output audio logs/webui/<run>/*.wav, microphone turns logs/webui/mic/
cd "$(dirname "$0")/.."
PAIR=${1:-vi,ko}
M=${2:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}
. .venv/bin/activate && . tools/env.sh
mkdir -p logs
DATA=${S2S_DATA:-$(cd .. && pwd)}          # default: the folder that holds this package (onevoice_btc_data/ and onevoice_btc_adv/ next to it)
echo "models $M | data $DATA | default pair $PAIR"
python -m s2s_edge serve --pair "$PAIR" --models "$M" --backend npu --port "${S2S_PORT:-8770}" --data_root "$DATA" ${S2S_NO_WARMUP:+--no-warmup} 2>&1 | tee -a "logs/serve_$(date +%Y%m%d).log"
