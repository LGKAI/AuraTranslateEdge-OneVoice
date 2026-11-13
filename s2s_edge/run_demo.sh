#!/usr/bin/env bash
# Live demo with a USB mic / speaker plugged INTO THE BOX. (Laptop mic/speaker instead: tools/box_serve.sh + tools/laptop_ptt.py.)
# usage: bash run_demo.sh <src> <tgt> [extra live-mode args]      e.g. S2S_MIC=plughw:1,0 S2S_SPK=plughw:1,0 bash run_demo.sh vi en --continuous
# Model set: S2S_MODELS, else the set tools/session0.sh validated (logs/models_ok.txt), else models_qcs8550 (AIBOX 8550).
cd "$(dirname "$0")"; . .venv/bin/activate; . tools/env.sh
M=${S2S_MODELS:-$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)}
python -m s2s_edge live --src "${1:-vi}" --tgt "${2:-en}" --models "$M" --backend npu "${@:3}"
