#!/bin/sh
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_PYTHON=${S2S_PYTHON:-/private/tmp/s2s-original-env/bin/python}
if [ ! -x "$TASK_PYTHON" ]; then
  echo "Không tìm thấy môi trường Python S2S. Kiểm tra S2S_PYTHON."
  exit 1
fi
"$TASK_PYTHON" "$TASK_ROOT/tools/start_s2s_web.py" || exit 1
open 'http://localhost:8766'
