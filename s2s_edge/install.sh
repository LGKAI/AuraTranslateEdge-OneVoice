#!/usr/bin/env bash
# Offline install (no network needed) on aarch64 Linux with a Hexagon v73 NPU (HSPTEK AIBOX 8550 = QCS8550; also IQ-9075).
#   wheelhouse_cp310/  Python 3.10, glibc >= 2.35 (Ubuntu 22.04: the AIBOX 8550 image)
#   wheelhouse/        Python 3.12, glibc >= 2.39 (Ubuntu 24.04)
# Both carry qai_appbuilder 2.51.40 = the QAIRT 2.51 runtime the context binaries in models_qcs8550/ were linked for.
# Everything goes into ./.venv inside this folder: nothing on the system (/usr/lib, apt) is touched.
set -e
cd "$(dirname "$0")"
PYV=$(python3 -c 'import sys; print("%d%d" % sys.version_info[:2])')
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)
echo "python3 = $PYV, glibc = $GLIBC, arch = $(uname -m)"
[ "$(uname -m)" = "aarch64" ] || echo "WARNING: not aarch64; the wheels are for aarch64 only"
case "$PYV" in
  310) WH=wheelhouse_cp310 ;;
  312) WH=wheelhouse ;;
  *)   echo "No wheelhouse for Python $PYV. Options: python3.10/3.12 from the OS, or a container (tools/docker_run.sh)."; exit 2 ;;
esac
[ -d "$WH" ] || { echo "missing $WH/"; exit 2; }
PIPWHL=$(ls "$WH"/pip-*.whl 2>/dev/null | head -1)
if ! python3 -m venv --clear .venv 2>/tmp/s2s_venv_err.txt; then
  # Debian/Ubuntu without the python3-venv package: the venv module exists but ensurepip does not -> make the venv without pip, then
  # install pip from the wheel shipped here (a pip wheel can run itself: python pip.whl/pip install ...)
  echo "python3 -m venv failed ($(tail -1 /tmp/s2s_venv_err.txt)); retrying without pip and bootstrapping pip from $PIPWHL"
  python3 -m venv --clear --without-pip .venv
  [ -n "$PIPWHL" ] || { echo "no pip wheel in $WH"; exit 2; }
  .venv/bin/python "$PIPWHL/pip" install --no-index "$PIPWHL"
fi
. .venv/bin/activate
python -m pip --version
python -m pip install --no-index --find-links "$WH" -r requirements.txt openpyxl
# optional extras: a missing wheel only disables a feature (python-crfsuite -> identifier CRF of the hybrid TTS normalizer)
for opt in python-crfsuite; do python -m pip install --no-index --find-links "$WH" "$opt" >/dev/null 2>&1 && echo "optional $opt installed" || echo "optional $opt not in $WH: feature disabled (OK)"; done
python -m pip install --no-index --find-links "$WH" qai_appbuilder || { echo "qai_appbuilder wheel rejected (glibc $GLIBC too old?). See tools/docker_run.sh"; exit 3; }
. tools/env.sh 2>/dev/null   # libatomic fallback + bundled QAIRT 2.51 libs, needed to import qai_appbuilder
python -c "import numpy, scipy, soundfile, sentencepiece, kaldi_native_fbank, onnxruntime, jiwer, sacrebleu, openpyxl, qai_appbuilder, os; print('imports OK; qai_appbuilder', os.path.dirname(qai_appbuilder.__file__))"
echo "Before running anything: . .venv/bin/activate; . tools/env.sh   (puts the bundled QAIRT 2.51 libs ahead of any QAIRT installed on the box)"
echo "Next: bash start.sh demo (web demo) | bash start.sh box (batch web UI) | bash start.sh check (NPU check only)"
