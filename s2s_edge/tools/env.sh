# Source AFTER activating .venv:  . .venv/bin/activate; . tools/env.sh
# The KIMQ8550 user guide installs QAIRT (e.g. 2.41) into /usr/lib and its Hexagon v73 skel into /usr/lib/rfsa/adsp. Our context binaries are
# linked for QAIRT 2.51 = the runtime bundled inside the qai_appbuilder wheel, so the wheel's libs must win:
#   * LD_LIBRARY_PATH must be set BEFORE python starts (the loader reads it once), so a plain `python ...` could pick /usr/lib/libQnnHtpV73Stub.so;
#   * qai_appbuilder only sets ADSP_LIBRARY_PATH when it is empty, so a pre-set system value would load the 2.41 skel with the 2.51 stub.
# The system dirs stay after ours (FastRPC also needs the box's own shell files). Set S2S_SYSTEM_QNN=1 to leave both variables untouched.
if [ "${S2S_SYSTEM_QNN:-0}" != "1" ]; then
  QAI=$(python -c "import qai_appbuilder, os; print(os.path.dirname(qai_appbuilder.__file__))" 2>/dev/null)
  if [ -n "$QAI" ] && [ -d "$QAI/libs" ]; then
    export LD_LIBRARY_PATH="$QAI/libs:$QAI${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ADSP_LIBRARY_PATH="$QAI/libs;/usr/lib/rfsa/adsp;/dsp"
  else
    echo "[env] qai_appbuilder not importable: run install.sh first" >&2
  fi
fi
echo "[env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
echo "[env] ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH:-}" >&2
