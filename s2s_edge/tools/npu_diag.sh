#!/usr/bin/env bash
# ON THE BOX, as the account that can use the NPU (root): find out why the NPU cannot be opened and try the library-path variants.
# Usage (package root):  bash tools/npu_diag.sh 2>&1 | tee /tmp/npu_diag.txt        then copy /tmp/npu_diag.txt back (adb pull) or paste it.
# Read-only except that it starts a few short python processes that try to open one NPU graph (asr_vi). Nothing outside this folder is changed.
cd "$(dirname "$0")/.."; . .venv/bin/activate 2>/dev/null
QAI=$(python -c "import importlib.util, os; s = importlib.util.find_spec('qai_appbuilder'); print(os.path.dirname(s.origin) if s else '')" 2>/dev/null)
echo "== user: $(id)"
echo "== NPU device files"; ls -l /dev/adsprpc-smd* /dev/*cdsp* 2>&1 | head; for d in /dev/adsprpc-smd /dev/adsprpc-smd-secure; do [ -r "$d" ] && [ -w "$d" ] && echo "$d: this user can read+write" || echo "$d: NO access for this user"; done
echo "== DSP subsystems"; for r in /sys/class/remoteproc/remoteproc*; do echo "$(cat $r/name 2>/dev/null) = $(cat $r/state 2>/dev/null)"; done
echo "== FastRPC daemons"; ps -eo user,pid,args | grep -iE "adsprpcd|cdsprpcd|fastrpc|sdsprpcd" | grep -v grep
echo "== system stack (box vendor)"; ls -l /usr/lib/libcdsprpc.so /usr/lib/rfsa/adsp/libQnnHtpV73Skel.so /usr/lib/libQnnHtpV73Stub.so 2>&1; strings /usr/lib/libQnnHtp.so 2>/dev/null | grep -m1 -E "v2\.[0-9]+\.[0-9]+"; dpkg -l 2>/dev/null | grep -iE " (adsprpc|qnn|qti-qnn) " | cut -c1-100
echo "== our runtime (qai_appbuilder wheel)"; echo "$QAI"; ls -l "$QAI"/libs 2>/dev/null | grep -iE "V73|Skel|Stub|System|Htp\.so" | head; strings "$QAI/libs/libQnnHtp.so" 2>/dev/null | grep -m1 -E "v2\.[0-9]+\.[0-9]+"; python -c "import qai_appbuilder as q; print('qai_appbuilder', getattr(q, '__version__', '?'))" 2>&1 | tail -1
echo "== kernel log about FastRPC / CDSP (last lines)"; dmesg 2>/dev/null | grep -iE "fastrpc|adsprpc|cdsp|unsigned|rpmsg" | tail -12
cat > /tmp/_npu_try.py <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
from s2s_edge.registry import Registry
reg = Registry("models_qcs8550/models.json", "npu")
g = reg.graph("asr_vi"); z = np.load(reg.gold_path("asr_vi"))
out = g(**{n[3:]: z[n] for n in z.files if n.startswith("in_")})
print("RESULT:", "OK, %d outputs" % len(out) if len(out) else "FAIL (no outputs)")
PYEOF
try() {  # name, then env assignments
  n=$1; shift; echo; echo "== variant $n"
  ( env "$@" timeout 120 python /tmp/_npu_try.py 2>&1 | grep -E "RESULT|ERROR|Failed|unsigned|status=|Traceback|Error" | cut -c1-230 | head -8 )
}
LIBS="$QAI/libs"; BASE="LD_LIBRARY_PATH=$LIBS:$QAI:$(pwd)/syslibs"
try "A (as start_box.sh does: wheel 2.51 libs first)" "$BASE" "ADSP_LIBRARY_PATH=$LIBS;/usr/lib/rfsa/adsp;/dsp"
try "B (system skel first, wheel 2.51 stub)"            "$BASE" "ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp;/dsp;$LIBS"
try "C (only the wheel skel)"                           "$BASE" "ADSP_LIBRARY_PATH=$LIBS"
try "D (box vendor libs only: expected to reject 2.51 context binaries)" "LD_LIBRARY_PATH=/usr/lib:$(pwd)/syslibs" "ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp;/dsp"
rm -f /tmp/_npu_try.py
echo; echo "== done. Send this whole output."
