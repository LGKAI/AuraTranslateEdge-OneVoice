#!/usr/bin/env bash
# One-shot environment survey for a new device / QDC session (about 1-2 minutes). Output: recon_<host>_<date>.txt next to this script's cwd.
# Usage: bash tools/recon.sh
OUT="recon_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
sec() { echo; echo "===== $* ====="; }
{
sec "date / host";            date -Is; hostname; uname -a
sec "os-release";             cat /etc/os-release 2>/dev/null | head -8
sec "soc";                    cat /sys/devices/soc0/machine /sys/devices/soc0/soc_id /sys/devices/soc0/family 2>/dev/null; grep -m1 -i "hardware\|model name" /proc/cpuinfo
sec "cpu";                    lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Architecture|Max MHz" | head -6
sec "memory";                 free -h
sec "disk";                   df -h / /home /tmp 2>/dev/null | head -6
sec "network (expect: none when testing offline)"; ip -br addr 2>/dev/null | head -10; (ping -c1 -W2 8.8.8.8 >/dev/null 2>&1 && echo "INTERNET: reachable") || echo "INTERNET: unreachable"
sec "glibc (wheel needs >= 2.35 for Python 3.10, >= 2.39 for 3.12)"; ldd --version 2>&1 | head -1
sec "python";                 python3 --version 2>&1; pip3 --version 2>&1 | head -1; ls /usr/bin/python3* 2>/dev/null | tr '\n' ' '; echo; python3 -c "import venv; print('venv OK')" 2>&1 | tail -1
sec "docker (fallback when the OS Python/glibc is too old)"; docker --version 2>&1 | head -1; docker images 2>/dev/null | head -5
sec "usb devices (USB microphone / speaker / PTT button)"; lsusb 2>/dev/null | head -12
sec "python packages (relevant)"; pip3 list 2>/dev/null | grep -iE "numpy|scipy|onnx|onnxruntime|qai|qnn|sentencepiece|kaldi|soundfile|sounddevice|torch|psutil|jiwer|sacrebleu"
sec "onnxruntime providers"; python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())" 2>&1 | tail -1
sec "qai_appbuilder";         python3 -c "import qai_appbuilder, sys; print('qai_appbuilder OK', getattr(qai_appbuilder,'__version__',''))" 2>&1 | tail -1
sec "QNN tools / libs";       which qnn-net-run qnn-context-binary-generator qnn-platform-validator 2>&1 | head; ls -d /opt/qcom* /opt/qairt* /usr/lib/qnn* 2>/dev/null
                              ls /usr/lib/aarch64-linux-gnu/libQnn* /usr/lib/libQnn* /opt/*/lib/*/libQnnHtp*.so 2>/dev/null | head -20
                              (find /usr /opt /lib /home -maxdepth 6 \( -name "libQnnHtp*.so" -o -name "libQnnSystem.so" -o -name "qnn-net-run" -o -name "libQnnDsp*.so" \) 2>/dev/null | head -20)
sec "HTP / DSP devices";      ls -l /dev/fastrpc* /dev/adsprpc* /dev/dri 2>/dev/null; ls /usr/lib/rfsa/adsp /usr/lib/dsp 2>/dev/null | head -20
sec "audio devices";          arecord -l 2>&1 | head -12; aplay -l 2>&1 | head -12; (pactl info 2>/dev/null | head -4)
sec "sound python";           python3 -c "import sounddevice as sd; print(sd.query_devices())" 2>&1 | tail -6
sec "HSPTek OS version (KIMQ8550 guide: Ubuntu 22.04 V3.0.0 expected; datasheet still says 20.04)"; cat /etc/os-version 2>/dev/null
sec "QAIRT already on the box (user guide pushes 2.41 libs to /usr/lib and the v73 unsigned skel to /usr/lib/rfsa/adsp)"
                              for f in /usr/lib/libQnnHtp.so /usr/lib/libQnnHtpV73Stub.so /usr/lib/rfsa/adsp/libQnnHtpV73Skel.so /usr/lib/libSNPE.so; do [ -f "$f" ] && echo "$f: $(strings "$f" 2>/dev/null | grep -m1 -oE 'v?2\.[0-9]{2}\.[0-9]+(\.[0-9]+)?')"; done
                              echo "ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH:-<unset>}  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
                              (command -v qnn-platform-validator >/dev/null && timeout 60 qnn-platform-validator --backend dsp --testBackend 2>&1 | tail -3) || echo "qnn-platform-validator not installed"
sec "DSP subsystems (cdsp must be running for the HTP)"; for r in /sys/class/remoteproc/*; do echo "$r: $(cat $r/name 2>/dev/null) $(cat $r/state 2>/dev/null)"; done
sec "thermal zones (datasheet: thermal TBD, typical 11.9 W, max 18 W on Ubuntu)"; for z in /sys/class/thermal/thermal_zone*; do printf "%s %s\n" "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"; done | head -40
sec "cpu frequency governors"; cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor 2>/dev/null | sort | uniq -c
sec "limits";                 ulimit -a | grep -E "open files|max memory|virtual" ; nproc
} 2>&1 | tee "$OUT"
echo; echo "saved $OUT  (copy it back for analysis)"
