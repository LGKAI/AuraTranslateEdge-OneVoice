#!/usr/bin/env bash
# Status screen for the demo video (step 2): network, NPU device, the last NPU check of every graph, NLLB-3.3B files, RAM.
# READ-ONLY: changes nothing, loads no model (safe while the web demo runs). Works as root or as system.
#   bash tools/demo_status.sh
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
B=$(tput bold 2>/dev/null); N=$(tput sgr0 2>/dev/null)
echo "${B}=== AuraTranslate Edge | $(hostname) | $(date '+%d/%m/%Y %H:%M:%S') | user $(id -un)${N}"
[ -r /proc/device-tree/model ] && echo "    board: $(tr -d '\0' < /proc/device-tree/model)"

echo "${B}--- Mạng${N}"
ip -brief link 2>/dev/null | awk '$1 != "lo" {printf "    %-14s %s\n", $1, $2}'
if ip route 2>/dev/null | grep -q '^default'; then echo "    default route: có"; else echo "    default route: KHÔNG có"; fi
if ping -c 2 -W 2 8.8.8.8 >/dev/null 2>&1; then echo "    ${B}!! ping 8.8.8.8 thành công: hộp VẪN CÒN MẠNG${N}"; else echo "    ping 8.8.8.8: không tới được  ->  ${B}OFFLINE${N}"; fi

echo "${B}--- NPU (Hexagon v73)${N}"
if [ -e /dev/adsprpc-smd ]; then
  echo "    $(ls -l /dev/adsprpc-smd | awk '{print $1, $3":"$4, $NF}')"
  if [ -r /dev/adsprpc-smd ] && [ -w /dev/adsprpc-smd ]; then echo "    user $(id -un) mở được NPU"; else echo "    user $(id -un) không mở trực tiếp; start.sh tự chạy model bằng user system (chủ của NPU)"; fi
else echo "    !! không thấy /dev/adsprpc-smd"; fi

M=$(cat logs/models_ok.txt 2>/dev/null)
echo "${B}--- Model set đã kiểm trên NPU: ${M:-chưa kiểm (chạy: bash start.sh check)}${N}"
G=$(ls -t logs/golden_npu_*.json 2>/dev/null | head -1)
if [ -n "$G" ]; then
  python3 - "$G" <<'PY'
import json, os, sys, time
p = sys.argv[1]; rows = json.load(open(p))
print(f"    lần kiểm: {time.strftime('%d/%m %H:%M', time.localtime(os.path.getmtime(p)))} ({os.path.basename(p)})")
for r in rows:
    t = f"{r['seconds'] * 1000:7.1f} ms" if r.get("seconds") is not None else "        -"
    print(f"    {'PASS' if r.get('ok') else 'FAIL'}  {r['graph']:<18} {t}")
print(f"    => {sum(1 for r in rows if r.get('ok'))}/{len(rows)} đồ thị PASS")
PY
else echo "    chưa có kết quả kiểm (logs/golden_npu_*.json): chạy  bash start.sh check"; fi
n33=$(ls models_qcs8550/nllb33_*.bin 2>/dev/null | wc -l)
if [ "$n33" -gt 0 ]; then
  echo "    NLLB-200 3.3B: $n33 đồ thị ($(du -ch models_qcs8550/nllb33_*.bin 2>/dev/null | tail -1 | cut -f1)), kiểm khi nạp; trạng thái hiện ở góc trên web demo"
  [ -f logs/mt33_check.json ] && echo "    (đã có logs/mt33_check.json từ tools/mt33_check.py)"
fi

echo "${B}--- Bộ nhớ${N}"
free -h | awk 'NR==1 {printf "    %-6s %8s %8s %8s\n", "", $1, $2, $NF} NR==2 {printf "    %-6s %8s %8s %8s\n", $1, $2, $3, $NF}'
T=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
[ -n "$T" ] && [ "$T" -gt 1000 ] 2>/dev/null && echo "    nhiệt độ cao nhất: $((T / 1000)) °C"
exit 0
