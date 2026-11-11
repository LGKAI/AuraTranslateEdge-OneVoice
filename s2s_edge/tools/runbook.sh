#!/usr/bin/env bash
# Staged runbook for the HSPTEK AIBOX 8550 (KIMQ8550). Run the stages in order; each one is small, logged and timed, and a failure
# does not stop the stages you asked for after it. Times in brackets are ESTIMATES before we have seen the box (see docs/device_plan.md).
#
#   A0 recon      [2 min]   OS / glibc / Python / QAIRT already on the box / cdsp / thermal / audio
#   A1 install    [5-15]    offline install from the package (install.sh), import qai_appbuilder
#   A2 golden     [2-5]     load every graph on the NPU and compare with the CPU reference (which graph is wrong, if any)
#   A3 perf       [5-10]    same golden run under HTP perf profiles BURST / HIGH_PERFORMANCE / NONE (system default)
#   A4 soak       [10-20]   thermal soak: decB (MT beam step) and the zh TTS graph back-to-back, latency drift first vs last minute
#   A5 smoke      [3-5]     2 BTC clips per direction, all 6 directions (end-to-end works?)
#   AL live       [2-4]     live-mode code path incl. warmup, with sample WAVs instead of a microphone (prints model load times)
#   AP parity     [10-15]   the same first-30 clips per direction the CPU reference ran on (eval/cpu_ref): does the NPU give the same quality?
#   A6 quick      [10-15]   BTC quick split (30 clips per direction = 180 clips), quality + latency
#   A7 dev        [35-50]   BTC dev split (about 550 clips): the baseline every method is compared against
#   A8 holdout    [25-35]   BTC holdout split (about 370 clips): run ONCE at the end with the final configuration
#   A9 noisy      [10-15 each] quick split on each noisy copy listed in NOISY (made with tools/mix_noise.py)
#
# Usage (package root):  DATA=/data/onevoice_btc_data bash tools/runbook.sh A0 A1 A2 A3 A4 A5 A6
#                         DATA=... TAG=beam_off EXTRA="--no_gold" bash tools/runbook.sh A6      # a method variant, tagged
#                         DATA=... NOISY="/data/btc_babble10 /data/btc_office5" bash tools/runbook.sh A9
# Env: MT (600m = pipeline A, default | 13b = pipeline B, NLLB-1.3B with beam 4 -> 2 -> greedy fallback), DATA (sample set), MODELS (default: logs/models_ok.txt or models_qcs8550/models.json), TAG (name of the variant), EXTRA (extra eval_btc args),
#      SPLIT (default eval/btc_split.json, created on first use), SOAK_S (seconds per soak graph, default 600).
set -u
cd "$(dirname "$0")/.."
TAG=${TAG:-${MT:-600m}}; D=logs/runbook_$(date +%Y%m%d_%H%M%S)_$TAG; mkdir -p "$D"
case "${MT:-600m}" in 13b|1.3b|B) DEFM=models_qcs8550_13b/models.json ;; *) DEFM=$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json) ;; esac
M=${MODELS:-$DEFM}                                       # MT=13b -> pipeline B (NLLB-1.3B), default pipeline A (NLLB-600M)
SPLIT=${SPLIT:-eval/btc_split.json}; EXTRA=${EXTRA:-}; SOAK_S=${SOAK_S:-600}
. .venv/bin/activate 2>/dev/null && . tools/env.sh
echo -e "stage\texit\tseconds\ttag\tmodels" > "$D/timing.tsv"
run() { n=$1; shift; echo "=== $n ($(date +%H:%M:%S))"; s=$(date +%s); ( "$@" ) > "$D/$n.log" 2>&1; rc=$?; t=$(( $(date +%s) - s ))
        echo -e "$n\t$rc\t$t\t$TAG\t$M" >> "$D/timing.tsv"; echo "exit $rc ($t s)"; tail -3 "$D/$n.log"; return $rc; }
need_data() { [ -n "${DATA:-}" ] || { echo "set DATA=/path/to/onevoice_btc_data"; return 1; }; [ -f "$SPLIT" ] || python tools/btc_split.py --data "$DATA" --out "$SPLIT"; }
btc() { sp=$1; shift; out="$D/btc_${sp}"; mkdir -p "$out"
        for d in vi-en en-vi vi-ko ko-vi vi-zh zh-vi; do
          python tools/eval_btc.py --data "${BTC_DATA:-$DATA}" --models "$M" --backend npu --dirs "$d" --split_file "$SPLIT" --split "$sp" --out "$out/btc_$d.json" $EXTRA "$@" || echo "!! $d failed"
        done
        python tools/eval_btc.py --summarize "$out"/btc_*.json; }
for st in "$@"; do
  case $st in
    A0) run A0_recon bash tools/recon.sh ;;
    A1) [ -d .venv ] || run A1_install bash install.sh; . .venv/bin/activate 2>/dev/null && . tools/env.sh
        run A1_import python -c "import qai_appbuilder, os; p=os.path.dirname(qai_appbuilder.__file__); print(p); print(sorted(os.listdir(os.path.join(p,'libs')))[:30])" ;;
    A2) run A2_golden python tools/golden.py check --models "$M" --backend npu ;;
    A3) for p in BURST HIGH_PERFORMANCE NONE; do run "A3_perf_$p" env S2S_PERF=$p python tools/golden.py check --models "$M" --backend npu; done
        grep -hE "^(PASS|FAIL)" "$D"/A3_perf_*.log | sed 's/\[.*\]//' > "$D/A3_summary.txt"; cat "$D/A3_summary.txt" ;;
    A4) for g in mt_decB tts_zh; do run "A4_soak_$g" python tools/thermal_soak.py --models "$M" --graph $g --seconds "$SOAK_S"; done ;;
    A5) need_data && run A5_smoke btc quick --n 2 ;;
    AP) need_data && run AP_parity bash -c '
          mkdir -p "'"$D"'/parity"
          for d in vi-en en-vi vi-ko ko-vi vi-zh zh-vi; do
            python tools/eval_btc.py --data "'"$DATA"'" --models "'"$M"'" --backend npu --dirs $d --n 30 --out "'"$D"'/parity/btc_$d.json"; done
          for ref in eval/cpu_ref/btc_cpu30*.json; do for f in "'"$D"'"/parity/btc_*.json; do
            for m in src_err mt_e2e heard; do python tools/compare_runs.py "$ref" "$f" --metric $m --n 500; done; done; done' ;;
    AL) run AL_live_vi_en python tools/live_selftest.py --wav samples/vi_0.wav --wav samples/vi_0.wav -- --src vi --tgt en --models "$M" --backend npu --continuous
        run AL_live_en_vi python tools/live_selftest.py --wav samples/en_0.wav -- --src en --tgt vi --models "$M" --backend npu --continuous ;;
    A6) need_data && run A6_quick btc quick ;;
    A7) need_data && run A7_dev btc dev ;;
    A8) need_data && run A8_holdout btc holdout ;;
    A9) need_data && for nd in ${NOISY:-}; do export BTC_DATA=$nd; run "A9_noisy_$(basename "$nd")" btc quick; unset BTC_DATA; done ;;
    *) echo "unknown stage $st" ;;
  esac
done
cp "$SPLIT" "$D/" 2>/dev/null; cp logs/golden_npu_*.json "$D/" 2>/dev/null
tar czf "$D.tgz" "$D" && echo "packed $D.tgz -> copy it back (scp root@<box>:$(pwd)/$D.tgz .)"
