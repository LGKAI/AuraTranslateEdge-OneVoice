#!/usr/bin/env bash
# Run ONCE at the start of a device session (QDC or board): survey, install, check every graph, smoke test, quick benchmark.
# Every step is logged to logs/session0_<date>/ and no step aborts the others; the folder is also packed as logs/session0_<date>.tgz to copy back.
# Model sets are tried in order until every graph passes the golden check; the chosen one is written to logs/models_ok.txt:
#   MODELS env var (if set), models_qcs8550/models.json (AIBOX 8550), models/models.json (IQ-9075 link)
# Usage (from the package root):  bash tools/session0.sh
set -u
cd "$(dirname "$0")/.."
D=logs/session0_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
run() { n=$1; shift; echo "=== $n"; s=$(date +%s); ( "$@" ) > "$D/$n.log" 2>&1; rc=$?; echo "exit $rc ($(( $(date +%s) - s )) s)" | tee -a "$D/$n.log"; tail -4 "$D/$n.log"; return $rc; }
run 00_recon       bash tools/recon.sh
[ -d .venv ] || run 01_install bash install.sh
. .venv/bin/activate 2>/dev/null && . tools/env.sh
run 02_import      python -c "import qai_appbuilder, os; print(qai_appbuilder.__file__); print(os.listdir(os.path.join(os.path.dirname(qai_appbuilder.__file__), 'libs'))[:20])"
M=""
for cand in ${MODELS:-} models_qcs8550/models.json models/models.json; do
  [ -f "$cand" ] || continue
  tag=$(dirname "$cand" | tr '/' '_')
  if run "03_golden_$tag" python tools/golden.py check --models "$cand" --backend npu; then M=$cand; break; fi
done
if [ -z "$M" ]; then M=${MODELS:-models_qcs8550/models.json}; [ -f "$M" ] || M=models/models.json; echo "!! no model set passed every graph; continuing with $M to collect logs"; fi
echo "$M" > logs/models_ok.txt; echo "model set: $M"
run 04_smoke_vi_en python -m s2s_edge file --src vi --tgt en --in samples/vi_0.wav --out "$D/vi_en.wav" --models "$M" --backend npu
run 05_smoke_en_vi python -m s2s_edge file --src en --tgt vi --in samples/en_0.wav --out "$D/en_vi.wav" --models "$M" --backend npu
run 06_bench_vi_en python -m s2s_edge.bench --src vi --tgt en --models "$M" --manifest eval/manifest_vi.json --runs 3 --limit 5 --logs "$D"
cp logs/golden_npu_*.json "$D/" 2>/dev/null
tar czf "$D.tgz" "$D" && echo "packed $D.tgz  -> copy it back"
