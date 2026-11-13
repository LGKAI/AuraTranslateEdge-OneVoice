#!/usr/bin/env bash
# One command for the whole device session on the AIBOX 8550: survey -> install -> golden check -> smoke -> sample set -> noise set -> pack logs.
# No step aborts the others; everything is logged under logs/vng_<date>/ and packed as logs/vng_<date>.tgz (adb pull it afterwards).
# Usage (package root):  bash tools/vng_run_all.sh /root/onevoice [n_per_direction=30]
#   /root/onevoice must contain onevoice_btc_data/ (sample set) and, optionally, onevoice_btc_adv/ (noise set).
#   n_per_direction: 30 = quick (about 70 min on the box, shift 1), 0 = every clip (shift 2).
# Rules of the shared box: nothing outside the package folder is modified (no /usr/lib, no network config, no passwords).
set -u
cd "$(dirname "$0")/.."
ROOT=${1:-$(cd .. && pwd)}; N=${2:-30}
D=logs/vng_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/steps.log"; }
[ -f VERSION.txt ] && { cat VERSION.txt | tee -a "$D/steps.log"; }
if [ -f ../SHA256SUMS ]; then say "checksum"; ( sha256sum -c ../SHA256SUMS --quiet && echo CHECKSUM_OK ) 2>&1 | tail -3 | tee -a "$D/steps.log"; fi
say "session0 (recon + install + golden check + smoke + bench)"; bash tools/session0.sh > "$D/session0.log" 2>&1; tail -5 "$D/session0.log"
M=$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json); say "model set: $M"
. .venv/bin/activate 2>/dev/null && . tools/env.sh 2>/dev/null
if [ -d "$ROOT/onevoice_btc_data" ]; then say "sample set, $N clips per direction"; bash tools/run_btc.sh "$ROOT/onevoice_btc_data" "$N" "$M" > "$D/run_btc.log" 2>&1; tail -8 "$D/run_btc.log"; else say "no $ROOT/onevoice_btc_data: sample set skipped"; fi
if [ -d "$ROOT/onevoice_btc_adv" ]; then say "noise set (25 sentences x 4 noise levels x 6 directions)"; bash tools/run_adv.sh "$ROOT/onevoice_btc_adv" 0 "$M" > "$D/run_adv.log" 2>&1; tail -20 "$D/run_adv.log"; else say "no $ROOT/onevoice_btc_adv: noise set skipped"; fi
cp -r logs/btc_*.tgz logs/adv_*.tgz logs/session0_*.tgz "$D/" 2>/dev/null
tar czf "$D.tgz" "$D" && say "packed $D.tgz  -> adb pull $(pwd)/$D.tgz"
