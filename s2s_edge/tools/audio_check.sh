#!/usr/bin/env bash
# Microphone / speaker check on the box (the KIMQ8550 carrier has no built-in mic or speaker: use a USB audio device or headset).
#   bash tools/audio_check.sh                 -> list devices
#   bash tools/audio_check.sh hw:1,0 hw:1,0   -> record 5 s from the capture device, report its level, play it back on the playback device
#   bash tools/audio_check.sh hw:1,0 - 120    -> record 120 s of room ambience to logs/office_ambience.wav (input for tools/mix_noise.py --kind file)
set -u
cd "$(dirname "$0")/.."; mkdir -p logs
if [ $# -eq 0 ]; then echo "== capture"; arecord -l; echo "== playback"; aplay -l; echo "pick hw:<card>,<device> and rerun with it"; exit 0; fi
CAP=$1; PLAY=${2:--}; SEC=${3:-5}
OUT=logs/mic_test.wav; [ "$SEC" -gt 30 ] && OUT=logs/office_ambience.wav
echo "recording $SEC s from $CAP -> $OUT (speak now unless recording ambience)"
arecord -D "plughw:${CAP#hw:}" -f S16_LE -r 16000 -c 1 -d "$SEC" "$OUT" || exit 1
python3 - "$OUT" <<'EOF'
import sys, wave, numpy as np
w = wave.open(sys.argv[1]); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9); peak = 20 * np.log10(np.abs(x).max() + 1e-9)
print(f"{len(x)/16000:.1f} s, RMS {rms:.1f} dBFS, peak {peak:.1f} dBFS, clipped samples {(np.abs(x) > 0.99).sum()}")
print("hint: speech should peak around -12 to -3 dBFS; below -30 dBFS raise the mic gain (alsamixer), clipping -> lower it")
EOF
[ "$PLAY" != "-" ] && { echo "playing back on $PLAY"; aplay -D "plughw:${PLAY#hw:}" "$OUT"; }
