#!/usr/bin/env bash
# Fallback for hosts whose own Python/glibc cannot take the qai_appbuilder wheel (e.g. Ubuntu 20.04: glibc 2.31, Python 3.8).
# Runs the package inside a Debian 12 container (glibc 2.36, Python 3.10) with the host NPU passed through:
#   - FastRPC device nodes (/dev/fastrpc-*, /dev/adsprpc-smd) and the ION/DMA heaps,
#   - the host's libcdsprpc.so (the user-space FastRPC driver belongs to the host image, not to the container).
# Offline preparation on a laptop with Docker:
#   docker pull --platform linux/arm64 python:3.10-slim-bookworm && docker save -o py310_arm64.tar python:3.10-slim-bookworm
# On the box:  docker load -i py310_arm64.tar ; bash tools/docker_run.sh bash tools/session0.sh
set -eu
cd "$(dirname "$0")/.."
IMG=${IMG:-python:3.10-slim-bookworm}
command -v docker >/dev/null || { echo "docker not installed on this image"; exit 2; }
DEV=""
for d in /dev/fastrpc-cdsp /dev/fastrpc-cdsp-secure /dev/fastrpc-adsp /dev/adsprpc-smd /dev/ion /dev/dma_heap/system /dev/dma_heap/qcom,system; do [ -e "$d" ] && DEV="$DEV --device $d"; done
[ -n "$DEV" ] || { echo "no FastRPC device node found: the NPU is not reachable from user space on this image"; exit 2; }
LIBS=""
for f in $(ls /usr/lib/libcdsprpc.so* /usr/lib/aarch64-linux-gnu/libcdsprpc.so* /vendor/lib64/libcdsprpc.so 2>/dev/null); do LIBS="$LIBS -v $f:/usr/lib/aarch64-linux-gnu/$(basename "$f"):ro"; done
[ -d /usr/lib/rfsa ] && LIBS="$LIBS -v /usr/lib/rfsa:/usr/lib/rfsa:ro"
[ -d /dsp ] && LIBS="$LIBS -v /dsp:/dsp:ro"
SND=""; [ -d /dev/snd ] && SND="--device /dev/snd"
echo "devices:$DEV $SND"; echo "libs:$LIBS"
# first run creates .venv inside the mounted package (Python 3.10 wheels), later runs reuse it
exec docker run --rm -it --network none $DEV $SND $LIBS -v "$PWD":/pkg -w /pkg "$IMG" \
  bash -c '[ -d .venv ] || bash install.sh; . .venv/bin/activate; "$@"' _ "${@:-bash}"
