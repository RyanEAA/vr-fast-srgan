#!/usr/bin/env bash
# One-time setup: Python env, Fast-SRGAN (pinned commit), ffmpeg, and the test clip. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
FAST_SRGAN_COMMIT=c23bf5fa3faf4ab95e7d295a4c69349702ab2510
PY=${PYTHON:-python3}

[ -d Fast-SRGAN ] || git clone https://github.com/HasnainRaz/Fast-SRGAN.git
git -C Fast-SRGAN checkout -q "$FAST_SRGAN_COMMIT"

[ -d env ] || "$PY" -m venv env
source env/bin/activate
python -m pip install -q -U pip
python -m pip install -r requirements.txt

# use the system ffmpeg if there is one, otherwise the binary that imageio-ffmpeg ships
if ! command -v ffmpeg >/dev/null; then
  ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" env/bin/ffmpeg
fi
ffmpeg -version | head -1
python -c "import torch, bench; print('torch', torch.__version__, '| upscaler device:', bench.pick_device())"

mkdir -p clips
[ -s clips/pano.mp4 ] || curl -fL -o clips/pano.mp4 \
  https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/pano.mp4
echo "Setup done. Next: ./run_benchmarks.sh   then   ./live.sh"
