#!/usr/bin/env bash
# Runs every experiment on the test clip and writes results/RESULTS.md.
# Takes ~5 minutes on an M4 Mac mini. Usage: ./run_benchmarks.sh [video]  (default clips/pano.mp4)
set -euo pipefail
cd "$(dirname "$0")"
source env/bin/activate
V=${1:-clips/pano.mp4}
B="python bench.py --video $V --out out --eval-region 120"

echo "== 1/4 quality + speed of each upscaler"
$B --upscaler none
$B --upscaler bicubic --match-bitrate
$B --upscaler lanczos
$B --upscaler fastsrgan
$B --upscaler fastsrgan --half
$B --upscaler fastsrgan --half --viewport 120

echo "== 2/4 full-res stream of only the visible region"
python native_vp.py "$V"

echo "== 3/4 CPU / RAM / GPU while playing at the video frame rate"
for m in native native_vp120 none bicubic lanczos fastsrgan fastsrgan_vp120; do
  python resources.py "$m"; sleep 2
done

echo "== 4/4 Fast-SRGAN speed vs crop size (1080p / 4K / 8K viewports)"
python crop_speed.py

python make_report.py
echo "Done: see results/RESULTS.md"
