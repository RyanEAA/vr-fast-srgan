#!/usr/bin/env bash
# Downloads extra sample 360 videos into clips/ (keeps only ones that download and are ~2:1).
cd "$(dirname "$0")"; mkdir -p clips; source env/bin/activate
fetch() {  # name url
  [ -s "clips/$1.mp4" ] && { echo "have $1"; return; }
  if curl -fsSL --max-time 300 -o "clips/$1.mp4.part" "$2"; then mv "clips/$1.mp4.part" "clips/$1.mp4"; else
    echo "FAILED $1 ($2)"; rm -f "clips/$1.mp4.part"; return; fi
  python - "$1" <<'PY'
import os, sys, cv2
p = f"clips/{sys.argv[1]}.mp4"; c = cv2.VideoCapture(p); w, h = c.get(3), c.get(4)
print(f"got {sys.argv[1]}: {int(w)}x{int(h)}")
if not h or abs(w / h - 2) > 0.15: print("  not 2:1 equirectangular, removing"); os.remove(p)
PY
}
# webvr.info sample (Brandon Jones), 1920x960
fetch bench_man "https://raw.githubusercontent.com/toji/webvr.info/master/samples/media/textures/vr_bench_man.mp4"
