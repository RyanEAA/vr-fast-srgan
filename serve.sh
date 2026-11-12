#!/usr/bin/env bash
# Recorded viewer (pre-rendered videos from ./run_benchmarks.sh, with split view): http://localhost:8765/viewer.html
cd "$(dirname "$0")"
echo "Open http://localhost:8765/viewer.html  (Ctrl-C to stop)"
python3 -m http.server 8765 --bind 127.0.0.1
