#!/bin/bash
# macOS double-click wrapper for run_benchmarks.sh
cd "$(dirname "$0")"
./run_benchmarks.sh
echo; read -n 1 -s -r -p "Done. Press any key to close."
