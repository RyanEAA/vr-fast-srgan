#!/bin/bash
# macOS double-click wrapper for live.sh
cd "$(dirname "$0")"
./live.sh
echo; read -n 1 -s -r -p "Done. Press any key to close."
