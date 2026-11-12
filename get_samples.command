#!/bin/bash
# macOS double-click wrapper for get_samples.sh
cd "$(dirname "$0")"
./get_samples.sh
echo; read -n 1 -s -r -p "Done. Press any key to close."
