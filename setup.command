#!/bin/bash
# macOS double-click wrapper for setup.sh
cd "$(dirname "$0")"
./setup.sh
echo; read -n 1 -s -r -p "Done. Press any key to close."
