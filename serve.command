#!/bin/bash
# macOS double-click wrapper for serve.sh
cd "$(dirname "$0")"
./serve.sh
echo; read -n 1 -s -r -p "Done. Press any key to close."
