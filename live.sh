#!/usr/bin/env bash
# Live viewer: runs the chosen technique in real time and graphs CPU/RAM/GPU.  Ctrl-C to stop.
cd "$(dirname "$0")"; source env/bin/activate
python live_server.py
