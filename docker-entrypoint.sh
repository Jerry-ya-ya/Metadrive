#!/bin/sh
set -eu

export DISPLAY="${DISPLAY:-:99}"
Xvfb "$DISPLAY" -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &

exec uvicorn webapp.backend:app --host 0.0.0.0 --port 4000
