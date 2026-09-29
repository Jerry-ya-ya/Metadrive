#!/bin/sh
set -eu

export DISPLAY="${DISPLAY:-:99}"
display_number="${DISPLAY#:}"
display_number="${display_number%%.*}"
case "$display_number" in
    ''|*[!0-9]*) echo "DISPLAY must use a numeric local screen, such as :99." >&2; exit 1 ;;
esac

lock_path="/tmp/.X${display_number}-lock"
socket_path="/tmp/.X11-unix/X${display_number}"
display_pid=""
if [ -r "$lock_path" ]; then
    display_pid="$(tr -d '[:space:]' < "$lock_path")"
fi

if [ -z "$display_pid" ] || [ ! -r "/proc/${display_pid}/comm" ] || \
   [ "$(cat "/proc/${display_pid}/comm")" != "Xvfb" ] || \
   grep -q '^State:.*Z' "/proc/${display_pid}/status"; then
    rm -f "$lock_path" "$socket_path"
    Xvfb "$DISPLAY" -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
fi

display_ready=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [ -S "$socket_path" ]; then
        display_ready=1
        break
    fi
    sleep 0.2
done
if [ "$display_ready" -ne 1 ]; then
    cat /tmp/xvfb.log >&2
    exit 1
fi

exec uvicorn webapp.backend:app --host 0.0.0.0 --port 4000
