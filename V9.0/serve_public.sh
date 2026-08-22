#!/usr/bin/env bash
# Start the app and expose it on a public HTTPS URL, in one command.
#
#   ./serve_public.sh
#
# Downloads cloudflared on first run (~55MB). No account, no card, no signup.
# HTTPS matters beyond looking tidy: browsers only allow getUserMedia in a secure
# context, so the microphone silently does nothing over plain http from another device.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8600}"
PY="${PY:-../.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"

CF="./.cloudflared.exe"
if [ ! -x "$CF" ]; then
  echo "Fetching cloudflared (one time)..."
  curl -sL -o "$CF" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
  chmod +x "$CF"
fi

echo "Starting the app on :$PORT ..."
PYTHONIOENCODING=utf-8 "$PY" -m uvicorn demo.app:app --host 127.0.0.1 --port "$PORT" \
  > app.log 2>&1 &
APP_PID=$!

# Loading the index and the ONNX session takes a few seconds; the tunnel would otherwise
# come up pointing at a closed port and hand out a URL that 502s.
echo -n "Waiting for the pipeline to load"
for _ in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  echo -n "."; sleep 2
done
echo

if ! curl -s -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
  echo "The app did not come up. Last lines of app.log:"; tail -20 app.log; kill "$APP_PID" 2>/dev/null || true; exit 1
fi

echo "Opening the tunnel ..."
"$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > tunnel.log 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 1 40); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" tunnel.log 2>/dev/null | head -1 || true)
  [ -n "$URL" ] && break
  sleep 2
done

if [ -z "$URL" ]; then
  echo "The tunnel did not report a URL. Last lines of tunnel.log:"; tail -20 tunnel.log
  kill "$APP_PID" "$TUNNEL_PID" 2>/dev/null || true; exit 1
fi

cat <<EOF

  ────────────────────────────────────────────────────────────
   LIVE:  $URL
  ────────────────────────────────────────────────────────────

  The URL is new every run, so re-check it right before you paste it anywhere.
  Keep this window open — closing it stops both the app and the tunnel.
  Set Windows to never sleep, or the link dies when the screen does.

  Ctrl-C to stop.

EOF

trap 'echo; echo "Stopping..."; kill "$APP_PID" "$TUNNEL_PID" 2>/dev/null || true' INT TERM
wait "$TUNNEL_PID"
