#!/usr/bin/env sh
# Start the app and expose it on a public HTTPS URL.
#
# Works the same inside a Codespace and on a laptop. Inside a Codespace it is the better
# option, because Codespaces' own port forwarding is not usable as a link you hand to
# someone else:
#
#   - port visibility silently reverts to private
#   - non-browser clients get a flat 401 no matter what the visibility says
#   - anonymous visitors hit an "about to access a development port" interstitial that
#     GitHub gives no way to switch off
#
# A Cloudflare quick tunnel has none of that. No account, no card, no sign-in wall, and
# the URL serves the app directly to anyone.
#
#   sh serve_public.sh
#
# Ctrl-C stops both processes.

set -eu

PORT="${PORT:-7860}"
APP_DIR="/app"
[ -d "$APP_DIR/index_store" ] || APP_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$APP_DIR/index_store/vectors.npy" ]; then
  echo "No index at $APP_DIR/index_store."
  echo "Build one first:  python -m retrieval.build_index --limit 1500 --out index_store"
  exit 1
fi

if [ -z "${SARVAM_API_KEY:-}" ]; then
  echo "WARNING: SARVAM_API_KEY is not set."
  echo "  Retrieval and the guardrails still work. Speech-to-text returns a typed error"
  echo "  rather than a fake transcript, and answers drop to labelled extractive mode."
  echo ""
fi

cd "$APP_DIR"

echo "Starting the app on :$PORT ..."
python -m uvicorn demo.app:app --host 127.0.0.1 --port "$PORT" > /tmp/app.log 2>&1 &
APP_PID=$!

# Poll rather than sleeping a fixed amount: loading 15k vectors and the ONNX session takes
# a few seconds on a fast machine and considerably longer on a cold 2-core Codespace.
echo "Waiting for it to answer ..."
i=0
while [ "$i" -lt 60 ]; do
  if python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=3).status==200 else 1)" 2>/dev/null; then
    break
  fi
  i=$((i + 1))
  sleep 2
done

if [ "$i" -ge 60 ]; then
  echo "The app never became healthy. Last 30 lines of /tmp/app.log:"
  tail -30 /tmp/app.log
  kill "$APP_PID" 2>/dev/null || true
  exit 1
fi

python -c "import urllib.request as u; print('  health:', u.urlopen('http://127.0.0.1:$PORT/health',timeout=5).read().decode()[:150])"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo ""
  echo "cloudflared is not installed, so the app is only reachable locally on :$PORT."
  echo "Install it, or rebuild the container — the Dockerfile now includes it."
  wait "$APP_PID"
fi

echo ""
echo "Opening a public tunnel ..."
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > /tmp/tunnel.log 2>&1 &
TUNNEL_PID=$!

trap 'kill "$APP_PID" "$TUNNEL_PID" 2>/dev/null || true' INT TERM EXIT

URL=""
i=0
while [ "$i" -lt 45 ]; do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/tunnel.log 2>/dev/null | head -1 || true)
  [ -n "$URL" ] && break
  i=$((i + 1))
  sleep 2
done

if [ -z "$URL" ]; then
  echo "The tunnel did not report a URL. Last 20 lines of /tmp/tunnel.log:"
  tail -20 /tmp/tunnel.log
  exit 1
fi

echo ""
echo "======================================================================"
echo "  LIVE:  $URL"
echo "======================================================================"
echo ""
echo "  Real HTTPS, so the microphone works. No sign-in, no interstitial."
echo "  Check it:  $URL/health"
echo ""
echo "  This URL lives as long as this process does. Leave it running, and"
echo "  re-check the link shortly before you submit it anywhere."
echo ""

wait "$APP_PID"
