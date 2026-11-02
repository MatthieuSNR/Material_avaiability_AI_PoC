#!/bin/bash
# Share the dashboard from this Mac through two tunnels (prototype only): ngrok and a Cloudflare quick tunnel
# (for networks that block ngrok, e.g. Roche). Both lead to the same app, behind the same password.
# The data stays on this Mac; the app is reachable only while this script runs.
# Requires APP_PASSWORD in .env (otherwise anyone with the link sees the data).
# Optional: NGROK_DOMAIN in .env (the free static domain of the ngrok account, e.g. name.ngrok-free.app):
# the link stays the same, so a visitor who clicked 'Visit Site' once does not see the ngrok page again for 7 days.
# The Cloudflare link (https://….trycloudflare.com) changes at each start; it is written to outputs/cloudflare_link.txt
# and copied to the clipboard (the ngrok screen hides the terminal). Without cloudflared, ngrok only.
# Usage (from the project root):  ./share.sh      Stop: Ctrl+C

cd "$(dirname "$0")"
if ! grep -qE '^APP_PASSWORD=.+' .env; then
    echo "Set APP_PASSWORD in .env before sharing the app."; exit 1
fi

venv/bin/streamlit run scripts/app.py --server.headless true --server.port 8505 > /dev/null 2>&1 &
STREAMLIT_PID=$!
trap 'kill $STREAMLIT_PID $CLOUDFLARE_PID 2>/dev/null' EXIT

if command -v cloudflared > /dev/null; then
    mkdir -p outputs
    CLOUDFLARE_LOG=outputs/cloudflare.log
    cloudflared tunnel --no-autoupdate --url http://localhost:8505 > "$CLOUDFLARE_LOG" 2>&1 &
    CLOUDFLARE_PID=$!
    for _ in $(seq 30); do   # the link appears in the log after a few seconds
        CLOUDFLARE_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CLOUDFLARE_LOG" | head -1)
        [ -n "$CLOUDFLARE_URL" ] && break
        sleep 1
    done
    if [ -n "$CLOUDFLARE_URL" ]; then
        echo "$CLOUDFLARE_URL" > outputs/cloudflare_link.txt
        printf %s "$CLOUDFLARE_URL" | pbcopy
        echo "Cloudflare link (copied to the clipboard): $CLOUDFLARE_URL"
    else
        echo "Cloudflare tunnel not started (see $CLOUDFLARE_LOG); ngrok only."
    fi
    sleep 3
else
    echo "cloudflared not installed (brew install cloudflared); ngrok only."
fi

NGROK_DOMAIN=$(grep -E '^NGROK_DOMAIN=' .env | cut -d= -f2-)
caffeinate -i ngrok http ${NGROK_DOMAIN:+--url=$NGROK_DOMAIN} 8505   # caffeinate: the Mac does not go to sleep while sharing
