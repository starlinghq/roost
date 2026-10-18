#!/usr/bin/env bash
# Serve harness for S1: start `s1app serve` under one gpu-lease, check /health and /sse,
# send 200 more SSE requests, time a 64-event stream, then stop the server.
# Writes server stderr (RSS checkpoints) and all curl output to results/serve-*.txt.
# Usage: ./serve-harness.sh   (run from anywhere; needs publish/s1app and publish/libs1kernels.so)
set -euo pipefail
cd "$(dirname "$0")"
URL=${S1_URL:-http://127.0.0.1:5199}
mkdir -p results
ERR=results/serve-stderr.txt
OUT=results/serve-curl.txt
: > "$ERR"; : > "$OUT"
export S1_URL="$URL"

../../tools/gpu-lease run --device 3090 --need-mib 500 --wait 3600 --stream F -- bash -c '
  publish/s1app serve 2>>"$0" &
  pid=$!
  trap "kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true" EXIT
  for _ in $(seq 1 100); do curl -sf "$1/health" >/dev/null 2>&1 && break; sleep 0.1; done
  {
    echo "# $(date -u +%FT%TZ) serve harness, server pid $pid"
    echo "## GET /health"; curl -si "$1/health"; echo
    echo "## GET /sse?n=8"; curl -siN "$1/sse?n=8"; echo
    echo "## 200 x GET /sse?n=8 (body discarded; status codes counted)"
    for _ in $(seq 1 200); do curl -s -o /dev/null -w "%{http_code}\n" "$1/sse?n=8"; done | sort | uniq -c
    echo "## GET /sse?n=64 timing (loopback)"
    curl -s -o /dev/null -w "time_starttransfer=%{time_starttransfer}s time_total=%{time_total}s http=%{http_code}\n" "$1/sse?n=64"
    echo "## GET /rss"; curl -s "$1/rss"
  } >>"$2" 2>&1
' "$ERR" "$URL" "$OUT"
