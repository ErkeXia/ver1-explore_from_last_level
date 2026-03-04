#!/usr/bin/env bash
set -euo pipefail

COOKIE_FILE="${1:-NYT_cookies.txt}"
OUT_DIR="NYT_raw_cw"
TARGET=100
START_DATE="2025-01-01"

mkdir -p "$OUT_DIR"

count=0
offset=0

echo "Using cookies: $COOKIE_FILE"
echo "Saving raw puzzles to: $OUT_DIR"
echo "Starting from: $START_DATE"
echo

while [ "$count" -lt "$TARGET" ]; do
  date=$(python - <<PY
from datetime import datetime, timedelta
start = datetime.fromisoformat("$START_DATE")
print((start + timedelta(days=$offset)).date().isoformat())
PY
)

  out="$OUT_DIR/$date.json"
  tmp="$OUT_DIR/$date.json.tmp"
  url="https://www.nytimes.com/svc/crosswords/v6/puzzle/daily/$date.json"

  # Skip if already downloaded
  if [ -f "$out" ]; then
    offset=$((offset+1))
    continue
  fi

  http_code=$(curl -sS -L \
    -H "Accept: application/json" \
    -b "$COOKIE_FILE" \
    -o "$tmp" \
    -w "%{http_code}" \
    "$url" || echo "000")

  if [ "$http_code" = "200" ]; then
    # Validate JSON
    if python -c "import json; json.load(open('$tmp','r',encoding='utf-8'))" >/dev/null 2>&1; then
      mv "$tmp" "$out"
      count=$((count+1))
      echo "[$count/$TARGET] saved $out"
    else
      rm -f "$tmp"
      echo "[skip] $date invalid JSON (got HTTP 200 but not JSON)"
    fi
  else
    rm -f "$tmp"
    echo "[skip] $date HTTP $http_code"
  fi

  offset=$((offset+1))
  sleep 1
done

echo
echo "Done. Downloaded $count puzzles into $OUT_DIR/"
