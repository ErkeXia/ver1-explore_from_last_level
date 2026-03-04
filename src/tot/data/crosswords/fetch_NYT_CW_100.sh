#!/usr/bin/env bash
set -euo pipefail

COOKIES="${1:-NYT_cookies.txt}"
OUT_DIR="NYT_raw_cw"
TARGET=100
START="2025-01-01"

mkdir -p "$OUT_DIR"

count=0
offset=0

while [ "$count" -lt "$TARGET" ]; do
  d="$(python - <<PY
from datetime import date, timedelta
y,m,dd = map(int, "$START".split("-"))
print((date(y,m,dd) + timedelta(days=$offset)).isoformat())
PY
)"
  url="https://www.nytimes.com/svc/crosswords/v6/puzzle/daily/$d.json"
  out="$OUT_DIR/$d.json"
  tmp="$out.tmp"

  if [ -f "$out" ]; then
    offset=$((offset+1))
    continue
  fi

  code="$(curl -sS -L \
    -b "$COOKIES" \
    -H 'User-Agent: Mozilla/5.0' \
    -H 'Accept: application/json' \
    -o "$tmp" -w "%{http_code}" \
    "$url" || true)"

  if [ "$code" = "200" ] && python - "$tmp" >/dev/null 2>&1 <<'PY'
import json, sys
j = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert "body" in j and isinstance(j["body"], list) and len(j["body"]) > 0
PY
  then
    mv "$tmp" "$out"
    count=$((count+1))
    echo "[$count/$TARGET] saved $out"
  else
    rm -f "$tmp"
    echo "[skip] $d (HTTP $code or invalid JSON)"
  fi

  offset=$((offset+1))
  sleep 1
done

echo "Done: downloaded $count puzzles into $OUT_DIR/"
