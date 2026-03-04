#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="NYT_raw_cw"
COOKIES="NYT_cookies.txt"
BASE="https://www.nytimes.com/svc/crosswords/v6/puzzle/daily"

mkdir -p "$RAW_DIR"

count=0

# Generate all dates in 2025 (portable: use python for date arithmetic)
python - <<'PY' > "$RAW_DIR/dates_2025.txt"
from datetime import date, timedelta
d = date(2025, 1, 1)
end = date(2026, 1, 1)
while d < end:
    print(d.isoformat())
    d += timedelta(days=1)
PY

while read -r d; do
  out="$RAW_DIR/$d.json"
  [ -f "$out" ] && continue

  tmp="$out.tmp"
  url="$BASE/$d.json"

  code="$(curl -sS -L \
    -b "$COOKIES" \
    -H 'User-Agent: Mozilla/5.0' \
    -H 'Accept: application/json' \
    -o "$tmp" -w "%{http_code}" \
    "$url" || true)"

  # Require HTTP 200 and valid JSON containing 'body'
  if [ "$code" = "200" ] && python - "$tmp" >/dev/null 2>&1 <<'PY'
import json, sys
j = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert "body" in j and isinstance(j["body"], list) and len(j["body"]) > 0
PY
  then
    mv "$tmp" "$out"
    count=$((count+1))
    echo "Saved $out ($count/100)"
  else
    rm -f "$tmp"
    echo "Skipped $d (HTTP $code or invalid JSON)"
  fi

  [ "$count" -ge 100 ] && break
done < "$RAW_DIR/dates_2025.txt"

echo "Done. Downloaded $count puzzles into $RAW_DIR/"
