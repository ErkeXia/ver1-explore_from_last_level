import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAW_DIR = Path("NYT_raw_cw")
OUT_FILE = Path("NYT_cw.json")

BLOCK_CHARS = {".", "#"}  # NYT grid commonly uses "." for blocks


def normalize_answer(ans: str) -> str:
    # Keep A–Z only; strip punctuation/spaces/hyphens.
    return re.sub(r"[^A-Z]", "", (ans or "").upper())


def find_puzzle_payload(obj: Any) -> Dict[str, Any]:
    """
    NYT's JSON may contain lots of extra fields. We look for the dict that has:
      size.rows/cols, grid, clues.across/down, answers.across/down
    Returns that dict.
    """
    queue = [obj]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            size = cur.get("size")
            clues = cur.get("clues")
            answers = cur.get("answers")
            grid = cur.get("grid")

            ok = (
                isinstance(size, dict)
                and "rows" in size and "cols" in size
                and isinstance(grid, list)
                and isinstance(clues, dict) and isinstance(answers, dict)
                and isinstance(clues.get("across"), list) and isinstance(clues.get("down"), list)
                and isinstance(answers.get("across"), list) and isinstance(answers.get("down"), list)
            )
            if ok:
                return cur

            # BFS
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    queue.append(v)

        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    queue.append(v)

    raise KeyError("Could not locate puzzle payload with size/grid/clues/answers.")


def extract_size(p: Dict[str, Any]) -> Tuple[int, int]:
    return int(p["size"]["rows"]), int(p["size"]["cols"])


def extract_grid(p: Dict[str, Any], rows: int, cols: int) -> List[str]:
    grid = p["grid"]
    if len(grid) != rows * cols:
        raise ValueError(f"grid length {len(grid)} != rows*cols {rows*cols}")
    return [str(c) for c in grid]


def has_rebus(p: Dict[str, Any]) -> bool:
    # Some NYT payloads include rebus info in fields like "rebuses".
    # Treat any non-empty rebus map as "has rebus".
    reb = p.get("rebuses") or p.get("rebus")
    if isinstance(reb, dict) and len(reb) > 0:
        return True
    if isinstance(reb, list) and len(reb) > 0:
        return True
    return False


def scan_slots(rows: int, cols: int, grid: List[str]) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    def is_block(r: int, c: int) -> bool:
        return grid[r * cols + c] in BLOCK_CHARS

    across: List[Tuple[int, int, int]] = []
    down: List[Tuple[int, int, int]] = []

    # Across slots in standard clue order
    for r in range(rows):
        c = 0
        while c < cols:
            if is_block(r, c):
                c += 1
                continue
            left_block = (c == 0) or is_block(r, c - 1)
            if left_block:
                cc = c
                L = 0
                while cc < cols and not is_block(r, cc):
                    L += 1
                    cc += 1
                if L >= 2:
                    across.append((r, c, L))
                c = cc
            else:
                c += 1

    # Down slots in standard clue order
    for r in range(rows):
        for c in range(cols):
            if is_block(r, c):
                continue
            above_block = (r == 0) or is_block(r - 1, c)
            if above_block:
                rr = r
                L = 0
                while rr < rows and not is_block(rr, c):
                    L += 1
                    rr += 1
                if L >= 2:
                    down.append((r, c, L))

    return across, down


def convert_one(raw_json_path: Path, puzzle_id: int) -> Optional[Dict[str, Any]]:
    raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    p = find_puzzle_payload(raw)

    if has_rebus(p):
        return None  # skip rebus puzzles

    rows, cols = extract_size(p)
    grid = extract_grid(p, rows, cols)

    clues_a = p["clues"]["across"]
    clues_d = p["clues"]["down"]
    ans_a = p["answers"]["across"]
    ans_d = p["answers"]["down"]

    across_slots, down_slots = scan_slots(rows, cols, grid)

    # Must align perfectly; otherwise skip (format mismatch or special puzzle)
    if not (len(across_slots) == len(clues_a) == len(ans_a)):
        return None
    if not (len(down_slots) == len(clues_d) == len(ans_d)):
        return None

    entries: List[Dict[str, Any]] = []
    eid = 0

    # Across entries
    for (r, c, L), clue, ans in zip(across_slots, clues_a, ans_a):
        a = normalize_answer(ans)
        if len(a) != L:
            return None  # length mismatch => likely rebus-ish; skip
        entries.append({
            "entry_id": eid,
            "direction": "across",
            "row": r,
            "col": c,
            "length": L,
            "clue": clue,
            "answer": a,
        })
        eid += 1

    # Down entries
    for (r, c, L), clue, ans in zip(down_slots, clues_d, ans_d):
        a = normalize_answer(ans)
        if len(a) != L:
            return None
        entries.append({
            "entry_id": eid,
            "direction": "down",
            "row": r,
            "col": c,
            "length": L,
            "clue": clue,
            "answer": a,
        })
        eid += 1

    return {
        "puzzle_id": puzzle_id,
        "rows": rows,
        "cols": cols,
        "entries": entries,
    }


def main():
    files = sorted(RAW_DIR.glob("*.json"))
    out: List[Dict[str, Any]] = []
    pid = 0
    skipped = 0

    for fp in files:
        try:
            obj = convert_one(fp, pid)
            if obj is None:
                skipped += 1
                continue
            out.append(obj)
            pid += 1
        except Exception:
            skipped += 1
            continue

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} puzzles to {OUT_FILE} (skipped {skipped})")


if __name__ == "__main__":
    main()
