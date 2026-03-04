import argparse
import json
from pathlib import Path
from collections import Counter

def extract_text(text_field) -> str:
    """
    NYT clue 'text' often looks like: [{"plain":"..."}] or [{"formatted":"...","plain":"..."}]
    """
    if isinstance(text_field, str):
        return text_field.strip()
    if isinstance(text_field, dict):
        return (text_field.get("plain") or text_field.get("formatted") or "").strip()
    if isinstance(text_field, list):
        parts = []
        for t in text_field:
            if isinstance(t, dict):
                parts.append(t.get("plain") or t.get("formatted") or "")
            elif isinstance(t, str):
                parts.append(t)
        return "".join(parts).strip()
    return ""

def load_puzzle_from_raw(raw_path: Path):
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    body = raw.get("body")
    if not isinstance(body, list) or not body:
        return None, "missing_body"

    p = body[0]
    if not isinstance(p, dict):
        return None, "bad_body0"

    dims = p.get("dimensions") or {}
    w = dims.get("width")
    h = dims.get("height")
    if not isinstance(w, int) or not isinstance(h, int):
        return None, "missing_dimensions"

    cells = p.get("cells")
    clues = p.get("clues")
    if not isinstance(cells, list) or not isinstance(clues, list):
        return None, "missing_cells_or_clues"

    # Skip ONLY true rebus / weird cells: multi-letter answers or non-alpha
    for c in cells:
        if not c:  # {} => black square
            continue
        ans = c.get("answer")
        if not isinstance(ans, str) or len(ans) != 1 or not ans.isalpha():
            return None, "rebus_or_nonalpha_cell"

    entries = []
    for entry_id, cl in enumerate(clues):
        if not isinstance(cl, dict):
            return None, "bad_clue_obj"

        cell_idxs = cl.get("cells")
        if not isinstance(cell_idxs, list) or not cell_idxs:
            return None, "missing_clue_cells"

        direction = (cl.get("direction") or "").strip()
        if direction.lower().startswith("a"):
            dir_short = "A"
        elif direction.lower().startswith("d"):
            dir_short = "D"
        else:
            return None, "unknown_direction"

        label = str(cl.get("label") or "")
        clue_text = extract_text(cl.get("text"))

        start = cell_idxs[0]
        row = start // w
        col = start % w
        length = len(cell_idxs)

        answer = "".join(cells[i].get("answer", "") for i in cell_idxs if cells[i])
        if len(answer) != length:
            return None, "answer_length_mismatch"

        entries.append({
            "entry_id": entry_id,
            "label": label,
            "direction": dir_short,   # "A" or "D"
            "row": row,
            "col": col,
            "length": length,
            "cells": cell_idxs,       # useful for constraints / rendering
            "clue": clue_text,
            "answer": answer.upper(),
        })

    puzzle = {
        "nyt_id": raw.get("id"),
        "date": raw.get("publicationDate") or raw_path.stem,
        "width": w,
        "height": h,
        "entries": entries,
    }
    return puzzle, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="NYT_raw_cw")
    ap.add_argument("--out", default="NYT_cw.json")
    ap.add_argument("--max_puzzles", type=int, default=100)
    ap.add_argument("--debug_skips", type=int, default=5)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.json"))

    dataset = []
    reasons = Counter()
    shown = 0

    for f in files:
        if len(dataset) >= args.max_puzzles:
            break
        try:
            puzzle, err = load_puzzle_from_raw(f)
        except Exception:
            puzzle, err = None, "exception_parse"

        if err is not None:
            reasons[err] += 1
            if shown < args.debug_skips:
                print(f"[SKIP] {f.name}: {err}")
                shown += 1
            continue

        puzzle["puzzle_id"] = len(dataset)  # start from 0, contiguous
        dataset.append(puzzle)

    out_obj = {
        "meta": {
            "source": "NYT",
            "count": len(dataset),
        },
        "puzzles": dataset,
        "skipped": dict(reasons),
    }

    Path(args.out).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} puzzles to {args.out} (skipped {sum(reasons.values())}).")
    if reasons:
        print("Skip reasons:", dict(reasons))

if __name__ == "__main__":
    main()
