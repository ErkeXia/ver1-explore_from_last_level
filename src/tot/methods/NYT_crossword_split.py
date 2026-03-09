import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tot.models import gpt
from tot.prompts.NYT_crossword_split import (
    splitter_repair_prompt,
    splitter_system_prompt,
    splitter_user_prompt,
)


@dataclass
class SubCrosswordRegion:
    region_id: int
    entry_ids: List[int]
    overlap_entry_ids: List[int]
    rationale: str = ""


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _coerce_int_list(items: Any) -> List[int]:
    if not isinstance(items, list):
        return []
    out: List[int] = []
    seen = set()
    for x in items:
        i = _coerce_int(x)
        if i is None or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty splitter response.")

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or l >= r:
        raise ValueError("No JSON object found in splitter response.")

    return json.loads(raw[l : r + 1])


def _build_neighbors_and_degree(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    cell_to_eids: Dict[int, List[int]] = {}
    all_eids: List[int] = []

    for e in entries:
        eid = _coerce_int(e.get("entry_id"))
        if eid is None:
            continue
        all_eids.append(eid)
        for c in e.get("cells", []):
            cell = _coerce_int(c)
            if cell is None:
                continue
            cell_to_eids.setdefault(cell, []).append(eid)

    neighbors_map: Dict[int, set] = {eid: set() for eid in all_eids}
    for refs in cell_to_eids.values():
        uniq = list(dict.fromkeys(refs))
        n = len(uniq)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = uniq[i], uniq[j]
                neighbors_map[a].add(b)
                neighbors_map[b].add(a)

    neighbors = {str(eid): sorted(list(neighbors_map.get(eid, set()))) for eid in all_eids}
    degree = {str(eid): len(neighbors_map.get(eid, set())) for eid in all_eids}
    return neighbors, degree


def _build_intersection_features(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build explicit pairwise intersection metadata for the splitter.
    """
    cell_to_eids: Dict[int, List[int]] = {}
    all_eids: List[int] = []

    for e in entries:
        eid = _coerce_int(e.get("entry_id"))
        if eid is None:
            continue
        all_eids.append(eid)
        for c in e.get("cells", []):
            cell = _coerce_int(c)
            if cell is None:
                continue
            cell_to_eids.setdefault(cell, []).append(eid)

    pair_to_cells: Dict[Tuple[int, int], List[int]] = {}
    for cell, refs in cell_to_eids.items():
        uniq = list(dict.fromkeys(refs))
        n = len(uniq)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = uniq[i], uniq[j]
                if a > b:
                    a, b = b, a
                pair_to_cells.setdefault((a, b), []).append(cell)

    pair_intersections: List[Dict[str, Any]] = []
    entry_map: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in all_eids}

    for (a, b), cells in sorted(pair_to_cells.items()):
        shared_cells = sorted(set(cells))
        rec = {
            "entry_a": a,
            "entry_b": b,
            "shared_cells": shared_cells,
            "shared_count": len(shared_cells),
        }
        pair_intersections.append(rec)

        ab = {
            "other_entry_id": b,
            "shared_cells": shared_cells,
            "shared_count": len(shared_cells),
        }
        ba = {
            "other_entry_id": a,
            "shared_cells": shared_cells,
            "shared_count": len(shared_cells),
        }
        entry_map[a].append(ab)
        entry_map[b].append(ba)

    for eid in entry_map:
        entry_map[eid].sort(key=lambda x: int(x["other_entry_id"]))

    return {
        "pair_intersections": pair_intersections,
        "entry_intersections": {str(eid): entry_map[eid] for eid in all_eids},
    }


def _build_grid_layout(width: int, height: int, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a lightweight grid mask to provide explicit spatial context.
    '.' means fillable crossword cell; '#' means blocked/non-fillable.
    """
    fillable = set()
    for e in entries:
        for c in e.get("cells", []):
            cell = _coerce_int(c)
            if cell is not None:
                fillable.add(cell)

    rows: List[str] = []
    for r in range(height):
        chars = []
        base = r * width
        for c in range(width):
            idx = base + c
            chars.append("." if idx in fillable else "#")
        rows.append("".join(chars))

    return {
        "legend": ".=fillable,#=block",
        "rows": rows,
    }


def build_puzzle_input_from_task(task, idx: int) -> Dict[str, Any]:
    task.env.reset(idx)
    entries: List[Dict[str, Any]] = []

    for spec in task.env.entries:
        entries.append(
            {
                "entry_id": int(spec.entry_id),
                "direction": str(spec.direction),
                "label": str(spec.label),
                "length": int(spec.length),
                "cells": [int(c) for c in spec.cells],
                "clue": str(spec.clue),
            }
        )

    neighbors, degree = _build_neighbors_and_degree(entries)
    intersection_features = _build_intersection_features(entries)
    width = int(task.env.width)
    height = int(task.env.height)
    grid_layout = _build_grid_layout(width, height, entries)
    payload = {
        "width": width,
        "height": height,
        "entries": entries,
        "degree": degree,
        "neighbors": neighbors,
        "intersection_features": intersection_features,
        "grid_layout": grid_layout,
        "constraints": {
            "region_size_target": [8, 10],
            "region_size_hard": [5, 15],
            "overlap_preference": "Encourage overlap between related regions when useful, but it is not mandatory per region.",
        },
    }
    return payload


def _compute_coverage(
    regions: List[SubCrosswordRegion], all_entry_ids: List[int]
) -> Dict[str, Any]:
    counts: Dict[int, int] = {eid: 0 for eid in all_entry_ids}
    for region in regions:
        for eid in region.entry_ids:
            if eid in counts:
                counts[eid] += 1

    uncovered = [eid for eid in all_entry_ids if counts[eid] == 0]
    return {
        "all_entries_covered": len(uncovered) == 0,
        "uncovered_entry_ids": uncovered,
        "entry_appearance_counts": {str(eid): counts[eid] for eid in all_entry_ids},
    }


def _validate_region_plan(
    plan: Dict[str, Any], puzzle: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    critical_issues: List[str] = []
    secondary_issues: List[str] = []
    entries = puzzle.get("entries", [])
    all_entry_ids = sorted(_coerce_int(e.get("entry_id")) for e in entries if _coerce_int(e.get("entry_id")) is not None)
    all_id_set = set(all_entry_ids)
    eid_to_dir: Dict[int, str] = {}
    for e in entries:
        eid = _coerce_int(e.get("entry_id"))
        if eid is None:
            continue
        d = str(e.get("direction", "")).strip().upper()
        if d.startswith("A"):
            eid_to_dir[eid] = "A"
        elif d.startswith("D"):
            eid_to_dir[eid] = "D"

    neighbors_raw = puzzle.get("neighbors", {})
    neighbors_map: Dict[int, set] = {}
    if isinstance(neighbors_raw, dict):
        for k, vals in neighbors_raw.items():
            eid = _coerce_int(k)
            if eid is None:
                continue
            nset = set()
            if isinstance(vals, list):
                for v in vals:
                    nv = _coerce_int(v)
                    if nv is not None:
                        nset.add(nv)
            neighbors_map[eid] = nset
    if not neighbors_map:
        neighbors, _ = _build_neighbors_and_degree(entries)
        for k, vals in neighbors.items():
            eid = _coerce_int(k)
            if eid is None:
                continue
            neighbors_map[eid] = set(vals)

    constraints = puzzle.get("constraints", {})
    hard_min, hard_max = 5, 15
    if isinstance(constraints.get("region_size_hard"), list) and len(constraints["region_size_hard"]) == 2:
        c0 = _coerce_int(constraints["region_size_hard"][0])
        c1 = _coerce_int(constraints["region_size_hard"][1])
        if c0 is not None and c1 is not None and c0 <= c1:
            hard_min, hard_max = c0, c1

    regions_raw = plan.get("regions")
    if not isinstance(regions_raw, list) or len(regions_raw) == 0:
        critical_issues.append("Missing or empty 'regions'.")
        normalized_empty = {
            "regions": [],
            "coverage": _compute_coverage([], all_entry_ids),
            "_critical_issues": critical_issues,
            "_secondary_issues": secondary_issues,
        }
        return False, critical_issues + secondary_issues, normalized_empty

    region_ids_seen = set()
    regions: List[SubCrosswordRegion] = []

    for i, raw_region in enumerate(regions_raw):
        if not isinstance(raw_region, dict):
            secondary_issues.append(f"Region[{i}] is not an object.")
            continue

        rid = _coerce_int(raw_region.get("region_id"))
        if rid is None:
            rid = i
            secondary_issues.append(f"Region[{i}] missing/invalid region_id; auto-assigned {rid}.")
        if rid in region_ids_seen:
            secondary_issues.append(f"Duplicate region_id detected: {rid}.")
        region_ids_seen.add(rid)

        raw_entry_ids = _coerce_int_list(raw_region.get("entry_ids"))
        invalid_entry_ids = [eid for eid in raw_entry_ids if eid not in all_id_set]
        if invalid_entry_ids:
            secondary_issues.append(
                f"Region {rid} contains hallucinated entry_ids not in puzzle: {invalid_entry_ids}."
            )
        entry_ids = [eid for eid in raw_entry_ids if eid in all_id_set]
        if len(entry_ids) == 0:
            secondary_issues.append(f"Region {rid} has empty valid entry_ids.")
            continue
        if len(entry_ids) < hard_min or len(entry_ids) > hard_max:
            secondary_issues.append(
                f"Region {rid} size {len(entry_ids)} out of hard bounds [{hard_min},{hard_max}]."
            )

        # Connectivity (critical): each region must be one connected component.
        entry_set = set(entry_ids)
        unvisited = set(entry_set)
        comp_sizes: List[int] = []
        while unvisited:
            start = next(iter(unvisited))
            stack = [start]
            unvisited.remove(start)
            sz = 0
            while stack:
                cur = stack.pop()
                sz += 1
                for nb in neighbors_map.get(cur, set()):
                    if nb in unvisited and nb in entry_set:
                        unvisited.remove(nb)
                        stack.append(nb)
            comp_sizes.append(sz)
        if len(comp_sizes) > 1:
            critical_issues.append(
                f"Region {rid} disconnected: {len(comp_sizes)} components with sizes {sorted(comp_sizes, reverse=True)}."
            )

        # Across/Down balance rule: |A-D| <= 5.
        a_count = sum(1 for eid in entry_ids if eid_to_dir.get(eid) == "A")
        d_count = sum(1 for eid in entry_ids if eid_to_dir.get(eid) == "D")
        if abs(a_count - d_count) > 5:
            secondary_issues.append(
                f"Region {rid} across/down imbalance too high: A={a_count}, D={d_count}, |A-D|={abs(a_count-d_count)} > 5."
            )

        raw_overlap_ids = _coerce_int_list(raw_region.get("overlap_entry_ids"))
        invalid_overlap_ids = [eid for eid in raw_overlap_ids if eid not in all_id_set]
        if invalid_overlap_ids:
            secondary_issues.append(
                f"Region {rid} contains hallucinated overlap_entry_ids not in puzzle: {invalid_overlap_ids}."
            )
        overlap_ids = [eid for eid in raw_overlap_ids if eid in all_id_set]
        bad_overlap = [eid for eid in overlap_ids if eid not in entry_ids]
        if bad_overlap:
            secondary_issues.append(f"Region {rid} overlap_entry_ids not in entry_ids: {bad_overlap}.")
        overlap_ids = [eid for eid in overlap_ids if eid in entry_ids]

        rationale = str(raw_region.get("rationale", "")).strip()
        regions.append(
            SubCrosswordRegion(
                region_id=rid,
                entry_ids=entry_ids,
                overlap_entry_ids=overlap_ids,
                rationale=rationale if rationale else "local intersection cluster",
            )
        )

    coverage = _compute_coverage(regions, all_entry_ids)
    if not coverage["all_entries_covered"]:
        critical_issues.append(
            "Coverage failure: uncovered_entry_ids="
            f"{coverage['uncovered_entry_ids']}"
        )

    all_issues = critical_issues + secondary_issues
    normalized = {
        "regions": [
            {
                "region_id": r.region_id,
                "entry_ids": r.entry_ids,
                "overlap_entry_ids": r.overlap_entry_ids,
                "rationale": r.rationale,
            }
            for r in regions
        ],
        "coverage": coverage,
        "_critical_issues": critical_issues,
        "_secondary_issues": secondary_issues,
    }
    return len(critical_issues) == 0, all_issues, normalized


def _build_splitter_prompt(puzzle: Dict[str, Any]) -> str:
    payload_json = json.dumps(puzzle, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{splitter_system_prompt}\n\n"
        f"{splitter_user_prompt.format(payload_json=payload_json)}"
    )


def _build_repair_prompt(
    puzzle: Dict[str, Any],
    issues: List[str],
    critical_issues: List[str],
    secondary_issues: List[str],
    previous_json: str,
) -> str:
    issue_text = "\n".join(f"- {i}" for i in issues)
    critical_text = "\n".join(f"- {i}" for i in critical_issues) if critical_issues else "- (none)"
    secondary_text = "\n".join(f"- {i}" for i in secondary_issues) if secondary_issues else "- (none)"
    payload_json = json.dumps(puzzle, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{splitter_system_prompt}\n\n"
        f"{splitter_repair_prompt.format(payload_json=payload_json, critical_issues=critical_text, secondary_issues=secondary_text, issues=issue_text, previous_json=previous_json)}"
    )


def split_puzzle_with_gpt41(
    puzzle: Dict[str, Any],
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Input: puzzle dict with width/height/entries (+ optional neighbors/degree)
    Output: region_plan dict matching schema (regions + coverage)
    """
    if "entries" not in puzzle or not isinstance(puzzle.get("entries"), list):
        raise ValueError("Puzzle must contain 'entries' as a list.")

    if (
        "neighbors" not in puzzle
        or "degree" not in puzzle
        or "intersection_features" not in puzzle
    ):
        neighbors, degree = _build_neighbors_and_degree(puzzle["entries"])
        intersection_features = _build_intersection_features(puzzle["entries"])
        puzzle = dict(puzzle)
        puzzle["neighbors"] = neighbors
        puzzle["degree"] = degree
        puzzle["intersection_features"] = intersection_features

    if (
        "grid_layout" not in puzzle
        and _coerce_int(puzzle.get("width")) is not None
        and _coerce_int(puzzle.get("height")) is not None
    ):
        width = int(_coerce_int(puzzle.get("width")))
        height = int(_coerce_int(puzzle.get("height")))
        puzzle = dict(puzzle)
        puzzle["grid_layout"] = _build_grid_layout(width, height, puzzle["entries"])

    print(
        f"[split] start model={model} temp={temperature} "
        f"max_retries={max_retries} entries={len(puzzle.get('entries', []))}"
    )

    prompt = _build_splitter_prompt(puzzle)
    best_plan: Dict[str, Any] = {"regions": [], "coverage": {"all_entries_covered": False, "uncovered_entry_ids": [], "entry_appearance_counts": {}}}
    best_issues: List[str] = []
    raw = ""

    for attempt in range(max_retries + 1):
        print(f"[split] attempt={attempt + 1}/{max_retries + 1} querying GPT")
        raw = gpt(prompt, model=model, temperature=temperature, max_tokens=2400, n=1, stop=None)[0]
        parsed: Optional[Dict[str, Any]] = None
        issues: List[str] = []

        try:
            parsed = _extract_json_object(raw)
        except Exception as e:
            issues.append(f"JSON parse failed: {e}")
            print(f"[split] parse_error={e}")
            print("[split] failed_raw_response_begin")
            print(raw)
            print("[split] failed_raw_response_end")

        if parsed is not None:
            ok, issues, normalized = _validate_region_plan(parsed, puzzle)
            best_plan = normalized
            best_issues = issues
            if ok:
                critical = best_plan.get("_critical_issues", [])
                secondary = best_plan.get("_secondary_issues", [])
                print(
                    f"[split] success attempt={attempt + 1} "
                    f"regions={len(best_plan.get('regions', []))} "
                    f"critical_issues={len(critical)} secondary_issues={len(secondary)}"
                )
                for idx, issue in enumerate(secondary, start=1):
                    print(f"[split] secondary_issue_{idx}: {issue}")
                print(
                    f"[split] note: repair optimized for coverage/connectivity; "
                    f"secondary diagnostics are reported but non-blocking."
                )
                return best_plan
            print(
                f"[split] validation_failed attempt={attempt + 1} "
                f"issues={len(issues)}"
            )
            print("[split] failed_parsed_json_begin")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            print("[split] failed_parsed_json_end")
            print("[split] failed_raw_response_begin")
            print(raw)
            print("[split] failed_raw_response_end")
            for idx, issue in enumerate(issues, start=1):
                print(f"[split] issue_{idx}: {issue}")
        else:
            best_issues = issues

        if attempt < max_retries:
            print(f"[split] building repair prompt for attempt={attempt + 2}")
            previous_json = json.dumps(parsed if parsed is not None else {"raw_response": raw}, ensure_ascii=False)
            critical = best_plan.get("_critical_issues", []) if parsed is not None else best_issues
            secondary = best_plan.get("_secondary_issues", []) if parsed is not None else []
            prompt = _build_repair_prompt(puzzle, best_issues, critical, secondary, previous_json)

    # Keep schema-compatible output and attach debug fields.
    print("[split] exhausted retries; returning best effort result")
    best_plan["_issues"] = best_issues
    best_plan["_raw_response"] = raw
    return best_plan


def split(
    puzzle: Dict[str, Any],
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Helper function for puzzle decomposition only.
    """
    return split_puzzle_with_gpt41(
        puzzle=puzzle,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )


def solve(
    task,
    idx: int,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Main entrypoint others can call for NYT sub-region workflow.
    Current stage: split only. Region solving/merging will be added later.
    """
    puzzle = build_puzzle_input_from_task(task, idx)
    region_plan = split(
        puzzle=puzzle,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    return {
        "problem_id": idx,
        "status": "split_only",
        "message": "Task split completed. Sub-region solving is not implemented yet.",
        "region_plan": region_plan,
    }
