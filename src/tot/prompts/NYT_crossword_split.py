"""
Prompts for NYT crossword region splitting (decomposition only).
"""

splitter_system_prompt_original = """You are a crossword decomposition planner.
You must output STRICT JSON only (no markdown, no backticks, no prose outside JSON).

Goal:
- Decompose one crossword puzzle into overlapping regions.
- Each region is a subset of entry_ids.
- Regions should be locally dense in intersections.

Optimization objectives (soft, but very important):
1. Connectivity first:
   - Prefer regions where entries form one connected component in the intersection graph.
   - Avoid regions that are mostly disconnected.
2. Keep nearby entries together:
   - Entries with nearby cells (based on row/col from cell indices) should usually be in the same region.
   - Prefer compact local neighborhoods over scattered/global groupings.
3. Encourage internal constraint flow:
   - Maximize intersections inside each region.
   - Avoid regions whose entries mostly intersect entries outside the region.
   - Use the provided `grid_layout` to keep regions spatially compact and local.
   - Use `intersection_features.pair_intersections` and `intersection_features.entry_intersections` explicitly.
   - Prefer regions where many entry pairs have non-empty shared_cells and good shared_count.
4. Directional balance:
   - Prefer each region to include a mix of Across and Down entries.
   - Avoid one-direction-only regions unless absolutely unavoidable.
5. Useful overlap across regions:
   - Use overlap_entry_ids to connect related neighboring regions and carry constraints across regions.
   - Prefer overlap entries that are structurally important connectors.

Output schema:
{
  "regions": [
    {
      "region_id": 0,
      "entry_ids": [12, 44, 55, 7, 8],
      "overlap_entry_ids": [44, 55],
      "rationale": "short <= 1 sentence"
    }
  ],
  "coverage": {
    "all_entries_covered": true,
    "uncovered_entry_ids": [],
    "entry_appearance_counts": {"12": 1, "44": 2}
  }
}

Required rules:
1. Coverage: every entry must appear in >= 1 region.
2. Size target per region: 8-14 entries.
3. Hard size bound per region: 5-20 entries.
4. Prefer overlap between related neighboring regions when useful for downstream consistency, but overlap is not mandatory for every region.
5. overlap_entry_ids must be a subset of entry_ids.
6. Across/Down balance in each region: absolute difference between #Across and #Down entries must be <= 5.
7. Do not hallucinate IDs: every id in entry_ids and overlap_entry_ids must exist in the input entries.
8. Each region's entries must form exactly one connected component in the intersection graph.
"""

splitter_system_prompt_short = """You are a crossword decomposition planner.
You must output STRICT JSON only (no markdown, no backticks, no prose outside JSON).

Goal:
- Decompose one crossword puzzle into overlapping regions.
- Each region is a subset of entry_ids.
- Regions should be locally dense in intersections.

Optimization objectives (soft, but very important):
1. Connectivity first:
   - Make every region a single connected component in the intersection graph.
2. Locality:
   - Keep nearby entries together using `cells` and `grid_layout` (compact spatial neighborhoods).
3. Internal constraints:
   - Maximize intersections inside each region using `intersection_features`.
   - Avoid regions whose entries mostly connect outside the region.
4. Direction mix:
   - Prefer a reasonable Across/Down balance in each region.
5. Overlap use:
   - Use overlap_entry_ids as optional bridges between related neighboring regions.
6. Size target:
   - Prefer 8-10 entries per region.

Output schema:
{
  "regions": [
    {
      "region_id": 0,
      "entry_ids": [12, 44, 55, 7, 8],
      "overlap_entry_ids": [44, 55],
      "rationale": "short <= 1 sentence"
    }
  ],
  "coverage": {
    "all_entries_covered": true,
    "uncovered_entry_ids": [],
    "entry_appearance_counts": {"12": 1, "44": 2}
  }
}

Required rules:
1. Coverage: every entry must appear in >= 1 region.
2. Hard size bound per region: 5-15 entries.
3. Prefer overlap between related neighboring regions when useful for downstream consistency, but overlap is not mandatory for every region.
4. overlap_entry_ids must be a subset of entry_ids.
5. Across/Down balance in each region: absolute difference between #Across and #Down entries must be <= 5.
6. Do not hallucinate IDs: every id in entry_ids and overlap_entry_ids must exist in the input entries.
7. Each region's entries must form exactly one connected component in the intersection graph.
"""

# Active system prompt (short variant). Keep original for side-by-side comparison.
splitter_system_prompt = splitter_system_prompt_short


splitter_user_prompt = """Split this crossword puzzle into overlapping regions.

Input payload JSON:
{payload_json}

Prioritize connected, local, intersection-dense regions with reasonable Across/Down balance.
Use explicit intersection features (`intersection_features`) as the primary structural signal.
Use `grid_layout` rows to preserve local spatial neighborhoods.

Return strict JSON only using the required schema.
"""


splitter_repair_prompt = """Your previous output violated constraints.

Original puzzle payload JSON (same puzzle/context as initial request):
{payload_json}

Primary repair objectives (must fix first):
1. Coverage: every entry_id from input must appear in at least one region.
2. Connectivity: each region's entry_ids must form exactly one connected component in the intersection graph.

Critical issues to fix now:
{critical_issues}

Other reported diagnostics (informational, lower priority during repair):
{secondary_issues}

Optional optimization preference (lower priority than primary objectives):
- Prefer 8-10 entries per region when possible.

Issues:
{issues}

Previous JSON:
{previous_json}

Re-output a corrected strict JSON that satisfies all constraints and schema.
Do not include markdown or explanation outside JSON.
"""
