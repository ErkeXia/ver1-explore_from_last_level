import re
from functools import partial

from tot.models import gpt, model_setup, llama_instruct
from tot.cache import FileCache

PROPOSE_CACHE = None
PROPOSE_MODE = "llama"


def _safe_cache_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", (s or "unknown"))


def y_output_from_env(task):
    return task.y_from_current_state(only_nonempty=True)


# ---------------- apply single action ----------------
def apply_action_to_y(task, x, parent_y, entry_id, word):
    task.set_status(x, parent_y)  # sync env to current y
    msg, _r_all, _done, _info = task.env.step(entry_id, word)
    if isinstance(msg, str) and msg.startswith("Invalid!"):
        return None
    return y_output_from_env(task)


def count_filled_entries(task, x, y_state: str) -> int:
    task.set_status(x, y_state)
    return sum(1 for fill in task.env.entry_fills if "_" not in fill)


def _norm_cached_rows(cached_list):
    out = []
    for row in (cached_list or []):
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append([str(row[0]).upper(), float(row[1])])
    return out


def _propose_single_entry(task, x, y_parent, entry_idx, rejected_words_by_entry=None, N=1):
    """
    Draft helper:
    Propose one candidate for exactly one entry and return (word, score) or None.
    Uses cache + avoid-list to reduce repeats.
    """
    spec = task.env.entries[entry_idx]
    line = task.get_entry_line(entry_idx)
    eid = spec.entry_id

    cached_list = _norm_cached_rows(PROPOSE_CACHE.get(line) or [])
    rejected = rejected_words_by_entry.get(eid, set()) if rejected_words_by_entry else set()
    cached_words = [w for w, _ in cached_list]

    # First try cached best not previously rejected.
    usable = [row for row in cached_list if row[0] != "NONE_SIGNAL" and row[0] not in rejected]
    if usable:
        usable.sort(key=lambda row: row[1], reverse=True)
        return usable[0][0], usable[0][1]

    avoid_words = sorted(set(cached_words + list(rejected)))
    system_prompt, user_prompt = task.propose_one_instruct_prompt_wrap(line, avoid_words=avoid_words)

    if PROPOSE_MODE == "gpt":
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        raw = gpt(full_prompt, n=N, stop=None, max_tokens=200)
        print(f"raw proposals from GPT {raw} for line {line}")
    else:
        raw = llama_instruct(user_prompt, system_prompt, n=N, stop=None, max_tokens=200)
        print(f"raw proposals from SLM {raw} for line {line}")

    y_action = task.propose_one_outputs_unwrap(x, y_parent, raw)
    if y_action == "NONE_SIGNAL":
        # Cache sentinel so we know this line may be exhausted.
        if ["NONE_SIGNAL", 0] not in cached_list:
            cached_list.append(["NONE_SIGNAL", 0])
            PROPOSE_CACHE.set(line, cached_list)
        return None

    if not y_action:
        return None

    word, score = y_action
    word = word.upper()

    if word not in cached_words:
        cached_list.append([word, score])
        PROPOSE_CACHE.set(line, cached_list)

    if word in rejected:
        return None
    return word, score


def propose_unfilled(task, parent_state, parent_index, x=None, rejected_words_by_entry=None, N=1):
    """
    Draft helper:
    Ask proposer for each currently unfilled entry and return one candidate per entry.
    """
    y_parent = parent_state["current"]
    print(f"____________________\nProposals (iterative round) for {y_parent}")
    task.set_status(x, y_parent)

    candidates = {}
    exhausted_entry_ids = set()

    for i, spec in enumerate(task.env.entries):
        current_fill = task.env.entry_fills[i]
        if "_" not in current_fill:
            continue

        y_action = _propose_single_entry(
            task=task,
            x=x,
            y_parent=y_parent,
            entry_idx=i,
            rejected_words_by_entry=rejected_words_by_entry or {},
            N=N,
        )

        if y_action is None:
            exhausted_entry_ids.add(spec.entry_id)
            continue

        word, score = y_action
        action_text = task.action_from_entry_word(spec.entry_id, word)
        if action_text is None:
            continue

        # Validate against parent state before conflict-resolution stage.
        if task.action_valid(spec.entry_id, word, x, y_parent):
            candidates[spec.entry_id] = {
                "entry_id": spec.entry_id,
                "word": word,
                "score": score,
                "action": action_text,
            }
        else:
            print(f"Filtered invalid action: {action_text}")

    return {
        "parent_y": y_parent,
        "parent_idx": parent_index,
        "candidates": candidates,
        "exhausted_entry_ids": exhausted_entry_ids,
    }


def resolve_conflicts(task, x, y_parent, candidates_by_eid):
    """
    Draft conflict resolution:
    - Keep all non-conflicting entries.
    - For each conflicting cell, keep entries supporting the best-scoring letter.
    - Drop other entries from that cell; iterate until stable.
    """
    task.set_status(x, y_parent)
    cell_letters = task.env._cell_letters_from_entry_fills()

    active = set(candidates_by_eid.keys())
    removed = {}

    # 1) Drop entries that directly conflict with already-fixed letters in parent state.
    for eid in list(active):
        eidx = task.env.eid_to_idx.get(int(eid))
        if eidx is None:
            active.discard(eid)
            continue
        spec = task.env.entries[eidx]
        word = candidates_by_eid[eid]["word"]
        invalid = False
        for j, cell in enumerate(spec.cells):
            existing = cell_letters.get(cell, "_")
            wch = word[j]
            if existing != "_" and existing != wch:
                invalid = True
                break
        if invalid:
            removed[eid] = {**candidates_by_eid[eid], "reason": "conflict_with_parent"}
            active.discard(eid)

    # 2) Resolve candidate-vs-candidate conflicts until stable.
    changed = True
    while changed:
        changed = False
        per_cell = {}

        for eid in active:
            eidx = task.env.eid_to_idx[int(eid)]
            spec = task.env.entries[eidx]
            word = candidates_by_eid[eid]["word"]
            score = candidates_by_eid[eid]["score"]
            for j, cell in enumerate(spec.cells):
                per_cell.setdefault(cell, []).append((eid, word[j], score))

        for cell, rows in per_cell.items():
            letters = set(ch for _, ch, _ in rows)
            if len(letters) <= 1:
                continue

            # Pick winning letter by max score; tie-break lexicographically for determinism.
            best_letter = sorted(
                letters,
                key=lambda ch: (
                    max(sc for _eid, c, sc in rows if c == ch),
                    ch,
                ),
                reverse=True,
            )[0]

            losers = [eid for eid, ch, _ in rows if ch != best_letter]
            for eid in losers:
                if eid in active:
                    removed[eid] = {**candidates_by_eid[eid], "reason": f"cell_conflict_{cell}"}
                    active.discard(eid)
                    changed = True

    kept = {eid: candidates_by_eid[eid] for eid in active}
    return kept, removed


def apply_candidates(task, x, y_parent, kept_candidates):
    """
    Apply kept candidates sequentially and return new y + applied/failed action lists.
    """
    y_current = y_parent
    applied_actions = []
    failed_actions = []

    ordered = sorted(kept_candidates.values(), key=lambda d: d["score"], reverse=True)
    for item in ordered:
        entry_id = item["entry_id"]
        word = item["word"]
        action_text = item["action"]
        score = item["score"]

        y_child = apply_action_to_y(task, x, y_current, entry_id, word)
        if y_child is None:
            failed_actions.append({**item, "reason": "apply_failed"})
            continue

        y_current = y_child
        applied_actions.append(
            {
                "entry_id": entry_id,
                "word": word,
                "action": action_text,
                "score": score,
            }
        )

    return y_current, applied_actions, failed_actions


def solve_v2(
    args,
    task,
    idx,
    slm="llama",
    instruct_model_arg=False,
    max_rounds=10,
    no_progress_limit=2,
):
    """
    Draft iterative NYT solver:
      1) propose one candidate for each unfilled entry
      2) delete conflicting proposals
      3) apply remaining proposals
      4) repeat until all entries are filled or solver gets stuck
    """
    global gpt
    global PROPOSE_CACHE, PROPOSE_MODE

    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    if slm and slm.lower() != "gpt":
        print(f"\n[Config] Setting proposal mode to SLM: {slm}")
        PROPOSE_MODE = slm
        PROPOSE_CACHE = FileCache(f"NYT_crossword/propose_cache_{_safe_cache_token(slm)}.json")
        model_setup(slm, instruct_model_arg)
    else:
        print("\n[Config] Setting proposal mode to GPT")
        PROPOSE_MODE = "gpt"
        PROPOSE_CACHE = FileCache(f"NYT_crossword/propose_cache_{_safe_cache_token(args.backend)}.json")

    x = task.get_input(idx)
    task.env.reset(idx)

    print("__Correct Answer Key (first 30 entries)__")
    for spec in task.env.entries[:30]:
        print(f"  {spec.direction}{spec.label} (e{spec.entry_id}): {spec.answer}")
    print(f"x = {x}\n")

    start_y = "[]"
    rejected_words_by_entry = {}
    all_states = []
    iteration_details = []
    no_progress_rounds = 0

    last_filled = count_filled_entries(task, x, start_y)
    current_y = start_y

    for r in range(1, max_rounds + 1):
        parent_state = {"step": None, "connect": None, "current": current_y, "created_order": 1}
        proposal_stage = propose_unfilled(
            task=task,
            parent_state=parent_state,
            parent_index=0,
            x=x,
            rejected_words_by_entry=rejected_words_by_entry,
            N=1,
        )
        candidates = proposal_stage["candidates"]
        exhausted = proposal_stage["exhausted_entry_ids"]

        if not candidates:
            print(f"[Round {r}] No candidates proposed. Stopping.")
            iteration_details.append(
                {
                    "iteration": r,
                    "mode": "iterative_propose_conflict_prune",
                    "status": "no_candidates",
                    "exhausted_entry_ids": sorted(list(exhausted)),
                }
            )
            break

        kept, removed = resolve_conflicts(task, x, current_y, candidates)
        for eid, item in removed.items():
            rejected_words_by_entry.setdefault(eid, set()).add(item["word"])

        next_y, applied_actions, failed_actions = apply_candidates(task, x, current_y, kept)
        for item in failed_actions:
            eid = item["entry_id"]
            rejected_words_by_entry.setdefault(eid, set()).add(item["word"])

        states = {
            0: [parent_state],
            1: [
                {
                    "step": "iterative_propose_conflict_prune",
                    "connect": 0,
                    "current": next_y,
                    "created_order": 2,
                    "applied_actions": applied_actions,
                }
            ],
        }
        all_states.append(states)

        filled_now = count_filled_entries(task, x, next_y)
        progress = filled_now - last_filled

        iteration_details.append(
            {
                "iteration": r,
                "mode": "iterative_propose_conflict_prune",
                "n_candidates": len(candidates),
                "n_kept_after_conflict": len(kept),
                "n_removed_by_conflict": len(removed),
                "n_applied": len(applied_actions),
                "n_apply_failed": len(failed_actions),
                "filled_entries_before": last_filled,
                "filled_entries_after": filled_now,
                "progress": progress,
                "exhausted_entry_ids": sorted(list(exhausted)),
                "result_grid": next_y,
            }
        )

        current_y = next_y
        info = task.test_output(idx, current_y)
        print(
            f"[Round {r}] filled={filled_now} progress={progress} "
            f"r_word={info['r_word']:.3f} r_letter={info['r_letter']:.3f} r_game={info['r_game']}"
        )

        if info.get("r_game", 0):
            print(f"[Round {r}] Puzzle solved.")
            break

        if progress <= 0:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        if no_progress_rounds >= no_progress_limit:
            print(f"[Round {r}] No progress for {no_progress_rounds} rounds. Stopping.")
            break

        last_filled = filled_now

    final_y = current_y
    final_info = task.test_output(idx, final_y)

    depth = len(all_states)
    nodes = 1 + depth

    print(f"__my ans_ \n{final_y}")
    print(
        f"[{idx}] depth={depth} nodes={nodes}  "
        f"r_word={final_info['r_word']:.3f}  r_letter={final_info['r_letter']:.3f}  r_game={final_info['r_game']}"
    )

    return 0, final_y, depth, all_states, nodes, [], iteration_details
