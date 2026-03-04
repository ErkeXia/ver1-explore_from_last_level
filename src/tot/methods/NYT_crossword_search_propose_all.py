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


# ---------------- propose all entries once ----------------
def get_proposals_v1(task, parent_state, parent_index, x=None, N=1):
    """
    Incomplete solver stage by design:
    - iterate all unfilled entries exactly once
    - ask proposer for one candidate per entry
    - apply valid actions sequentially into one child state
    """
    y_parent = parent_state["current"]
    print(f"____________________\nProposals (single pass) for {y_parent}")

    task.set_status(x, y_parent)

    actions = []
    for i, spec in enumerate(task.env.entries):
        current_fill = task.env.entry_fills[i]
        if "_" not in current_fill:
            continue

        position = f"e{spec.entry_id}"
        line = task.get_entry_line(i)

        cached_list = PROPOSE_CACHE.get(line) or []
        existing_words = [item[0] for item in cached_list if isinstance(item, (list, tuple)) and len(item) >= 1]

        if not cached_list:
            system_prompt, user_prompt = task.propose_one_instruct_prompt_wrap(line, avoid_words=existing_words)

            if PROPOSE_MODE == "gpt":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                raw = gpt(full_prompt, n=N, stop=None, max_tokens=200)
                print(f"raw proposals from GPT {raw} for line {line}")
            else:
                raw = llama_instruct(user_prompt, system_prompt, n=N, stop=None, max_tokens=200)
                print(f"raw proposals from SLM {raw} for line {line}")

            y_action = task.propose_one_outputs_unwrap(x, y_parent, raw)

            if y_action == "NONE_SIGNAL":
                print(f"  - Model indicated no valid word for {position}.")
                PROPOSE_CACHE.set(line, [["NONE_SIGNAL", 0]])
                continue

            if y_action:
                word, score = y_action
                cached_list.append([word, score])
                PROPOSE_CACHE.set(line, cached_list)
                print(f"  + New proposal found for {position}: {word}")
            else:
                continue

        # pick best cached word for this single-pass stage
        filtered = [row for row in cached_list if row and row[0] != "NONE_SIGNAL"]
        if not filtered:
            continue

        filtered.sort(key=lambda row: row[1], reverse=True)
        word, score = filtered[0]

        full_action = task.action_from_entry_word(spec.entry_id, word)
        if full_action is None:
            continue

        if task.action_valid(spec.entry_id, word, x, y_parent):
            actions.append((spec.entry_id, word, full_action, score))
        else:
            print(f"Filtered invalid action: {full_action}")

    if not actions:
        return {
            "parent_y": y_parent,
            "parent_idx": parent_index,
            "child_y": y_parent,
            "actions": [],
        }

    y_current = y_parent
    applied_actions = []
    for entry_id, word, action_text, score in actions:
        y_child = apply_action_to_y(task, x, y_current, entry_id, word)
        if y_child is None:
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

    return {
        "parent_y": y_parent,
        "parent_idx": parent_index,
        "child_y": y_current,
        "actions": applied_actions,
    }


def solve_v1(
    args,
    task,
    idx,
    slm="llama",
    instruct_model_arg=False,
):
    """
    Incomplete NYT solver:
    only performs the first stage (propose all entries once), then stops.
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
    parent_state = {"step": None, "connect": None, "current": start_y, "created_order": 1}

    proposal_result = get_proposals_v1(task, parent_state, 0, x=x, N=1)
    sol_y = proposal_result["child_y"]

    states = {
        0: [parent_state],
        1: [
            {
                "step": "propose_all_entries_once",
                "connect": 0,
                "current": sol_y,
                "actions": proposal_result.get("actions", []),
                "created_order": 2,
            }
        ],
    }

    nodes = 2
    depth = 1
    all_states = [states]

    info = task.test_output(idx, sol_y)
    print(f"__my ans_ \n{sol_y}")
    print(
        f"[{idx}] depth={depth} nodes={nodes}  "
        f"r_word={info['r_word']:.3f}  r_letter={info['r_letter']:.3f}  r_game={info['r_game']}"
    )

    iteration_details = [
        {
            "iteration": 1,
            "mode": "single_pass_propose_all_entries_once",
            "proposed_actions": proposal_result.get("actions", []),
            "result_grid": sol_y,
        }
    ]

    # Keep return shape compatible with existing experiment scripts.
    return 0, sol_y, depth, all_states, nodes, [], iteration_details
