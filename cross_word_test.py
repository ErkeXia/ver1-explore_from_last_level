import argparse
import sys
import time
import openai
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import gpt
from functools import partial
import torch
import transformers

model = 'gpt-3.5-turbo'
args = argparse.Namespace(backend=model, temperature=0.7, task='crossword', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
gpt = partial(gpt, model=args.backend, temperature=args.temperature)

states = {}   # depth -> list[{'step','connect','current'}]
nodes = 0

# ---------------- tiny glue: env -> "Output:\n..." ----------------
def y_output_from_env(env):
    rows = []
    for i in range(5):
        rows.append([env.board[i*5 + j] for j in range(5)])
    return "Output:\n" + "\n".join(" ".join(r) for r in rows) + "\n"

# ---------------- apply single action ----------------
def apply_action_to_y(task, x, parent_y, action_line):
    task.set_status(x, parent_y)                    # sync env to current y
    _msg, _r_all, _done, _info = task.env.step(action_line)  # apply 'h3. apple' or 'v2. water'
    return y_output_from_env(task.env)              # serialize back to Output-grid

# ---------------- propose children ----------------
def get_proposals_v1(task, parent_state, parent_index, feedback=None, x=None, K=10, M=5):
    y_parent = parent_state['current']

    propose_prompt = task.propose_prompt_wrap(x, y_parent)
    raw = gpt(propose_prompt, n=K, stop=None, max_tokens=200)  # ask for K samples
    print(f"raw proposals from gpt {raw}")

    # Parse lines like "h3. apple (high)" into y+action strings
    y_action_variants = task.propose_outputs_unwrap(x, y_parent, raw, n_max_propose=M)
    print(f"Actions applied {y_action_variants}")
    children = []
    for action in y_action_variants:
        last = [ln for ln in action.strip().split("\n") if ln]
        last = last[-1] if last else ""
        if not last or (last[0] not in ("h", "v")):
            continue
        y_child = apply_action_to_y(task, x, y_parent, last)
        children.append((y_parent, parent_index, y_child))
    return children

# ---------------- score children ----------------
def get_values_v1(task, x, ys, n_eval=1):
    vals = []
    for y in ys:
        print(f"Evaluation \n" + y)
        score_obj = task.evaluate(x, y, n_evaluate_sample=n_eval)  # {'sure','maybe','impossible'}
        s = score_obj.get('sure', 0)
        m = score_obj.get('maybe', 0)
        i = score_obj.get('impossible', 0)
        vals.append(s + 0.5*m - i)
    return vals

# ---------------- check solved ----------------
def check_answer(candidates, task=None, x=None):
    for i, y in enumerate(candidates):
        info = task.test_output(task.xs.index(x), y)  # uses env internally
        if info.get('r_game', 0) == 1:
            return i, y
        if "_" not in y:
            return i, y
    return None, None

# ---------------- DFS core ----------------
def reasoning_dfs(task, x, max_depth=12, branch=5, K=10, M=20):
    global nodes, states

    # init storage if needed
    if 0 not in states:
        states[0] = []

    # root node (blank grid) if none exists at depth 0
    if not states[0]:
        y0 = "Output:\n" + "\n".join(["_ _ _ _ _"]*5) + "\n"
        root = {'step': None, 'connect': None, 'current': y0}
        states[0] = [root]
        nodes = 1

    stack = [(0, 0)]

    while stack:
        depth, idx_in_level = stack.pop()
        parent_state = states[depth][idx_in_level]
        print(f"depth {depth} with idx {idx_in_level} parent state {parent_state}")

        if depth >= max_depth:
            continue

        new_ys_triplets = []
        attempt = 0 
        while(new_ys_triplets == [] and attempt < 3):
            new_ys_triplets = get_proposals_v1(task, parent_state, idx_in_level, x=x, K=K, M=M)
            attempt += 1
        
        print(f'__new ys__ {new_ys_triplets}')
        if not new_ys_triplets:
            continue

        child_ys = [t[2] for t in new_ys_triplets]
        values = get_values_v1(task, x, child_ys, n_eval=1)

        ranked = sorted(zip(new_ys_triplets, values), key=lambda z: z[1], reverse=True)[:branch]

        # ensure next layer
        if (depth + 1) not in states:
            states[depth + 1] = []

        print(f"rank: {ranked}")
        
        # store children nodes
        states[depth + 1].extend([
            {'step': trip[0], 'connect': trip[1], 'current': trip[2]} for (trip, _v) in ranked
        ])
        nodes += len(ranked)

        # check if solved
        cand_y = [trip[2] for (trip, _v) in ranked]
        idx_sol, ans = check_answer(cand_y, task=task, x=x)
        if ans is not None:
            return idx_sol, ans, depth + 1

        # push onto DFS stack (highest score explored first)

        start_idx = len(states[depth + 1]) - len(ranked)
        for local_pos in range(len(ranked) - 1, -1, -1):
            child_idx_global = start_idx + local_pos
            stack.append((depth + 1, child_idx_global))

    # fallback
    last_depth = max((d for d in states if states[d]), default=0)
    fallback_y = states[last_depth][0]['current'] if states[last_depth] else "Output:\n" + "\n".join(["_ _ _ _ _"]*5) + "\n"
    print("DFS could not find exact solution, returning a likely candidate.")
    return 0, fallback_y, last_depth

task = MiniCrosswordsTask()
results = []

idx = 1
x = task.get_input(idx)

with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    
    print(x)
    print(task.env.ans_gt)

    sol_idx, sol_y, depth = reasoning_dfs(task, x, max_depth=12, branch=5, K=5, M=5)
    info = task.test_output(idx, sol_y)

    results.append({
        'idx': idx,
        'depth': depth,
        'nodes': nodes,
        'y': sol_y,
        'info': info
    })
    print(f"__state__ {states}")
    
    print(f"__my ans_ \n" + sol_y)
    
    
    correct_words = task.env.ans_gt
    
    print("__Correct Answer Key__")
    
    print("Horizontal:")
    for i in range(5):
        print(f"  h{i+1}. {correct_words[i]}")

    print("Vertical:")
    for i in range(5, 10):
        print(f"  v{i-5+1}. {correct_words[i]}")

    print(f"[{idx}] depth={depth} nodes={nodes}  r_word={info['r_word']:.3f}  r_letter={info['r_letter']:.3f}  r_game={info['r_game']}")