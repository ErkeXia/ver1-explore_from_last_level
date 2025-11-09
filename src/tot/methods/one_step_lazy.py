import itertools
import time
import copy
import numpy as np
from functools import partial
from tot.models import gpt, model_setup, llama_instruct
from tot.cache import FileCache

# --- Global Cache Registry ---
# We use dictionaries to handle caches for different models simultaneously
PROPOSE_CACHES = {}
VALUE_CACHES = {}

propose_num = value_num = 0
propose_time = value_time = 0

def y_output_from_env(env):
    rows = []
    for i in range(5):
        rows.append([env.board[i*5 + j] for j in range(5)])
    return "Output:\n" + "\n".join(" ".join(r) for r in rows) + "\n"

def apply_action_to_y(task, x, parent_y, action_line):
    task.set_status(x, parent_y)
    _msg, _r_all, _done, _info = task.env.step(action_line)
    return y_output_from_env(task.env)

# ---------------- propose children ----------------
def get_proposals_v1(task, parent_state, parent_index, x=None, K=5, M=3, model='llama'):
    """
    Generates proposals using the specified model (gpt or llama).
    """
    y_parent = parent_state['current']
    print(f"[{model.upper()}] Proposing for state:\n{y_parent.strip()}")
    
    task.set_status(x, y_parent)
    
    # Ensure cache exists for this model
    if model not in PROPOSE_CACHES:
        PROPOSE_CACHES[model] = FileCache(f"crossword_propose_{model}.json")
    current_cache = PROPOSE_CACHES[model]

    actions = []
    # 1. Gather all potential moves
    for i, (ans, data, status) in enumerate(zip(task.env.ans, task.env.data, task.env.status)):
        if '_' not in ans: continue

        position = f"h{i+1}" if i < 5 else f"v{i-5+1}"
        line = f'{data}: {" ".join(ans.lower())}'
        
        # Check cache first
        cached_result = current_cache.get(line)
        if cached_result:
            word, score = cached_result
            actions.append((f"{position}. {word}", score))
            continue
        
        # If not in cache, query the specific model
        system_prompt, user_prompt = task.propose_one_instruct_prompt_wrap(line)
        
        if model == 'gpt':
             full_prompt = f"{system_prompt}\n\n{user_prompt}"
             raw = gpt(full_prompt, n=K, stop=None, max_tokens=200)
        else:
             # Default to llama_instruct for 'llama' or other local models
             raw = llama_instruct(user_prompt, system_prompt, n=K, stop=None, max_tokens=200)

        y_action = task.propose_one_outputs_unwrap(x, y_parent, raw)
        if y_action:
            word, score = y_action
            actions.append((f"{position}. {word}", score))
            current_cache.set(line, (word, score))
            
    # 2. Filter invalid actions
    valid_actions = []
    for action, score in actions:
        if task.action_valid(action, x, y_parent):
            valid_actions.append((action, score))
        else:
            pass # print(f"Filtered invalid action: {action}")

    # 3. Rank and select top M
    if not valid_actions: return []
    valid_actions.sort(key=lambda item: item[1], reverse=True)
    top_actions = [action for action, score in valid_actions[:M]]

    # 4. Create child nodes
    children = []
    for action_line in top_actions:
        y_child = apply_action_to_y(task, x, y_parent, action_line)
        children.append({
            'parent_y': y_parent, 'parent_idx': parent_index, 
            'child_y': y_child, 'action': action_line
        })
    return children

# ---------------- score children ----------------
def get_values_v1(task, x, ys, n_eval=1, model='llama'):
    """
    Gets values for a list of states using the specified model.
    """
    # Ensure cache exists for this model
    if model not in VALUE_CACHES:
        VALUE_CACHES[model] = FileCache(f"crossword_value_{model}.json")
    current_cache = VALUE_CACHES[model]

    vals = []
    for y in ys:
        # We assume task.evaluate now supports a 'model' and 'cache' argument
        # matching the updates you likely made to crossword.py
        score_obj = task.evaluate(x, y, n_evaluate_sample=n_eval, model=model, cache=current_cache)
        s = score_obj.get('sure', 0)
        m = score_obj.get('maybe', 0)
        i = score_obj.get('impossible', 0)
        vals.append(s + 0.5*m - i)
    return vals

# ... (check_answer and count_filled_words remain the same) ...
def check_answer(candidates, task=None, x=None):
    for i, y in enumerate(candidates):
        info = task.test_output(task.xs.index(x), y)
        if info.get('r_game', 0) == 1: return i, y
        if "_" not in y: return i, y
    return None, None

def count_filled_words(y_grid_string: str) -> int:
    lines = y_grid_string.strip().split('\n')
    if not lines or "Output:" not in lines[0]: return 0
    grid_lines = lines[1:]
    board = [char for line in grid_lines[:5] for char in line.split()]
    if len(board) != 25: return 0
    count = 0
    for i in range(5):
        if '_' not in board[i*5 : (i+1)*5]: count += 1 # Horizontal
        if '_' not in board[i::5]: count += 1 # Vertical
    return count

# ---------------- DFS core (Hybrid) ----------------
def reasoning_dfs(task, x, start_grid, max_depth=12, branch=5, K=5, M=5):
    nodes = 1
    states = {0: [{'step': None, 'connect': None, 'current': start_grid, 'score': 0}]}
    stack = [(0, 0)] 

    while stack:
        depth, idx_in_level = stack.pop()
        parent_state = states[depth][idx_in_level]
        current_y = parent_state['current']
        print(f"\n--- Depth {depth}, Node {idx_in_level} ---")

        if depth >= max_depth: continue

        # === Phase 1: Llama proposes and initially values children ===
        llama_children_data = get_proposals_v1(task, parent_state, idx_in_level, x=x, K=K, M=M, model='llama')
        if not llama_children_data:
             print("Llama could not find any valid moves.")
             # Fallback to GPT immediately if Llama finds nothing
             # (Will be handled by Phase 3 logic below if we don't continue here)
        
        child_ys = [d['child_y'] for d in llama_children_data]
        llama_values = get_values_v1(task, x, child_ys, n_eval=1, model='llama')
        
        # Sort Llama's candidates by Llama's own value estimate
        ranked_llama_candidates = sorted(zip(llama_children_data, llama_values), key=lambda x: x[1], reverse=True)

        # === Phase 2: GPT Verification Loop ===
        best_verified_child = None
        
        print(f"Verifying {len(ranked_llama_candidates)} Llama candidates with GPT...")
        for child_data, llama_val in ranked_llama_candidates:
            # Ask GPT to value this specific child
            gpt_val = get_values_v1(task, x, [child_data['child_y']], n_eval=1, model='gpt')[0]
            
            print(f"  Action: {child_data['action']} | Llama Val: {llama_val:.2f} | GPT Val: {gpt_val:.2f} | Req: {depth + 1}")
            
            # Threshold check: GPT value must be >= new depth
            if gpt_val >= (depth + 1):
                print(f"  >>> Candidate ACCEPTED by GPT (Value {gpt_val:.2f} >= Depth {depth + 1})")
                # Found a good one!
                best_verified_child = (child_data, gpt_val)
                break
            else:
                 print(f"  >>> Candidate REJECTED by GPT. Trying next...")

        # === Phase 3: GPT Fallback (if needed) ===
        final_children_to_push = []
        
        if best_verified_child:
             # If we found a verified child, that's our path forward.
             final_children_to_push.append(best_verified_child)
        else:
             # If all Llama candidates failed, ask GPT for proposals directly.
             print("\n!!! All Llama candidates rejected. Switching to GPT for proposals. !!!")
             gpt_children_data = get_proposals_v1(task, parent_state, idx_in_level, x=x, K=K, M=M, model='gpt')
             if gpt_children_data:
                 # We verify these too, to get their scores and ensure they meet the threshold.
                 # Or, if we trust GPT completely, we just take the top one. 
                 # Let's value them to be consistent and get a score for visualization.
                 child_ys_gpt = [d['child_y'] for d in gpt_children_data]
                 gpt_values = get_values_v1(task, x, child_ys_gpt, n_eval=1, model='gpt')
                 ranked_gpt = sorted(zip(gpt_children_data, gpt_values), key=lambda x: x[1], reverse=True)
                 
                 # Take the single best one from GPT as the fallback path
                 print(f"GPT proposed {len(ranked_gpt)} actions. Taking the best one: {ranked_gpt[0][0]['action']} (Val: {ranked_gpt[0][1]:.2f})")
                 final_children_to_push.append(ranked_gpt[0])
             else:
                 print("GPT also failed to find proposals. Dead end.")

        # === Phase 4: Push selected child(ren) to stack ===
        if (depth + 1) not in states: states[depth + 1] = []
        
        # In this hybrid single-path mode, we typically only push ONE child to continue the DFS.
        # If you wanted a wider search, you could push more 'verified' children in Phase 2.
        for child_data, score in final_children_to_push:
            states[depth + 1].append({
                'step': child_data['action'],
                'connect': child_data['parent_idx'],
                'current': child_data['child_y'],
                'score': score
            })
            # Add to stack to continue exploration
            new_node_idx = len(states[depth + 1]) - 1
            stack.append((depth + 1, new_node_idx))
            nodes += 1

            # Check if this move solved it
            idx_sol, ans = check_answer([child_data['child_y']], task=task, x=x)
            if ans is not None: return idx_sol, ans, depth + 1, states, nodes

    # Fallback if stack empties without solution
    last_depth = max((d for d in states if states[d]), default=0)
    fallback_y = states[last_depth][0]['current'] if states[last_depth] else start_grid
    return 0, fallback_y, last_depth, states, nodes

def solve_v1(args, task, idx, slm='llama', instruct_model_arg=True, do_validate=False):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    
    # Set up the local model if required
    if slm and slm.lower() != 'gpt':
        model_setup(slm, instruct_model_arg, TGI_arg=False)

    x = task.get_input(idx)
    start_y = "Output:\n" + "\n".join(["_ _ _ _ _"]*5) + "\n"

    print("\n" + "="*40 + "\n  RUNNING HYBRID (LLAMA + GPT VERIFY) SOLVER  \n" + "="*40)
    
    # Run the new hybrid DFS
    sol_idx, sol_y, depth, states, nodes = reasoning_dfs(task, x, start_y, max_depth=12, branch=5, K=5, M=5)

    info = task.test_output(idx, sol_y)
    print(f"\nFinal Answer:\n{sol_y}")
    print(f"[{idx}] Stats: r_word={info['r_word']:.3f}, r_letter={info['r_letter']:.3f}, r_game={info['r_game']}")

    # Wrap in lists to maintain compatibility with your experiment runner's expectations
    return sol_idx, sol_y, depth, [states], nodes, [], []