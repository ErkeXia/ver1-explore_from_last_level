import itertools
import time
import copy
import numpy as np
from functools import partial
# from tot.models import gpt
from tot.models import gpt, base_model, model_setup, llama_instruct
from tot.cache import FileCache

import json
import os

PROPOSE_CACHE = None
PROPOSE_MODE = 'llama'
VISITED_CACHE = None

propose_num = value_num = 0
propose_time = value_time = 0

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
def get_proposals_v1(task, parent_state, parent_index, feedback=None, x=None, K=5, M=3, N=5):
    y_parent = parent_state['current']
    print(f"____________________\nProposals for {y_parent}")
    
    task.set_status(x, y_parent)
    actions = []
    TARGET_CANDIDATES = N
    
    for i, (ans, data, status) in enumerate(zip(task.env.ans, task.env.data, task.env.status)):
        if '_' not in ans: continue

        position = f"h{i+1}" if i < 5 else f"v{i-5+1}"
        ans = ' '.join(ans.lower())
        line = f'{data}: {ans}'
        
        # 1. Retrieve existing proposals (List of [word, score] lists)
        # We use empty list [] default if nothing cached
        cached_list = PROPOSE_CACHE.get(line) or []
        
        # Extract just the words for the "avoid list"
        existing_words = [item[0] for item in cached_list]
        
        # 2. Check if we need more proposals
        attempts = 0
        MAX_RETRIES = TARGET_CANDIDATES
        
        print(f"use cache {len(cached_list)} for line {line}\n")
        
        while len(cached_list) < TARGET_CANDIDATES and attempts < MAX_RETRIES:
            system_prompt, user_prompt = task.propose_one_instruct_prompt_wrap(line, avoid_words=existing_words)
            
            if PROPOSE_MODE == 'gpt':
                 full_prompt = f"{system_prompt}\n\n{user_prompt}"
                 raw = gpt(full_prompt, n=1, stop=None, max_tokens=200)
                 print(f"raw proposals from GPT {raw} for line {line}")
            else:
                 raw = llama_instruct(user_prompt, system_prompt, n=1, stop=None, max_tokens=200)
                 print(f"raw proposals from SLM {raw} for line {line}")
            
            y_action = task.propose_one_outputs_unwrap(x, y_parent, raw)
            
            # --- NEW LOGIC START ---
            if y_action == "NONE_SIGNAL":
                print(f"  - Model indicated no more valid words for {position}.")
                break # Stop trying to generate more words for this line
            # --- NEW LOGIC END ---
            
            if y_action:
                word, score = y_action
                if word not in existing_words:
                    cached_list.append([word, score])
                    existing_words.append(word)
                    PROPOSE_CACHE.set(line, cached_list)
                    print(f"  + New proposal found for {position}: {word}")
                else:
                    print(f"  - Model repeated existing word: {word}")
            
            attempts += 1
            
        # 3. Add all cached proposals (old + new) to the current actions list
        for word, score in cached_list:
            full_action = f"{position}. {word}"
            actions.append((full_action, score))
            
    valid_actions = []
    for action, score in actions:
        if task.action_valid(action, x, y_parent):
            valid_actions.append((action, score))
        else:
            print(f"Filtered invalid action: {action}")

    # 3. Rank the valid actions and select the top M
    if not valid_actions:
        return []
        
    valid_actions.sort(key=lambda item: item[1], reverse=True)
    
    children = []
    for action_line, score in valid_actions:
        # Check if we have collected enough children
        if len(children) >= M:
            break

        y_child = apply_action_to_y(task, x, y_parent, action_line)
        
        # --- Evaluate with GPT and filter invalid grids ---
        # gpt_evaluate returns None if "impossible" is found in any line
        print(f"possible action {action_line}")
        eval_result = task.evaluate_state(x, y_child, model = 'gpt')
        if eval_result is None:
            print(f"Child rejected by GPT evaluation: Action '{action_line}' created an impossible state.")
            continue
        # -------------------------------------------------------------

        # Return a dictionary that includes the action taken
        children.append({
            'parent_y': y_parent, 
            'parent_idx': parent_index, 
            'child_y': y_child, 
            'action': action_line
        })
        
        
    return children

# ---------------- score children ----------------
def get_values_v1(task, x, ys, n_eval=1):
    vals = []
    for y in ys:
        print(f"Evaluation \n" + y)
        score_obj = task.score(x, y, n_evaluate_sample=n_eval, model = PROPOSE_MODE)  # {'sure','maybe','impossible'}
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

def count_filled_words(y_grid_string: str) -> int:
    """
    Parses a grid string and counts how many full 5-letter words
    (horizontal and vertical) have been completed (i.e., contain no underscores).
    """
    lines = y_grid_string.strip().split('\n')
    if not lines or "Output:" not in lines[0]:
        return 0
    grid_lines = lines[1:]

    # Create a flat list of all 25 characters on the board
    board = []
    for line in grid_lines:
        board.extend(line.split())

    if len(board) != 25:
        return 0 # Malformed board

    filled_word_count = 0

    # 1. Check for 5 complete horizontal words
    for i in range(5):
        row = board[i*5 : (i+1)*5]
        if '_' not in row:
            filled_word_count += 1

    # 2. Check for 5 complete vertical words
    for i in range(5):
        column = board[i::5] # Slicing trick to get every 5th element
        if '_' not in column:
            filled_word_count += 1
            
    return filled_word_count

# ---------------- DFS core ----------------
def reasoning_dfs(task, x, start_grid, max_depth=12, branch=5, K=5, M=5):
    global nodes, states
    completeness_bonus = 3

    nodes = 0
    states = {}
    completeness_bonus = 3

    # Initialize the search with the provided starting grid
    states[0] = [{'step': None, 'connect': None, 'current': start_grid, 'created_order': 1}]
    nodes = 1
    
    stack = [(0, 0)]

    while stack:
        depth, idx_in_level = stack.pop()
        parent_state = states[depth][idx_in_level]
        print(f"depth {depth} with idx {idx_in_level} parent state {parent_state}")
        
        current_y = parent_state['current']
        if VISITED_CACHE.get(current_y):
            print(f"Skipping duplicate state at depth {depth}")
            continue
        VISITED_CACHE.set(current_y, True)

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

        child_ys = [t['child_y'] for t in new_ys_triplets]
        values = get_values_v1(task, x, child_ys, n_eval=1)

        # ranked = sorted(zip(new_ys_triplets, values), key=lambda z: z[1], reverse=True)[:branch]
        
        combined_scores = []
        for i, y_child in enumerate(child_ys):
            llm_score = values[i]
            filled_words = count_filled_words(y_child)
            # New score = LLM's quality score + bonus for each completed word
            # combined_score = llm_score + (completeness_bonus * filled_words)
            combined_score = llm_score
            combined_scores.append(combined_score)
            print(f"Child {i}: LLM Score={llm_score:.2f}, Filled Words={filled_words}, Combined Score={combined_score:.2f}")

        # 2. Sort the original data based on the new combined scores
        # We zip the original triplets and values with their new scores, then sort
        combined_data = zip(new_ys_triplets, values, combined_scores)
        sorted_data = sorted(combined_data, key=lambda item: item[2], reverse=True)
        ranked = [(triplet, value) for triplet, value, score in sorted_data[:branch]]
        ranked_data = sorted_data[:branch]
        
        # ensure next layer
        if (depth + 1) not in states:
            states[depth + 1] = []

        print(f"rank: {ranked}")
        
        for data_dict, _, score in ranked_data:
            nodes += 1 # Increment counter for each new node created
            states[depth + 1].append({
                'step': data_dict['action'], 
                'connect': data_dict['parent_idx'],
                'current': data_dict['child_y'],
                'score': score,
                'created_order': nodes # Store the unique creation order
            })
        
        # store children nodes
        # states[depth + 1].extend([
        #     {'step': trip[0], 'connect': trip[1], 'current': trip[2]} for (trip, _v) in ranked
        # ])
        # nodes += len(ranked)

        # check if solved
        cand_y = [d['child_y'] for d, _, _ in ranked_data]
        idx_sol, ans = check_answer(cand_y, task=task, x=x)
        if ans is not None:
            return idx_sol, ans, depth + 1, states, nodes

        # push onto DFS stack (highest score explored first)

        start_idx = len(states[depth + 1]) - len(ranked)
        for local_pos in range(len(ranked) - 1, -1, -1):
            child_idx_global = start_idx + local_pos
            stack.append((depth + 1, child_idx_global))

    # fallback
    last_depth = max((d for d in states if states[d]), default=0)
    fallback_y = states[last_depth][0]['current'] if states[last_depth] else "Output:\n" + "\n".join(["_ _ _ _ _"]*5) + "\n"
    print("DFS could not find exact solution, returning a likely candidate.")
    return 0, fallback_y, last_depth, states, nodes

def evaluate(task, x, y):
    
    sure_lst = task.gpt_evaluate(x, y)
    pruned_y = task.prune_grid_by_sure_list(x, y, sure_lst)
    print(f"Original Llama Output:\n{y}")
    print(f"Grid after GPT Pruning:\n{pruned_y}")
        
    return pruned_y, sure_lst


def solve_v1(args, task, idx, slm = 'llama', instruct_model_arg = False, do_validate = True):
    global gpt, validate_time
    global PROPOSE_CACHE, PROPOSE_MODE, VISITED_CACHE

    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    visited_filename = f"temp_visited_nodes_{idx}.json"
    VISITED_CACHE = FileCache(visited_filename)
    
    if slm and slm.lower() != 'gpt':
        print(f"\n[Config] Setting proposal mode to SLM: {slm}")
        PROPOSE_MODE = slm
        PROPOSE_CACHE = FileCache(f"crossword_propose_cache_{slm}.json")
        model_setup(slm, instruct_model_arg, TGI_arg=False)
    else:
        print("\n[Config] Setting proposal mode to GPT")
        PROPOSE_MODE = 'gpt'
        PROPOSE_CACHE = FileCache("crossword_propose_cache_gpt.json")
    
    try:
        x = task.get_input(idx)  # input
        
        correct_words = task.env.ans_gt
        
        print("__Correct Answer Key__")
        
        print("Horizontal:")
        for i in range(5):
            print(f"  h{i+1}. {correct_words[i]}")

        print("Vertical:")
        for i in range(5, 10):
            print(f"  v{i-5+1}. {correct_words[i]}")
        
        print(f'x = {x}\n')
        
        start_y = "Output:\n" + "\n".join(["_ _ _ _ _"]*5) + "\n"
        all_states = []
        gpt_eval_results = []
        iteration_details = [] 

        max_iterations = 3

        for i in range(max_iterations):
            
            sol_idx, sol_y, depth, states, nodes = reasoning_dfs(task, x, start_grid=start_y, max_depth=12, branch=5, K=5, M=5)
            all_states.append(states)
            
            if not do_validate:
                iteration_details.append({
                    "iteration": i + 1,
                    "llama_result_grid": sol_y,
                    "gpt_pruned_grid": "N/A (Validation skipped)"
                })
                break
            
            pruned_y, sure_lst = evaluate(task, x, sol_y)
            gpt_eval_results.append(sure_lst)
            
            iteration_details.append({
                "iteration": i + 1,
                "llama_result_grid": sol_y,
                "gpt_pruned_grid": pruned_y
            })
                    
            if pruned_y == sol_y:
                print("\nGPT pruning resulted in no changes. Halting refinement.")
                break
            
            start_y = pruned_y
            
        info = task.test_output(idx, sol_y)
        print(f"__state__ {states}")
        
        print(f"__my ans_ \n" + sol_y)
        

        print(f"[{idx}] depth={depth} nodes={nodes}  r_word={info['r_word']:.3f}  r_letter={info['r_letter']:.3f}  r_game={info['r_game']}")
        return sol_idx, sol_y, depth, all_states, nodes, gpt_eval_results, iteration_details
    finally:
        if os.path.exists(VISITED_CACHE.cache_file):
            try:
                os.remove(VISITED_CACHE.cache_file)
                print(f"[Cleanup] Deleted temporary visited cache: {visited_filename}")
            except OSError as e:
                print(f"[Cleanup Error] Could not delete {visited_filename}: {e}")

def get_time():
    global propose_num, propose_time, value_num, value_time
    print(f"propose num: {propose_num}, propose time per num: {(propose_time/propose_num):.6f}")
    print(f"value num: {value_num}, value time per num: {(value_time/value_num):.6f}")
    return propose_num, value_num, validate_num, (propose_time/propose_num), (value_time/value_num), (validate_time/validate_num)
