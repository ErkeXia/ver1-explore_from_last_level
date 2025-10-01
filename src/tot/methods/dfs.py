import itertools
import time
import copy
import numpy as np
from functools import partial
# from tot.models import gpt
from tot.models import gpt, base_model, model_setup

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
def get_proposals_v1(task, parent_state, parent_index, feedback=None, x=None, K=5, M=3):
    y_parent = parent_state['current']

    propose_prompt = task.propose_prompt_wrap(x, y_parent)
    raw = base_model(propose_prompt, n=K, stop=None, max_tokens=200)
    # raw = gpt(propose_prompt, n=K, stop=None, max_tokens=200)  # ask for K samples
    print(f"raw proposals from llama {raw}")

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
def reasoning_dfs(task, x, max_depth=12, branch=5, K=5, M=3):
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

def validate(task, x, f_step):
    global validate_time, validators
    start = time.perf_counter()
    validate_prompt = task.validate_sys_prompt_wrap(x, f_step)
    print(f'Validate prompt: {validate_prompt}')
    validate_outputs = gpt(validate_prompt, n=1, stop=None)
    elapsed = time.perf_counter() - start
    validate_time += elapsed
    # print(validate_outputs)
    validators.append(validate_outputs[0])
    redo_s, feedback = task.validate_unwrap(validate_outputs[0])
    return redo_s, feedback

def evaluate(task, x, f_step):
    global validate_time, correctness_r, suggestion_r
    start = time.perf_counter()
    correctness_prompt, suggest_prompt = task.evaluate_sys_prompt_wrap(x, f_step)
    # print(f'Correctness prompt: {correctness_prompt} \n suggest_prompt: {suggest_prompt}')
    correctness_output = gpt(correctness_prompt, n=1, stop=None)[0]
    print(f'correctness output: {correctness_output}')
    correctness_r.append(correctness_output)
    correctness = task.correctness_unwrap(correctness_output)
    if correctness == 'No':
        suggest_output = gpt(suggest_prompt, n=1, stop=None)[0]
        print(f'suggest output: {suggest_output}')
        redo_s, feedback = task.suggest_unwrap(suggest_output)
        suggestion_r.append(suggest_output)
    else:
        redo_s = -1
        feedback = correctness
    elapsed = time.perf_counter() - start
    validate_time += elapsed
    return redo_s, feedback


def retrieve_steps(num_steps, idx, y):
    step = num_steps - 1
    thought_chain = []
    chain_index = []
    intermediate_state = []
    while step >= 0:
        print(f'step: {step}, idx: {idx}')
        thought_chain.append(states[step+1][idx]["step"])
        next_idx = states[step+1][idx]['connect']
        intermediate_state.append(states[step][next_idx]["current"])
        # assert(states[step][idx]["connect"] == thoughts[step][idx])
        chain_index.append(idx)
        idx = next_idx
        step -= 1
    chain_index.append(0)
    chain_index.reverse()
    thought_chain.reverse()
    intermediate_state.reverse()
    return intermediate_state, thought_chain, chain_index

def solve_v1(args, task, idx, slm = 'llama', instruct_model_arg = False, do_validate = True):
    global gpt
    global thoughts, connection, steps, validators, correctness_r, suggestion_r
    global value_num, value_time
    global propose_num, propose_time
    global validate_num, validate_time
    global nodes
    global states, repeat_value
    global instruct_model
    
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    instruct_model = model_setup(slm, instruct_model_arg, TGI_arg = False)
    
    # nodes = 1
    # propose_num = value_num = 0
    # propose_time = value_time = 0
    # validate_num = validate_time = 0
    # repeat_value = 0
    
    # thoughts = [[] for _ in range(task.steps)]
    # connection = [[] for _ in range(task.steps)]
    # steps = [[] for _ in range(task.steps)]
    # states = [[] for _ in range(task.steps + 1)]
    
    # all_states = []
    # validators = []
    # correctness_r = []
    # suggestion_r = []
    # all_thoughts = []
    # llama_ans = []
    
    # print(gpt)
    
    states = {}   # depth -> list[{'step','connect','current'}]
    nodes = 0
    
    idx = 1
    x = task.get_input(idx)  # input
    print(f'x = {x}\n')
    
    # states[0].append({'step': '', 'connect': 0, 'current': x})
    
    # prev_level = ['']
    # val_count = 0
    # step = 0
    # single = 0
    # while(val_count < 1): # call large model for at most three times
    #     idx, y, st = reasoning(task, step, x, feedback = None, single = single)
    #     print(f'--States-- {states}')
        
    #     all_thoughts.append(copy.deepcopy(thoughts))
    #     all_states.append(copy.deepcopy(states))
    #     intermediate_state, thought_chain, chain_index = retrieve_steps(st, idx, y)
    #     llama_ans.append(thought_chain)
    #     print(f'Retrieve steps: {thought_chain} \n Chainindex: {chain_index}')
    #     if not do_validate:
    #         print(f"Output from reasoning! idx: {idx} \n y: {y} \n st: {st}")
    #         evaluation = {'validators': validators, 'correctness': correctness_r, 'suggestions': suggestion_r}
    #         return x, thought_chain, all_thoughts, evaluation, llama_ans, nodes, all_states

    #     validate_num += 1
    #     # validate_outputs = validate(task, x, thought_chain)
    #     # validators.append(validate_outputs)
    #     # redo_s, feedback = task.validate_unwrap(validate_outputs)
    #     # redo_s, feedback = validate(task, x, thought_chain)
    #     redo_s, feedback = evaluate(task, x, thought_chain)
    #     print(f'redo {redo_s} feedback: {feedback}')
        
    #     possible_steps = [p for p in feedback.split("\n") if any(ch.isdigit() for ch in p)]
        
    #     if(redo_s == -1):
    #         break
    #     if(feedback != ""):
    #         # prev_level = [feedback]
    #         prev_level = possible_steps
    #         step = redo_s + 1
    #         prev_idx = chain_index[redo_s]
    #         print(f'possible steps: {possible_steps}, connect: {prev_idx}')
    #         # states[step] = [{'step': feedback, 'connect': prev_idx, 'current': task.manage_state(states[redo_s][prev_idx]["current"], feedback)}]
    #         states[step] = [{'step': possible_step, 'connect': prev_idx, 'current': task.manage_state(states[redo_s][prev_idx]["current"], possible_step)} for possible_step in possible_steps]
    #         print(f"before thoughts{thoughts} steps{steps}")
    #         # thoughts[redo_s][prev_idx] = feedback
    #         # steps[redo_s][prev_idx] = feedback
    #         thoughts[redo_s] = possible_steps
    #         steps[redo_s] = possible_steps
    #         print(f"after thoughts{thoughts} steps{steps}")
    #     else:
    #         if(redo_s == 0):
    #             prev_level = ['']
    #             single = 0
    #         else:
    #             prev_level = thoughts[redo_s - 1]
    #             single = None
    #         step = redo_s
    #     print(f'prev_level {prev_level} \nstep {step}\nsingle{single if single else -1}')
    #     # print(f'The validate result: \n {validate_outputs}\n')
    #     val_count += 1
        
        # print(f'Receive result from reasoning:\n{y} \n with index {idx}\n')
    
        # print("Thoughts: \n")
        # for i,ts in enumerate(thoughts):
        #     print(f'step {i} \n')
        #     for t in ts:
        #         print(f'{t} \n')
        #     print(connection[i])
        
        # print("Index: \n")
        # print(connection)
        
        # print("Steps: \n")
        # for i,ts in enumerate(steps):
        #     print(f'step {i} \n')
        #     for t in ts:
        #         print(f'{t} \n')
    # print(f'Repeat value time: {repeat_value}')
    # avg_validate = validate_time/validate_num
    # print(f'validate average time: {avg_validate}')
    # evaluation = {'validators': validators, 'correctness': correctness_r, 'suggestions': suggestion_r}
    sol_idx, sol_y, depth = reasoning_dfs(task, x, max_depth=12, branch=5, K=5, M=3)
    info = task.test_output(idx, sol_y)

    # results.append({
    #     'idx': idx,
    #     'depth': depth,
    #     'nodes': nodes,
    #     'y': sol_y,
    #     'info': info
    # })
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
    return sol_idx, sol_y, depth, states
      
def get_time():
    global propose_num, propose_time, value_num, value_time
    print(f"propose num: {propose_num}, propose time per num: {(propose_time/propose_num):.6f}")
    print(f"value num: {value_num}, value time per num: {(value_time/value_num):.6f}")
    return propose_num, value_num, validate_num, (propose_time/propose_num), (value_time/value_num), (validate_time/validate_num)

# def get_proposals(task, x, y): 
#     global propose_num, propose_time
#     propose_num += 1
#     propose_prompt = task.propose_prompt_wrap(x, y)
#     start = time.perf_counter()
#     proposals = gpt(propose_prompt, n=1, stop=None, max_tokens=200)[0].split('\n')
#     elapsed = time.perf_counter() - start
#     propose_time += elapsed
#     proposals = [s for s in proposals if not ("Input" in s or "steps" in s)]
#     return [y + _ + '\n' for _ in proposals]

# def solve(args, task, idx, to_print=True):
#     global gpt
#     global thoughts
#     global value_num, value_time
#     global propose_num, propose_time
#     nodes_num = 1
    
#     thoughts = [[] for _ in range(task.steps)]
#     gpt = partial(gpt, model=args.backend, temperature=args.temperature)
#     propose_num = value_num = 0
#     propose_time = value_time = 0
#     print(gpt)
#     x = task.get_input(idx)  # input
#     print(f"x = {x}")
#     ys = ['']  # current output candidates
#     infos = []
#     for step in range(task.steps):
#         # generation
#         new_ys = [get_proposals(task, x, y) for y in ys]
#         new_ys = list(itertools.chain(*new_ys))
#         ids = list(range(len(new_ys)))
#         # evaluation
#         if args.method_evaluate == 'vote':
#             values = get_votes(task, x, new_ys, args.n_evaluate_sample)
#         elif args.method_evaluate == 'value':
#             values = get_values(task, x, new_ys, args.n_evaluate_sample)

#         # if step == task.steps - 1:
#         #     print(f"Reach final layer! \n x = {x} \n new_ys = {new_ys} \n value = {values}")
#         # selection
#         if args.method_select == 'sample':
#             ps = np.array(values) / sum(values)
#             select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
#         elif args.method_select == 'greedy':
#             select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
#         select_new_ys = [new_ys[select_id] for select_id in select_ids]

#         # log
#         if to_print: 
#             sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
#             print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
#         infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
#         ys = select_new_ys
#         thoughts[step] = ys
#         nodes_num += len(ys)
#         ans = check_answer(ys)
#         if ans != None:
#             print("Find final answer!\n")
#             return x, [ans], {'steps': infos}, thoughts, nodes_num
#     if to_print: 
#         print(ys)
#     return x, ys, {'steps': infos}, thoughts, nodes_num
