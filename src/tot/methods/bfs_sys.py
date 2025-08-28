import itertools
import time
import copy
import numpy as np
from functools import partial
from tot.models import gpt, llama_instruct, model_setup, base_model

propose_num = value_num = 0
propose_time = value_time = 0

def chunk_list(input_list, ind = 2):
    for i in range(0, len(input_list), ind):
        yield input_list[i:i+ind]

def get_value_batch(task, x, s_list, n_evaluate_sample):
    global value_num, value_time, repeat_value
    value_num += len(s_list)
    
    if len(s_list) == 0:
        return [], [] 
    
    prompts = [task.value_sys_prompt_wrap(x, s) for s in s_list]
    # if cache_value and user in task.value_cache:
    #     print(f'cache')
    #     return task.value_cache[user]
    # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    system, _ = prompts[0]
    
    users = [user for _, user in prompts]
    
    valid_counts = []
    valid_outputs_lst = []
    
    
    keywords = {'likely', 'impossible', 'sure'}
    for user_list in chunk_list(users):
        print(f'Batch value for {user_list}')
        outputs_B = llama_instruct(user_list, system, n=n_evaluate_sample, stop=None, max_tokens = 200, query_task = 'value')
        for outputs in outputs_B:
            valid_outputs = [
                o for o in outputs
                if any(k in o.strip().split('\n')[-1] for k in keywords)
            ]
            valid_counts.append(len(valid_outputs))
            valid_outputs_lst.append(valid_outputs)
            
    assert(len(users) == len(valid_counts))
    remains = [n_evaluate_sample - a for a in valid_counts]
    return valid_outputs_lst, remains


def get_value_outputs(task, x, s, n_evaluate_sample, cache_value=True):
    global value_num, value_time, repeat_value
    # value_num += 1
    
    system, user = task.value_sys_prompt_wrap(x, s)
    # if cache_value and user in task.value_cache:
    #     print(f'cache')
    #     return task.value_cache[user]
    # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    num = n_evaluate_sample
    value_outputs = []
    max_attempts = 5
    attempt = 0
    
    start = time.perf_counter()
    
    while(num > 0 and attempt < max_attempts):
        outputs = llama_instruct([user], system, n=num, stop=None, max_tokens = 200, query_task = 'value')[0]
        keywords = {'likely', 'impossible', 'sure'}
        valid_outputs = [
            o for o in outputs
            if any(k in o.strip().split('\n')[-1] for k in keywords)
        ]
        valid_count = len(valid_outputs)
        # print(f'Number of value needed is {num}, this time we have {valid_count} valid output')
        num -= valid_count
        value_outputs.extend(valid_outputs)
        attempt += 1
    
    repeat_value += (attempt - 1)    
    if(attempt == max_attempts):
        print('Reach max attempts')
        
    elapsed = time.perf_counter() - start
    value_time += elapsed
    
    # print(f'The valid outputs are {value_outputs}')
    # value = task.value_outputs_unwrap(x, "", value_outputs)
    # print(f'The value is {value}')
    # if cache_value:
    #     task.value_cache[user] = value
    return value_outputs

def get_values_v2(task, x, ys, n_evaluate_sample, cache_value=True):
    global value_num
    value_num += len(ys)
    
    s_list = [s for _,_,s in ys]
    valid_outputs_lst, remains = get_value_batch(task, x, s_list, n_evaluate_sample)
    
    values = []
    
    print(f'--Remains-- {remains}')

    for idx, remain in enumerate(remains):
        if remain > 0:
            value_outputs = get_value_outputs(task, x, s_list[idx], remain, cache_value=cache_value)
            valid_outputs_lst[idx].extend(value_outputs)
        assert(len(valid_outputs_lst[idx]) == n_evaluate_sample)
        value = task.value_outputs_unwrap(x, "", valid_outputs_lst[idx])
        print(f'Get value for s: {s_list[idx]} value: {value}')
        values.append(value)

    return values

def get_value(task, x, s, n_evaluate_sample, cache_value=True):
    global value_num, value_time, repeat_value
    value_num += 1
    
    num = n_evaluate_sample
    value_outputs = []
    max_attempts = 5
    attempt = 0
    
    start = time.perf_counter()
    
    KEYWORDS = {"likely", "impossible", "sure"}

    if instruct_model:
        system, user = task.value_sys_prompt_wrap_instruct(x, s)
        if cache_value and user in task.value_cache:
            print("cache")
            return task.value_cache[user]

        gen = lambda n: llama_instruct(user, system, n=n, stop=None, max_tokens=200, query_task="value")
    else:
        prompt = task.value_sys_prompt_wrap(x, s)
        if cache_value and s in task.value_cache:
            print("cache")
            return task.value_cache[s]

        gen = lambda n: base_model(prompt, n=n, max_tokens=200)

    while num > 0 and attempt < max_attempts:
        outputs = gen(num)
        valid_outputs = [
            o for o in outputs
            if any(k in o.strip().split("\n")[-1] for k in KEYWORDS)
        ]
        valid_count = len(valid_outputs)
        num -= valid_count
        value_outputs.extend(valid_outputs)
        attempt += 1
    
    repeat_value += (attempt - 1)    
    if(attempt == max_attempts):
        print('Reach max attempts')
        
    elapsed = time.perf_counter() - start
    value_time += elapsed
    
    value = task.value_outputs_unwrap(x, "", value_outputs)
    if cache_value:
        task.value_cache[user] = value
    return value

def get_values_v1(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for p,i,s in ys:  # each partial output
        if s in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, s, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[s] = value
        print(f'Get value for p: {p}  s: {s} value: {value}')
        values.append(value)
    return values

def get_proposals_v1(task, current, index, feedback = None): 
    print(f'\nGetting proposals for index {index} with current = {current}')
    global propose_num, propose_time
    propose_num += 1
    start = time.perf_counter()
    
    if instruct_model:
        
        system, user = task.propose_sys_prompt_wrap_instruct(current)
        proposals = llama_instruct(user, system, n=1, stop=None, query_task = 'propose')[0].split('\n')
    else:
        prompt = task.propose_sys_prompt_wrap(current)
        proposals = base_model(prompt, n=1, max_tokens=200)[0].split('\n')
    # print(proposals)
    elapsed = time.perf_counter() - start
    propose_time += elapsed
    proposals, states_new = task.propose_prompt_unwrap(current, proposals)
    # print(f'The proposals for {y} is \n {proposals}')
    return [(proposal, index, state_new) for proposal, state_new in zip(proposals, states_new)]

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

def check_answer(reasoning_steps): #This is only for game of 24
    for i,s in enumerate(reasoning_steps):
        if len(s.split()) == 1 and float(s.strip()) == 24:
            print("Found the answer! \n")
            return i,s
    return 0,None

def reasoning(task, step, x, feedback = None, single = None):
    global nodes
    #if prev_level only one element(first node or refinement), single signal the index of previous thoughts
    #this should be improved
    while step < 3:
        new_ys = [get_proposals_v1(task, state['current'], i, feedback) for i, state in enumerate(states[step])]
        # else:
        # new_ys = [get_proposals_v1(task, x, y, single, feedback) for state in states[step]]
        feedback = None
        single = None
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        
        #SLM evaluate
        print(f'--Get value for--: {new_ys}')
        values = get_values_v1(task, x, new_ys, 3)  #n_evaluate_sample=3
        #Select top ans
        select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:5] #n_select_sample=5
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        #log
        # print(f'-- new step of {step}\n')
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {new_ys}\n-- values -- {values}\n-- sorted_new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        #update thoughts tree
        states[step + 1] = [{'step': p, 'connect': i, 'current': s} for (p, i, s) in select_new_ys]
        prev_level = [y for (y,i,s) in select_new_ys]
        indices = [i for (y,i,s) in select_new_ys]
        reasoning_steps = [s for (y,i,s) in select_new_ys]
        
        print(f"--Reasoning-- {reasoning_steps}")
        
        thoughts[step] = prev_level
        connection[step] = indices
        steps[step] = reasoning_steps
        nodes += len(prev_level)
        
        step += 1
        print(f'--Step--: {step}')
        idx, ans = check_answer(reasoning_steps)
        if ans != None:
            print("Find final answer!\n")
            return idx, ans, step
    print("Could not find answer, return most probable steps\n")
    return 0, states[step][0]['step'], step

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

def correctness_check(task, x, f_step):
    global validate_time, correctness_r
    correctness_prompt, locate_prompt = task.s3_evaluate_sys_prompt_wrap(x, f_step)
    correctness_output = gpt(correctness_prompt, n=1, stop=None)[0]
    print(f'correctness output: {correctness_output}')
    correctness_r.append(correctness_output)
    correctness = task.correctness_unwrap(correctness_output)
    if correctness == 'No':
        return 0, ""
    return -1, correctness
    

def evaluate_3steps(task, x, f_step):
    global validate_time, correctness_r, locate_r, suggestion_r
    start = time.perf_counter()
    correctness_prompt, locate_prompt = task.s3_evaluate_sys_prompt_wrap(x, f_step)
    
    correctness_output = gpt(correctness_prompt, n=1, stop=None)[0]
    print(f'correctness output: {correctness_output}')
    correctness_r.append(correctness_output)
    correctness = task.correctness_unwrap(correctness_output)
    
    if correctness == 'No':
        locate_output = gpt(locate_prompt, n=1, stop=None)[0]
        print(f'locate output: {locate_output}')
        wrong_step, current_numbers = task.locate_unwrap(locate_output)
        print(f'wrong step: {wrong_step}, current numbers: {current_numbers}')
        
        gpt_propose_prompt = task.eval_gpt_propose_prompt_wrap(current_numbers)
        proposals = gpt(gpt_propose_prompt, n=1, stop = None)[0]
        print(f"Gpt proposals {proposals}")
        redo_s = wrong_step - 1
        feedback = proposals
        
        locate_r.append(locate_output)
        suggestion_r.append(proposals)
        
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

def solve_v1(args, task, idx, slm = 'llama', do_validate = True):
    global gpt
    global thoughts, connection, steps, validators, correctness_r, suggestion_r, locate_r
    global value_num, value_time
    global propose_num, propose_time
    global validate_num, validate_time
    global nodes
    global states, repeat_value
    global instruct_model
    
    instruct_model = False
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    model_setup(slm, TGI_arg = False)
    if 'instruct' in slm.lower():
        instruct_model = True
        
    
    nodes = 1
    propose_num = value_num = 0
    propose_time = value_time = 0
    validate_num = validate_time = 0
    repeat_value = 0
    
    thoughts = [[] for _ in range(task.steps)]
    connection = [[] for _ in range(task.steps)]
    steps = [[] for _ in range(task.steps)]
    states = [[] for _ in range(task.steps + 1)]
    
    all_states = []
    validators = []
    correctness_r = []
    suggestion_r = []
    locate_r = []
    all_thoughts = []
    llama_ans = []
    
    print(gpt)
    
    x = task.get_input(idx)  # input
    print(f'x = {x}\n')
    
    states[0].append({'step': '', 'connect': 0, 'current': x})
    print(states[0])
    
    # prev_level = ['']
    val_count = 0
    step = 0
    single = 0
    while(val_count < 3): # call large model for at most three times
        idx, y, st = reasoning(task, step, x, feedback = None, single = single)
        print(f'--States-- {states}')
        
        all_thoughts.append(copy.deepcopy(thoughts))
        all_states.append(copy.deepcopy(states))
        intermediate_state, thought_chain, chain_index = retrieve_steps(st, idx, y)
        llama_ans.append(thought_chain)
        print(f'Retrieve steps: {thought_chain} \n Chainindex: {chain_index}')
        if not do_validate:
            print(f"Output from reasoning! idx: {idx} \n y: {y} \n st: {st}")
            evaluation = {'validators': validators, 'correctness': correctness_r,'locate': locate_r, 'suggestions': suggestion_r}
            return x, thought_chain, all_thoughts, evaluation, llama_ans, nodes, all_states

        validate_num += 1
        redo_s, feedback = evaluate_3steps(task, x, thought_chain)
        print(f'redo {redo_s} feedback: {feedback}')
        
        possible_steps = [p for p in feedback.split("\n") if any(ch.isdigit() for ch in p)]
        
        if(redo_s == -1):
            break
        if(feedback != ""):
            # prev_level = [feedback]
            prev_level = possible_steps
            step = redo_s + 1
            prev_idx = chain_index[redo_s]
            print(f'possible steps: {possible_steps}, connect: {prev_idx}')
            # states[step] = [{'step': feedback, 'connect': prev_idx, 'current': task.manage_state(states[redo_s][prev_idx]["current"], feedback)}]
            states[step] = [{'step': possible_step, 'connect': prev_idx, 'current': task.manage_state(states[redo_s][prev_idx]["current"], possible_step)} for possible_step in possible_steps]
            print(f"before thoughts{thoughts} steps{steps}")
            # thoughts[redo_s][prev_idx] = feedback
            # steps[redo_s][prev_idx] = feedback
            thoughts[redo_s] = possible_steps
            steps[redo_s] = possible_steps
            print(f"after thoughts{thoughts} steps{steps}")
        else:
            if(redo_s == 0):
                prev_level = ['']
                single = 0
            else:
                prev_level = thoughts[redo_s - 1]
                single = None
            step = redo_s
        print(f'prev_level {prev_level} \nstep {step}\nsingle{single if single else -1}')
        # print(f'The validate result: \n {validate_outputs}\n')
        val_count += 1
        
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
    print(f'Repeat value time: {repeat_value}')
    avg_validate = validate_time/validate_num
    print(f'validate average time: {avg_validate}')
    evaluation = {'validators': validators, 'correctness': correctness_r, 'locate': locate_r, 'suggestions': suggestion_r}
    return x, feedback, all_thoughts, evaluation, llama_ans, nodes, all_states
      
def get_time():
    global propose_num, propose_time, value_num, value_time
    print(f"propose num: {propose_num}, propose time per num: {(propose_time/propose_num):.6f}")
    print(f"value num: {value_num}, value time per num: {(value_time/value_num):.6f}")
    return propose_num, value_num, validate_num, (propose_time/propose_num), (value_time/value_num), (validate_time/validate_num)

