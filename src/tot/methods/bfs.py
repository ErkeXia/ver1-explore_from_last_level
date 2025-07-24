import itertools
import time
import copy
import numpy as np
from functools import partial
from tot.models import gpt
from tot.models import llama

propose_num = value_num = 0
propose_time = value_time = 0

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y, feedback = None): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def get_value_gpt(task, x, y, n_evaluate_sample, cache_value=True):
    global value_num, value_time
    value_num += 1
    value_prompt = task.gpt_value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    start = time.perf_counter()
    value_outputs = gpt_e(value_prompt, n=n_evaluate_sample, stop=None, max_tokens=200)
    # print(value_outputs)
    elapsed = time.perf_counter() - start
    value_time += elapsed
    value = task.gpt_value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_value(task, x, y, n_evaluate_sample, model, cache_value=True):
    global value_num, value_time
    if model == 'gpt-4.1-mini':
        return get_value_gpt(task, x, y, n_evaluate_sample, cache_value=True)
    value_num += 1
    system, user = task.value_prompt_wrap(x, y)
    if cache_value and user in task.value_cache:
        return task.value_cache[user]
    # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    num = n_evaluate_sample
    value_outputs = []
    max_attempts = 5
    attempt = 0
    
    start = time.perf_counter()
    
    while(num > 0 and attempt < max_attempts):
        outputs = llama(user, system, n=num, stop=None, max_tokens = 200)
        keywords = {'likely', 'impossible', 'sure'}
        valid_outputs = [
            s for s in outputs
            if any(k in s.strip().split('\n')[-1] for k in keywords)
        ]
        valid_count = len(valid_outputs)
        # print(f'Number of value needed is {num}, this time we have {valid_count} valid output')
        num -= valid_count
        value_outputs.extend(valid_outputs)
        attempt += 1
        
    if(attempt == max_attempts):
        print('Reach max attempts')
        
    elapsed = time.perf_counter() - start
    value_time += elapsed
    
    # print(f'The valid outputs are {value_outputs}')
    value = task.value_outputs_unwrap(x, y, value_outputs)
    # print(f'The value is {value}')
    if cache_value:
        task.value_cache[user] = value
    return value

def get_proposals_v1(task, x, y, index, model, feedback = None): 
    # print(f'Getting proposals from index {index} with y = {y}')
    global propose_num, propose_time
    propose_num += 1
    if model == 'llama':
        system, user = task.propose_prompt_wrap(x, y)
        # proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
        start = time.perf_counter()
        proposals = llama(user, system, n=1, stop=None)[0].split('\n')
        elapsed = time.perf_counter() - start
        propose_time += elapsed
        proposals = task.propose_prompt_unwrap(proposals)
    elif model == 'gpt-4.1-mini':
        propose_prompt = task.propose_prompt_wrap_4_1(x, y)
        start = time.perf_counter()
        proposals = gpt_e(propose_prompt, n=1, stop=None, max_tokens=200)[0].split('\n')
        elapsed = time.perf_counter() - start
        propose_time += elapsed
        proposals = [s for s in proposals if not ("Input" in s or "steps" in s)]
    else:
        print("Error! Model invalid!")
    print(f'The proposals for {y} is \n {proposals}')
    return [(y + _ + '\n', index , _) for _ in proposals]

def get_values_v1(task, x, ys, n_evaluate_sample, model, cache_value=True):
    values = []
    local_value_cache = {}
    for y,i,s in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            # print(f'getting value for {y}')
            value = get_value(task, x, y, n_evaluate_sample, model = model, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def validate(task, x, f_step):
    f_step.reverse()
    validate_prompt = task.validate_prompt_wrap(x, f_step)
    print(f'Validate prompt: {validate_prompt}')
    validate_outputs = gpt_v(validate_prompt, n=1, stop=None) 
    # print(validate_outputs)
    return validate_outputs[0]

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

def check_answer(prev_level): #This is only for game of 24
    for i,y in enumerate(prev_level):
        if get_current_numbers(y) == '24' or 'Answer' in y or 'answer' in y:
            print("Found the answer! \n")
            return i,y
    return 0,None

def reasoning(task, step, x, prev_level, feedback = None, single = None, model = 'llama'):
    global nodes
    #if prev_level only one element(first node or refinement), single signal the index of previous thoughts
    #this should be improved
    while step < task.steps:
        # print(f'Start reasoning with step {step}\n')
        # print(f'number of prev level{len(prev_level)}')
        if(len(prev_level) > 5):
            print("Error! \n")
            print(prev_level)
            return 0
        if single == None:
            new_ys = [get_proposals_v1(task, x, y, i, model = model, feedback = feedback) for i,y in enumerate(prev_level)]
        else:
            new_ys = [get_proposals_v1(task, x, y, single, model = model, feedback = feedback) for y in prev_level]
        feedback = None
        single = None
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        
        #SLM evaluate
        values = get_values_v1(task, x, new_ys, 3, model = model)  #n_evaluate_sample=3
        #Select top ans
        select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:5] #n_select_sample=5
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        
        #log
        print(f'-- new step of {step}\n')
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {new_ys}\n-- values -- {values}\n-- sorted_new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        #update thoughts tree
        prev_level = [y for (y,i,s) in select_new_ys]
        indices = [i for (y,i,s) in select_new_ys]
        reasoning_steps = [s for (y,i,s) in select_new_ys]
        
        thoughts[step] = prev_level
        connection[step] = indices
        steps[step] = reasoning_steps
        nodes += len(prev_level)
        
        step += 1
        idx, ans = check_answer(prev_level)
        if ans != None:
            print("Find final answer!\n")
            return idx, ans, step
    print("Could not find answer, return most probable steps\n")
    return 0, prev_level[0], step

def retrieve_steps(num_steps, idx, y):
    step = num_steps - 1
    thought_chain = []
    chain_index = []
    while step >= 0:
        thought_chain.append(steps[step][idx])
        chain_index.append(idx)
        idx = connection[step][idx]
        step -= 1
    return thought_chain, chain_index

def solve_v1(args, task, idx, do_validate = True, slm = 'llama'):
    global gpt_e, gpt_v
    global thoughts
    global connection
    global steps
    global value_num, value_time
    global propose_num, propose_time
    global nodes
    if slm != 'llama':
        gpt_e = partial(gpt, model=slm, temperature=args.temperature)
    gpt_v = partial(gpt, model=args.backend, temperature=args.temperature)
    
    nodes = 1
    propose_num = value_num = 0
    propose_time = value_time = 0
    
    thoughts = [[] for _ in range(task.steps)]
    connection = [[] for _ in range(task.steps)]
    steps = [[] for _ in range(task.steps)]
    
    validators = []
    all_thoughts = []
    llama_ans = []
    
    print(gpt_e)
    
    x = task.get_input(idx)  # input
    print(f'x = {x}\n')
    
    prev_level = ['']
    val_count = 0
    step = 0
    single = 0
    while(val_count < 3): # call large model for at most three times
        idx, y, st = reasoning(task, step, x, prev_level, feedback = None, single = single, model = slm)
        all_thoughts.append(copy.deepcopy(thoughts))
        
        thought_chain, chain_index = retrieve_steps(st, idx, y)
        chain_index.reverse()
        llama_ans.append(thought_chain)
        print(f'Retrieve steps: {thought_chain} \n Chainindex: {chain_index}')
        
        if not do_validate:
            print(f"Output from reasoning! idx: {idx} \n y: {y} \n st: {st}")
            return x, y, all_thoughts, validators, llama_ans, nodes

        validate_outputs = validate(task, x, thought_chain)
        validators.append(validate_outputs)
        redo_s, feedback = task.validate_unwrap(validate_outputs)
        print(f'redo{redo_s} feedback: {feedback}')
        if(redo_s == -1):
            return x, feedback, all_thoughts, validators, llama_ans, nodes
        if(feedback != ""):
            prev_level = [feedback]
            step = redo_s + 1
            single = chain_index[step - 1]
            thoughts[redo_s][single] = feedback
            steps[redo_s][single] = feedback
        else:
            if(redo_s == 0):
                prev_level = ['']
                single = 0
            else:
                prev_level = thoughts[redo_s - 1]
                single = None
            step = redo_s
        print(f'prev_level {prev_level} \nstep {step}\nsingle{single if single else -1}')
        print(f'The validate result: \n {validate_outputs}\n')
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
            
    return x, feedback, all_thoughts, validators, llama_ans, nodes
      
def get_time():
    global propose_num, propose_time, value_num, value_time
    print(f"propose num: {propose_num}, propose time per num: {(propose_time/propose_num):.6f}")
    print(f"value num: {value_num}, value time per num: {(value_time/value_num):.6f}")
    return propose_num, value_num, (propose_time/propose_num), (value_time/value_num)  

# def solve(args, task, idx, to_print=True):
#     global gpt
#     global thoughts
#     global connection
#     gpt = partial(gpt, model=args.backend, temperature=args.temperature)
#     thoughts = [[] for _ in range(task.steps)]
#     connection = [[] for _ in range(task.steps)]
#     print(gpt)
#     x = task.get_input(idx)  # input
#     ys = ['']  # current output candidates
#     infos = []
#     for step in range(task.steps):
#         # generation
#         if args.method_generate == 'sample':
#             new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
#         elif args.method_generate == 'propose':
#             new_ys = [get_proposals(task, x, y) for y in ys]
#         new_ys = list(itertools.chain(*new_ys))
#         ids = list(range(len(new_ys)))
#         # evaluation
#         if args.method_evaluate == 'vote':
#             values = get_votes(task, x, new_ys, args.n_evaluate_sample)
#         elif args.method_evaluate == 'value':
#             values = get_values(task, x, new_ys, args.n_evaluate_sample)

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
    
#     if to_print: 
#         print(ys)
#     return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
