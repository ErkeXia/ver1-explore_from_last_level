import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

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



def get_proposals_v1(task, x, y, index, feedback = None): 
    print(f'Getting proposals from index {index} with y = {y}')
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    print(f'The proposals for {y} is \n {proposals}')
    return [(y + _ + '\n', index) for _ in proposals]

def get_values_v1(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y,i in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def check_answer(prev_level): #This is only for game of 24
    for y in prev_level:
        if 'Answer' in y:
            print("Found the answer! \n")
            return y
    return None

def reasoning(task, step, x, prev_level, feedback = None, single = None): 
    #if prev_level only one element(first node or refinement), single signal the index of previous thoughts
    #this should be improved
    while step < 2:
        print(f'Start reasoning with step {step}\n')
        print(f'number of prev level{len(prev_level)}')
        if(len(prev_level) > 5):
            print("Error! \n")
            print(prev_level)
            return 0
        if single == None:
            new_ys = [get_proposals_v1(task, x, y, i, feedback) for i,y in enumerate(prev_level)]
        else:
            new_ys = [get_proposals_v1(task, x, y, single, feedback) for y in prev_level]
        feedback = None
        single = None
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        
        #SLM evaluate
        values = get_values_v1(task, x, new_ys, 3)  #n_evaluate_sample=3
        #Select top ans
        select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:5] #n_select_sample=5
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        
        #log
        print(f'-- new step of {step}\n')
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f'-- new_ys --: {new_ys}\n-- values -- {values}\n-- sorted_new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        #update thoughts tree
        prev_level = [y for (y,i) in select_new_ys]
        indices = [i for (y,i) in select_new_ys]
        thoughts[step] = prev_level
        connection[step] = indices
        step += 1
        ans = check_answer(prev_level)
        if ans != None:
            return ans
    return prev_level 

def solve_v1(args, task, idx):
    global gpt
    global thoughts
    global connection
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    thoughts = [[] for _ in range(task.steps)]
    connection = [[] for _ in range(task.steps)]
    print(gpt)
    x = task.get_input(idx)  # input
    x = "4 5 10 10"
    print(f'x = {x}\n')
    prev_level = ['']
    ys = reasoning(task, 0, x, prev_level, feedback = None, single = 0)
    print("Thoughts: \n")
    for i,ts in enumerate(thoughts):
        print(f'step {i} \n')
        for t in ts:
            print(f'{t} \n')
        print(connection[i])
    print("Index: \n")
    print(connection)
    return ys
        

def solve(args, task, idx, to_print=True):
    global gpt
    global thoughts
    global connection
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    thoughts = [[] for _ in range(task.steps)]
    connection = [[] for _ in range(task.steps)]
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}