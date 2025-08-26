import argparse
import sys
import time
import openai
# from tot.methods.bfs import solve, solve_v1
from tot.methods.dfs import solve
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import gpt
from functools import partial
import torch
import transformers

model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='crossword', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
gpt = partial(gpt, model=args.backend, temperature=args.temperature)

def get_proposal(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposal = gpt(propose_prompt, n=1, stop=None, max_tokens=200)
    # proposal= proposal[0].split('\n')
    print(proposal)
    candidates = task.propose_outputs_unwrap(x, y, proposal, n_max_propose=5)
    print(candidates)
    return proposal


task = MiniCrosswordsTask()
x = task.get_input(0)
print(x)
y = ""
proposal = get_proposal(task, x, y)
