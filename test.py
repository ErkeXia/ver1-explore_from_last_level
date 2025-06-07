import argparse
import sys
from tot.methods.bfs import solve, solve_v1
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage

with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    model = 'gpt-3.5-turbo'
    args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    task = Game24Task()
    ys = solve_v1(args, task, 900)
    # ys, infos = solve(args, task, 900)

    print("The final answer is: \n")
    print(ys)
    print(gpt_usage(model))