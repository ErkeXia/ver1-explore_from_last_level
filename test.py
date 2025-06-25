import argparse
import sys
from tot.methods.bfs import solve, solve_v1
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage, llmaa_usage

with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    model = 'gpt-4'
    args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

    task = Game24Task()
    x, y = solve_v1(args, task, 800)
    if "Answer" in y:
        y = y[(y.find('Answer') + 7):].strip()
    # ys, infos = solve(args, task, 900)
    print(f'check answer: {task.check_answer(x, y)}')
    print("The final answer is: \n")
    print(y)
    print(gpt_usage(model))
    print(llmaa_usage())