import argparse
import sys
import time
# from tot.methods.bfs import solve, solve_v1
from tot.tasks.game24 import Game24Task, update_list
import torch
import transformers

model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
task = Game24Task()
y = "13 - 11 = 2"
print(update_list("1 11 12 13", y))
    # start = time.perf_counter()
    # x, y = solve_v1(args, task, 500, do_validate = False)
    # elapsed = time.perf_counter() - start
    # print(f"{elapsed:.6f} seconds")
    # if "Answer" in y:
    #     y = y[(y.find('Answer') + 7):].strip()
    # ys, infos = solve(args, task, 900)
    # print(f'check answer: {task.check_answer(x, y)}')
# print("The final answer is: \n")
# print(y)
# print(gpt_usage(model))
# print(llama_usage())