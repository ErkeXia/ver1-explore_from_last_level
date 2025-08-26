import argparse
import sys
import time
import openai
# from tot.methods.bfs import solve, solve_v1
from tot.methods.bfs_sys import solve_v1, check_answer, get_time
from tot.tasks.game24 import Game24Task
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import gpt_usage, llama_usage
import torch
import transformers
# models = openai.Model.list()
# for model in models['data']:
#     print(f"Model ID: {model['id']}")
print(f"!!!Version:{transformers.__version__}")

print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should print your GPU model
model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

# task2 = MiniCrosswordsTask()
# x = task2.get_input(0)
# print(x)

task = Game24Task()
y = "13 - 11 = 2 (left: 1 2 12)\n2 * 12 = 24 (left: 1 24)\n1 * 24 = 24 (left: 24)"
# print(task.check_multistep_solution("1 11 12 13", y))
# check_answer(['24', '12', '576', '10', '6'])
with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    start = time.perf_counter()
    x, y, all_thoughts, validators, llama_ans, nodes, all_states = solve_v1(args, task, 352, slm = 'llama', do_validate = True)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.6f} seconds")
    # if "Answer" in y:
    #     y = y[(y.find('Answer') + 7):].strip()
    # ys, infos = solve(args, task, 900)
    if "Answer" in y:
        y_clean = y[(y.find('Answer') + 7):].strip()
    else:
        y_clean = y.strip()
    print(f'check answer: {task.check_answer(x, y_clean)}')
    print("The final answer is: \n")
    print(y)
    gpt_stats = gpt_usage(model)
    llama_stats = llama_usage()
    propose_num, value_num, validate_num, propose_avg, value_avg, validate_avg = get_time()
    print(f'gpt_stats: {gpt_stats}')
    print(f'llama_stats: {llama_stats}')
    print(f'propose_num:{propose_num} value_num {value_num}, propose_avg {propose_avg}, value_avg{value_avg}')
    print(llama_ans)
    print(gpt_usage(model))
    print(llama_usage())