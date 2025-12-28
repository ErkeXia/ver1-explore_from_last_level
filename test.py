import argparse
import sys
import time
import openai
import os
# from tot.methods.bfs import solve, solve_v1
# from tot.methods.bfs_sys import solve_v1, check_answer, get_time
# from tot.tasks.game24 import Game24Task
# from tot.methods.dfs import solve_v1
# from tot.methods.one_step_lazy import solve_v1
from tot.methods.crossword_search import solve_v1


from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import gpt_usage, llama_usage
from tot.cache import FileCache
import torch
import transformers
# models = openai.Model.list()
# for model in models['data']:
#     print(f"Model ID: {model['id']}")
print(f"!!!Version:{transformers.__version__}")

print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should print your GPU model
model = 'gpt-3.5-turbo'
args = argparse.Namespace(backend=model, temperature=0.7, task='crossword', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = MiniCrosswordsTask(file="miniNYT.json")
# task2 = MiniCrosswordsTask()
# x = task2.get_input(0)
# print(x)

# task = Game24Task()
# y = "13 - 11 = 2 (left: 1 2 12)\n2 * 12 = 24 (left: 1 24)\n1 * 24 = 24 (left: 24)"
# print(task.check_multistep_solution("1 11 12 13", y))
# check_answer(['24', '12', '576', '10', '6'])



with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f

    # for i in range(10):
    #     idx = i
    #     x = task.get_input(idx)
    #     print(x)
        
    start = time.perf_counter()
    sol_idx, sol_y, depth, states, nodes, gpt_eval_results, iteration_details = solve_v1(args, task, 2, slm = 'llama', do_validate = False, instruct_model_arg = True)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.6f} seconds")
    # if "Answer" in y:
    #     y = y[(y.find('Answer') + 7):].strip()
    # ys, infos = solve(args, task, 900)
    # if "Answer" in y:
    #     y_clean = y[(y.find('Answer') + 7):].strip()
    # else:
    #     y_clean = y.strip()
    # print(f'check answer: {task.check_answer(x, y_clean)}')
    # print("The final answer is: \n")
    # print(y)
    # gpt_stats = gpt_usage(model)
    # llama_stats = llama_usage()
    # propose_num, value_num, validate_num, propose_avg, value_avg, validate_avg = get_time()
    # print(f'gpt_stats: {gpt_stats}')
    # print(f'llama_stats: {llama_stats}')
    # print(f'propose_num:{propose_num} value_num {value_num}, propose_avg {propose_avg}, value_avg{value_avg}')
    # print(llama_ans)
    # print(gpt_usage(model))
    # print(llama_usage())
    
    
# def run_demo():
#     """Demonstrates the basic usage of the FileCache class."""
    
#     demo_cache_file = "demo_cache.json"
#     print(f"--- Running FileCache Demo ---")
#     print(f"Using temporary cache file: '{demo_cache_file}'\n")

#     # 1. Initialize the cache.
#     # If demo_cache.json exists from a previous run, it will be loaded.
#     cache = FileCache(demo_cache_file)

#     # 2. Define a key and a value to store.
#     clue_key = "A popular fruit: a _ p _ _"
#     proposal_value = ("apple", 0.9) # Storing a tuple (word, score)

#     # 3. Use .set() to store the value. This writes to the file.
#     print(f"Setting cache for key: '{clue_key}'")
#     cache.set(clue_key, proposal_value)
#     print("Value set successfully.\n")

#     # 4. Use .get() to retrieve the stored value.
#     retrieved_value = cache.get(clue_key)
#     print(f"Retrieving value for key: '{clue_key}'")
#     print(f"  -> Retrieved: {retrieved_value}\n")

#     # 5. Try to get a key that does not exist.
#     non_existent_key = "An unknown clue: _ _ _ _ _"
#     retrieved_value = cache.get(non_existent_key)
#     print(f"Retrieving value for a non-existent key: '{non_existent_key}'")
#     print(f"  -> Retrieved: {retrieved_value} (as expected)\n")
    
#     # 6. Clean up the created cache file.
#     print(f"--- Demo Finished. Cleaning up. ---")
        
# run_demo()
