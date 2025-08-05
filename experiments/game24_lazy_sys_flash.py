import argparse
import sys
from tot.methods.bfs_sys import solve_v1, get_time
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage, reset, llama_usage
import json
import time
from tqdm import tqdm
from contextlib import redirect_stdout
model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()
def test_game24_lazy(start=400, end=449):
    for i in tqdm(range(start, end + 1), desc="Testing lazy"):
        reset()
        start = time.perf_counter()
        x, y, thoughts, validators, llama_ans, nodes, all_states = solve_v1(args, task, i, slm = 'llama', do_validate = True)
        elapsed = time.perf_counter() - start
        gpt_stats = gpt_usage(model)
        llama_stats = llama_usage()
        propose_num, value_num, propose_avg, value_avg = get_time()
        result = {
            "seed": i,
            "x": x,
            "answer": y,
            "states": all_states,
            "thoughts": thoughts,
            "llama_ans": llama_ans,
            "gpt_prompt_tokens": gpt_stats["prompt_tokens"],
            "gpt_completion_tokens": gpt_stats["completion_tokens"],
            "llama_prompt_tokens": llama_stats["llama_prompt_tokens"],
            "llama_completion_tokens": llama_stats["llama_completion_tokens"],
            "propose_num": propose_num,
            "value_num": value_num,
            "propose_avg": propose_avg,
            "value_avg": value_avg,
            "total_time": elapsed,
            "validation": validators,
            "nodes": nodes
        }
        with open("./results/lazy_sys_game24_flash_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

with open('output.txt', 'w', buffering=1) as f:
    sys.stdout = f
    test_game24_lazy(351,375)
