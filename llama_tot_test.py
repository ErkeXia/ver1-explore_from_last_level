import argparse
import sys
from tot.methods.bfs import solve_v1, get_time
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage, reset, llmaa_usage
import json
from tqdm import tqdm
from contextlib import redirect_stdout


model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
task = Game24Task()
def test_game24_llama_tot(start=400, end=449):
    for i in tqdm(range(start, end + 1), desc="Testing llama tot"):
        reset()
        start = time.perf_counter()
        x, y, thoughts = solve_v1(args, task, i, do_validate = False)
        elapsed = time.perf_counter() - start
        gpt_stats = gpt_usage(model)
        llama_stats = llama_usage(model)
        propose_num, value_num, propose_avg, value_avg = get_time()
        result = {
            "seed": i,
            "x": x,
            "answer": y,
            "thoughts": thoughts,
            "gpt_prompt_tokens": gpt_stats["prompt_tokens"],
            "gpt_completion_tokens": gpt_stats["completion_tokens"],
            "llama_prompt_tokens": llama_stats["llama_prompt_tokens"],
            "llama_completion_tokens": llama_stats["llama_completion_tokens"],
            "propose_num": propose_num,
            "value_num": value_num,
            "propose_avg": propose_avg,
            "value_avg": value_avg,
            "total_time": elapsed
        }
        with open("llama_tot_results_ver2.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

def test_game24_lazy(start=400, end=449):
    for i in tqdm(range(start, end + 1), desc="Testing lazy"):
        reset()
        start = time.perf_counter()
        x, y, thoughts = solve_v1(args, task, i, do_validate = True)
        elapsed = time.perf_counter() - start
        gpt_stats = gpt_usage(model)
        llama_stats = llama_usage(model)
        propose_num, value_num, propose_avg, value_avg = get_time()
        result = {
            "seed": i,
            "x": x,
            "answer": y,
            "thoughts": thoughts,
            "gpt_prompt_tokens": gpt_stats["prompt_tokens"],
            "gpt_completion_tokens": gpt_stats["completion_tokens"],
            "llama_prompt_tokens": llama_stats["llama_prompt_tokens"],
            "llama_completion_tokens": llama_stats["llama_completion_tokens"],
            "propose_num": propose_num,
            "value_num": value_num,
            "propose_avg": propose_avg,
            "value_avg": value_avg,
            "total_time": elapsed
        }
        with open("lazy_game24_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
test_game24_llama_tot(300,300)
test_game24_lazy(300,300)

# with open("gpt4_results.json", "w") as f:
#     json.dump(results, f, indent=2)
