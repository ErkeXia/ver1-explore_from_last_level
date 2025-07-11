import argparse
import sys
from tot.methods.bfs import solve, solve_v1
from tot.tasks.game24 import Game24Task
from tot.models import gpt_usage
import json
from tqdm import tqdm
from contextlib import redirect_stdout


model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()
def test_game24_range(start=800, end=900):
    results = []
    for i in tqdm(range(start, end + 1), desc="Testing Game24"):
        with open("solve_v1_output.log", "a") as f, redirect_stdout(f):
            x, y = solve_v1(args, task, i)
        if "Answer" in y:
            y_clean = y[(y.find('Answer') + 7):].strip()
        else:
            y_clean = y.strip()

        is_correct, feedback = task.check_answer(x, y_clean)
        results.append({
            'index': i,
            'input': x,
            'output': y_clean,
            'correct': is_correct,
            'feedback': feedback
        })
        tqdm.write(f"Index {i}: {'True' if is_correct else 'False'}")
    return results
results = test_game24_range()
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
