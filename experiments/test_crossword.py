import argparse
import sys
import json
import time
from tqdm import tqdm
from contextlib import redirect_stdout

# Import your core solving logic and task definition
from tot.methods.dfs_cache import solve_v1
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import reset # To reset token counters if you add them later

def run_crossword_experiment(start_idx=0, end_idx=9, log_file="crossword_output.txt", results_file="crossword_results.jsonl"):
    """
    Runs the Llama+GPT refinement solver on a range of crossword problems
    and saves the results.
    """
    print(f"Running experiment for crossword problems {start_idx} to {end_idx}.")
    print(f"Detailed logs will be saved to: {log_file}")
    print(f"Structured results will be saved to: {results_file}")

    # Standard setup for the solver's args
    args = argparse.Namespace(backend='gpt-4o', temperature=0.7)
    task = MiniCrosswordsTask()

    # Redirect all print statements to the log file
    with open(log_file, 'w', buffering=1) as f, redirect_stdout(f):
        # Use tqdm for a nice progress bar in the console
        for i in tqdm(range(start_idx, end_idx + 1), desc="Solving Crosswords"):
            # Reset any global counters if you have them (good practice)
            reset()
            
            # Start timer
            start_time = time.perf_counter()

            # --- Call your main solving function ---
            # It now returns the final node count as the fifth item
            _, final_y, depth, all_states, nodes = solve_v1(
                args, task, i, slm='llama', instruct_model_arg=True, do_validate=True
            )
            
            # Stop timer
            elapsed_time = time.perf_counter() - start_time

            # Get final reward metrics by re-evaluating the final answer
            info = task.test_output(i, final_y)

            # --- Assemble the results dictionary as requested ---
            result = {
                "problem_id": i,
                "total_time": elapsed_time,
                "final_answer": final_y,
                "final_depth": depth,
                "total_nodes_in_last_iteration": nodes,
                "r_word": info.get('r_word', 0),
                "r_letter": info.get('r_letter', 0),
                "r_game": info.get('r_game', 0),
                "all_states": all_states,
            }

            # Append the JSON object to the results file
            # This is safer than writing all at once, in case of a crash
            with open('./results/' + results_file, "a") as rf:
                rf.write(json.dumps(result) + "\n")
                
    print(f"\nExperiment finished. Results saved in '{results_file}'.")

if __name__ == '__main__':
    # This block runs when you execute the script directly
    run_crossword_experiment(start_idx=0, end_idx=9)
