import argparse
import sys
import json
import time
import os
from tqdm import tqdm
from contextlib import redirect_stdout

# Import your core solving logic and task definition
# from tot.methods.dfs_cache import solve_v1
from tot.methods.crossword_search import solve_v1
from tot.tasks.crosswords import MiniCrosswordsTask
from tot.models import reset # To reset token counters if you add them later

def run_crossword_experiment(indices=None, start_idx=0, end_idx=9, log_file="./logs/llama_valid_crossword_output.txt", results_file="./results/llama_valid_crossword_results.jsonl", data = None):
    """
    Runs the Llama+GPT refinement solver on a specific list or range of crossword problems.
    
    Args:
        indices (list[int], optional): A specific list of problem IDs to run. 
                                       If None, uses start_idx and end_idx.
        start_idx (int): Start index for range (used if indices is None).
        end_idx (int): End index for range (used if indices is None).
    """
    # Determine which puzzles to run
    if indices is not None:
        puzzles_to_run = indices
        print(f"Running experiment for {len(indices)} specific puzzles: {indices}")
    else:
        puzzles_to_run = range(start_idx, end_idx + 1)
        print(f"Running experiment for crossword problems {start_idx} to {end_idx}.")
        
    print(f"Detailed logs will be saved to: {log_file}")
    print(f"Structured results will be saved to: {results_file}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Standard setup for the solver's args
    args = argparse.Namespace(backend='gpt-3.5-turbo', temperature=0.7)
    if data != None:
        task = MiniCrosswordsTask(data)
    else:
        task = MiniCrosswordsTask()

    # Redirect all print statements to the log file
    with open(log_file, 'w', buffering=1) as f, redirect_stdout(f):
        # Use tqdm for a nice progress bar in the console
        for i in tqdm(puzzles_to_run, desc="Solving Crosswords"):
            # Reset any global counters if you have them (good practice)
            reset()
            
            # Start timer
            start_time = time.perf_counter()

            # --- Call your main solving function ---
            # NOTE: Ensure 'do_validate' matches your desired experiment mode (True for refinement, False for single-pass)
            _, final_y, depth, all_states, nodes, gpt_results, iteration_details = solve_v1(
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
                "gpt_eval_results": gpt_results,
                "iteration_details": iteration_details,
                "all_states": all_states,
            }

            # Append the JSON object to the results file
            # This is safer than writing all at once, in case of a crash
            with open(results_file, "a") as rf:
                rf.write(json.dumps(result) + "\n")
                
    print(f"\nExperiment finished. Results saved in '{results_file}'.")

if __name__ == '__main__':
    # The specific list of puzzles you want to test
    # TARGET_INDICES = [40, 113, 117, 124, 2, 12, 15, 16, 38, 44]
    TARGET_INDICES = [0, 1, 2, 3, 4, 5]
    # Run the experiment with this list
    log_file="./logs/llama_NYT_crossword_search_inorder_output.txt"
    results_file="./results/llama_NYT_crossword_search_inorder_results.jsonl"
    task = "miniNYT.json"
    run_crossword_experiment(indices=TARGET_INDICES, log_file = log_file, results_file = results_file, data = task)