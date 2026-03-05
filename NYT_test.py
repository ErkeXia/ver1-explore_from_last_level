import argparse
import sys
import time

from tot.methods.NYT_crossword_search_iterative_propose import solve_v2
from tot.tasks.NYT_crosswords import NYTCrosswordsTask
from tot.models import reset, gpt_usage, llama_usage


def main():
    model = "gpt-3.5-turbo"
    args = argparse.Namespace(
        backend=model,
        temperature=0.7,
    )

    task = NYTCrosswordsTask(file="NYT_cw.json")

    # Single-case settings
    problem_idx = 0
    slm = "qwen"
    instruct_model = True

    with open("output.txt", "w", buffering=1, encoding="utf-8") as f:
        sys.stdout = f

        reset()
        start = time.perf_counter()

        sol_idx, sol_y, depth, states, nodes, gpt_eval_results, iteration_details = solve_v2(
            args=args,
            task=task,
            idx=problem_idx,
            slm=slm,
            instruct_model_arg=instruct_model,
            max_rounds=10,
            no_progress_limit=2,
        )

        elapsed = time.perf_counter() - start
        info = task.test_output(problem_idx, sol_y)

        print(f"elapsed_sec={elapsed:.6f}")
        print(f"solution_index={sol_idx}")
        print(f"depth={depth}")
        print(f"nodes={nodes}")
        print(f"r_word={info.get('r_word', 0.0):.4f}")
        print(f"r_letter={info.get('r_letter', 0.0):.4f}")
        print(f"r_game={info.get('r_game', 0)}")
        print(f"final_y={sol_y}")
        print(f"iterations={len(iteration_details)}")
        print(f"iteration_details={iteration_details}")
        print(f"gpt_eval_results={gpt_eval_results}")
        print(f"gpt_usage={gpt_usage(model)}")
        print(f"slm_usage={llama_usage()}")


if __name__ == "__main__":
    main()
