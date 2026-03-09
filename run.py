import argparse
import json
import sys
import time

from tot.methods.NYT_crossword_split import solve
from tot.models import gpt_usage, reset
from tot.tasks.NYT_crosswords import NYTCrosswordsTask


def run(args):
    task = NYTCrosswordsTask(file=args.data_file)

    with open(args.output_file, "w", buffering=1, encoding="utf-8") as f:
        sys.stdout = f

        reset()
        start = time.perf_counter()
        result = solve(
            task=task,
            idx=args.problem_idx,
            model=args.backend,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        elapsed = time.perf_counter() - start

        print(f"elapsed_sec={elapsed:.6f}")
        print(f"backend={args.backend}")
        print(f"problem_idx={args.problem_idx}")
        print(f"status={result.get('status', 'unknown')}")
        print(f"message={result.get('message', '')}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"gpt_usage={gpt_usage(args.backend)}")


def parse_args():
    p = argparse.ArgumentParser(description="NYT crossword split runner (writes to output.txt).")
    p.add_argument("--backend", type=str, choices=["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4.1"], default="gpt-4.1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--data_file", type=str, default="NYT_cw.json")
    p.add_argument("--problem_idx", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--output_file", type=str, default="output.txt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
