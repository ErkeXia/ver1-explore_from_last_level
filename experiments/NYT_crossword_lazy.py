"""
NYT Crossword Lazy Experiment

What this tests:
- Task: NYT crossword puzzles from `src/tot/data/crosswords/NYT_cw.json` (or `--data_file`).
- Solver: `tot.methods.NYT_crossword_search.solve_v1`.
- State representation: JSON list of entry fills keyed by `entry_id`.

Model roles:
- Propose model (SLM): controlled by `--slm` (default `qwen`, which maps to Qwen/Qwen3-8B in `model_setup`).
- Evaluation/pruning model: controlled by `--eval_model` (default `gpt`).
- GPT backend for evaluation/value calls: `--backend` (default `gpt-3.5-turbo`).
- Value scoring uses the same model family configured by the solver run (`solve_v1` internals).

Outputs:
- Log file: `./logs/NYT_crossword_lazy_qwen_output.txt` (default, configurable).
- Result file: `./results/NYT_crossword_lazy_qwen_results.jsonl` (default, configurable).
- Result persistence mode: JSONL append, one JSON object written immediately after each puzzle finishes.
  This preserves completed puzzle results even if the run is interrupted later.
"""

import argparse
import json
import os
import time
from contextlib import redirect_stdout

from tqdm import tqdm

from tot.methods.NYT_crossword_search import solve_v1
from tot.tasks.NYT_crosswords import NYTCrosswordsTask
from tot.models import reset, gpt_usage, llama_usage


def upsert_checkpoint_record(jsonl_path: str, checkpoint_record: dict):
    """
    Keep at most one checkpoint record per problem_id in the JSONL file.
    Non-checkpoint lines are preserved.
    """
    pid = checkpoint_record.get("problem_id")
    existing = []

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # Preserve malformed lines as-is to avoid destructive behavior.
                    existing.append(raw.rstrip("\n"))
                    continue

                is_same_checkpoint = (
                    isinstance(obj, dict)
                    and obj.get("record_type") == "checkpoint"
                    and obj.get("problem_id") == pid
                )
                if not is_same_checkpoint:
                    existing.append(json.dumps(obj, ensure_ascii=False))

    existing.append(json.dumps(checkpoint_record, ensure_ascii=False))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ln in existing:
            f.write(ln + "\n")


def parse_index_list(raw: str):
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out if out else None


def run_nyt_crossword_experiment(
    data_file: str = "NYT_cw.json",
    indices=None,
    start_idx: int = 0,
    end_idx: int = 9,
    backend: str = "gpt-3.5-turbo",
    slm: str = "qwen",
    eval_model: str = "gpt",
    temperature: float = 0.7,
    do_validate: bool = True,
    instruct_model: bool = True,
    checkpoint_every: int = 0,
    log_file: str = "./logs/NYT_crossword_lazy_qwen_output.txt",
    results_file: str = "./results/NYT_crossword_lazy_qwen_results.jsonl",
):
    if indices is not None:
        puzzles_to_run = list(indices)
    else:
        puzzles_to_run = list(range(start_idx, end_idx + 1))

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    task = NYTCrosswordsTask(file=data_file)
    args = argparse.Namespace(backend=backend, temperature=temperature)

    run_meta = {
        "task_file": data_file,
        "slm": slm,
        "eval_model": eval_model,
        "backend": backend,
        "temperature": temperature,
        "do_validate": do_validate,
        "indices": puzzles_to_run,
    }

    # Persist one run-meta record for traceability.
    with open(results_file, "a", encoding="utf-8") as rf:
        rf.write(json.dumps({"record_type": "run_meta", **run_meta}, ensure_ascii=False) + "\n")

    with open(log_file, "w", buffering=1, encoding="utf-8") as lf, redirect_stdout(lf):
        print("=== NYT Crossword Lazy Experiment ===")
        print(json.dumps(run_meta, indent=2))
        print(f"Total puzzles: {len(puzzles_to_run)}")

        def checkpoint_writer(payload: dict):
            rec = {"record_type": "checkpoint", **payload}
            upsert_checkpoint_record(results_file, rec)

        ckpt_fn = checkpoint_writer if checkpoint_every and checkpoint_every > 0 else None

        try:
            for idx in tqdm(puzzles_to_run, desc="Solving NYT Crosswords"):
                reset()
                start = time.perf_counter()

                sol_idx, sol_y, depth, all_states, nodes, gpt_eval_results, iteration_details = solve_v1(
                    args=args,
                    task=task,
                    idx=idx,
                    slm=slm,
                    eval_model=eval_model,
                    instruct_model_arg=instruct_model,
                    do_validate=do_validate,
                    checkpoint_fn=ckpt_fn,
                    checkpoint_every=checkpoint_every,
                )

                elapsed = time.perf_counter() - start
                info = task.test_output(idx, sol_y)

                result = {
                    "record_type": "puzzle_result",
                    "problem_id": idx,
                    "solve_time_sec": elapsed,
                    "solution_index": sol_idx,
                    "final_state_y": sol_y,
                    "final_depth": depth,
                    "total_nodes_in_last_iteration": nodes,
                    "metrics": {
                        "r_word": info.get("r_word", 0.0),
                        "r_letter": info.get("r_letter", 0.0),
                        "r_game": info.get("r_game", 0),
                    },
                    "gpt_eval_results": gpt_eval_results,
                    "iteration_details": iteration_details,
                    "all_states": all_states,
                    "token_usage": {
                        "gpt": gpt_usage(backend),
                        "slm": llama_usage(),
                    },
                }

                # Append immediately so completed puzzle results survive interruptions.
                with open(results_file, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps(result, ensure_ascii=False) + "\n")
        except KeyboardInterrupt:
            print("\n[Interrupted] Received Ctrl+C. Completed puzzle results are already persisted in JSONL.")
            raise

    print(f"Finished. Logs: {log_file}")
    print(f"Results JSONL: {results_file}")


def build_parser():
    p = argparse.ArgumentParser(description="Run NYT crossword solver experiment with Qwen proposer + GPT eval.")
    p.add_argument("--data_file", type=str, default="NYT_cw.json")
    p.add_argument("--indices", type=str, default="", help='Comma-separated puzzle ids, e.g. "0,1,2"')
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=9)
    p.add_argument("--backend", type=str, default="gpt-3.5-turbo", help="GPT backend for evaluation calls.")
    p.add_argument("--slm", type=str, default="qwen", help="Small model proposer in model_setup (qwen/llama/...).")
    p.add_argument("--eval_model", type=str, default="gpt", help='Use "gpt" to evaluate with backend above.')
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--checkpoint_every", type=int, default=0, help="Emit DFS checkpoint every N created nodes (0 disables).")
    p.add_argument("--do_validate", action="store_true", default=True)
    p.add_argument("--no_validate", action="store_true")
    p.add_argument("--instruct_model", action="store_true", default=True)
    p.add_argument("--base_model", action="store_true")
    p.add_argument("--log_file", type=str, default="./logs/NYT_crossword_lazy_qwen_output.txt")
    p.add_argument("--results_file", type=str, default="./results/NYT_crossword_lazy_qwen_results.jsonl")
    return p


if __name__ == "__main__":
    parser = build_parser()
    cli = parser.parse_args()

    idx_list = parse_index_list(cli.indices)
    do_validate = False if cli.no_validate else cli.do_validate
    instruct_model = False if cli.base_model else cli.instruct_model

    run_nyt_crossword_experiment(
        data_file=cli.data_file,
        indices=idx_list,
        start_idx=cli.start_idx,
        end_idx=cli.end_idx,
        backend=cli.backend,
        slm=cli.slm,
        eval_model=cli.eval_model,
        temperature=cli.temperature,
        do_validate=do_validate,
        instruct_model=instruct_model,
        checkpoint_every=cli.checkpoint_every,
        log_file=cli.log_file,
        results_file=cli.results_file,
    )
