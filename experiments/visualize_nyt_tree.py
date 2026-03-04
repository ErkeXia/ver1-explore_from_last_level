import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import graphviz

from tot.tasks.NYT_crosswords import NYTCrosswordsTask


DEFAULT_RESULTS_FILE = "./results/NYT_crossword_lazy_qwen_results.jsonl"
DEFAULT_OUTPUT_BASE = "./results/NYT_visualizations"


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def _pick_record(records: List[Dict[str, Any]], problem_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    run_meta = None
    puzzle_result = None
    checkpoints = []

    for rec in records:
        rtype = rec.get("record_type")
        if rtype == "run_meta":
            run_meta = rec
            continue
        if rec.get("problem_id") != problem_id:
            continue
        if rtype == "puzzle_result":
            puzzle_result = rec
        elif rtype == "checkpoint":
            checkpoints.append(rec)

    if puzzle_result is not None:
        return puzzle_result, run_meta
    if checkpoints:
        return checkpoints[-1], run_meta
    return None, run_meta


def _extract_state_trees(record: Dict[str, Any]) -> Tuple[List[Dict[str, List[Dict[str, Any]]]], str]:
    rtype = record.get("record_type", "")
    if rtype == "puzzle_result":
        trees = record.get("all_states", []) or []
        return trees, "puzzle_result"

    # checkpoint variants
    if "states" in record and isinstance(record["states"], dict):
        return [record["states"]], "checkpoint"

    if "all_states" in record and isinstance(record["all_states"], list):
        return record["all_states"], "checkpoint"

    return [], "unknown"


def _node_uid(depth_key: str, idx: int, node_data: Dict[str, Any]) -> str:
    order = node_data.get("created_order")
    if order is not None:
        return f"N{order}"
    return f"D{depth_key}_I{idx}"


def _render_tree_ids_only(
    state_tree: Dict[str, List[Dict[str, Any]]],
    output_prefix: str,
    title: str,
) -> None:
    dot = graphviz.Digraph(
        comment=title,
        graph_attr={"rankdir": "TB", "splines": "ortho"},
        node_attr={"shape": "ellipse", "fontname": "Courier New", "fontsize": "10"},
        edge_attr={"arrowsize": "0.7"},
    )

    # create nodes
    for depth_key, nodes_at_depth in state_tree.items():
        for idx, node in enumerate(nodes_at_depth):
            uid = _node_uid(depth_key, idx, node)
            dot.node(f"{depth_key}-{idx}", uid)

    # create edges
    for depth_key, nodes_at_depth in state_tree.items():
        depth = _safe_int(depth_key, 0)
        if depth == 0:
            continue
        for idx, node in enumerate(nodes_at_depth):
            parent_idx = node.get("connect")
            if parent_idx is None:
                continue
            parent_id = f"{depth-1}-{parent_idx}"
            child_id = f"{depth_key}-{idx}"
            dot.edge(parent_id, child_id)

    dot.render(output_prefix, format="png", view=False, cleanup=True)


def _render_grid(task: NYTCrosswordsTask, problem_id: int, y_state: str) -> str:
    try:
        x = task.get_input(problem_id)
        return task.render_grid_only(x, y_state)
    except Exception as e:
        return f"[Grid render failed] {e}\nRaw state:\n{y_state}\n"


def _write_nodes_detail(
    task: NYTCrosswordsTask,
    problem_id: int,
    trees: List[Dict[str, List[Dict[str, Any]]]],
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"NYT Node Details - Problem {problem_id}\n")
        f.write("=" * 72 + "\n\n")

        for it, tree in enumerate(trees, start=1):
            f.write(f"--- Iteration {it} ---\n")
            depth_keys = sorted(tree.keys(), key=lambda k: _safe_int(k, 0))
            for depth_key in depth_keys:
                nodes = tree.get(depth_key, [])
                for idx, node in enumerate(nodes):
                    uid = _node_uid(depth_key, idx, node)
                    action = node.get("step")
                    score = node.get("score")
                    connect = node.get("connect")
                    created_order = node.get("created_order")
                    y_state = node.get("current", "[]")
                    grid = _render_grid(task, problem_id, y_state)

                    f.write(f"Node {uid}\n")
                    f.write(f"  depth: {depth_key}\n")
                    f.write(f"  index_in_depth: {idx}\n")
                    f.write(f"  connect(parent_idx): {connect}\n")
                    f.write(f"  created_order: {created_order}\n")
                    f.write(f"  action: {action}\n")
                    f.write(f"  score: {score}\n")
                    f.write("  grid:\n")
                    for ln in grid.rstrip("\n").splitlines():
                        f.write(f"    {ln}\n")
                    f.write("\n")
            f.write("\n")


def _write_summary(
    task: NYTCrosswordsTask,
    problem_id: int,
    run_meta: Optional[Dict[str, Any]],
    chosen_record: Dict[str, Any],
    source_type: str,
    trees: List[Dict[str, List[Dict[str, Any]]]],
    out_path: str,
) -> None:
    task.env.reset(problem_id)
    clues_text = task.env.render_clues()
    gt_ans = task.env.render_gt_ans()

    total_nodes = 0
    for tree in trees:
        for nodes in tree.values():
            total_nodes += len(nodes)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"NYT Visualization Summary - Problem {problem_id}\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Source record type: {source_type}\n")
        if run_meta:
            f.write(f"Task file: {run_meta.get('task_file', 'N/A')}\n")
            f.write(f"SLM: {run_meta.get('slm', 'N/A')}\n")
            f.write(f"Eval model: {run_meta.get('eval_model', 'N/A')}\n")
            f.write(f"Backend: {run_meta.get('backend', 'N/A')}\n")
        f.write(f"Iterations visualized: {len(trees)}\n")
        f.write(f"Total nodes visualized: {total_nodes}\n\n")

        if chosen_record.get("record_type") == "puzzle_result":
            metrics = chosen_record.get("metrics", {})
            f.write("--- Final Metrics ---\n")
            f.write(f"r_word: {metrics.get('r_word', 'N/A')}\n")
            f.write(f"r_letter: {metrics.get('r_letter', 'N/A')}\n")
            f.write(f"r_game: {metrics.get('r_game', 'N/A')}\n")
            f.write(f"solve_time_sec: {chosen_record.get('solve_time_sec', 'N/A')}\n")
            f.write("\n")

        f.write("--- Clues ---\n")
        f.write(clues_text + "\n")
        f.write("--- Ground Truth Answers ---\n")
        f.write(gt_ans + "\n")


def visualize_nyt_case(
    problem_id: int,
    results_file: str = DEFAULT_RESULTS_FILE,
    output_base: str = DEFAULT_OUTPUT_BASE,
) -> None:
    records = _load_jsonl(results_file)
    chosen, run_meta = _pick_record(records, problem_id)
    if chosen is None:
        print(f"No record found for problem_id={problem_id} in {results_file}")
        return

    task_file = (run_meta or {}).get("task_file", "NYT_cw.json")
    task = NYTCrosswordsTask(file=task_file)
    trees, source_type = _extract_state_trees(chosen)
    if not trees:
        print(f"No tree data found for problem_id={problem_id}.")
        return

    out_dir = os.path.join(output_base, f"problem_{problem_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Summary
    summary_path = os.path.join(out_dir, "summary.txt")
    _write_summary(task, problem_id, run_meta, chosen, source_type, trees, summary_path)

    # Detailed node dump
    details_path = os.path.join(out_dir, "node_details.txt")
    _write_nodes_detail(task, problem_id, trees, details_path)

    # Trees (node-id-only)
    for i, tree in enumerate(trees, start=1):
        title = f"NYT Tree IDs - problem {problem_id} - iteration {i}"
        prefix = os.path.join(out_dir, f"tree_ids_iteration_{i}")
        _render_tree_ids_only(tree, prefix, title)

    print(f"Visualization written to: {out_dir}")
    print(f" - summary.txt")
    print(f" - node_details.txt")
    print(f" - tree_ids_iteration_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NYT solver trees with node IDs + separate node detail dump.")
    parser.add_argument("problem_id", type=int, help="Puzzle id to visualize.")
    parser.add_argument("--file", type=str, default=DEFAULT_RESULTS_FILE, help="NYT experiment JSONL path.")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_BASE, help="Base output folder.")
    args = parser.parse_args()

    visualize_nyt_case(problem_id=args.problem_id, results_file=args.file, output_base=args.out)
