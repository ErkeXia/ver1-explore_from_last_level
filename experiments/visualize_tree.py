import json
import graphviz
import os
import argparse
# import argparse  # Import argparse for command-line arguments
from tot.tasks.crosswords import MiniCrosswordsTask

# --- Configuration ---
# Change this to point to the specific results file you want to visualize
RESULTS_FILE = "./results/llama_NYT_crossword_search_results.jsonl" 
BASE_OUTPUT_DIR = "./results/visualizations/NYT_crossword"
DATA_FILE = "miniNYT.json"

def format_grid_for_display(grid_string: str, compact=True) -> str:
    """Formats the grid string for better display."""
    if "Output:" in grid_string:
        lines = grid_string.strip().split('\n')[1:]
        return "\n".join([line.replace(" ", "") for line in lines]) if compact else "\n".join(lines)
    return grid_string

def create_summary_file(result_data, output_path):
    """Creates a text file with summary information about the run."""
    problem_id = result_data.get("problem_id", "N/A")
    num_iterations = len(result_data.get("all_states", []))
    # Safely calculate total nodes, handling potential missing data
    total_nodes = 0
    all_states = result_data.get("all_states", [])
    if all_states:
         total_nodes = sum(sum(len(nodes) for nodes in state_tree.values()) for state_tree in all_states)

    task = MiniCrosswordsTask(file=DATA_FILE)
    task.env.reset(problem_id)
    correct_grid_str = "\n".join([" ".join(task.env.board_gt[i*5:(i+1)*5]) for i in range(5)])
    
    # --- NEW: Extract Clues ---
    clues = task.env.data
    horizontal_clues = clues[:5]
    vertical_clues = clues[5:]

    with open(output_path, "w") as f:
        f.write("="*50 + "\n")
        f.write(f"  Summary for Crossword Problem ID: {problem_id}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Refinement Iterations (Explorations): {num_iterations}\n")
        f.write(f"Total Nodes Explored (All Iterations): {total_nodes}\n\n")
        
        # --- NEW: Write Clues to Summary ---
        f.write("--- Hints (Clues) ---\n")
        f.write("Horizontal:\n")
        for i, clue in enumerate(horizontal_clues):
            f.write(f"  h{i+1}. {clue}\n")
        f.write("Vertical:\n")
        for i, clue in enumerate(vertical_clues):
            f.write(f"  v{i+1}. {clue}\n")
        f.write("\n")

        f.write("--- Correct Answer ---\n")
        f.write(correct_grid_str + "\n\n")
        
        f.write("--- Iteration Details ---\n")
        iteration_details = result_data.get("iteration_details", [])
        if iteration_details:
            for details in iteration_details:
                f.write(f"\n--- Iteration {details['iteration']} ---\n")
                f.write("Llama Result Grid (before GPT eval):\n")
                f.write(format_grid_for_display(details.get('llama_result_grid', ''), compact=False) + "\n\n")
                f.write("GPT Pruned Grid (start of next iteration):\n")
                f.write(format_grid_for_display(details.get('gpt_pruned_grid', ''), compact=False) + "\n")
        else:
             f.write("No iteration details available.\n")

        f.write("\n--- GPT Evaluation Results ---\n")
        gpt_results = result_data.get("gpt_eval_results", [])
        if gpt_results:
            for i, sure_list in enumerate(gpt_results):
                f.write(f"Iteration {i+1}: GPT was 'sure' about word indices: {sure_list}\n")
        else:
             f.write("No GPT evaluation results available.\n")

def visualize_single_case(results_file_path: str, problem_id_to_find: int):
    if not os.path.exists(results_file_path):
        print(f"Error: Results file not found at '{results_file_path}'")
        return

    # Determine model suffix based on filename
    model_suffix = ""
    if "gpt" in os.path.basename(results_file_path).lower():
        model_suffix = "_gpt"
    elif "llama" in os.path.basename(results_file_path).lower():
         model_suffix = "_llama"

    # 1. Find the specific result
    target_result_data = None
    with open(results_file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("problem_id") == problem_id_to_find:
                    target_result_data = data
                    break
            except json.JSONDecodeError:
                continue
    
    if target_result_data is None:
        print(f"Error: Problem ID {problem_id_to_find} not found in '{results_file_path}'.")
        return

    # 2. Create output folder
    case_output_dir = os.path.join(BASE_OUTPUT_DIR, f"problem_{problem_id_to_find}")
    os.makedirs(case_output_dir, exist_ok=True)
    print(f"\nVisualizing case for Problem ID {problem_id_to_find}.")
    print(f"Output will be saved to: '{case_output_dir}'")

    # 3. Create summary file with suffix
    summary_filename = f"summary_data{model_suffix}.txt"
    summary_path = os.path.join(case_output_dir, summary_filename)
    create_summary_file(target_result_data, summary_path)
    print(f"  - Summary data saved to '{summary_filename}'")

    # 4. Visualize trees
    all_states_list = target_result_data.get("all_states", [])
    for i, state_tree in enumerate(all_states_list):
        iteration_num = i + 1
        dot = graphviz.Digraph(
            comment=f'Search Tree - ID {problem_id_to_find} - Iteration {iteration_num}',
            graph_attr={'rankdir': 'TB', 'splines': 'ortho'},
            node_attr={'shape': 'box', 'fontname': 'Courier New', 'fontsize': '10'},
            edge_attr={'arrowsize': '0.7', 'fontsize': '8'}
        )
        for depth_str, nodes_at_depth in state_tree.items():
            for idx, node_data in enumerate(nodes_at_depth):
                node_id, score = f"{depth_str}-{idx}", node_data.get('score', 0)
                grid_display = format_grid_for_display(node_data['current'])
                
                created_order = node_data.get('created_order', 'NA')
                
                label = f"#{created_order}\n{grid_display}\nScore: {score:.2f}"
                dot.node(node_id, label)
        for depth_str, nodes_at_depth in state_tree.items():
            depth = int(depth_str)
            if depth == 0: continue
            for idx, node_data in enumerate(nodes_at_depth):
                child_id, parent_idx = f"{depth_str}-{idx}", node_data.get('connect')
                if parent_idx is not None:
                    parent_id, action = f"{depth - 1}-{parent_idx}", node_data.get('step', '').strip()
                    if action: dot.edge(parent_id, child_id, label=f" {action} ")
        
        # Append suffix to image filename
        image_filename = f"tree_iteration_{iteration_num}{model_suffix}"
        output_path_prefix = os.path.join(case_output_dir, image_filename)
        try:
            dot.render(output_path_prefix, format='png', view=False, cleanup=True)
            print(f"  - Tree image saved to '{image_filename}.png'")
        except graphviz.backend.execute.ExecutableNotFound:
            print("Graphviz Error: 'dot' command not found.")
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the search tree for a specific crossword problem.")
    parser.add_argument("problem_id", type=int, help="The ID of the problem to visualize.")
    # Optional: Override the RESULTS_FILE from command line
    parser.add_argument("--file", type=str, default=RESULTS_FILE, help="Path to the results JSONL file.")
    args = parser.parse_args()
    
    visualize_single_case(args.file, args.problem_id)
