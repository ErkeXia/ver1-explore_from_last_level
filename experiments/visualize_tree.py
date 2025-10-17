import json
import graphviz
import os
import argparse  # Import argparse for command-line arguments
from tot.tasks.crosswords import MiniCrosswordsTask

# --- Configuration ---
RESULTS_FILE = "./results/crossword_results.jsonl"
BASE_OUTPUT_DIR = "./results/visualizations"

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
    total_nodes = sum(sum(len(nodes) for nodes in state_tree.values()) for state_tree in result_data.get("all_states", []))

    task = MiniCrosswordsTask()
    task.env.reset(problem_id)
    correct_grid_str = "\n".join([" ".join(task.env.board_gt[i*5:(i+1)*5]) for i in range(5)])

    with open(output_path, "w") as f:
        f.write("="*50 + "\n")
        f.write(f"  Summary for Crossword Problem ID: {problem_id}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Refinement Iterations (Explorations): {num_iterations}\n")
        f.write(f"Total Nodes Explored (All Iterations): {total_nodes}\n\n")
        f.write("--- Correct Answer ---\n")
        f.write(correct_grid_str + "\n\n")
        
        f.write("--- Iteration Details ---\n")
        iteration_details = result_data.get("iteration_details", [])
        for details in iteration_details:
            f.write(f"\n--- Iteration {details['iteration']} ---\n")
            f.write("Llama Result Grid (before GPT eval):\n")
            f.write(format_grid_for_display(details['llama_result_grid'], compact=False) + "\n\n")
            f.write("GPT Pruned Grid (start of next iteration):\n")
            f.write(format_grid_for_display(details['gpt_pruned_grid'], compact=False) + "\n")

        f.write("\n--- GPT Evaluation Results ---\n")
        gpt_results = result_data.get("gpt_eval_results", [])
        for i, sure_list in enumerate(gpt_results):
            f.write(f"Iteration {i+1}: GPT was 'sure' about word indices: {sure_list}\n")

def visualize_single_case(results_file_path: str, problem_id_to_find: int):
    """
    Finds a specific problem ID in the results file, and creates a dedicated
    folder for its summary and tree visualizations.
    """
    if not os.path.exists(results_file_path):
        print(f"Error: Results file not found at '{results_file_path}'")
        return

    # 1. Find the specific result for the given problem ID
    target_result_data = None
    with open(results_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("problem_id") == problem_id_to_find:
                target_result_data = data
                break
    
    if target_result_data is None:
        print(f"Error: Problem ID {problem_id_to_find} not found in '{results_file_path}'.")
        return

    # 2. Create a dedicated output folder for this case
    case_output_dir = os.path.join(BASE_OUTPUT_DIR, f"problem_{problem_id_to_find}")
    os.makedirs(case_output_dir, exist_ok=True)
    print(f"\nVisualizing case for Problem ID {problem_id_to_find}.")
    print(f"Output will be saved to: '{case_output_dir}'")

    # 3. Create the summary file inside the new folder
    summary_path = os.path.join(case_output_dir, "summary_data.txt")
    create_summary_file(target_result_data, summary_path)
    print(f"  - Summary data saved to 'summary_data.txt'")

    # 4. Visualize each tree and save it inside the new folder
    all_states_list = target_result_data.get("all_states", [])
    for i, state_tree in enumerate(all_states_list):
        iteration_num = i + 1
        dot = graphviz.Digraph(
            comment=f'Search Tree - ID {problem_id_to_find} - Iteration {iteration_num}',
            graph_attr={'rankdir': 'TB', 'splines': 'ortho'},
            node_attr={'shape': 'box', 'fontname': 'Courier New', 'fontsize': '10'},
            edge_attr={'arrowsize': '0.7', 'fontsize': '8'}
        )
        # (The graph generation logic is the same as before)
        for depth_str, nodes_at_depth in state_tree.items():
            for idx, node_data in enumerate(nodes_at_depth):
                node_id, score = f"{depth_str}-{idx}", node_data.get('score', 0)
                grid_display = format_grid_for_display(node_data['current'])
                dot.node(node_id, f"{grid_display}\nScore: {score:.2f}")
        for depth_str, nodes_at_depth in state_tree.items():
            depth = int(depth_str)
            if depth == 0: continue
            for idx, node_data in enumerate(nodes_at_depth):
                child_id, parent_idx = f"{depth_str}-{idx}", node_data.get('connect')
                if parent_idx is not None:
                    parent_id, action = f"{depth - 1}-{parent_idx}", node_data.get('step', '').strip()
                    if action: dot.edge(parent_id, child_id, label=f" {action} ")
        
        output_filename = os.path.join(case_output_dir, f"tree_iteration_{iteration_num}")
        try:
            dot.render(output_filename, format='png', view=False, cleanup=True)
            print(f"  - Tree image saved to 'tree_iteration_{iteration_num}.png'")
        except graphviz.backend.execute.ExecutableNotFound:
            print("Graphviz Error: 'dot' command not found. Please install Graphviz system software.")
            return

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize the search tree for a specific crossword problem.")
    parser.add_argument("problem_id", type=int, help="The ID of the problem to visualize (e.g., 0, 1, 2...).")
    args = parser.parse_args()
    
    visualize_single_case(RESULTS_FILE, args.problem_id)

# import json
# import graphviz
# import os
# from tot.tasks.crosswords import MiniCrosswordsTask

# # --- Configuration ---
# RESULTS_FILE = "./results/crossword_results.jsonl"
# OUTPUT_DIR = "./results/visualizations"

# def format_grid_for_display(grid_string: str, compact=True) -> str:
#     """Formats the grid string for better display."""
#     if "Output:" in grid_string:
#         lines = grid_string.strip().split('\n')[1:]
#         if compact:
#             # For graph nodes, use a compact format
#             return "\n".join([line.replace(" ", "") for line in lines])
#         else:
#             # For the text file, keep the spaces for readability
#             return "\n".join(lines)
#     return grid_string

# def create_summary_file(result_data, output_path):
#     """Creates a text file with summary information about the run."""
#     problem_id = result_data.get("problem_id", "N/A")
#     num_iterations = len(result_data.get("all_states", []))
#     total_nodes = sum(
#         sum(len(nodes) for nodes in state_tree.values())
#         for state_tree in result_data.get("all_states", [])
#     )

#     task = MiniCrosswordsTask()
#     task.env.reset(problem_id)
#     correct_grid_str = "\n".join([" ".join(task.env.board_gt[i*5:(i+1)*5]) for i in range(5)])

#     with open(output_path, "w") as f:
#         f.write("="*50 + "\n")
#         f.write(f"  Summary for Crossword Problem ID: {problem_id}\n")
#         f.write("="*50 + "\n\n")

#         f.write(f"Refinement Iterations (Explorations): {num_iterations}\n")
#         f.write(f"Total Nodes Explored (All Iterations): {total_nodes}\n\n")

#         f.write("--- Correct Answer ---\n")
#         f.write(correct_grid_str + "\n\n")

#         # --- NEW: Add detailed iteration results ---
#         f.write("--- Iteration Details ---\n")
#         iteration_details = result_data.get("iteration_details", [])
#         if iteration_details:
#             for details in iteration_details:
#                 f.write(f"\n--- Iteration {details['iteration']} ---\n")
#                 f.write("Llama Result Grid (before GPT eval):\n")
#                 f.write(format_grid_for_display(details['llama_result_grid'], compact=False) + "\n\n")
#                 f.write("GPT Pruned Grid (start of next iteration):\n")
#                 f.write(format_grid_for_display(details['gpt_pruned_grid'], compact=False) + "\n")
#         else:
#             f.write("No iteration details found.\n")

#         f.write("\n--- GPT Evaluation Results ---\n")
#         gpt_results = result_data.get("gpt_eval_results", [])
#         if gpt_results:
#             for i, sure_list in enumerate(gpt_results):
#                 f.write(f"Iteration {i+1}: GPT was 'sure' about word indices: {sure_list}\n")
#         else:
#             f.write("No GPT evaluation data found.\n")

# def visualize_crossword_trees(results_file_path: str):
#     """
#     Reads the first result and visualizes the search tree(s),
#     and creates a summary data file.
#     """
#     if not os.path.exists(results_file_path):
#         print(f"Error: Results file not found at '{results_file_path}'")
#         return

#     with open(results_file_path, "r") as f:
#         first_line = f.readline()
#         if not first_line:
#             print("Error: The results file is empty.")
#             return
#         result_data = json.loads(first_line)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     summary_path = os.path.join(OUTPUT_DIR, "summary_data.txt")
#     create_summary_file(result_data, summary_path)
#     print(f"Summary data saved to '{summary_path}'")

#     all_states_list = result_data.get("all_states", [])
#     for i, state_tree in enumerate(all_states_list):
#         iteration_num = i + 1
#         dot = graphviz.Digraph(
#             comment=f'Search Tree - Iteration {iteration_num}',
#             graph_attr={'rankdir': 'TB', 'splines': 'ortho'},
#             node_attr={'shape': 'box', 'fontname': 'Courier New', 'fontsize': '10'},
#             edge_attr={'arrowsize': '0.7', 'fontsize': '8'}
#         )
#         for depth_str, nodes_at_depth in state_tree.items():
#             for idx, node_data in enumerate(nodes_at_depth):
#                 node_id = f"{depth_str}-{idx}"
#                 grid_display = format_grid_for_display(node_data['current'])
#                 score = node_data.get('score', 0)
#                 label = f"{grid_display}\nScore: {score:.2f}"
#                 dot.node(node_id, label)
#         for depth_str, nodes_at_depth in state_tree.items():
#             depth = int(depth_str)
#             if depth == 0: continue
#             for idx, node_data in enumerate(nodes_at_depth):
#                 child_id = f"{depth_str}-{idx}"
#                 parent_idx = node_data.get('connect')
#                 if parent_idx is not None:
#                     parent_id = f"{depth - 1}-{parent_idx}"
#                     action = node_data.get('step', '').strip()
#                     if action:
#                         dot.edge(parent_id, child_id, label=f" {action} ")
#         output_filename = os.path.join(OUTPUT_DIR, f"tree_iteration_{iteration_num}")
#         try:
#             dot.render(output_filename, format='png', view=False, cleanup=True)
#             print(f"Successfully saved tree for Iteration {iteration_num} to '{output_filename}.png'")
#         except graphviz.backend.execute.ExecutableNotFound:
#             print("Graphviz Error: 'dot' command not found.")
#             return

# if __name__ == '__main__':
#     visualize_crossword_trees(RESULTS_FILE)