import json
import graphviz
import os
import argparse

# --- Configuration ---
RESULTS_FILE = "./results/game24_sys_MS_eval_results.jsonl"
BASE_OUTPUT_DIR = "./results/visualizations"

def format_node_label(text: str) -> str:
    """Cleans and formats the multi-line text for a Graphviz node."""
    # Escape backslashes and quotes, and left-align the text
    return text.replace('\\', '\\\\').replace('"', '\\"').strip() + '\\l'

def create_summary_file(result_data, output_path):
    """Creates a text file with summary information about the Game of 24 run."""
    seed = result_data.get("seed", "N/A")
    problem_input = result_data.get("x", "N/A")
    final_answer_path = result_data.get("answer", ["No solution found."])
    if isinstance(final_answer_path, list) and final_answer_path:
        final_answer_path = final_answer_path[-1] # Get the last valid thought chain

    num_iterations = len(result_data.get("all_states", []))
    total_nodes = result_data.get("nodes", 0)
    
    with open(output_path, "w") as f:
        f.write("="*50 + "\n")
        f.write(f"  Summary for Game of 24 Problem ID (Seed): {seed}\n")
        f.write("="*50 + "\n\n")

        f.write(f"Problem Input: {problem_input}\n")
        f.write(f"Final Answer Found: {final_answer_path}\n\n")

        f.write(f"Refinement Iterations (Explorations): {num_iterations}\n")
        f.write(f"Total Nodes Explored in Final Iteration: {total_nodes}\n\n")

        f.write("--- GPT Evaluation and Feedback ---\n")
        # --- MODIFICATION START ---
        # Read all the specified keys for GPT's feedback
        validators = result_data.get("validation", [])
        correctness = result_data.get("correctness_r", [])
        locate = result_data.get("locate", [])
        suggestions = result_data.get("suggestion_r", [])
        
        has_feedback = any([validators, correctness, locate, suggestions])

        if not has_feedback:
            f.write("No GPT evaluation data found in the results file.\n")
        else:
            if validators:
                f.write("\n-- Validation (Overall Check) --\n")
                for i, text in enumerate(validators):
                    f.write(f"Iteration {i+1}:\n{text}\n\n")
            
            if correctness:
                f.write("\n-- Correctness Check --\n")
                for i, text in enumerate(correctness):
                    f.write(f"Iteration {i+1}:\n{text}\n\n")

            if locate:
                f.write("\n-- Error Location --\n")
                for i, text in enumerate(locate):
                    f.write(f"Iteration {i+1}:\n{text}\n\n")

            if suggestions:
                f.write("\n-- Suggestions --\n")
                for i, text in enumerate(suggestions):
                    f.write(f"Iteration {i+1}:\n{text}\n\n")
        # --- MODIFICATION END ---

def visualize_game24_trees(results_file_path: str, seed_to_find: int):
    """
    Finds a specific problem seed in the Game of 24 results and creates a
    dedicated folder for its summary and tree visualizations.
    """
    if not os.path.exists(results_file_path):
        print(f"Error: Results file not found at '{results_file_path}'")
        return

    # 1. Find the specific result for the given problem seed
    target_result_data = None
    with open(results_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("seed") == seed_to_find:
                target_result_data = data
                break
    
    if target_result_data is None:
        print(f"Error: Seed {seed_to_find} not found in '{results_file_path}'.")
        return

    # 2. Create a dedicated output folder for this case
    case_output_dir = os.path.join(BASE_OUTPUT_DIR, f"game24_problem_{seed_to_find}")
    os.makedirs(case_output_dir, exist_ok=True)
    print(f"\nVisualizing case for Problem Seed {seed_to_find}.")
    print(f"Output will be saved to: '{case_output_dir}'")

    # 3. Create the summary file
    summary_path = os.path.join(case_output_dir, "summary_data.txt")
    create_summary_file(target_result_data, summary_path)
    print(f"  - Summary data saved to 'summary_data.txt'")

    # 4. Visualize each tree from the 'all_states' list
    all_states_list = target_result_data.get("states", [])
    for i, state_tree in enumerate(all_states_list):
        iteration_num = i + 1
        dot = graphviz.Digraph(
            comment=f'Game of 24 Search Tree - Iteration {iteration_num}',
            graph_attr={'rankdir': 'TB', 'splines': 'ortho'},
            node_attr={'shape': 'box', 'fontname': 'Courier New', 'fontsize': '9'},
            edge_attr={'arrowsize': '0.7', 'fontsize': '8'}
        )

        # The state_tree is a list of lists: state_tree[depth][node_index]
        for depth, nodes_at_depth in enumerate(state_tree):
            for idx, node_data in enumerate(nodes_at_depth):
                node_id = f"{depth}-{idx}"
                label = format_node_label(node_data.get('current', ''))
                dot.node(node_id, label)

                if depth > 0:
                    parent_idx = node_data.get('connect')
                    if parent_idx is not None:
                        parent_id = f"{depth - 1}-{parent_idx}"
                        action = node_data.get('step', '').strip()
                        if action:
                            # --- BUG FIX ---
                            # Replaced the undefined 'child_id' with the correct 'node_id'
                            dot.edge(parent_id, node_id, label=f" {action} ")
        
        output_filename = os.path.join(case_output_dir, f"tree_iteration_{iteration_num}")
        try:
            dot.render(output_filename, format='png', view=False, cleanup=True)
            print(f"  - Tree image saved to 'tree_iteration_{iteration_num}.png'")
        except graphviz.backend.execute.ExecutableNotFound:
            print("Graphviz Error: 'dot' command not found. Please install Graphviz system software.")
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the search tree for a specific Game of 24 problem.")
    parser.add_argument("seed", type=int, help="The seed of the problem to visualize (e.g., 351, 400...).")
    args = parser.parse_args()
    
    visualize_game24_trees(RESULTS_FILE, args.seed)

