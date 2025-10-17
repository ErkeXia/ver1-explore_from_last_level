import json
import os

# --- Configuration ---
# The path to your results file, as specified in your experiment script.
RESULTS_FILE = "./results/game24_sys_MS_eval2_results.jsonl"

def find_specific_cases(jsonl_filepath: str) -> list[int]:
    """
    Analyzes a Game of 24 results file to find specific cases.

    This function searches for cases that meet two criteria:
    1. The number of GPT validation steps ('validate_num') is greater than 1.
    2. The final answer string begins with the word "Answer".

    Args:
        jsonl_filepath: The path to the .jsonl results file.

    Returns:
        A list of integers, where each integer is the seed of a matching case.
        Returns an empty list if no cases match or the file is not found.
    """
    matching_seeds = []
    
    if not os.path.exists(jsonl_filepath):
        print(f"Error: Results file not found at '{jsonl_filepath}'")
        return matching_seeds

    print(f"Analyzing results from: {jsonl_filepath}\n")

    with open(jsonl_filepath, "r") as f:
        for i, line in enumerate(f):
            try:
                result_data = json.loads(line)

                # --- Condition 1: Check validation time (number of validations) ---
                validation_count = result_data.get("validate_num", 0)
                is_multi_validation = validation_count > 1

                # --- Condition 2: Check the final answer format ---
                answer_path = result_data.get("answer", [])
                starts_with_answer = False
                if isinstance(answer_path, list) and answer_path:
                    # The final answer is the last element in the thought chain
                    final_step = answer_path[-1]
                    if isinstance(final_step, str) and final_step.strip().startswith("Answer"):
                        starts_with_answer = True

                # --- Check if both conditions are met ---
                if is_multi_validation and starts_with_answer:
                    seed = result_data.get("seed")
                    if seed is not None:
                        matching_seeds.append(seed)
                        print(f"Found a matching case! -> Seed: {seed}")

            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON on line {i+1}. Skipping.")

    if not matching_seeds:
        print("No cases found that match both criteria.")
        
    return matching_seeds

if __name__ == '__main__':
    # This block runs when you execute the script directly
    found_cases = find_specific_cases(RESULTS_FILE)
    
    if found_cases:
        print(f"\nSummary: Found {len(found_cases)} matching case(s) with seeds: {found_cases}")