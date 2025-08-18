import argparse
import sys
from tot.tasks.game24 import Game24Task
import json
from tqdm import tqdm
from contextlib import redirect_stdout

model = 'gpt-4'
args = argparse.Namespace(backend=model, temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()

def compute_average_stats_lazy(jsonl_path):
    stats_sum = {}
    count = 0

    # Store problem-wise stats for better formatting
    problem_stats = []

    with open(jsonl_path, 'r') as file:
        correct = 0
        incorrect = 0
        valid_res = 0
        invalid_res = 0
        suggest_res = 0
        
        explore_time = [0, 0, 0]
        
        for line in file:
            data = json.loads(line)
            x = data["x"]
            y = data["answer"]
            if "Answer" in y:
                y_clean = y[(y.find('Answer') + 7):].strip()
            else:
                y_clean = y.strip()
            is_correct, feedback = task.check_answer(x, y_clean)
            seed = data["seed"]

            # Track stats for each problem in one line
            problem_stat = {
                "seed": seed,
                "x": x,
                "is_correct": is_correct,
                "explore_count": len(data["validation"]),
                "answer": data["answer"],
                "feedback": feedback,
            }

            if is_correct:
                correct += 1
            else:
                incorrect += 1
            count += 1

            # validations = data["validation"]
            # for validation in validations:
            #     redo_s, feedback = task.validate_unwrap(validation)
            #     if (redo_s == -1):
            #         valid_res += 1
            #     elif (feedback != ""):
            #         suggest_res += 1
            #     else:
            #         invalid_res += 1

            # explore_time[len(validations)-1] += 1

            for key, value in data.items():
                if isinstance(value, (int, float)):
                    stats_sum[key] = stats_sum.get(key, 0) + value
                    problem_stat[key] = value  # Store key-value pair for the problem

            problem_stats.append(problem_stat)

    # Compute averages
    averages = {key: total / count for key, total in stats_sum.items()}

    # Print results for each problem
    print(f"\n{'Seed':<10} {'Input':<15} {'Correct':<10} {'Explore Count':<15} {'nodes':<10} {'Time':<10} {'gpt_prompt_tokens':<20} {'gpt_completion_tokens':<25} {'llama_prompt_tokens':<20} {'llama_completion_tokens':<25} {'Answer':<35}  {'Feedback':<30} ")
    for stat in problem_stats:
        print(f"{stat['seed']:<10} {stat['x']:<15} {stat['is_correct']:<10} {stat['explore_count']:<15} {stat['nodes']:<10} {stat['total_time']:<10.2f} {stat.get('gpt_prompt_tokens', 'N/A'):<20} {stat.get('gpt_completion_tokens', 'N/A'):<25} {stat.get('llama_prompt_tokens', 'N/A'):<20} {stat.get('llama_completion_tokens', 'N/A'):<25} {stat['answer']:<35} {stat['feedback']:<30}")

    # Print summary stats for the entire dataset
    print("\nSummary of the Dataset:")
    print(f"Correct count: {correct}")
    print(f"Incorrect count: {incorrect}")
    print(f"Valid response: {valid_res}")
    print(f"Invalid response: {invalid_res}")
    print(f"Suggest response: {suggest_res}")
    print(f"Explore time distribution: {explore_time}")

    return averages

def compute_average_stats_llama(jsonl_path):
    stats_sum = {}
    count = 0

    # Store problem-wise stats for better formatting
    problem_stats = []

    with open(jsonl_path, 'r') as file:
        correct = 0
        incorrect = 0
                
        for line in file:
            data = json.loads(line)
            x = data["x"]
            y = data["answer"]
            y_clean = y.strip()
            is_correct, feedback = task.check_multistep_solution(x, y_clean)
            seed = data["seed"]

            # Track stats for each problem in one line
            problem_stat = {
                "seed": seed,
                "x": x,
                "is_correct": is_correct,
                "explore_layers": len(data["thoughts"][0]),
                "answer": data["answer"].replace('\n',''),
                "feedback": feedback,
            }

            if is_correct:
                correct += 1
            else:
                incorrect += 1
            count += 1

            for key, value in data.items():
                if isinstance(value, (int, float)):
                    stats_sum[key] = stats_sum.get(key, 0) + value
                    problem_stat[key] = value  # Store key-value pair for the problem

            problem_stats.append(problem_stat)

    # Compute averages
    averages = {key: total / count for key, total in stats_sum.items()}

    # Print results for each problem
    print(f"\n{'Seed':<10} {'Input':<15} {'Correct':<10} {'Explore Layer':<15} {'nodes':<10} {'Time':<10} {'llama_prompt_tokens':<20} {'llama_completion_tokens':<25} {'Feedback':<80} {'Answer':<35} ")
    for stat in problem_stats:
        print(f"{stat['seed']:<10} {stat['x']:<15} {stat['is_correct']:<10} {stat['explore_layers']:<15} {stat['nodes']:<10} {stat['total_time']:<10.2f} {stat.get('llama_prompt_tokens', 'N/A'):<20} {stat.get('llama_completion_tokens', 'N/A'):<25} {stat['feedback']:<80}  {stat['answer']:<35}")

    # Print summary stats for the entire dataset
    print("\nSummary of the Dataset:")
    print(f"Correct count: {correct}")
    print(f"Incorrect count: {incorrect}")

    return averages


# Example usage
jsonl_path = "./results/lazy_sys_game24_flash_new_results.jsonl"  # Replace with your path

with open('./stats/sys_lazy_flash_stats.txt', 'w', buffering=1) as f:
    sys.stdout = f

    averages = compute_average_stats_lazy(jsonl_path)

    # Print averages to file
    print("\nAverages for each statistic:")
    for key, avg in averages.items():
        print(f"{key}: {avg:.4f}")
