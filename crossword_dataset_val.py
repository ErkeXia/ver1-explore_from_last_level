import json
import os
from collections import Counter

REPORT_FILE = "dataset_validation_report.json"

def analyze_report():
    if not os.path.exists(REPORT_FILE):
        print(f"Error: Report file '{REPORT_FILE}' not found.")
        print("Please run 'validate_dataset.py' first.")
        return

    print(f"Reading report from {REPORT_FILE}...")
    with open(REPORT_FILE, 'r') as f:
        data = json.load(f)

    detailed_results = data.get("detailed_results", {})
    total_tasks = data.get("total_tasks", 0)

    # 1. Aggregate Stats
    # Count how many puzzles have X valid words
    validity_distribution = Counter()
    # Store pairs of (task_id, valid_count) for sorting later
    task_validity_scores = []

    for task_id, result in detailed_results.items():
        # Ensure we handle potential errors in validation gracefully
        if "valid_word_count" in result:
            count = result["valid_word_count"]
            validity_distribution[count] += 1
            task_validity_scores.append((int(task_id), count))

    # 2. Print Distribution Stats
    print("\n=== Puzzle Validity Distribution ===")
    print(f"Total Puzzles Checked: {total_tasks}")
    print("Valid Words per Puzzle | Number of Puzzles")
    print("-----------------------|------------------")
    # Sort by word count descending (10 down to 0)
    for word_count in sorted(validity_distribution.keys(), reverse=True):
        num_puzzles = validity_distribution[word_count]
        percentage = (num_puzzles / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"{word_count:22d} | {num_puzzles:6d} ({percentage:.1f}%)")

    # 3. Get Top 10 Indices
    # Sort primarily by valid_word_count (descending), secondarily by task_id (ascending) for stability
    task_validity_scores.sort(key=lambda x: (-x[1], x[0]))
    
    top_10_indices = [idx for idx, score in task_validity_scores[:10]]
    
    print("\n=== Top 10 'Cleanest' Puzzles ===")
    print(f"Indices of puzzles with the most valid words (Max: {task_validity_scores[0][1]}/10):")
    print(top_10_indices)

if __name__ == '__main__':
    analyze_report()