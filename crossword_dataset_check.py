import json
import os
import time
from tqdm import tqdm
from functools import partial
import argparse

# Import your existing infrastructure
from tot.tasks.crosswords import MiniCrosswordsEnv
from tot.models import gpt

# Configuration
OUTPUT_FILE = "dataset_validation_report.json"

# Define the validation prompt
VALIDATION_SYSTEM_PROMPT = """You are an expert lexicographer for standard English crossword puzzles. 
Your job is to validate whether a given list of words are acceptable entries in a standard mini crossword.
Acceptable entries include:
- Standard English words found in common dictionaries.
- Common proper nouns (names of famous people, major cities, countries, etc.).
- Standard abbreviations (e.g., 'USA', 'ETC').

Unacceptable entries include:
- obscure gibberish or non-words (like 'MAMOU' if it's considered too obscure).
- random strings of letters.

For each word provided, reply with exactly "Valid" or "Invalid".
"""

VALIDATION_USER_PROMPT_TEMPLATE = """Assess these 10 words:
{words_list}

Output format (one line per word, matching the order above):
1. [Word] - [Valid/Invalid]
2. [Word] - [Valid/Invalid]
...
"""

def validate_all_tasks(model_name="gpt-3.5-turbo", temperature=0.0):
    print(f"Starting dataset validation using {model_name}...")
    
    # Initialize environment to load data
    env = MiniCrosswordsEnv()
    total_tasks = len(env)
    print(f"Loaded {total_tasks} tasks from {env.file.name if hasattr(env.file, 'name') else 'dataset'}.")

    # Prepare GPT partial function
    gpt_validator = partial(gpt, model=model_name, temperature=temperature)

    results = {}
    total_invalid_words_found = 0

    for i in tqdm(range(total_tasks), desc="Validating Tasks"):
        # Load the ground truth for this task
        env.reset(i)
        ground_truth_words = env.ans_gt
        
        # Format words for the prompt (e.g., "1. APPLE\n2. PEAR...")
        formatted_list = "\n".join([f"{j+1}. {word}" for j, word in enumerate(ground_truth_words)])
        
        # Create the full prompt
        full_prompt = f"{VALIDATION_SYSTEM_PROMPT}\n\n{VALIDATION_USER_PROMPT_TEMPLATE.format(words_list=formatted_list)}"
        
        # Get LLM judgement
        try:
            response = gpt_validator(full_prompt, n=1, stop=None, max_tokens=500)[0]
            
            # Parse the response
            task_valid_count = 0
            task_invalid_words = []
            
            lines = response.strip().split('\n')
            for j, line in enumerate(lines):
                # Expected format: "1. WORD - Status"
                if '-' in line:
                    parts = line.split('-')
                    word = parts[0].split('.')[1].strip()
                    status = parts[1].strip().lower()
                    
                    if 'invalid' in status:
                        task_invalid_words.append(word)
                    elif 'valid' in status:
                        task_valid_count += 1
            
            total_invalid_words_found += len(task_invalid_words)
            
            results[i] = {
                "valid_word_count": task_valid_count,
                "invalid_words": task_invalid_words,
                "all_words": ground_truth_words
            }

        except Exception as e:
            print(f"\nError validating task {i}: {e}")
            results[i] = {"error": str(e)}

    # Calculate summary stats
    tasks_perfect = sum(1 for r in results.values() if r.get("valid_word_count") == 10)
    
    summary = {
        "total_tasks": total_tasks,
        "tasks_with_perfect_10_valid_words": tasks_perfect,
        "total_invalid_words_detected": total_invalid_words_found,
        "detailed_results": results
    }

    # Save complete report
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Validation Complete ===")
    print(f"Tasks with all 10 words valid: {tasks_perfect}/{total_tasks}")
    print(f"Total invalid words detected across dataset: {total_invalid_words_found}")
    print(f"Detailed report saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    # You can optionally add argparse here to choose model/temperature
    validate_all_tasks()