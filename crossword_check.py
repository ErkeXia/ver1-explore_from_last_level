import json
import os
import sys

# Default path based on your project structure
# Adjust this if your data is located elsewhere
DEFAULT_DATA_FILE = "./src/tot/data/crosswords/mini0505.json"

def count_input_problems(file_path):
    """
    Counts the number of problems in a standard JSON list file
    like 'mini0505.json'.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please check the path and try again.")
        return

    try:
        print(f"Reading file: '{file_path}'...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = len(data)
            print(f"\nSuccess! Found {count} crossword problems in this file.")
        else:
            print("\nError: Unexpected JSON structure. Expected a list of problems.")
            
    except json.JSONDecodeError as e:
        print(f"\nError: Failed to parse valid JSON from file: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    # Use command line argument if provided, otherwise use default
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE
    count_input_problems(target_file)