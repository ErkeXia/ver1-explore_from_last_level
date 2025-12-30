import json
import os
import shutil

# --- Configuration ---
# Update this path if your cache directory is different
# Based on your tot/cache.py, it should be in src/tot/caches/ or similar
CACHE_DIR = "./src/tot/caches" 
TARGET_FILES = [
    "crossword_propose_cache_gpt.json",
    "crossword_propose_cache_llama.json"
]

def migrate_cache_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping: '{filepath}' not found.")
        return

    print(f"Processing: {filepath}")
    
    # 1. Create a backup
    backup_path = filepath + ".bak"
    shutil.copy2(filepath, backup_path)
    print(f"  - Backup created at: {backup_path}")

    # 2. Load data
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"  - Error decoding JSON. Skipping.")
            return

    # 3. Transform data
    migrated_count = 0
    new_data = {}
    
    for key, value in data.items():
        # Check if value matches old format: [str, number]
        # Example: ["apple", 1.0]
        is_old_format = (
            isinstance(value, list) and 
            len(value) == 2 and 
            isinstance(value[0], str) and 
            isinstance(value[1], (int, float))
        )

        if is_old_format:
            # Wrap it in a list to make it [[word, score]]
            new_data[key] = [value]
            migrated_count += 1
        else:
            # Keep as is (already new format or empty)
            new_data[key] = value

    # 4. Save updated data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"  - Migration complete. Updated {migrated_count} entries.")

if __name__ == "__main__":
    # Ensure we are looking in the right place relative to where the script is run
    # If the script is in root, and caches are in tot/caches, adjust path logic if needed
    
    # Auto-detect common cache locations if the default doesn't exist
    possible_dirs = [
        CACHE_DIR,
        "./tot/caches",
        "./caches",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tot/caches")
    ]
    
    found_dir = None
    for d in possible_dirs:
        if os.path.isdir(d):
            found_dir = d
            break
            
    if not found_dir:
        print(f"Error: Could not find cache directory. Checked: {possible_dirs}")
        exit(1)
        
    print(f"Using cache directory: {found_dir}")

    for filename in TARGET_FILES:
        full_path = os.path.join(found_dir, filename)
        migrate_cache_file(full_path)