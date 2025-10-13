import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, "caches")

class FileCache:
    """
    A simple file-based cache that reliably stores files in the project's
    'caches' directory, regardless of where the script is run from.
    """
    def __init__(self, filename="crossword_cache.json"):
        # The filename passed in is now just the name of the file, not the path.
        # We construct the full, reliable path by joining it with our CACHE_DIR.
        self.cache_file = os.path.join(CACHE_DIR, filename)
        self.data = {}
        
        # Ensure the cache directory exists.
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load existing cache from file upon initialization
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content: # Ensure file is not empty before loading
                        self.data = json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache file '{self.cache_file}'. Starting fresh. Error: {e}")
                self.data = {}

    def get(self, key: str):
        """Return cached result if present, else None."""
        return self.data.get(key)

    def set(self, key: str, value):
        """Store result and write to disk."""
        self.data[key] = value
            
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)