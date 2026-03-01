import json
from pathlib import Path

def load_history(history_path):
    with open(history_path, "r") as f:
        return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
