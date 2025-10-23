import yaml
import os
import sys

# Make sure the config path is relative to the project root
def load_config(path=None):
    if path is None:
        # Determine the correct path based on execution context
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        path = os.path.join(project_root, "config", "config.yaml")
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {path}")
        sys.exit(1)