#!/usr/bin/env python
"""
Clean Strategy Runner - Run the strategy with log cleaning
"""
import os
import sys
import time
import logging

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Import the log sanitizer first
from src.log_sanitizer import filter_log_file, find_and_fix_sensitive_logs

def main():
    # Clean existing logs
    print("Cleaning existing logs before running strategy...")
    find_and_fix_sensitive_logs()
    
    # Run the main strategy
    print("Running strategy...")
    try:
        from src.main import main as run_main
        run_main()
    except Exception as e:
        print(f"Error running strategy: {str(e)}")
    finally:
        # Clean logs after running
        print("Filtering logs to remove any sensitive information...")
        find_and_fix_sensitive_logs()
        filter_log_file()  # Make sure current strategy log is cleaned
    
    print("Strategy execution completed with filtered logs!")

if __name__ == "__main__":
    main()
