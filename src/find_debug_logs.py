import os
import re
import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def find_debug_log_sources():
    """Find all sources that might be adding debug logs with token information"""
    # Get the source directory
    src_dir = Path(__file__).parent / 'src'
    
    # Find files with potential token logging
    token_patterns = [
        r"logging\.(?:debug|info|warning|error|critical).*token",
        r"logging\.(?:debug|info|warning|error|critical).*client_id",
        r"\[DEBUG\]",
        r"DEBUG.*token",
        r"debug.*token"
    ]
    
    suspicious_files = []
    for pattern in token_patterns:
        for python_file in src_dir.rglob("*.py"):
            with open(python_file, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if re.search(pattern, content, re.IGNORECASE):
                    suspicious_files.append((python_file, pattern))
                    logging.info(f"Found potential token logging in {python_file} matching pattern: {pattern}")

    # Check for monkey patching of the logging module
    for python_file in src_dir.rglob("*.py"):
        with open(python_file, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if "logging.orig_" in content or "orig_logging" in content or "patch" in content and "logging" in content:
                suspicious_files.append((python_file, "Potential logging monkey patching"))
                logging.info(f"Found potential logging modification in {python_file}")

    return suspicious_files

def check_strategy_log_debug_entries():
    """Check strategy.log for debug entries related to tokens"""
    log_file = Path(__file__).parent / 'logs' / 'strategy.log'
    token_patterns = [
        r"\[DEBUG\].*client_id=.*token",
        r"\[DEBUG\].*access_token"
    ]
    
    found_entries = []
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as file:
            for i, line in enumerate(file):
                for pattern in token_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        found_entries.append((i+1, line.strip()))
                        logging.info(f"Found token debug entry at line {i+1}: {line.strip()}")
                        
    return found_entries

if __name__ == "__main__":
    print("Searching for debug logging of tokens...")
    suspicious_files = find_debug_log_sources()
    debug_entries = check_strategy_log_debug_entries()
    
    print("\n--- Summary Report ---")
    if suspicious_files:
        print(f"Found {len(suspicious_files)} files with potential token logging:")
        for file, pattern in suspicious_files:
            print(f" - {file} (matched: {pattern})")
    else:
        print("No suspicious token logging found in source files.")
        
    if debug_entries:
        print(f"\nFound {len(debug_entries)} debug entries with token information:")
        for line_num, content in debug_entries[:5]:
            print(f" - Line {line_num}: {content}")
        if len(debug_entries) > 5:
            print(f" ... and {len(debug_entries) - 5} more entries")
    else:
        print("\nNo token debug entries found in log file.")
