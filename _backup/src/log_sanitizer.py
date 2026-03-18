import os
import sys
import re
import time
import logging
from pathlib import Path
import atexit

def filter_log_file(log_file_path='logs/strategy.log'):
    """Filter sensitive information from log file"""
    try:
        if not os.path.exists(log_file_path):
            logging.error(f"Log file not found: {log_file_path}")
            return
            
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        filtered_lines = []
        removed_lines = 0
        sensitive_info_count = 0
        option_data_count = 0
        
        for line in lines:
            # Skip option chain data structure logs completely
            if "Sample option data structure:" in line or "Option data structure fields:" in line:
                removed_lines += 1
                option_data_count += 1
                continue
                
            # Filter sensitive authentication information
            if "[DEBUG] get_fyers_client:" in line:
                line = "[DEBUG] get_fyers_client: <CREDENTIALS FILTERED>\n"
                sensitive_info_count += 1
            
            # Filter other types of auth tokens
            line = re.sub(
                r'(\[DEBUG\] get_fyers_client: client_id=)[^,\s]+', 
                r'\1***FILTERED***', 
                line
            )
            
            line = re.sub(
                r'(access_token_head=)[^,\s\n]+', 
                r'\1***FILTERED***', 
                line
            )
            
            line = re.sub(
                r'(token_combo=)[^,\s\n]+', 
                r'\1***FILTERED***', 
                line
            )
            
            filtered_lines.append(line)
            
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
            
        print(f"Successfully filtered sensitive information from {log_file_path}")
        print(f"  - Removed {removed_lines} lines with excessive option data")
        print(f"  - Filtered {sensitive_info_count} lines with sensitive authentication data")
    except Exception as e:
        print(f"Error filtering log file: {str(e)}")

def start_log_monitor(log_file_path='logs/strategy.log', check_interval=5):
    """Start a background monitor to filter logs in real-time"""
    import threading
    
    def monitor_thread():
        print(f"Starting log monitor for {log_file_path} (check every {check_interval} seconds)")
        while True:
            try:
                filter_log_file(log_file_path)
                time.sleep(check_interval)
            except Exception as e:
                print(f"Error in log monitor: {str(e)}")
                time.sleep(check_interval)
    
    # Start the monitor thread
    thread = threading.Thread(target=monitor_thread, daemon=True)
    thread.start()
    
    # Register cleanup at exit
    atexit.register(filter_log_file, log_file_path)
    
    return thread

def find_and_fix_sensitive_logs(directory='logs'):
    """Find and filter sensitive information from all log files in directory"""
    log_pattern = re.compile(r'\.log$|\.log\.')
    fixed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if log_pattern.search(file):
                log_path = os.path.join(root, file)
                print(f"Checking {log_path}...")
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for sensitive auth information
                    has_auth_info = "[DEBUG] get_fyers_client:" in content and ("client_id=" in content or "access_token" in content)
                    
                    # Check for option chain data structure logs
                    has_option_data = "Sample option data structure:" in content or "Option data structure fields:" in content
                    
                    if has_auth_info or has_option_data:
                        print(f"Found sensitive info in {log_path}, filtering...")
                        filter_log_file(log_path)
                        fixed_count += 1
                except Exception as e:
                    print(f"Error processing {log_path}: {str(e)}")
    
    print(f"Filtered sensitive information from {fixed_count} log files")

if __name__ == "__main__":
    # Check if we should run as a background monitor
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        monitor_thread = start_log_monitor()
        print("Log monitor started in background. Press Ctrl+C to stop.")
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Log monitor stopping...")
    else:
        # Default: clean all logs immediately
        find_and_fix_sensitive_logs()
        # Filter current strategy log
        filter_log_file()
