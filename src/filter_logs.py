import re
import os
import logging
from pathlib import Path

def filter_sensitive_log_file(log_file_path):
    """
    Filters sensitive information from a log file
    
    Args:
        log_file_path: Path to the log file
    """
    try:
        # Check if file exists
        if not os.path.exists(log_file_path):
            logging.error(f"Log file not found: {log_file_path}")
            return False
            
        # Read the original content
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Filter sensitive information
        # Filter client_id
        filtered_content = re.sub(
            r'(\[DEBUG\] get_fyers_client: client_id=)[^,\s]+', 
            r'\1***FILTERED***', 
            content
        )
        
        # Filter access token
        filtered_content = re.sub(
            r'(access_token_head=)[^,\s\n]+', 
            r'\1***FILTERED***', 
            filtered_content
        )
        
        # Filter token combo
        filtered_content = re.sub(
            r'(token_combo=)[^,\s\n]+', 
            r'\1***FILTERED***', 
            filtered_content
        )
        
        # Write back to the file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(filtered_content)
            
        logging.info(f"Successfully filtered sensitive information from {log_file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error filtering log file: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Path to the strategy log file
    log_file = "logs/strategy.log"
    
    # Filter sensitive information
    if filter_sensitive_log_file(log_file):
        print(f"Successfully filtered sensitive information from {log_file}")
    else:
        print(f"Failed to filter sensitive information from {log_file}")
