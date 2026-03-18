import re

def filter_log_file():
    # Log file path
    log_file_path = 'logs/strategy.log'
    
    # Read the original content
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Filter each line
    filtered_lines = []
    for line in lines:
        # Replace sensitive information with placeholders
        if "[DEBUG] get_fyers_client:" in line:
            # Filter client_id
            line = re.sub(r'client_id=[^,\s]+', 'client_id=***FILTERED***', line)
            # Filter access_token
            line = re.sub(r'access_token_head=[^,\s\n]+', 'access_token_head=***FILTERED***', line)
            # Filter token combo
            line = re.sub(r'token_combo=[^,\s\n]+', 'token_combo=***FILTERED***', line)
        filtered_lines.append(line)
    
    # Write back to the file
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
    
    print(f"Filtered sensitive information from {log_file_path}")

# Run the filter function
if __name__ == "__main__":
    filter_log_file()
