"""
Patching script for OpenInterestStrategy class
This script will patch the strategy.py file to add any missing methods required for initialization
"""

import os
import re
import shutil
import datetime

# Backup the original file
strategy_file = "src/strategy.py"
backup_file = f"src/strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py.bak"

print(f"Backing up {strategy_file} to {backup_file}")
shutil.copy2(strategy_file, backup_file)

# Read the original file content
with open(strategy_file, 'r') as f:
    content = f.read()

# Define the methods we need to ensure exist
run_self_diagnostic_method = '''
    def run_self_diagnostic(self):
        """
        Run a self-diagnostic check to verify key components are functioning
        """
        logging.info("Running self-diagnostic check...")
        diagnostics_passed = True
        
        # Check 1: Test authentication
        try:
            from src.token_helper import is_token_valid
            token_valid = is_token_valid()
            if token_valid:
                logging.info("✓ Authentication token is valid")
            else:
                logging.error("✗ Authentication token is invalid or expired")
                diagnostics_passed = False
        except Exception as e:
            logging.error(f"✗ Authentication check failed: {str(e)}")
            diagnostics_passed = False
            
        # Check 2: Test API connectivity
        try:
            from src.fyers_api_utils import get_nifty_spot_price
            spot_price = get_nifty_spot_price()
            if spot_price and spot_price > 0:
                logging.info(f"✓ API connectivity verified - Nifty spot price: {spot_price}")
            else:
                logging.error("✗ Failed to fetch Nifty spot price - API connection issue")
                diagnostics_passed = False
        except Exception as e:
            logging.error(f"✗ API connectivity check failed: {str(e)}")
            diagnostics_passed = False
            
        # Check 3: Test option chain retrieval 
        try:
            from src.nse_data_new import get_nifty_option_chain
            option_chain = get_nifty_option_chain()
            if option_chain is not None and not option_chain.empty:
                logging.info(f"✓ Option chain retrieval verified - Got {len(option_chain)} options")
            else:
                logging.error("✗ Failed to retrieve option chain data")
                diagnostics_passed = False
        except Exception as e:
            logging.error(f"✗ Option chain retrieval check failed: {str(e)}")
            diagnostics_passed = False
            
        # Check 4: Test Excel file access
        try:
            import pandas as pd
            test_df = pd.DataFrame([{"test": "data"}])
            test_path = "logs/diagnostic_test.xlsx"
            with pd.ExcelWriter(test_path, engine='openpyxl') as writer:
                test_df.to_excel(writer, index=False)
            import os
            os.remove(test_path)
            logging.info("✓ Excel file writing and access verified")
        except Exception as e:
            logging.error(f"✗ Excel file access check failed: {str(e)}")
            diagnostics_passed = False
            
        if diagnostics_passed:
            logging.info("✓✓✓ All diagnostic checks passed! Strategy ready to run.")
        else:
            logging.error("✗✗✗ Some diagnostic checks failed. Please check the logs for details.")
            
        return diagnostics_passed
'''

clear_logs_method = '''
    def clear_logs(self):
        """Clear log file for a fresh start to the trading day"""
        try:
            log_file = 'logs/strategy.log'
            if os.path.exists(log_file):
                # Keep existing logs by backing up current log file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f'logs/strategy_{timestamp}.log.bak'
                
                # Copy to backup before clearing
                if os.path.getsize(log_file) > 0:
                    with open(log_file, 'r') as src, open(backup_file, 'w') as dst:
                        dst.write(src.read())
                    logging.info(f"Log file backed up to {backup_file}")
                    
                # Clear the current log file
                with open(log_file, 'w') as f:
                    f.write(f"Log file cleared on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                logging.info("Log file has been cleared for new trading day")
        except Exception as e:
            logging.warning(f"Error clearing logs: {str(e)}")
            # Continue execution even if log clearing fails
'''

# Check if the methods exist already
run_self_diagnostic_exists = 'def run_self_diagnostic' in content
clear_logs_exists = 'def clear_logs' in content

# Find the class definition
class_match = re.search(r'class OpenInterestStrategy:', content)
if not class_match:
    print("ERROR: Could not find OpenInterestStrategy class definition")
    exit(1)

# Find a good place to add the methods (after __init__ method)
init_end_match = re.search(r'def __init__.*?\n(\s+)self\.max_unrealized_loss_pct = 0', content, re.DOTALL)
if init_end_match:
    indent = init_end_match.group(1)
    insertion_point = init_end_match.end()
    
    # Prepare methods with correct indentation
    methods_to_add = ""
    
    if not clear_logs_exists:
        print("Adding clear_logs method")
        methods_to_add += clear_logs_method
        
    if not run_self_diagnostic_exists:
        print("Adding run_self_diagnostic method")
        methods_to_add += run_self_diagnostic_method
    
    # Insert methods after __init__
    new_content = content[:insertion_point] + "\n" + methods_to_add + content[insertion_point:]
    
    # Write the modified file
    with open(strategy_file, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {strategy_file} successfully")
else:
    print("ERROR: Could not find a suitable insertion point")
    exit(1)

# Also modify the initialize_day method to handle missing methods gracefully
initialize_day_pattern = re.compile(r'def initialize_day\(self\).*?# Run self-diagnostic check\s+diagnostics_passed = self\.run_self_diagnostic\(\)', re.DOTALL)
initialize_day_match = initialize_day_pattern.search(content)

if initialize_day_match:
    replacement = '''def initialize_day(self):
        """Reset variables for a new trading day"""
        # Check for valid token before starting the trading day
        try:
            # Clear the logs for a fresh start
            try:
                self.clear_logs()
            except AttributeError:
                # If clear_logs doesn't exist, create it directly
                logging.info("clear_logs method not found, creating backup log now")
                log_file = 'logs/strategy.log'
                if os.path.exists(log_file):
                    # Keep existing logs by backing up current log file
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = f'logs/strategy_{timestamp}.log.bak'
                    
                    # Copy to backup before clearing
                    if os.path.getsize(log_file) > 0:
                        with open(log_file, 'r') as src, open(backup_file, 'w') as dst:
                            dst.write(src.read())
                        logging.info(f"Log file backed up to {backup_file}")
                        
                    # Clear the current log file
                    with open(log_file, 'w') as f:
                        f.write(f"Log file cleared on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                    logging.info("Log file has been cleared for new trading day")
            
            # Reset all state variables for a clean start
            self.reset_state()
            
            access_token = ensure_valid_token()
            if access_token:
                self.fyers = get_fyers_client(check_token=False)  # Token already checked
                logging.info("Authentication verified for today's trading session")
            else:
                logging.error("Failed to obtain valid access token for today's session")
                return False
                
            # Close any existing websocket connection
            if self.data_socket:
                try:
                    self.data_socket.close_connection()
                    logging.info("Closed previous websocket connection")
                except Exception as e:
                    logging.error(f"Error closing previous websocket: {str(e)}")
                    
            self.data_socket = None
              # Reset trading variables
            self.reset_state()
            
            # Run self-diagnostic check
            try:
                diagnostics_passed = self.run_self_diagnostic()'''
                
    # Replace the initialize_day method
    with open(strategy_file, 'r') as f:
        content = f.read()
    
    new_content = initialize_day_pattern.sub(replacement, content)
    
    with open(strategy_file, 'w') as f:
        f.write(new_content)
    
    print("Updated initialize_day method to handle missing methods gracefully")
else:
    print("WARNING: Could not find initialize_day method to update")

print("Patch completed successfully")
