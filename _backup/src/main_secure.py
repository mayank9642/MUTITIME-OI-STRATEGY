import schedule
import time
import logging
import datetime
import os
import sys
import pytz
import threading
import re
import atexit

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure a filter to prevent sensitive information from being logged
class SensitiveInfoFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'msg'):
            return True
            
        # Check if this is a string message that might contain sensitive info
        if isinstance(record.msg, str):
            # Check for token debug logs and filter them
            if "[DEBUG] get_fyers_client:" in record.msg:
                record.msg = "[DEBUG] get_fyers_client: <credentials filtered>"
                return True
                
            # Check for any sensitive tokens in messages
            if "client_id=" in record.msg and "access_token" in record.msg:
                # Filter out client_id and token information
                record.msg = re.sub(r'client_id=[^,\s]+', 'client_id=***FILTERED***', record.msg)
                record.msg = re.sub(r'access_token_head=[^,\s\n]+', 'access_token_head=***FILTERED***', record.msg)
                record.msg = re.sub(r'token_combo=[^,\s\n]+', 'token_combo=***FILTERED***', record.msg)
                
        return True  # Always allow the log record, but with filtered content

# Function to sanitize log files
def filter_log_file(log_file_path='logs/strategy.log'):
    try:
        if not os.path.exists(log_file_path):
            return
            
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Filter sensitive information
        filtered_content = re.sub(
            r'(\[DEBUG\] get_fyers_client: client_id=)[^,\s]+', 
            r'\1***FILTERED***', 
            content
        )
        
        filtered_content = re.sub(
            r'(access_token_head=)[^,\s\n]+', 
            r'\1***FILTERED***', 
            filtered_content
        )
        
        filtered_content = re.sub(
            r'(token_combo=)[^,\s\n]+', 
            r'\1***FILTERED***', 
            filtered_content
        )
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(filtered_content)
    except Exception as e:
        print(f"Error filtering log file: {str(e)}")

# Setup a background thread to periodically sanitize the log file
def start_log_sanitizer():
    def sanitizer_thread():
        while True:
            filter_log_file()
            time.sleep(30)  # Check every 30 seconds
    
    thread = threading.Thread(target=sanitizer_thread, daemon=True)
    thread.start()
    return thread

# Register log file sanitization at program exit
atexit.register(filter_log_file)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Setup logging with our sensitive info filter
logging.basicConfig(
    filename='logs/strategy.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add our filter to the root logger
root_logger = logging.getLogger()
root_logger.addFilter(SensitiveInfoFilter())

# Start the log sanitizer
sanitizer_thread = start_log_sanitizer()

# Use src namespace for imports
from src.strategy import OpenInterestStrategy
from src.config import load_config
from src.token_helper import ensure_valid_token
from src.auth import generate_access_token

def job(strategy):
    """Run the strategy job at specified intervals"""
    try:
        logging.info("Heartbeat: job running at " + str(datetime.datetime.now()))
        strategy.run_strategy()
    except Exception as e:
        logging.error(f"Error in scheduled job: {str(e)}")

def check_past_nine_twenty():
    """Check if current IST time is past 9:20 AM"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    
    # Check if current time is past 9:20 AM
    return (now.hour > 9) or (now.hour == 9 and now.minute >= 20)

def prompt_for_immediate_run():
    """Prompt user if they want to run immediately bypassing 9:20 AM check"""
    response = input("It is after 9:20 AM IST. Do you want to skip the 9:20 check and run immediately? (y/n): ")
    return response.lower().startswith('y')

def main():
    """Main function to start the strategy"""
    try:
        # Clear the log file for a new day
        with open('logs/strategy.log', 'w') as f:
            f.write(f"Log file cleared on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logging.info("Log file has been cleared for new trading day")
        
        # Verify authentication token
        access_token = ensure_valid_token()
        if not access_token:
            logging.error("Failed to get valid authentication token. Please run the auth script.")
            print("Authentication failed. Please run python -m src.auth to authenticate.")
            return
        
        logging.info("Authentication verified for today's trading session")
        
        # Initialize strategy
        strategy = OpenInterestStrategy()
        
        # Run diagnostics to ensure everything is ready
        logging.info("Running self-diagnostic check...")
        diagnostic_success = strategy.run_diagnostics()
        if not diagnostic_success:
            logging.error("Strategy diagnostics failed. See log for details.")
            print("Strategy diagnostics failed. Check logs/strategy.log for details.")
            return
        
        logging.info("Strategy initialized successfully.")
        
        # Check if current time is after 9:20 AM IST
        if check_past_nine_twenty():
            logging.info("Current IST time {} is after 9:20. Prompting user for immediate run.".format(
                datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')))
            
            if prompt_for_immediate_run():
                logging.info("User chose to bypass 9:20 logic - running immediately")
                strategy.run_strategy()
            else:
                logging.info("User chose to wait for scheduled run at 9:20 AM IST tomorrow")
                schedule.every().day.at("09:20").do(job, strategy=strategy)
        else:
            # Schedule the strategy to run at 9:20 AM IST
            logging.info("Scheduling strategy to run at 9:20 AM IST")
            schedule.every().day.at("09:20").do(job, strategy=strategy)
        
        # Also run every hour for monitoring
        schedule.every(1).hours.do(job, strategy=strategy)
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Strategy manually stopped by user")
        print("\nStrategy manually stopped by user")
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting strategy with secure logging...")
    main()
