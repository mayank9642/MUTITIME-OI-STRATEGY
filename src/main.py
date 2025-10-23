import schedule
import time
import logging
import datetime
import os
import sys
import pytz

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use src namespace for imports
from src.strategy import OpenInterestStrategy
from src.config import load_config
from src.token_helper import ensure_valid_token
from src.auth import generate_access_token

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='logs/strategy.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add a filter to remove sensitive information from logs
class SensitiveInfoFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        
        # Filter sensitive authentication info
        if message.startswith("[DEBUG] get_fyers_client:"):
            # Replace the message with a filtered version
            record.msg = "[DEBUG] get_fyers_client: <CREDENTIALS FILTERED>"
            return True
            
        # Filter repetitive option chain data structure logs
        if "Sample option data structure:" in message:
            # Drop this message entirely to reduce log size
            return False
            
        return True

# Apply filter to root logger
logging.getLogger().addFilter(SensitiveInfoFilter())

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
        logging.info("Starting Open Interest Option Buying Strategy (manual mode)...")
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        market_open = datetime.time(9, 15)
        analysis_time = datetime.time(9, 20)
        # Ensure we have a valid access token before starting
        logging.info("Checking authentication status...")
        access_token = ensure_valid_token()
        if not access_token:
            logging.warning("No valid access token found. Attempting to generate one...")
            access_token = generate_access_token()
            if not access_token:
                logging.error("Failed to generate access token. Please run auth.py separately.")
                raise Exception("Authentication failed")
        # Create strategy instance
        strategy_instance = OpenInterestStrategy()
        # Initialize strategy for the day
        if not strategy_instance.initialize_day():
            logging.error("Failed to initialize strategy. Exiting.")
            raise Exception("Strategy initialization failed")
        logging.info("Strategy initialized successfully.")

        # --- LOGIC: before 9:15, between 9:15-9:20, after 9:20 ---
        while True:
            now = datetime.datetime.now(ist)
            current_time = now.time()
            if current_time < market_open:
                logging.info(f"Current IST time {current_time.strftime('%H:%M:%S')} is before market open (09:15). Waiting for market to open...")
                print(f"Waiting for market to open (09:15 IST). Current time: {current_time.strftime('%H:%M:%S')}")
                time.sleep(10)
                continue
            elif market_open <= current_time < analysis_time:
                logging.info(f"Current IST time {current_time.strftime('%H:%M:%S')} is after market open but before 9:20. Waiting for 9:20...")
                print(f"Market is open. Waiting for 9:20 (OI analysis). Current time: {current_time.strftime('%H:%M:%S')}")
                # Wait until 9:20
                while True:
                    now = datetime.datetime.now(ist)
                    if now.time() >= analysis_time:
                        break
                    time.sleep(5)
                # At 9:20, run strategy
                logging.info("It's 9:20 IST. Running OI analysis and strategy.")
                strategy_instance.run_strategy()
                return {"success": True, "message": "Strategy executed at 9:20"}
            elif current_time >= analysis_time:
                # After 9:20, prompt user
                logging.info(f"Current IST time {current_time.strftime('%H:%M:%S')} is after 9:20. Prompting user for immediate run.")
                print("It is after 9:20 AM IST. Do you want to skip the 9:20 logic and run immediately? (y/n): ", end="")
                response = input()
                if response.lower().startswith('y'):
                    logging.info("User chose to bypass 9:20 logic - running immediately")
                    strategy_instance.run_strategy(force_analysis=True)
                    return {"success": True, "message": "Strategy executed after 9:20 (user bypass)"}
                else:
                    logging.info("User chose not to bypass 9:20 logic. Exiting.")
                    print("Exiting. Please run the script before 9:20 for normal operation.")
                    return {"success": False, "message": "User did not bypass 9:20 logic"}
            else:
                # Should not reach here
                logging.error("Unexpected time logic in main(). Exiting.")
                return {"success": False, "error": "Unexpected time logic"}
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        return {"success": False, "error": str(e)}

def run_immediate():
    """Run the strategy immediately once, without scheduling"""
    try:
        logging.info("Starting Open Interest Option Buying Strategy (immediate mode)...")
        
        # Ensure we have a valid access token before starting
        logging.info("Checking authentication status...")
        access_token = ensure_valid_token()
        
        if not access_token:
            logging.warning("No valid access token found. Attempting to generate one...")
            access_token = generate_access_token()
            
            if not access_token:
                logging.error("Failed to generate access token. Please run auth.py separately.")
                raise Exception("Authentication failed")
        
        # Create strategy instance
        strategy_instance = OpenInterestStrategy()
        # Initialize strategy for the day
        if not strategy_instance.initialize_day():
            logging.error("Failed to initialize strategy. Exiting.")
            raise Exception("Strategy initialization failed")
        logging.info("Strategy initialized successfully.")

        # Run the strategy once immediately
        result = strategy_instance.run_strategy()
        logging.info("Strategy execution completed.")
        return result
    except Exception as e:
        logging.error(f"Error running immediate strategy: {str(e)}")
        return {"success": False, "error": str(e)}

def run_scheduled():
    """Run the strategy with normal scheduling"""
    try:
        logging.info("Starting Open Interest Option Buying Strategy (scheduled mode)...")
        
        # Ensure we have a valid access token before starting
        logging.info("Checking authentication status...")
        access_token = ensure_valid_token()
        
        if not access_token:
            logging.warning("No valid access token found. Attempting to generate one...")
            access_token = generate_access_token()
            
            if not access_token:
                logging.error("Failed to generate access token. Please run auth.py separately.")
                raise Exception("Authentication failed")
        
        logging.info("Authentication successful. Creating strategy instance...")
        
        # Create strategy instance
        strategy_instance = OpenInterestStrategy()
        # Initialize strategy for the day
        if not strategy_instance.initialize_day():
            logging.error("Failed to initialize strategy. Exiting.")
            raise Exception("Strategy initialization failed")
        logging.info("Strategy initialized successfully.")

        # Schedule jobs
        # Correct way to schedule for weekdays at 9:15 AM
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
            schedule.every().__getattribute__(day).at("09:15").do(strategy_instance.initialize_day)
        
        # Run the job every minute during market hours
        schedule.every(1).minutes.do(lambda: job(strategy_instance))
        
        logging.info("Strategy scheduled. Running main loop...")
        
        # Main loop
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Strategy execution stopped by user.")
        if 'strategy_instance' in locals():
            strategy_instance.cleanup()
        return {"success": False, "error": "User interrupted"}
    except Exception as e:
        logging.critical(f"Unhandled exception: {str(e)}")
        if 'strategy_instance' in locals():
            strategy_instance.cleanup()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Manual mode: robust time-based logic for user scenarios
    logging.info("[HEARTBEAT] Strategy main loop starting in manual mode.")
    main()