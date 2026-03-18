"""
Run the automated options trading strategy
"""
import os
import sys
import logging
import logging.handlers
import datetime
import subprocess
import atexit

# Configure logging
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure root logger with both console and file handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Clear any existing handlers and close them to avoid file locking
for handler in logger.handlers[:]:
    try:
        handler.close()
    except Exception:
        pass
    logger.removeHandler(handler)

# Create formatter (compact, single-line timestamps) to match requested log style
# Format: 2026-03-16 08:57:37 - INFO - message (no PID prefix)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Add console handler with stream lock for thread safety
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.createLock()
logger.addHandler(console_handler)

# Add file handler for main log with file lock



# Add file handler for strategy log only (avoid double file handlers)
strategy_log_handler = logging.handlers.RotatingFileHandler(
    'logs/strategy.log',
    maxBytes=10*1024*1024,
    backupCount=5,
    delay=True
)
strategy_log_handler.setFormatter(formatter)
logger.addHandler(strategy_log_handler)

logging.info("Logging configured with console and file handlers")

# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logging.info(f"Added current directory to Python path: {current_dir}")

# Print Python path for debugging
logging.info(f"Python path includes: {sys.path}")

# --- PID / lockfile handling ---
LOCKFILE = os.path.join(current_dir, 'run_strategy.pid')

def is_pid_running(pid):
    """Return True if a process with pid is running on Windows using tasklist."""
    try:
        # Use tasklist to check for the pid; returns non-empty if running
        cmd = ['tasklist', '/FI', f'PID eq {int(pid)}', '/FO', 'CSV', '/NH']
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
        out = out.strip()
        return bool(out) and '"' in out
    except Exception:
        return False

def write_lockfile():
    pid = os.getpid()
    try:
        with open(LOCKFILE, 'w') as f:
            f.write(f"{pid}\n")
        logging.info(f"Wrote lockfile {LOCKFILE} with PID {pid}")
    except Exception as e:
        logging.error(f"Failed to write lockfile {LOCKFILE}: {e}")

def remove_lockfile():
    try:
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
            logging.info(f"Removed lockfile {LOCKFILE}")
    except Exception as e:
        logging.error(f"Failed to remove lockfile {LOCKFILE}: {e}")

def check_existing_lockfile():
    """If a lockfile exists and the PID is running, refuse to start. If stale, remove it."""
    try:
        if not os.path.exists(LOCKFILE):
            return True
        with open(LOCKFILE, 'r') as f:
            content = f.read().strip()
        if not content:
            # empty file, remove and continue
            remove_lockfile()
            return True
        pid = int(content.splitlines()[0])
        if is_pid_running(pid):
            logging.error(f"Another run_strategy process appears to be running with PID {pid}. Aborting startup.")
            return False
        else:
            logging.info(f"Found stale lockfile (PID {pid} not running). Removing and continuing.")
            remove_lockfile()
            return True
    except Exception as e:
        logging.error(f"Error while checking lockfile {LOCKFILE}: {e}")
        return True

# Register cleanup
atexit.register(remove_lockfile)
# --- end PID / lockfile handling ---

def monkey_patch_option_symbol_conversion():
    """Apply symbol conversion to all relevant functions"""
    try:
        # Import the symbol formatter
        from src.symbol_utils import convert_option_symbol_format
        import src.nse_data_new
        import pandas as pd
        
        # Save original function
        original_get_option_chain = src.nse_data_new.get_nifty_option_chain
        
        # Create patched version that fixes symbols
        def patched_get_option_chain(*args, **kwargs):
            result = original_get_option_chain(*args, **kwargs)
            
            if isinstance(result, pd.DataFrame) and 'symbol' in result.columns:
                # Log original symbols first
                if not result.empty:
                    logging.info("Original option symbols:")
                    for i, symbol in enumerate(result['symbol'].iloc[:5]):
                        logging.info(f"  {i+1}. {symbol}")
                
                # Apply the conversion to all symbols
                logging.info("Converting option symbols to Fyers API format")
                result['symbol'] = result['symbol'].apply(convert_option_symbol_format)
                
                # Log the converted symbols
                if not result.empty:
                    logging.info("Converted option symbols:")
                    for i, symbol in enumerate(result['symbol'].iloc[:5]):
                        logging.info(f"  {i+1}. {symbol}")
            
            return result
            
        # Apply the patch
        src.nse_data_new.get_nifty_option_chain = patched_get_option_chain
        logging.info("Option symbol format conversion applied")
        
        return True
    except Exception as e:
        logging.error(f"Failed to apply option symbol conversion: {e}")
        return False

def apply_websocket_patch():
    """Apply the improved websocket implementation"""
    try:
        # Import the improved websocket module and monkey patch
        from src.improved_websocket import enhanced_start_market_data_websocket
        import src.fyers_api_utils
        
        # Save the original function
        original_websocket_fn = src.fyers_api_utils.start_market_data_websocket
        
        # Replace with improved version
        src.fyers_api_utils.start_market_data_websocket = enhanced_start_market_data_websocket
        logging.info("Replaced standard websocket with improved implementation")
        
        return True
    except Exception as e:
        logging.error(f"Failed to apply websocket patch: {e}")
        return False

def run_strategy():
    """Run the trading strategy with all fixes applied"""
    logging.info("RUNNING AUTOMATED OPTIONS TRADING STRATEGY WITH FIXES")
    # Apply necessary patches
    if not monkey_patch_option_symbol_conversion():
        return False
        
    if not apply_websocket_patch():
        return False
    
    try:
        # Import the strategy
        from src.strategy import OpenInterestStrategy
        
        # Create strategy instance
        strategy = OpenInterestStrategy()
        
        # Initialize for trading day
        logging.info("Initializing strategy for trading day...")
        init_success = strategy.initialize_day()
        if not init_success:
            logging.error("Failed to initialize strategy - check logs for details")
            return False
        # Always wait for market open and 9:20 before running strategy
        logging.info("Waiting for market open and 9:20 before running OI analysis and strategy...")
        wait_result = strategy.wait_for_market_open()
        # wait_for_market_open returns a dict with success status; proceed only if successful
        if isinstance(wait_result, dict) and not wait_result.get('success', False):
            logging.error(f"wait_for_market_open indicated failure: {wait_result}")
            return False

        # Now run the main strategy loop (OI analysis, subscriptions, monitoring, trade execution)
        logging.info("Starting main strategy execution (OI analysis, subscriptions, monitoring)")
        strategy_result = strategy.run_strategy()
        if isinstance(strategy_result, dict):
            if not strategy_result.get('success', False):
                logging.error(f"Strategy reported failure: {strategy_result}")
            else:
                logging.info(f"Strategy reported success: {strategy_result}")
        else:
            logging.info(f"Strategy.run_strategy returned: {strategy_result}")

        # Log final status
        logging.info("Strategy execution completed")
        return True
        
    except Exception as e:
        import traceback
        logging.error(f"Error running strategy: {e}")
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check for existing lockfile / running instance before starting
    try:
        if not check_existing_lockfile():
            logging.error("Aborting run due to existing active run_strategy process.")
            logging.shutdown()
            sys.exit(1)
        # Write our lockfile so other attempts can detect us
        write_lockfile()
        logging.info(f"Starting run_strategy with PID {os.getpid()}")
        run_strategy()
    finally:
        # Ensure lockfile removed on exit
        remove_lockfile()
        logging.shutdown()
