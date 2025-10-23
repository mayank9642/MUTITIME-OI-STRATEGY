"""
Run the automated options trading strategy
"""
import os
import sys
import logging
import logging.handlers
import datetime

# Configure logging
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure root logger with both console and file handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Add console handler with stream lock for thread safety
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.createLock()
logger.addHandler(console_handler)

# Add file handler for main log with file lock
file_handler = logging.handlers.RotatingFileHandler(
    'logs/strategy_run.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5  # Keep 5 backup files
)
file_handler.setFormatter(formatter)
file_handler.createLock()
logger.addHandler(file_handler)

# Add file handler for strategy log with file lock
strategy_log_handler = logging.handlers.RotatingFileHandler(
    'logs/strategy.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5  # Keep 5 backup files
)
strategy_log_handler.setFormatter(formatter)
strategy_log_handler.createLock()
logger.addHandler(strategy_log_handler)

logging.info("Logging configured with console and file handlers")

# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logging.info(f"Added current directory to Python path: {current_dir}")

# Print Python path for debugging
logging.info(f"Python path includes: {sys.path}")

def monkey_patch_option_symbol_conversion():
    """Apply symbol conversion to all relevant functions"""
    try:
        # Import the symbol formatter
        from src.symbol_formatter import convert_option_symbol_format
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
    logging.info("="*80)
    logging.info("RUNNING AUTOMATED OPTIONS TRADING STRATEGY WITH FIXES")
    logging.info("="*80)
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
        # Check if market is open, if not wait for it to open
        ist_now = strategy.get_ist_datetime()
        current_time = ist_now.time()
        market_open_time = datetime.time(9, 15)
        analysis_time = datetime.time(9, 20)

        if current_time < analysis_time:
            logging.info("Waiting for 9:20 before running OI analysis and strategy...")
            result = strategy.wait_for_market_open()
        else:
            # Run the strategy with force analysis to take a trade
            logging.info("It's 9:20 or later. Running strategy with force analysis enabled to take a trade...")
            result = strategy.run_strategy(force_analysis=True)
        
        # Log final status
        logging.info("Strategy execution completed")
        return True
        
    except Exception as e:
        import traceback
        logging.error(f"Error running strategy: {e}")
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_strategy()
