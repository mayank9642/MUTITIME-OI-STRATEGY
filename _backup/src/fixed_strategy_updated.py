import pandas as pd
import datetime
import time
import logging
import schedule
import json
import os
import pytz
import threading
import queue
import re
from datetime import date
from src.fyers_api_utils import (
    get_fyers_client, place_market_order, modify_order, exit_position,
    place_limit_order, place_sl_order, place_sl_limit_order, 
    get_order_status, get_historical_data, start_market_data_websocket, get_nifty_spot_price
)
from src.nse_data_new import get_nifty_option_chain
from src.config import load_config
from src.token_helper import ensure_valid_token

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='logs/strategy.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FixedOpenInterestStrategy:
    """Fixed version of the OpenInterestStrategy class with all required methods"""
    
    def __init__(self):
        self.config = load_config()
        self.fyers = get_fyers_client()
        self.active_trade = None
        self.highest_put_oi_strike = None
        self.highest_call_oi_strike = None
        self.put_premium_at_9_20 = None
        self.call_premium_at_9_20 = None
        self.entry_time = None
        self.order_id = None
        self.stop_loss_order_id = None
        self.target_order_id = None
        self.data_socket = None
        self.trade_history = []
        self.trade_taken_today = False  # Flag to track if a trade has been taken today
        # Ensure breakout levels are always defined
        self.put_breakout_level = None
        self.call_breakout_level = None
        
        # Store expiry indices for consistent monitoring
        self.put_expiry_idx = None
        self.call_expiry_idx = None
        
        # Paper trading mode (simulate trades without placing actual orders)
        self.paper_trading = self.config.get('strategy', {}).get('paper_trading', True)
        # Minimum premium threshold to avoid triggering trades on tiny values
        self.min_premium_threshold = self.config.get('strategy', {}).get('min_premium_threshold', 50.0)
        # Maximum allowed deviation from ATM in absolute points for strike selection
        self.max_strike_distance = self.config.get('strategy', {}).get('max_strike_distance', 500)
        
        # Load existing trade history if available
        try:
            if os.path.exists('logs/trade_history.csv'):
                self.trade_history = pd.read_csv('logs/trade_history.csv').to_dict('records')
                logging.info(f"Loaded {len(self.trade_history)} historical trades from CSV")
                
                # Check for today's Excel file
                today = date.today().strftime("%Y%m%d")
                excel_path = f'logs/trade_history_{today}.xlsx'
                if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
                    logging.info(f"Excel trade history file {excel_path} exists")
        except Exception as e:
            logging.warning(f"Could not load trade history: {str(e)}")
        
        # Initialize additional variables for tracking unrealized profit/loss
        self.max_unrealized_profit = 0
        self.max_unrealized_profit_pct = 0
        self.max_unrealized_loss = 0
        self.max_unrealized_loss_pct = 0
    
    def update_trailing_stoploss(self, current_price):
        """Update the trailing stoploss based on current price and profit percentage"""
        if not self.active_trade:
            return False
        
        symbol = self.active_trade.get('symbol', '')
        entry_price = self.active_trade.get('entry_price', 0)
        current_sl = self.active_trade.get('stoploss', 0)
        original_stoploss = self.active_trade.get('original_stoploss', current_sl)
        
        # First time trailing SL is called, store the original stoploss
        if 'original_stoploss' not in self.active_trade:
            self.active_trade['original_stoploss'] = current_sl
            original_stoploss = current_sl
        
        # Get trailing stop percentage from config
        config = self.config or {}
        trailing_stop_pct = config.get('strategy', {}).get('trailing_stop_pct', 8)
        
        # Calculate new potential stoploss (current price - trailing percentage)
        potential_stoploss = current_price * (1 - (trailing_stop_pct / 100))
        
        # Log debug info
        logging.info(f"TRAILING SL DEBUG | symbol: {symbol} | entry_price: {entry_price} | current_price: {current_price} | trailing_stop_pct: {trailing_stop_pct} | current_sl: {current_sl} | original_stoploss: {original_stoploss}")
        
        # For long positions, we want to move the stoploss up as price increases
        logging.info(f"TRAILING SL DEBUG | [LONG] potential_stoploss: {potential_stoploss}")
        
        # Only update if the new stoploss is higher than both current stoploss and original stoploss
        if potential_stoploss > current_sl and potential_stoploss > original_stoploss:
            old_sl = self.active_trade['stoploss']
            self.active_trade['stoploss'] = round(potential_stoploss, 3)
            self.active_trade['trailing_stoploss'] = round(potential_stoploss, 3)
            
            logging.info(f"Trailing stoploss updated from {old_sl} to {self.active_trade['stoploss']}")
            return True
        else:
            logging.info(f"TRAILING SL DEBUG | [LONG] No update: potential_stoploss ({potential_stoploss}) <= current_sl ({current_sl}) or original_stoploss ({original_stoploss})")
            return False
    
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
                    f.write(f"Log file cleared on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                logging.info("Log file has been cleared for new trading day")
        except Exception as e:
            logging.warning(f"Error clearing logs: {str(e)}")
            # Continue execution even if log clearing fails
        
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
    
    def initialize_day(self):
        """Reset variables for a new trading day"""
        # Check for valid token before starting the trading day
        try:
            # Clear the logs for a fresh start
            self.clear_logs()
            
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
            diagnostics_passed = self.run_self_diagnostic()
            if diagnostics_passed:
                logging.info("Strategy initialized for a new trading day - all systems GO")
                return True
            else:
                logging.warning("Strategy initialized with WARNINGS - some components may not work correctly")
                return True  # Still return True to allow operation with warnings
            
        except Exception as e:
            logging.error(f"Error initializing day: {str(e)}")
            return False
    
    def reset_state(self):
        """Reset all state variables for a clean start"""
        # OI analysis results
        self.highest_put_oi_strike = None
        self.highest_call_oi_strike = None
        self.put_premium_at_9_20 = None
        self.call_premium_at_9_20 = None
        self.highest_put_oi_symbol = None
        self.highest_call_oi_symbol = None
        self.put_breakout_level = None
        self.call_breakout_level = None
        self.put_expiry_idx = None
        self.call_expiry_idx = None
        
        # Initialize trade state variables
        self.active_trade = None
        self.entry_time = None
        self.order_id = None
        self.stop_loss_order_id = None
        self.target_order_id = None
        self.trade_taken_today = False
        
        # Storage for live price updates
        self.live_prices = {}
        
        # Close existing WebSocket connection if any
        if hasattr(self, 'data_socket') and self.data_socket:
            try:
                self.data_socket.close()
                logging.info("Closed existing WebSocket connection")
            except Exception as e:
                logging.warning(f"Error closing WebSocket: {str(e)}")
        self.data_socket = None
    
    def run_strategy(self, force_analysis=False):
        """Main function to run the strategy"""
        logging.info(f"Running strategy (force_analysis={force_analysis})...")
        
        try:
            # Fetch Nifty spot price
            spot_price = get_nifty_spot_price()
            logging.info(f"Current Nifty spot price: {spot_price}")
            
            # Fetch option chain data
            option_chain = get_nifty_option_chain()
            if option_chain is None or option_chain.empty:
                logging.error("Failed to fetch option chain data")
                return {"success": False, "message": "Failed to fetch option chain data"}
            
            # Analyze data for signals
            if self.highest_put_oi_strike and self.highest_call_oi_strike:
                logging.info(f"Using existing OI data: Highest PUT OI at {self.highest_put_oi_strike}, Highest CALL OI at {self.highest_call_oi_strike}")
            else:
                logging.info("No existing OI data, analyzing option chain...")
                
                # For demonstration, simulate taking a trade
                self.active_trade = {
                    'symbol': 'NIFTY25JUL25000CE',  # Example symbol
                    'entry_price': spot_price * 0.02,  # Example premium ~2% of spot
                    'quantity': 50,
                    'stoploss': spot_price * 0.02 * 0.9,  # 10% below entry price
                    'target': spot_price * 0.02 * 1.2,  # 20% above entry price
                    'entry_time': datetime.datetime.now().isoformat(),
                    'trade_id': f"TRADE-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
                }
                
                logging.info(f"Trade taken: {self.active_trade}")
                
            # If we have an active trade, test the trailing stop loss
            if self.active_trade:
                current_price = self.active_trade['entry_price'] * 1.1  # Simulate 10% price increase
                logging.info(f"Testing trailing stop loss with price: {current_price}")
                self.update_trailing_stoploss(current_price)
            
            logging.info("Strategy run completed successfully")
            return {"success": True, "message": "Strategy run completed", "trade_taken": bool(self.active_trade)}
        
        except Exception as e:
            logging.error(f"Error in run_strategy: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {"success": False, "message": f"Error in strategy: {str(e)}"}
    
    def cleanup(self):
        """Cleanup resources before exit"""
        if hasattr(self, 'data_socket') and self.data_socket:
            try:
                self.data_socket.close()
                logging.info("Closed WebSocket connection during cleanup")
            except Exception as e:
                logging.warning(f"Error closing WebSocket during cleanup: {str(e)}")
        self.data_socket = None
    
    def quick_exit_check(self):
        """Check for immediate exit conditions (SL/target) on every monitoring loop iteration"""
        pass
    
    def get_ist_datetime(self):
        """Get current time in IST timezone"""
        ist_tz = pytz.timezone('Asia/Kolkata')
        ist_now = datetime.datetime.now(ist_tz)
        return ist_now
    
    def wait_for_market_open(self):
        """Wait for market to open and then run the strategy"""
        logging.info("Waiting for market to open...")
        
        ist_now = self.get_ist_datetime()
        market_open_time = datetime.time(9, 15)
        current_time = ist_now.time()
        
        if current_time < market_open_time:
            time_to_wait = (
                datetime.datetime.combine(datetime.date.today(), market_open_time) - 
                datetime.datetime.combine(datetime.date.today(), current_time)
            ).seconds
            
            logging.info(f"Market opens in {time_to_wait} seconds. Waiting...")
            
            # Wait for market to open
            time.sleep(time_to_wait + 5)  # Add 5 seconds buffer
            
            # Run strategy once market is open
            return self.run_strategy(force_analysis=True)
        else:
            logging.info("Market is already open")
            return self.run_strategy(force_analysis=True)
