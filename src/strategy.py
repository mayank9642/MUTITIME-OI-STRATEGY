"""
Fixed version of the strategy file with proper update_trailing_stoploss implementation
"""
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import json
import numpy as np
import traceback
import sys
import requests
import threading
import websocket
from collections import defaultdict
from src.fyers_api_utils import get_fyers_client
from src.fixed_improved_websocket import enhanced_start_market_data_websocket
from src.order_manager import OrderManager
import re

class OpenInterestStrategy:
    def __init__(self):
        # Initialize your strategy here
        self.active_trade = {}
        self.live_prices = {}
        # DataFrame to store LTPs for each contract (symbol, expiry, strike, option_type)
        self.ltp_df = pd.DataFrame(columns=["symbol", "expiry", "strike", "option_type", "ltp"])
        self.config = {}
        self.paper_trading = True
        self.market_closed = False
        self.trade_taken_today = False
        self.put_breakout_level = 0
        self.call_breakout_level = 0
        self.highest_put_oi_strike = 0
        self.highest_call_oi_strike = 0
        self.fyers = get_fyers_client()
        self.min_premium_threshold = self.config.get('strategy', {}).get('min_premium_threshold', 50.0)
        self.entry_time = None
        self.max_strike_distance = self.config.get('strategy', {}).get('max_strike_distance', 500)
        self.trade_history = []
        self.order_manager = OrderManager(paper_trading=self.paper_trading)
        self._ws_lock = threading.Lock()
        
        # Load today's trade history if file exists
        today = datetime.now().strftime('%Y%m%d')
        excel_path = f'logs/trade_history_{today}.xlsx'
        csv_path = 'logs/trade_history.csv'
        
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                self.trade_history = df.to_dict('records')
                logging.info(f"Loaded existing trade history from {excel_path}")
            except Exception as e:
                logging.error(f"Error loading trade history from {excel_path}: {e}")
        elif os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self.trade_history = df.to_dict('records')
                logging.info(f"Loaded existing trade history from {csv_path}")
            except Exception as e:
                logging.error(f"Error loading trade history from {csv_path}: {e}")
                
    def update_trailing_stoploss(self, current_price):
        """
        Update the trailing stoploss based on current price and profit percentage.
        """
        if not self.active_trade:
            logging.warning("No active trade found when updating trailing stoploss")
            return False

        symbol = self.active_trade.get('symbol', '')
        entry_price = self.active_trade.get('entry_price', 0)
        current_sl = self.active_trade.get('stoploss', 0)
        original_stoploss = self.active_trade.get('original_stoploss', current_sl)
        
        # Validate the current_price is reasonable for this symbol
        if current_price <= 0 or current_price > 5000:
            logging.warning(f"Invalid price {current_price} for {symbol} in update_trailing_stoploss - ignoring update")
            return False
            
        # Additional validation to make sure we don't mix up CE and PE prices
        symbol_type = "unknown"
        if "CE" in symbol:
            symbol_type = "CE"
        elif "PE" in symbol:
            symbol_type = "PE"
        
        # Verify the price looks reasonable compared to entry price (no more than 50% decrease or 200% increase)
        if current_price < entry_price * 0.5 or current_price > entry_price * 3.0:
            logging.warning(f"Price for {symbol} ({symbol_type}) looks suspicious: entry={entry_price}, current={current_price} - needs verification")
            logging.warning("Running additional validation to prevent incorrect stoploss update")
            # Get the price directly from DataFrame with explicit contract match
            expiry = self.active_trade.get('expiry', None)
            strike = self.active_trade.get('strike', None)
            option_type = self.active_trade.get('option_type', None)
            df_row = self.ltp_df[
                (self.ltp_df.symbol == symbol) &
                (self.ltp_df.expiry == expiry) &
                (self.ltp_df.strike == strike) &
                (self.ltp_df.option_type == option_type)
            ]
            if not df_row.empty:
                live_price = float(df_row.iloc[0]['ltp'])
                if abs(live_price - current_price) > entry_price * 0.1:
                    logging.warning(f"Possible price mixup detected! Provided price: {current_price}, DataFrame live price: {live_price}")
                    logging.warning(f"Using verified live price instead for {symbol}")
                    current_price = live_price

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

        # Only update if the new stoploss is higher than both current stoploss and original_stoploss
        if potential_stoploss > current_sl and potential_stoploss > original_stoploss:
            old_sl = self.active_trade['stoploss']
            self.active_trade['stoploss'] = round(potential_stoploss, 3)
            self.active_trade['trailing_stoploss'] = round(potential_stoploss, 3)

            logging.info(f"Trailing stoploss updated from {old_sl} to {self.active_trade['stoploss']}")
            # --- Broker-side trailing stoploss update ---
            if not self.paper_trading and hasattr(self, 'stop_loss_order_id') and self.stop_loss_order_id:
                try:
                    from src.fyers_api_utils import modify_order
                    response = modify_order(self.fyers, self.stop_loss_order_id, stop_price=self.active_trade['stoploss'])
                    if response and response.get('s') == 'ok':
                        logging.info(f"Broker stoploss order modified: {self.stop_loss_order_id} to {self.active_trade['stoploss']}")
                    else:
                        logging.error(f"Failed to modify broker stoploss order: {response}")
                except Exception as e:
                    logging.error(f"Exception while modifying broker stoploss order: {e}")
            return True
        else:
            logging.info(f"TRAILING SL DEBUG | [LONG] No update: potential_stoploss ({potential_stoploss}) <= current_sl ({current_sl}) or original_stoploss ({original_stoploss})")
            return False

    def validate_fyers_symbols(self, symbols):
        """
        Validate option symbols against Fyers master contract (or API) before subscribing.
        Returns only valid symbols.
        """
        # Placeholder: In production, fetch valid symbols from Fyers API or master file
        # For now, assume all symbols are valid except those with None or empty string
        valid_symbols = [s for s in symbols if s and isinstance(s, str) and len(s) > 10]
        invalid_symbols = [s for s in symbols if s not in valid_symbols]
        if invalid_symbols:
            logging.warning(f"Some symbols are invalid and will not be subscribed: {invalid_symbols}")
        return valid_symbols

    def identify_high_oi_strikes(self):
        """
        Analyze option chain data to identify highest OI strikes for CE and PE.
        Sets self.highest_call_oi_strike, self.highest_put_oi_strike, self.highest_call_oi_symbol, self.highest_put_oi_symbol,
        self.call_breakout_level, self.put_breakout_level for trade monitoring.
        Returns True if analysis is successful, False otherwise.
        """
        try:
            from src.fetch_option_oi import fetch_option_oi  # Corrected import
            oi_data = fetch_option_oi()
            if oi_data is None or len(oi_data) == 0:
                logging.error("OI analysis failed: No option chain data returned.")
                return False
            ce_df = oi_data[oi_data['option_type'] == 'CE']
            pe_df = oi_data[oi_data['option_type'] == 'PE']
            if ce_df.empty or pe_df.empty:
                logging.error("OI analysis failed: No CE or PE data available.")
                return False
            ce_df = ce_df[ce_df['ltp'].notnull()]
            pe_df = pe_df[pe_df['ltp'].notnull()]
            if ce_df.empty or pe_df.empty:
                logging.error("OI analysis failed: No CE or PE contracts with valid LTP.")
                return False
            # --- Enhanced Strike Selection Logic ---
            # Sort by OI, filter by max_strike_distance from ATM
            spot_price = self.live_prices.get('NSE:NIFTY', None)
            if spot_price is None:
                spot_price = ce_df['strike'].median()  # fallback
            atm_strike = round(spot_price / 100) * 100 if spot_price else None
            max_distance = self.max_strike_distance
            min_premium = self.min_premium_threshold
            # Filter CE/PE by strike distance
            ce_filtered = ce_df[(ce_df['strike'] >= atm_strike - max_distance) & (ce_df['strike'] <= atm_strike + max_distance)]
            pe_filtered = pe_df[(pe_df['strike'] >= atm_strike - max_distance) & (pe_df['strike'] <= atm_strike + max_distance)]
            # Sort by OI descending
            ce_sorted = ce_filtered.sort_values('oi', ascending=False)
            pe_sorted = pe_filtered.sort_values('oi', ascending=False)
            # Select first strike with premium above threshold
            highest_call_row = None
            for _, row in ce_sorted.iterrows():
                if row['ltp'] >= min_premium:
                    highest_call_row = row
                    break
            if highest_call_row is None and not ce_sorted.empty:
                highest_call_row = ce_sorted.iloc[0]  # fallback to highest OI
            highest_put_row = None
            for _, row in pe_sorted.iterrows():
                if row['ltp'] >= min_premium:
                    highest_put_row = row
                    break
            if highest_put_row is None and not pe_sorted.empty:
                highest_put_row = pe_sorted.iloc[0]  # fallback to highest OI
            if highest_call_row is None or highest_put_row is None:
                logging.error("OI analysis failed: No suitable CE/PE strike found above premium threshold.")
                return False
            self.highest_call_oi_strike = int(highest_call_row['strike'])
            self.highest_put_oi_strike = int(highest_put_row['strike'])
            self.highest_call_oi_symbol = highest_call_row['symbol']
            self.highest_put_oi_symbol = highest_put_row['symbol']
            breakout_pct = self.config.get('strategy', {}).get('breakout_pct', 10)
            self.call_breakout_level = float(highest_call_row['ltp']) * (1 + breakout_pct / 100)
            self.put_breakout_level = float(highest_put_row['ltp']) * (1 + breakout_pct / 100)
            logging.info(f"OI analysis: Highest CE strike={self.highest_call_oi_strike}, symbol={self.highest_call_oi_symbol}, breakout={self.call_breakout_level}")
            logging.info(f"OI analysis: Highest PE strike={self.highest_put_oi_strike}, symbol={self.highest_put_oi_symbol}, breakout={self.put_breakout_level}")
            return True
        except Exception as e:
            logging.error(f"Error in identify_high_oi_strikes: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def subscribe_to_valid_symbols(self, symbols):
        """
        Subscribe only to valid symbols for monitoring.
        """
        valid_symbols = self.validate_fyers_symbols(symbols)
        if not valid_symbols:
            logging.error("No valid symbols to subscribe for monitoring.")
            return
        # Start websocket subscription for valid symbols
        self.data_socket = enhanced_start_market_data_websocket(valid_symbols, self.ws_price_update)
        logging.info(f"Subscribed to valid symbols: {valid_symbols}")

    def ws_price_update(self, symbol, key, ticks, raw_ticks):
        """
        Callback function to handle WebSocket price updates.
        Accepts symbol, key, ticks, raw_ticks as per the callback handler's call signature.
        Uses canonical symbol as the key for self.live_prices and logging.
        Logs both incoming and canonical symbols for diagnostics.
        """
        try:
            # --- PATCH: Ensure only exact contract is updated ---
            canonical_symbol = self.get_canonical_symbol(symbol)
            # Parse expiry, strike, option_type from canonical_symbol
            match = re.match(r"NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d+)(CE|PE)", canonical_symbol)
            expiry = strike = option_type = None
            if match:
                expiry = match.group(1)
                strike = int(match.group(2))
                option_type = match.group(3)
            # Extract LTP from ticks (assume ticks is a dict with 'ltp')
            ltp = ticks.get('ltp') if isinstance(ticks, dict) else None
            if ltp is not None:
                # Update live_prices only for the exact contract
                self.live_prices[canonical_symbol] = ltp
                # Update ltp_df for the exact contract
                df_row = self.ltp_df[
                    (self.ltp_df.symbol == canonical_symbol) &
                    (self.ltp_df.expiry == expiry) &
                    (self.ltp_df.strike == strike) &
                    (self.ltp_df.option_type == option_type)
                ]
                if not df_row.empty:
                    self.ltp_df.loc[df_row.index, 'ltp'] = ltp
                else:
                    # Insert new row for this contract
                    new_row = {
                        'symbol': canonical_symbol,
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': option_type,
                        'ltp': ltp
                    }
                    self.ltp_df = pd.concat([self.ltp_df, pd.DataFrame([new_row])], ignore_index=True)
                logging.debug(f"Tick update: {canonical_symbol} | expiry: {expiry} | strike: {strike} | type: {option_type} | ltp: {ltp}")
            else:
                logging.warning(f"Tick update missing LTP for {canonical_symbol}: {ticks}")
        except Exception as e:
            logging.error(f"Error in ws_price_update: {str(e)}")

    # Other essential method skeletons
    def run_diagnostic(self):
        """Run a self-diagnostic check to verify key components are functioning"""
        # Implementation would go here
        pass

    def save_trade_history(self):
        """Save trade history to both CSV and Excel files with proper error handling and column order"""
        import pandas as pd
        from datetime import date
        try:
            # Define required columns in order
            columns = [
                'Entry DateTime', 'Index', 'Symbol', 'Direction', 'Entry Price',
                'Exit DateTime', 'Exit Price', 'Stop Loss', 'Target', 'Trailing SL',
                'Quantity', 'Brokerage', 'P&L', 'Margin Required', '% Gain/Loss',
                'max up', 'max down', 'max up %', 'max down %'
            ]
            df = pd.DataFrame(self.trade_history)
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            df = df[columns]  # Ensure column order
            # Save to CSV
            df.to_csv('logs/trade_history.csv', index=False)
            # Save to Excel with today's date
            today = date.today().strftime('%Y%m%d')
            excel_path = f'logs/trade_history_{today}.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            logging.info(f"Trade history saved to CSV and Excel: {excel_path}")
        except Exception as e:
            logging.error(f"Error saving trade history: {str(e)}")

    def update_aggregate_stats(self):
        """Update aggregate statistics file with new trade data"""
        # Implementation would go here
        return datetime.now()

    def wait_for_market_open(self):
        """Wait for market to open (09:15) and then for 9:20 before running OI analysis and the rest of the strategy"""
        try:
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            current_time = ist_now.time()
            market_open_time = datetime.strptime("09:15", "%H:%M").time()
            analysis_time = datetime.strptime("09:20", "%H:%M").time()
            # Wait for market open (09:15)
            while current_time < market_open_time:
                mins, secs = divmod((datetime.combine(ist_now.date(), market_open_time) - datetime.combine(ist_now.date(), current_time)).total_seconds(), 60)
                logging.info(f"Market not open yet. Waiting... Current time: {current_time.strftime('%H:%M:%S')}, Market opens in: {int(mins)}m {int(secs)}s")
                time.sleep(10)
                ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                current_time = ist_now.time()
            logging.info("Market is now open. Waiting for 9:20 to perform OI analysis...")
            # Wait for 9:20
            while current_time < analysis_time:
                mins, secs = divmod((datetime.combine(ist_now.date(), analysis_time) - datetime.combine(ist_now.date(), current_time)).total_seconds(), 60)
                logging.info(f"Waiting for 9:20... Current time: {current_time.strftime('%H:%M:%S')}, OI analysis in: {int(mins)}m {int(secs)}s")
                time.sleep(10)
                ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                current_time = ist_now.time()
            logging.info("It's 9:20 or later. Running strategy and OI analysis...")
            return self.run_strategy(force_analysis=True)
        except Exception as e:
            logging.error(f"Error in wait_for_market_open: {str(e)}")
            logging.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def clear_logs(self):
        """Clear log file for a fresh start to the trading day"""
        try:
            log_file = 'logs/strategy.log'
            if os.path.exists(log_file):
                # Keep existing logs by backing up current log file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f'logs/strategy_{timestamp}.log.bak'
                
                # Copy to backup before clearing
                if os.path.getsize(log_file) > 0:
                    with open(log_file, 'r') as src, open(backup_file, 'w') as dst:
                        dst.write(src.read())
                    logging.info(f"Log file backed up to {backup_file}")
                    
                # Clear the current log file
                with open(log_file, 'w') as f:
                    f.write(f"Log file cleared on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                logging.info("Log file has been cleared for new trading day")
                return True
            return False
        except Exception as e:
            logging.error(f"Error clearing logs: {str(e)}")
            return False
        
    def initialize_day(self):
        """Initialize strategy for the day including setting up necessary state"""
        try:
            # Clear logs for a fresh start
            self.clear_logs()
            
            logging.info("Initializing strategy for the day")
            # Reset daily state variables
            self.trade_taken_today = False
            self.market_closed = False
            self.put_breakout_level = 0
            self.call_breakout_level = 0
            self.highest_put_oi_strike = 0
            self.highest_call_oi_strike = 0
            
            # Clear any active trades from previous day
            self.active_trade = {}
            
            # --- WebSocket subscription for all relevant symbols ---
            # Remove index subscription: only subscribe to options needed for breakout monitoring
            symbols = []
            if hasattr(self, 'highest_put_oi_symbol') and self.highest_put_oi_symbol:
                symbols.append(self.highest_put_oi_symbol)
            if hasattr(self, 'highest_call_oi_symbol') and self.highest_call_oi_symbol:
                symbols.append(self.highest_call_oi_symbol)
            if self.active_trade and 'symbol' in self.active_trade:
                trade_symbol = self.active_trade['symbol']
                if trade_symbol and trade_symbol not in symbols:
                    symbols.append(trade_symbol)
            logging.info(f"Subscribing to symbols: {symbols}")
            self.subscribe_to_valid_symbols(symbols)
            logging.info(f"WebSocket subscription started for symbols: {symbols}")
            logging.info("Strategy initialization complete")
            return True
        except Exception as e:
            logging.error(f"Error initializing strategy for the day: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def generate_daily_report(self):
        """Generate a summary report of the day's trading activity"""
        # Implementation would go here
        pass
        
    def run_strategy(self, force_analysis=False):
        """
        Main method to run the strategy logic
        
        Args:
            force_analysis (bool): Whether to force OI analysis regardless of time constraints
        
        Returns:
            dict: Result of strategy execution with success status and message
        """
        try:
            logging.info("Running Open Interest Option Buying Strategy")
            # Get current time in IST
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            current_time = ist_now.time()
            market_open_time = datetime.strptime("09:15", "%H:%M").time()
            market_close_time = datetime.strptime("15:30", "%H:%M").time()
            # Check if market is closed
            if self.market_closed or current_time >= market_close_time:
                logging.info("Market is closed. Skipping strategy execution.")
                return {"success": False, "message": "Market closed"}
            # Check if today is a weekday
            if ist_now.weekday() > 4:
                logging.info("Today is weekend. Market closed.")
                return {"success": False, "message": "Weekend"}
            # Check if trade already taken today
            if self.trade_taken_today and not force_analysis:
                logging.info("Trade already taken today. Skipping strategy execution.")
                return {"success": True, "message": "Trade already taken today"}
            # Wait for market open if needed
            if current_time < market_open_time:
                logging.info("Market not open yet. Waiting for market open...")
                return self.wait_for_market_open()
            # Step 1: OI analysis at/after 9:20
            analysis_time = datetime.strptime("09:20", "%H:%M").time()
            if force_analysis or (current_time >= analysis_time):
                logging.info("Performing OI analysis...")
                oi_result = self.identify_high_oi_strikes()
                if not oi_result:
                    logging.error("OI analysis failed. Exiting strategy run.")
                    return {"success": False, "message": "OI analysis failed"}
                logging.info(f"Breakout levels: PUT={self.put_breakout_level}, CALL={self.call_breakout_level}")
                # --- Original Trade Entry Logic ---                # Allow trade if either leg is valid (not both required)
                if (self.highest_call_oi_symbol and self.call_breakout_level) or (self.highest_put_oi_symbol and self.put_breakout_level):
                    breakout_detected = self.monitor_for_breakout()
                    if breakout_detected:
                        logging.info("Breakout detected and trade executed successfully")
                    else:
                        logging.info("No breakout detected during monitoring period")
                else:
                    logging.warning("Trade entry skipped: missing symbol or breakout level.")
            # Position management is handled by continuous_position_monitor thread
            logging.info("Strategy execution completed successfully")
            return {"success": True, "message": "Strategy executed successfully"}
        except Exception as e:
            logging.error(f"Error in run_strategy: {str(e)}")
            logging.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
            
    def unsubscribe_non_triggered_symbol(self, triggered_symbol, all_symbols):
        """Unsubscribe from the symbol(s) where trade was not triggered."""
        non_triggered = [s for s in all_symbols if s != triggered_symbol]
        # Assuming your data_socket has an unsubscribe method
        if hasattr(self.data_socket, 'unsubscribe'):
            for s in non_triggered:
                self.data_socket.unsubscribe(s)
                logging.info(f"Unsubscribed from {s} after trade triggered for {triggered_symbol}")
        else:
            logging.warning("WebSocket unsubscribe method not available. Manual unsubscribe required.")

    def retry_websocket_connection(self, symbols, max_retries=3, delay=5):
        """Retry websocket connection if it fails."""
        for attempt in range(1, max_retries + 1):
            try:
                self.data_socket = enhanced_start_market_data_websocket(
                    symbols=symbols,
                    callback_handler=self.ws_price_update
                )
                logging.info(f"WebSocket connection established on attempt {attempt} for symbols: {symbols}")
                return True
            except Exception as e:
                logging.error(f"WebSocket connection attempt {attempt} failed: {str(e)}")
                time.sleep(delay)
        logging.error(f"All {max_retries} websocket connection attempts failed for symbols: {symbols}")
        return False

    def monitor_for_breakout(self):
        """Continuously monitor both CE and PE option premiums for breakout using websocket for real-time data"""
        try:
            logging.info("Monitoring for breakout on both CE and PE...")
            symbols_to_monitor = []
            breakout_levels = {}
            if self.put_breakout_level and self.highest_put_oi_symbol:
                symbols_to_monitor.append(self.highest_put_oi_symbol)
                breakout_levels[self.get_canonical_symbol(self.highest_put_oi_symbol)] = self.put_breakout_level
            if self.call_breakout_level and self.highest_call_oi_symbol:
                symbols_to_monitor.append(self.highest_call_oi_symbol)
                breakout_levels[self.get_canonical_symbol(self.highest_call_oi_symbol)] = self.call_breakout_level
            if not symbols_to_monitor:
                logging.info("No valid option symbols to monitor for breakout.")
                return
            logging.info(f"Subscribing to both option symbols for breakout monitoring: {symbols_to_monitor}")
            if not self.retry_websocket_connection(symbols_to_monitor):
                logging.error("Could not establish websocket connection after retries. Aborting breakout monitoring.")
                return            
            logging.info(f"WebSocket subscription started for symbols: {symbols_to_monitor}")
            canonical_symbols = [self.get_canonical_symbol(s) for s in symbols_to_monitor]
            
            while True:
                for symbol, canonical_symbol in zip(symbols_to_monitor, canonical_symbols):
                    # --- PATCH: Use DataFrame-based LTP for price decisions ---
                    df_row = self.ltp_df[self.ltp_df.symbol == canonical_symbol]
                    price = float(df_row.iloc[0]['ltp']) if not df_row.empty else None
                    breakout_level = breakout_levels[canonical_symbol]
                    option_type = "unknown"
                    if "CE" in canonical_symbol:
                        option_type = "CE"
                    elif "PE" in canonical_symbol:
                        option_type = "PE"
                    logging.info(f"MONITOR: {canonical_symbol} ({option_type}) price={price} (Breakout: {breakout_level})")
                    if price is not None:
                        if price >= breakout_level:
                            logging.info(f"Breakout detected for {canonical_symbol} ({option_type}) at price {price} >= breakout level {breakout_level}. Executing trade entry.")
                            trade_result = self.execute_trade(symbol=canonical_symbol, side='BUY', entry_price=price)
                            if trade_result:
                                logging.info(f"Trade entry successful for {canonical_symbol} at price {price}.")
                            else:
                                logging.error(f"Trade entry failed for {canonical_symbol} at price {price}.")
                            self.unsubscribe_non_triggered_symbol(triggered_symbol=canonical_symbol, all_symbols=symbols_to_monitor)
                            return True
                time.sleep(2)
            return False
        except Exception as e:
            logging.error(f"Error monitoring for breakout: {str(e)}")
            return None

    def log_trade_update(self):
        """Log trade update and monitoring info after entry, including P&L, max up/down, trailing SL"""
        if not self.active_trade:
            return
        symbol = self.active_trade.get('symbol')
        entry_price = self.active_trade.get('entry_price')
        stoploss = self.active_trade.get('stoploss')
        target = self.active_trade.get('target')
        quantity = self.active_trade.get('quantity')
        entry_time = self.active_trade.get('entry_time')
        # Always use tick DataFrame LTP for the exact contract
        current_price = self.get_active_trade_ltp()
        if current_price is None:
            logging.error(f"No tick DataFrame LTP available for active trade contract. Skipping trade update.")
            return
        logging.info(f"TRADE_UPDATE | Symbol: {symbol} | Entry: {entry_price} | LTP: {current_price} (source: tick DataFrame) | SL: {self.active_trade['stoploss']} | Target: {target}")
        # Calculate P&L
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
        # Track max up/down
        max_up = self.active_trade.get('max_up', None)
        max_up_pct = self.active_trade.get('max_up_pct', None)
        max_down = self.active_trade.get('max_down', None)
        max_down_pct = self.active_trade.get('max_down_pct', None)
        trailing_sl = stoploss
        # Update max up only if unrealized profit increases
        if pnl > 0 and (max_up is None or pnl > max_up):
            self.active_trade['max_up'] = pnl
            self.active_trade['max_up_pct'] = pnl_pct
        # Update max down only if unrealized loss increases (more negative)
        if pnl < 0 and (max_down is None or pnl < max_down):
            self.active_trade['max_down'] = pnl
            self.active_trade['max_down_pct'] = pnl_pct
        # Trailing SL logic: only trail if profit exceeds 20%
        profit_threshold = 20
        if pnl_pct >= profit_threshold:
            profit_above_20 = current_price - (entry_price * 1.2)
            if profit_above_20 > 0:
                new_sl = entry_price + 0.5 * (current_price - entry_price)
                if new_sl > stoploss:
                    self.active_trade['stoploss'] = round(new_sl, 2)
                    trailing_sl = self.active_trade['stoploss']
                    logging.info(f"Trailing SL updated to {trailing_sl} after exceeding {profit_threshold}% profit.")
        # Ensure max_down and max_down_pct are floats for formatting
        max_down_val = float(self.active_trade.get('max_down', 0) or 0)
        max_down_pct_val = float(self.active_trade.get('max_down_pct', 0) or 0)
        max_up_val = float(self.active_trade.get('max_up', 0) or 0)
        max_up_pct_val = float(self.active_trade.get('max_up_pct', 0) or 0)
        logging.info(f"TRADE_UPDATE | Symbol: {symbol} | Entry: {entry_price} | LTP: {current_price} (source: tick DataFrame) | SL: {self.active_trade['stoploss']} | Target: {target} | P&L: {pnl:.2f} ({pnl_pct:.2f}%) | MaxUP: {max_up_val:.2f} ({max_up_pct_val:.2f}%) | MaxDN: {max_down_val:.2f} ({max_down_pct_val:.2f}%) | Trailing SL: {self.active_trade['stoploss']}")
        logging.info(f"TRADE_MONITOR | Monitoring {symbol} for SL/Target/Exit conditions...")

    def cleanup(self):
        """Cleanup resources before exiting"""
        try:
            logging.info("Cleaning up strategy resources")
            # Force exit any open trade before shutdown
            if self.active_trade and not self.active_trade.get('exit_reason'):
                logging.info("Forcing exit of open trade during cleanup to ensure exit is logged.")
                self.process_exit(exit_reason="FORCED_CLEANUP")
            # Save any pending data
            self.save_trade_history()
            # Close any connections
            if self.fyers:
                # Close any active websocket connections, etc.
                pass
            logging.info("Cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            return False
        return True

    def get_ist_datetime(self):
        """Return current datetime in IST timezone"""
        return datetime.now(pytz.timezone('Asia/Kolkata'))

    def get_canonical_symbol(self, symbol):
        """
        Convert any incoming symbol (raw or exchange-formatted) to the canonical format used for logging and processing.
        Ensures every unique contract (expiry, strike, type) gets a unique symbol.
        Logs original and converted symbol for diagnostics.
        """
        import re
        import logging
        orig_symbol = symbol
        # If already in Fyers format, return as is
        if symbol.startswith('NSE:') and (symbol.endswith('CE') or symbol.endswith('PE')):
            logging.info(f"[SYMBOL MAP] Already canonical: {symbol}")
            return symbol
        # Try to match NIFTY options: NIFTY07AUG25C24550 or NIFTY07AUG25P24550
        match = re.match(r'NIFTY(\d{2})([A-Z]{3})(\d{2})([CP])(\d+)', symbol)
        if match:
            year, month, day, opt_type, strike = match.groups()
            fyers_symbol = f"NSE:NIFTY{day}{month.upper()}{year}{strike}{'CE' if opt_type=='C' else 'PE'}"
            logging.info(f"[SYMBOL MAP] {orig_symbol} -> {fyers_symbol}")
            return fyers_symbol
        # Fallback: use convert_option_symbol_format if available
        try:
            from src.symbol_formatter import convert_option_symbol_format
            converted = convert_option_symbol_format(symbol)
            logging.info(f"[SYMBOL MAP] {orig_symbol} -> {converted}")
            return converted
        except Exception as e:
            logging.error(f"[SYMBOL MAP] Error converting {orig_symbol}: {e}")
            return symbol

    def stop_price_monitoring(self, symbol=None):
        """Stop all price monitoring and unsubscribe from all symbols after trade exit."""
        if hasattr(self, 'data_socket') and self.data_socket:
            if hasattr(self.data_socket, 'unsubscribe_all') and symbol is None:
                self.data_socket.unsubscribe_all()
                logging.info("Unsubscribed from all symbols after trade exit.")
            elif hasattr(self.data_socket, 'unsubscribe'):
                # Unsubscribe from the given symbol
                if symbol:
                    self.data_socket.unsubscribe(symbol)
                    logging.info(f"Unsubscribed from symbol: {symbol}")
            # Try to close the websocket/data socket if possible
            if hasattr(self.data_socket, 'close'):
                try:
                    self.data_socket.close()
                    logging.info("Closed data socket after trade exit.")
                except Exception as e:
                    logging.error(f"Error closing data socket: {e}")
            self.data_socket = None
        logging.info("Stopped all price monitoring after trade exit.")

    def calculate_fyers_option_charges(self, entry_price, exit_price, quantity, state='maharashtra'):
        """
        Calculate total brokerage and all statutory charges for a round-trip options trade (buy+sell) on Fyers.
        Returns approximately ₹50 for a typical Nifty option round trip trade.
        """
        # Turnover for each leg
        buy_turnover = entry_price * quantity
        sell_turnover = exit_price * quantity
        # Brokerage per leg
        buy_brokerage = min(20, 0.0003 * buy_turnover)
        sell_brokerage = min(20, 0.0003 * sell_turnover)
        # STT: 0.05% on sell-side premium for options (corrected from 0.0625%)
        stt = 0.0005 * sell_turnover
        # Exchange Transaction Charges: 0.00345% on premium (both legs) (corrected from 0.053%)
        buy_exch_txn = 0.0000345 * buy_turnover
        sell_exch_txn = 0.0000345 * sell_turnover
        # SEBI Charges: 0.0001% on turnover (both legs)
        buy_sebi = 0.000001 * buy_turnover
        sell_sebi = 0.000001 * sell_turnover
        # GST: 18% on (Brokerage + Exchange Transaction Charges) (both legs)
        buy_gst = 0.18 * (buy_brokerage + buy_exch_txn)
        sell_gst = 0.18 * (sell_brokerage + sell_exch_txn)
        # Stamp Duty (Maharashtra): 0.003% on buy-side turnover only (max ₹300/day)
        stamp_duty = 0.00003 * buy_turnover
        if state.lower() == 'maharashtra':
            stamp_duty = min(stamp_duty, 300)
        # Round all charges to 2 decimals for reporting
        breakdown = {
            'buy_brokerage': round(buy_brokerage, 2),
            'sell_brokerage': round(sell_brokerage, 2),
            'buy_exch_txn': round(buy_exch_txn, 2),
            'sell_exch_txn': round(sell_exch_txn, 2),
            'buy_sebi': round(buy_sebi, 2),
            'sell_sebi': round(sell_sebi, 2),
            'buy_gst': round(buy_gst, 2),
            'sell_gst': round(sell_gst, 2),
            'stamp_duty': round(stamp_duty, 2),
            'stt': round(stt, 2)
        }
        total = sum(breakdown.values())
        return round(total, 2), breakdown

    def stop_tick_consumer(self):
        """Stop the tick consumer thread cleanly."""
        if hasattr(self, '_tick_consumer_thread') and self._tick_consumer_thread:
            self._tick_consumer_thread_stop = True
            if self._tick_consumer_thread.is_alive():
                logging.info("Waiting for old tick consumer thread to stop...")
                self._tick_consumer_thread.join(timeout=2)
            self._tick_consumer_thread = None
            logging.info("Old tick consumer thread stopped.")

    def start_tick_consumer(self):
        """Start a new tick consumer thread for the current data_socket."""
        if not self.data_socket or not hasattr(self.data_socket, 'tick_queue'):
            logging.warning("No tick_queue found on data_socket; skipping tick consumer thread.")
            return
        if hasattr(self, '_tick_consumer_thread') and self._tick_consumer_thread and self._tick_consumer_thread.is_alive():
            logging.info("Tick consumer thread already running.")
            return
        self._tick_consumer_thread_stop = False
        import threading
        def tick_consumer():
            logging.info("Tick queue consumer thread started.")
            while not getattr(self, '_tick_consumer_thread_stop', False):
                try:
                    tick = self.data_socket.tick_queue.get(timeout=2)
                    symbol = tick.get('symbol')
                    if self.active_trade and symbol == self.active_trade.get('symbol'):
                        ltp = tick.get('ltp')
                        if ltp is not None:
                            self.live_prices[symbol] = float(ltp)
                            logging.info(f"[TICK CONSUMER] {symbol} LTP updated to {ltp}")
                except Exception:
                    continue
            logging.info("Tick consumer thread exiting.")
        self._tick_consumer_thread = threading.Thread(target=tick_consumer, name="TickQueueConsumer", daemon=True)
        self._tick_consumer_thread.start()
