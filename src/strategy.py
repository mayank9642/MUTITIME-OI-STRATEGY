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

        # Only update if the new stoploss is higher than both current stoploss and original stoploss
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

    def identify_high_oi_strikes(self):
        """Identify strikes with highest open interest at 9:20 AM using live option chain data"""
        try:
            self.put_breakout_level = None
            self.call_breakout_level = None
            from src.fyers_api_utils import get_nifty_spot_price
            from src.nse_data_new import get_nifty_option_chain
            spot_price = get_nifty_spot_price()
            logging.info(f"Current Nifty spot price: {spot_price}")
            atm_strike = round(spot_price / 100) * 100
            max_distance = self.max_strike_distance
            min_premium = self.min_premium_threshold
            # --- PUT LEG ---
            put_found = False
            put_chain = None  # Ensure variable is always defined
            for expiry_idx in range(3):
                option_chain = get_nifty_option_chain(expiry_idx)
                if option_chain is None or option_chain.empty:
                    continue
                put_chain = option_chain[(option_chain['option_type'] == 'PE') & (option_chain['strikePrice'] >= atm_strike - max_distance) & (option_chain['strikePrice'] <= atm_strike + max_distance)]
                if not put_chain.empty:
                    put_oi_sorted = put_chain.sort_values('openInterest', ascending=False)
                    for _, row in put_oi_sorted.iterrows():
                        strike_premium = float(row['lastPrice'])
                        if strike_premium >= min_premium:
                            self.highest_put_oi_strike = int(row['strikePrice'])
                            self.put_premium_at_9_20 = strike_premium
                            self.highest_put_oi_symbol = row['symbol']
                            self.put_breakout_level = round(strike_premium * 1.10, 1)
                            self.put_expiry_idx = expiry_idx
                            put_found = True
                            break
                if put_found:
                    break
            # --- CALL LEG ---
            call_found = False
            call_chain = None  # Ensure variable is always defined
            for expiry_idx in range(3):
                option_chain = get_nifty_option_chain(expiry_idx)
                if option_chain is None or option_chain.empty:
                    continue
                # FIX: use option_chain, not call_chain, in the filter below
                call_chain = option_chain[(option_chain['option_type'] == 'CE') & (option_chain['strikePrice'] >= atm_strike - max_distance) & (option_chain['strikePrice'] <= atm_strike + max_distance)]
                if not call_chain.empty:
                    call_oi_sorted = call_chain.sort_values('openInterest', ascending=False)
                    for _, row in call_oi_sorted.iterrows():
                        strike_premium = float(row['lastPrice'])
                        if strike_premium >= min_premium:
                            self.highest_call_oi_strike = int(row['strikePrice'])
                            self.call_premium_at_9_20 = strike_premium
                            self.highest_call_oi_symbol = row['symbol']
                            self.call_breakout_level = round(strike_premium * 1.10, 1)
                            self.call_expiry_idx = expiry_idx
                            call_found = True
                            break
                if call_found:
                    break
            logging.info(f"Selected strikes - PUT: {self.highest_put_oi_strike} (Premium: {self.put_premium_at_9_20}, Breakout: {self.put_breakout_level}, Expiry: {self.put_expiry_idx})")
            logging.info(f"Selected strikes - CALL: {self.highest_call_oi_strike} (Premium: {self.call_premium_at_9_20}, Breakout: {self.call_breakout_level}, Expiry: {self.call_expiry_idx})")
            return put_found or call_found
        except Exception as e:
            logging.error(f"Error identifying high OI strikes: {str(e)}")
            self.put_breakout_level = None
            self.call_breakout_level = None
            return False

    def process_exit(self, exit_reason="manual", exit_price=None):
        """Process exit consistently for all exit types (stoploss, target, time, market close)"""
        if not self.active_trade:
            logging.warning("No active trade to exit")
            return False
        symbol = self.active_trade.get('symbol')
        canonical_symbol = self.get_canonical_symbol(symbol) if symbol else None
        entry_price = self.active_trade.get('entry_price')
        exit_time_actual = datetime.now(pytz.timezone('Asia/Kolkata'))
        quantity = self.active_trade.get('quantity')
        # If exit_price is None (e.g., time-based exit), use last known price
        if exit_price is None:
            exit_price = self.live_prices.get(symbol) or self.active_trade.get('last_known_price') or entry_price
            logging.info(f"No explicit exit price provided. Using last known price for exit: {exit_price}")
        # Always define trailing_sl before use
        trailing_sl = self.active_trade.get('stoploss', '')
        # Calculate brokerage/charges (round trip)
        brokerage, charges_breakdown = self.calculate_fyers_option_charges(entry_price, exit_price, quantity, state='maharashtra')
        gross_pnl = (exit_price - entry_price) * quantity if exit_price is not None else 0
        net_pnl = gross_pnl - brokerage
        # Calculate max up/down
        max_up = self.active_trade.get('max_up', '')
        max_down = self.active_trade.get('max_down', '')
        max_up_pct = self.active_trade.get('max_up_pct', '')
        max_down_pct = self.active_trade.get('max_down_pct', '')
        # Update trade record in history
        idx = self.active_trade.get('trade_record_idx')
        margin_required = entry_price * quantity
        if idx is not None and idx < len(self.trade_history):
            self.trade_history[idx].update({
                'Exit DateTime': exit_time_actual.strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Price': exit_price,
                'P&L': round(net_pnl, 2),
                '% Gain/Loss': round((net_pnl / (entry_price * quantity)) * 100, 2) if entry_price and quantity else '',
                'Trailing SL': trailing_sl,
                'max up': round(max_up, 2) if isinstance(max_up, (int, float)) else '',
                'max down': round(max_down, 2) if isinstance(max_down, (int, float)) else '',
                'max up %': round(max_up_pct, 2) if isinstance(max_up_pct, (int, float)) else '',
                'max down %': round(max_down_pct, 2) if isinstance(max_down_pct, (int, float)) else '',
                'Brokerage': round(brokerage, 2),
                'Margin Required': round(margin_required, 2),
            })
        logging.info(f"Exiting trade: {symbol} | Reason: {exit_reason} | Exit Price: {exit_price}")
        logging.info(f"TRADE_EXIT | Symbol: {symbol} | Entry: {entry_price} | Exit: {exit_price} | Quantity: {quantity} | P&L: {net_pnl:.2f} ({(net_pnl / (entry_price * quantity)) * 100 if entry_price and quantity else 0:.2f}%) | MaxUP: {self.active_trade.get('max_up', 0):.2f} | MaxDN: {self.active_trade.get('max_down', 0):.2f} | Trailing SL: {self.active_trade['stoploss']} | Exit Time: {exit_time_actual.strftime('%Y-%m-%d %H:%M:%S')} | Reason: {exit_reason}")
        self.active_trade['exit_reason'] = exit_reason
        self.active_trade['exit_price'] = exit_price
        self.active_trade['exit_time_actual'] = exit_time_actual
        try:
            self.save_trade_history()
            logging.info("Trade history saved to file after exit.")
            logging.info("If you have frozen the first row in Excel, it will not impact the code or data saving.")
        except Exception as e:
            logging.error(f"Error saving trade history after exit: {str(e)}")
        # Unsubscribe from symbol before clearing active_trade
        self.stop_price_monitoring(canonical_symbol)
        # Remove symbol from live_prices to prevent further updates
        if canonical_symbol in self.live_prices:
            del self.live_prices[canonical_symbol]
        self.active_trade = {}
        logging.info("All trades exited and logged. Trade exit complete.")
        return True

    def execute_trade(self, symbol, side, entry_price):
        """Execute the option trade with correct lot size for Nifty options"""
        try:
            traded_symbol = self.get_canonical_symbol(symbol)  # Ensure traded_symbol is defined early
            qty = 75  # Nifty lot size (update if changed by exchange)
            notional_value = entry_price * qty
            if entry_price < self.min_premium_threshold:
                logging.warning(f"Trade rejected: Premium value ({entry_price}) is below threshold ({self.min_premium_threshold})")
                return None
            if self.paper_trading:
                logging.info(f"PAPER TRADING MODE - Symbol: {traded_symbol}, Price: {entry_price}")
            else:
                logging.info(f"LIVE TRADING - Symbol: {traded_symbol}, Price: {entry_price}")
            logging.info(f"Trade Size: {qty} lots, Notional Value: {notional_value}")
            order_response = None
            if self.paper_trading:
                order_response = {'s': 'ok', 'id': f'PAPER-{int(time.time())}'}
                logging.info(f"Paper trade simulated: {traded_symbol} {side} {qty}")
                # Log initial LTP and P&L
                logging.info(f"TRADE_UPDATE | Symbol: {traded_symbol} | Entry: {entry_price} | LTP: {entry_price} | SL: {self.active_trade['stoploss']} | Target: {self.active_trade['target']} | P&L: 0.00 (0.00%) | MaxUP: 0.00 (0.00%) | MaxDN: 0.00 (0.00%) | Trailing SL: {self.active_trade['stoploss']}")
            else:
                from src.fyers_api_utils import place_market_order
                order_response = place_market_order(self.fyers, traded_symbol, qty, side)
            if order_response and order_response.get('s') == 'ok':
                self.order_id = order_response.get('id')
                if not self.entry_time:
                    self.entry_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                exit_time = self.entry_time + timedelta(minutes=30)
                config = self.config or {}
                stoploss_pct = config.get('strategy', {}).get('stoploss_pct', 20)
                risk_reward_ratio = config.get('strategy', {}).get('risk_reward_ratio', 2)
                stoploss_factor = 1 - (stoploss_pct / 100)
                stoploss_price = round(entry_price * stoploss_factor, 1)
                risk_amount = entry_price - stoploss_price
                target_gain = risk_amount * risk_reward_ratio
                target_price = round(entry_price + target_gain, 1)
                index = 'NIFTY'
                direction = 'BUY' if side.upper() == 'BUY' else 'SELL'
                margin_required = entry_price * qty
                # Calculate brokerage for reporting (entry only, exit will be added on exit)
                brokerage, _ = self.calculate_fyers_option_charges(entry_price, entry_price, qty, state='maharashtra')
                trade_record = {
                    'Entry DateTime': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Index': index,
                    'Symbol': traded_symbol,
                    'Direction': direction,
                    'Entry Price': entry_price,
                    'Exit DateTime': '',
                    'Exit Price': '',
                    'Stop Loss': stoploss_price,
                    'Target': target_price,
                    'Trailing SL': '',
                    'Quantity': qty,
                    'Brokerage': round(brokerage, 2),
                    'P&L': '',
                    'Margin Required': round(margin_required, 2),
                    '% Gain/Loss': '',
                    'max up': '',
                    'max down': '',
                    'max up %': '',
                    'max down %': '',
                }
                # Extract expiry, strike, option_type from traded_symbol
                import re
                expiry = strike = option_type = None
                match = re.match(r"NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d+)(CE|PE)", traded_symbol)
                if match:
                    expiry = match.group(1)
                    strike = int(match.group(2))
                    option_type = match.group(3)
                self.active_trade = {
                    'symbol': traded_symbol,
                    'quantity': qty,
                    'entry_price': entry_price,
                    'entry_time': self.entry_time,
                    'stoploss': stoploss_price,
                    'target': target_price,
                    'exit_time': exit_time,
                    'paper_trade': self.paper_trading,
                    'trade_record_idx': len(self.trade_history),  # Track index for update on exit
                    'expiry': expiry,
                    'strike': strike,
                    'option_type': option_type,
                }
                self.trade_taken_today = True
                logging.info("Daily trade limit: Trade has been taken for today.")
                logging.info(f"=== {'PAPER' if self.paper_trading else 'LIVE'} TRADE EXECUTED ===")
                logging.info(f"Symbol: {traded_symbol}")
                logging.info(f"Entry Price: {entry_price}")
                logging.info(f"Quantity: {qty} lots")
                logging.info(f"Entry Time: {self.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logging.info(f"Stoploss: {self.active_trade['stoploss']}")
                logging.info(f"Target: {self.active_trade['target']}")
                logging.info(f"Exit Time Limit: {self.active_trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                logging.info(f"========================")
                logging.info(f"Trade symbol: {traded_symbol}, Expiry index: {self.put_expiry_idx if 'PE' in traded_symbol else self.call_expiry_idx}")
                logging.info(f"Actual premium at trade time: {entry_price}")
                self.trade_history.append(trade_record)
                try:
                    self.save_trade_history()
                except Exception as e:
                    logging.error(f"Error saving trade history: {str(e)}")
                symbols_to_subscribe = [traded_symbol]
                logging.info(f"(Re)subscribing to traded option symbol for live price updates: {symbols_to_subscribe}")
                # --- ENSURE ONLY TRADED SYMBOL IS SUBSCRIBED AFTER TRADE ENTRY ---
                self.stop_tick_consumer()
                # Close old WebSocket connection completely
                if hasattr(self, 'data_socket') and hasattr(self.data_socket, 'close_connection'):
                    try:
                        self.data_socket.close_connection()
                        logging.info("Closed old data socket after trade entry.")
                    except Exception as e:
                        logging.error(f"Error closing data socket: {e}")
                self.data_socket = None
                # Open a new WebSocket/data socket for only the traded symbol
                from src.fixed_improved_websocket import enhanced_start_market_data_websocket
                self.data_socket = enhanced_start_market_data_websocket(
                    symbols=[traded_symbol],
                    callback_handler=self.ws_price_update
                )
                self.start_tick_consumer()
                logging.info(f"WebSocket subscription started for only traded symbol: {traded_symbol}")
                # Clear live_prices except for traded symbol
                self.live_prices = {traded_symbol: self.live_prices.get(traded_symbol, entry_price)}
                logging.info(f"live_prices after trade entry: {self.live_prices}")
                # --- FYERS MARKET DATA UNSUBSCRIBE FOR NON-TRADED SYMBOLS ---
                symbols_to_unsubscribe = [s for s in self.live_prices.keys() if s != traded_symbol]
                if hasattr(self, 'fyers') and hasattr(self.fyers, 'unsubscribe') and symbols_to_unsubscribe:
                    try:
                        self.fyers.unsubscribe(symbols=symbols_to_unsubscribe, data_type="SymbolUpdate")
                        logging.info(f"Unsubscribed from non-traded symbols: {symbols_to_unsubscribe}")
                    except Exception as e:
                        logging.error(f"Error unsubscribing from non-traded symbols: {e}")
                # --- END FYERS MARKET DATA UNSUBSCRIBE ---
                # --- END ENSURE ONLY TRADED SYMBOL IS SUBSCRIBED ---
                # Place broker-side stoploss order and save its ID
                sl_side = 'SELL' if side == 'BUY' else 'BUY'
                if not self.paper_trading:
                    from src.fyers_api_utils import place_sl_order
                    sl_order_response = place_sl_order(self.fyers, traded_symbol, qty, sl_side, stoploss_price)
                    if sl_order_response and sl_order_response.get('s') == 'ok':
                        self.stop_loss_order_id = sl_order_response.get('id')
                        logging.info(f"Placed broker-side stoploss order: {self.stop_loss_order_id} at {stoploss_price}")
                    else:
                        logging.error(f"Failed to place broker-side stoploss order: {sl_order_response}")
                self.continuous_position_monitor()
                return True
            else:
                logging.error(f"Order placement failed: {order_response}")
                return False
        except Exception as e:
            logging.error(f"Error executing trade: {str(e)}")
            return False

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
            self.data_socket = enhanced_start_market_data_websocket(
                symbols=symbols,
                callback_handler=self.ws_price_update
            )
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
                return            logging.info(f"WebSocket subscription started for symbols: {symbols_to_monitor}")
            canonical_symbols = [self.get_canonical_symbol(s) for s in symbols_to_monitor]
            
            while True:
                for symbol, canonical_symbol in zip(symbols_to_monitor, canonical_symbols):
                    # Get the price from the EXACT symbol key to prevent mixups
                    price = self.live_prices.get(canonical_symbol)
                    breakout_level = breakout_levels[canonical_symbol]
                    
                    # Determine if this is a CE or PE symbol
                    option_type = "unknown"
                    if "CE" in canonical_symbol:
                        option_type = "CE"
                    elif "PE" in canonical_symbol:
                        option_type = "PE"
                    
                    logging.info(f"MONITOR: {canonical_symbol} ({option_type}) price={price} (Breakout: {breakout_level})")
                    
                    if price is not None:
                        if price >= breakout_level:
                            logging.info(f"BREAKOUT DETECTED: {canonical_symbol} ({option_type}) at premium {price} >= {breakout_level}")
                            # Execute the trade immediately when breakout is detected
                            self.execute_trade(canonical_symbol, "BUY", price)
                            self.unsubscribe_non_triggered_symbol(canonical_symbol, canonical_symbols)
                            return True
                time.sleep(2)
            return False
        except Exception as e:
            logging.error(f"Error monitoring for breakout: {str(e)}")
            return None

    def continuous_position_monitor(self):
        """Continuously monitor the position for adjustments and exits"""
        if not self.active_trade:
            logging.info("No active trade to monitor")
            return
        try:
            symbol = self.active_trade.get('symbol')
            exit_time = self.active_trade.get('exit_time')
            entry_time = self.active_trade.get('entry_time')
            while self.active_trade:
                current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                # Strict 30-min hard limit using entry_time
                if entry_time and (current_time - entry_time).total_seconds() >= 30 * 60:
                    logging.info(f"Strict 30-min limit: Exiting trade after 30 minutes.")
                    self.process_exit(exit_reason="MAX_DURATION")
                    break
                # Check for exit_time (legacy logic)                if exit_time and current_time >= exit_time:
                    logging.info("Exit time reached. Exiting position.")
                    self.process_exit(exit_reason="time")
                    break
                
                # Always get the latest LTP from the DataFrame for the active trade symbol
                df_row = self.ltp_df[self.ltp_df.symbol == symbol]
                if not df_row.empty:
                    current_price = float(df_row.iloc[-1].get('ltp', 0))
                    price_source = "DataFrame LTP"
                else:
                    current_price = self.live_prices.get(symbol) or self.active_trade.get('last_known_price')
                    price_source = "live_prices/last_known_price"

                if current_price:
                    logging.info(f"Position monitor price for {symbol}: {current_price} (source: {price_source})")
                    self.active_trade['last_known_price'] = current_price
                    stoploss = self.active_trade.get('stoploss')
                    target = self.active_trade.get('target')
                    logging.info(f"SL/Target check: Current: {current_price}, SL: {stoploss}, Target: {target}")
                    self.log_trade_update()
                    if current_price <= stoploss:
                        logging.info(f"Stoploss hit. Exiting position at defined stoploss: {stoploss}. Current price: {current_price}")
                        self.process_exit(exit_reason="stoploss", exit_price=stoploss)
                        break
                    elif current_price >= target:
                        logging.info(f"Target hit. Exiting position at defined target: {target}. Current price: {current_price}")
                        self.process_exit(exit_reason="target", exit_price=target)
                        break
                exit_time = self.active_trade.get('exit_time')
                if exit_time is not None and datetime.now(pytz.timezone('Asia/Kolkata')) >= exit_time:
                    logging.info("Exit time reached. Exiting position.")
                    self.process_exit(exit_reason="time")
                    break
                # If trade was exited in process_exit, break loop
                if not self.active_trade:
                    break
                time.sleep(5)
            logging.info(f"Stopped monitoring for {symbol}. Trade exited.")
        except Exception as e:
            logging.error(f"Error in continuous_position_monitor: {str(e)}")
            return False

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
        # Fetch live price if available
        current_price = self.live_prices.get(symbol) or self.active_trade.get('last_known_price', entry_price)
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
        logging.info(f"TRADE_UPDATE | Symbol: {symbol} | Entry: {entry_price} | LTP: {current_price} | SL: {self.active_trade['stoploss']} | Target: {target} | P&L: {pnl:.2f} ({pnl_pct:.2f}%) | MaxUP: {max_up_val:.2f} ({max_up_pct_val:.2f}%) | MaxDN: {max_down_val:.2f} ({max_down_pct_val:.2f}%) | Trailing SL: {self.active_trade['stoploss']}")
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
            logging.info(f"[SYMBOL MAP] {orig_symbol} → {fyers_symbol}")
            return fyers_symbol
        # Fallback: use convert_option_symbol_format if available
        try:
            from src.symbol_formatter import convert_option_symbol_format
            converted = convert_option_symbol_format(symbol)
            logging.info(f"[SYMBOL MAP] {orig_symbol} → {converted}")
            return converted
        except Exception as e:
            logging.error(f"[SYMBOL MAP] Error converting {orig_symbol}: {e}")
            return symbol

    def ws_price_update(self, symbol, key, ticks, raw_ticks):
        """
        Callback function to handle WebSocket price updates.
        Accepts symbol, key, ticks, raw_ticks as per the callback handler's call signature.
        Uses canonical symbol as the key for self.live_prices and logging.
        Logs both incoming and canonical symbols for diagnostics.
        """
        try:
            with self._ws_lock:  # Ensure thread safety
                canonical_symbol = self.get_canonical_symbol(symbol)
                ltp = ticks.get('ltp', 0)
                # Extract expiry, strike, option_type from symbol string (assumes format: NSE:UNDERLYINGYYMMDDSTRIKEOPT)
                import re
                match = re.match(r"NSE:[A-Z]+(\d{2}[A-Z]\d{2})(\d+)(CE|PE)", canonical_symbol)
                expiry = strike = option_type = None
                if match:
                    expiry = match.group(1)
                    strike = int(match.group(2))
                    option_type = match.group(3)
                else:
                    # fallback: try to get from ticks if available
                    expiry = ticks.get('expiry')
                    strike = ticks.get('strike')
                    option_type = ticks.get('option_type')
                logging.info(f"WS CALLBACK FULL TICK: symbol={symbol}, canonical={canonical_symbol}, expiry={expiry}, strike={strike}, option_type={option_type}, ws_ticks={ticks}")

                # Save all received contract data into a DataFrame
                tick_data = ticks.copy()
                tick_data['symbol'] = canonical_symbol
                tick_data['expiry'] = expiry
                tick_data['strike'] = strike
                tick_data['option_type'] = option_type
                # Remove old entry if exists (match all contract details)
                self.ltp_df = self.ltp_df[
                    ~(
                        (self.ltp_df.symbol == canonical_symbol) &
                        (self.ltp_df.expiry == expiry) &
                        (self.ltp_df.strike == strike) &
                        (self.ltp_df.option_type == option_type)
                    )
                ]
                # Append new tick data
                self.ltp_df = pd.concat([
                    self.ltp_df,
                    pd.DataFrame([tick_data])
                ], ignore_index=True)

                # After trade entry, use only the required symbol's data for trade logic
                if self.active_trade:
                    traded_symbol = self.active_trade.get('symbol')
                    traded_expiry = self.active_trade.get('expiry')
                    traded_strike = self.active_trade.get('strike')
                    traded_option_type = self.active_trade.get('option_type')
                    # Only use tick if all contract details match
                    if not (
                        canonical_symbol == traded_symbol and
                        expiry == traded_expiry and
                        strike == traded_strike and
                        option_type == traded_option_type
                    ):
                        logging.info(f"IGNORED TICK: {canonical_symbol} (not active trade contract)")
                        return
                    # Use only the correct contract's tick data for trade logic
                    df_row = self.ltp_df[
                        (self.ltp_df.symbol == traded_symbol) &
                        (self.ltp_df.expiry == traded_expiry) &
                        (self.ltp_df.strike == traded_strike) &
                        (self.ltp_df.option_type == traded_option_type)
                    ]
                    if not df_row.empty:
                        self.active_trade['last_known_price'] = float(df_row.iloc[0].get('ltp', 0))
                        logging.info(f"LTP UPDATE FOR ACTIVE TRADE: {traded_symbol} {self.active_trade['last_known_price']}")
                else:
                    # Handle breakout detection for both CE and PE symbols
                    option_type = None
                    if 'CE' in canonical_symbol:
                        option_type = 'CE'
                    elif 'PE' in canonical_symbol:
                        option_type = 'PE'
                    if canonical_symbol.startswith('NSE:NIFTY') and 0 < ltp < 5000:
                        self.live_prices[canonical_symbol] = ltp
                        if option_type == 'CE' and hasattr(self, 'call_breakout_level') and self.call_breakout_level:
                            if canonical_symbol == self.get_canonical_symbol(self.highest_call_oi_symbol or ''):
                                if ltp >= self.call_breakout_level:
                                    logging.info(f"BREAKOUT DETECTED IN CALLBACK: {canonical_symbol} (CE) at premium {ltp} >= {self.call_breakout_level}")
                        elif option_type == 'PE' and hasattr(self, 'put_breakout_level') and self.put_breakout_level:
                            if canonical_symbol == self.get_canonical_symbol(self.highest_put_oi_symbol or ''):
                                if ltp >= self.put_breakout_level:
                                    logging.info(f"BREAKOUT DETECTED IN CALLBACK: {canonical_symbol} (PE) at premium {ltp} >= {self.put_breakout_level}")
                        logging.info(f"No active trade. Updated price for symbol: {canonical_symbol}, LTP: {ltp}")
        except Exception as e:
            logging.error(f"Error in ws_price_update: {e}")
            logging.error(traceback.format_exc())

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
