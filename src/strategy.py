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
from src.regulatory_checks import run_regulatory_checks
import re

class OpenInterestStrategy:
    def __init__(self):
        # Initialize your strategy here
        self.active_trade = {}
        self.live_prices = {}
        # DataFrame to store LTPs for each contract (symbol, expiry, strike, option_type)
        self.ltp_df = pd.DataFrame(columns=["symbol", "expiry", "strike", "option_type", "ltp"])
        # Load configuration from config/config.yaml when available
        try:
            from src.config import load_config
            self.config = load_config()
        except Exception:
            # Fallback to empty config if loading fails
            self.config = {}
        # Run lightweight regulatory checks & guidance (logs warnings/instructions)
        try:
            status = run_regulatory_checks(self.config)
            if not status.get('ok'):
                logging.warning("Regulatory checks reported issues: %s", status.get('notes'))
            else:
                logging.info("Regulatory checks completed: %s", status.get('notes'))
        except Exception:
            logging.exception("Failed to run regulatory checks")
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
        # Load persistent balance (for PAPER mode) and performance state
        self.current_balance = None
        self.initial_balance = None
        # Pass fyers client into OrderManager so it can call broker APIs when not in paper mode
        try:
            self.order_manager = OrderManager(broker_api=self.fyers, paper_trading=self.paper_trading)
        except Exception:
            self.order_manager = OrderManager(paper_trading=self.paper_trading)
        # Register callback so OrderManager can notify us immediately when a GTT triggers
        try:
            self.order_manager.on_gtt_triggered = lambda order: threading.Thread(target=self._on_order_manager_gtt_callback, args=(order,), daemon=True).start()
        except Exception:
            logging.debug("Could not register on_gtt_triggered callback on order_manager")
        self._ws_lock = threading.Lock()

        # Load persisted balance info (if present)
        try:
            self.load_balance()
        except Exception:
            logging.debug("Could not load persisted balance; using defaults")
        # For log rate-limiting and aggregation
        self._last_logged_ltp = {}
        self._last_logged_time = {}
        # Minimum seconds between WS_UPDATE logs per symbol
        self._ws_update_min_seconds = 3
        # Minimum absolute change in LTP to trigger a WS_UPDATE log (or percent threshold)
        self._ws_update_min_change = 0.5

        # Rate-limit PAPER STATUS logs
        self._last_paper_status_time = 0
        self._paper_status_min_seconds = 5
        # When True, downgrade websocket INFO tick logs to DEBUG to avoid flooding during monitoring
        self._suppress_ws_update_info = False

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
        trailing_stop_pct = float(config.get('strategy', {}).get('trailing_stop_pct', 8))

        # Calculate new potential stoploss (current price * (1 - trailing_pct))
        potential_stoploss = current_price * (1 - (trailing_stop_pct / 100.0))

        logging.debug(f"TRAILING SL DEBUG | symbol: {symbol} | entry_price: {entry_price} | current_price: {current_price} | trailing_stop_pct: {trailing_stop_pct} | current_sl: {current_sl} | original_stoploss: {original_stoploss}")

        # Only update if the new stoploss is higher than both current stoploss and original stoploss
        if potential_stoploss > current_sl and potential_stoploss > original_stoploss:
            old_sl = self.active_trade.get('stoploss')
            # Round to 2 decimals for option prices (matches logging elsewhere)
            self.active_trade['stoploss'] = round(potential_stoploss, 2)
            self.active_trade['trailing_stoploss'] = round(potential_stoploss, 2)
            logging.info(f"Trailing stoploss updated from {old_sl} to {self.active_trade['stoploss']}")
            return True
        else:
            logging.debug(f"TRAILING SL DEBUG | No update: potential_stoploss ({potential_stoploss}) <= current_sl ({current_sl}) or original_stoploss ({original_stoploss})")
            return False
        
    def start_exit_monitor(self):
        """
        Start a background exit monitor thread which watches for SL/Target hits and closes the trade.
        This function returns immediately after launching the daemon thread. The actual
        monitoring loop runs in `_exit_monitor_loop`.
        """
        # If already running, don't start another
        if hasattr(self, '_exit_monitor_thread') and getattr(self, '_exit_monitor_thread'):
            if getattr(self, '_exit_monitor_thread').is_alive():
                logging.info("[EXIT MONITOR] Already running")
                return True

        def _exit_monitor_loop():
            logging.info("[EXIT MONITOR] Started for active trade")
            while self.active_trade and not self.market_closed:
                try:
                    current = self.get_active_trade_ltp()
                    if current is None:
                        time.sleep(1)
                        continue
                    # Enforce maximum trade duration to avoid premium melt-down
                    try:
                        entry_time = self.active_trade.get('entry_time')
                        if entry_time is not None:
                            # Use IST-aware comparison if entry_time stored with timezone
                            now_dt = datetime.now(pytz.timezone('Asia/Kolkata'))
                            try:
                                elapsed_secs = (now_dt - entry_time).total_seconds()
                            except Exception:
                                # Fallback if entry_time is naive or not a datetime
                                try:
                                    elapsed_secs = time.time() - float(entry_time.timestamp())
                                except Exception:
                                    elapsed_secs = 0
                            max_mins = int(self.config.get('strategy', {}).get('max_trade_duration_minutes', 30))
                            if elapsed_secs and elapsed_secs >= (max_mins * 60):
                                logging.info(f"[EXIT MONITOR][TIMEOUT] Active trade exceeded max duration ({max_mins} minutes). Closing trade.")
                                # Use current LTP as exit price if available
                                try:
                                    self._close_active_trade(exit_reason='TIMEOUT', exit_price=current)
                                except Exception:
                                    logging.exception("[EXIT MONITOR][ERROR] Failed to close trade on timeout")
                                return
                    except Exception:
                        logging.debug("[EXIT MONITOR] Error while checking max trade duration; continuing monitor")
                    sl = self.active_trade.get('stoploss')
                    tgt = self.active_trade.get('target')
                    # For long BUY trades: exit when price <= SL or >= Target
                    if sl is not None and current <= sl:
                        self._close_active_trade(exit_reason='STOPLOSS', exit_price=current)
                        return
                    if tgt is not None and current >= tgt:
                        self._close_active_trade(exit_reason='TARGET', exit_price=current)
                        return
                except Exception:
                    logging.exception("[EXIT MONITOR] Exception in exit monitor")
                time.sleep(1)
            logging.info("[EXIT MONITOR] Exiting")

        import threading as _thr
        self._exit_monitor_thread = _thr.Thread(target=_exit_monitor_loop, name="ExitMonitorThread", daemon=True)
        self._exit_monitor_thread.start()
        return True
        
    def monitor_exit(self):
        if not hasattr(self, '_exit_monitor_thread') or not getattr(self, '_exit_monitor_thread'):
            self._exit_monitor_thread = threading.Thread(target=self.start_exit_monitor, name="ExitMonitor", daemon=True)
            self._exit_monitor_thread.start()
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

    # ...existing code...

        # For long positions, we want to move the stoploss up as price increases
    # ...existing code...

        # Only update if the new stoploss is higher than both current stoploss and original_stoploss
        if potential_stoploss > current_sl and potential_stoploss > original_stoploss:
            old_sl = self.active_trade['stoploss']
            self.active_trade['stoploss'] = round(potential_stoploss, 3)
            self.active_trade['trailing_stoploss'] = round(potential_stoploss, 3)
            logging.info(f"[TRAILING_SL][UPDATE] Trailing stoploss updated from {old_sl} to {self.active_trade['stoploss']}")
            # --- Broker-side trailing stoploss update ---
            if not self.paper_trading and hasattr(self, 'stop_loss_order_id') and self.stop_loss_order_id:
                try:
                    from src.fyers_api_utils import modify_order
                    response = modify_order(self.fyers, self.stop_loss_order_id, stop_price=self.active_trade['stoploss'])
                    if response and response.get('s') == 'ok':
                        logging.info(f"[TRAILING_SL][BROKER] Broker stoploss order modified: {self.stop_loss_order_id} to {self.active_trade['stoploss']}")
                    else:
                        logging.error(f"[TRAILING_SL][BROKER][ERROR] Failed to modify broker stoploss order: {response}")
                except Exception as e:
                    logging.error(f"[TRAILING_SL][BROKER][ERROR] Exception while modifying broker stoploss order: {e}")
            return True
        else:
            # ...existing code...
            return False

    def validate_fyers_symbols(self, symbols):
        """
        Validate option symbols against Fyers master contract (or API) before subscribing.
        Returns only valid symbols.
        """
        # Robust validation + canonicalization before subscribing to Fyers websocket.
        # Use the project's canonicalizer (get_canonical_symbol) when available so
        # we always send Fyers-compatible, compact symbols like "NSE:NIFTY04NOV2527450CE".
        canonical = []
        invalid = []
        try:
            for s in symbols:
                if not s or not isinstance(s, str):
                    invalid.append(s)
                    continue
                try:
                    cs = self.get_canonical_symbol(s)
                except Exception:
                    # Fallback to identity if conversion fails
                    cs = s
                # Basic sanity checks: must start with NSE: and end with CE/PE and be reasonably long
                if isinstance(cs, str) and cs.startswith('NSE:') and (cs.endswith('CE') or cs.endswith('PE')) and len(cs) >= 12:
                    canonical.append(cs)
                else:
                    invalid.append(s)
        except Exception:
            logging.exception("Exception while canonicalizing symbols")

        # Deduplicate while preserving order
        seen = set()
        valid_symbols = []
        for s in canonical:
            if s not in seen:
                seen.add(s)
                valid_symbols.append(s)

        if invalid:
            logging.warning(f"Some symbols are invalid or could not be canonicalized and will not be subscribed: {invalid}")

        if not valid_symbols:
            logging.error("No valid symbols after canonicalization - subscription aborted.")

        return valid_symbols

    def identify_high_oi_strikes(self):
        """
        Analyze option chain data to identify highest OI strikes for CE and PE.
        Sets self.highest_call_oi_strike, self.highest_put_oi_strike, self.highest_call_oi_symbol, self.highest_put_oi_symbol,
        self.call_breakout_level, self.put_breakout_level for trade monitoring.
        Returns True if analysis is successful, False otherwise.
        """
        try:
            from src.fetch_option_oi_fyers import fetch_option_oi_fyers
            oi_data = fetch_option_oi_fyers(self.fyers, symbol="NSE:NIFTY50-INDEX", strikecount=20)
            if oi_data is None or len(oi_data) == 0:
                logging.error("[OI_ANALYSIS][ERROR] No option chain data returned from Fyers.")
                return False
            # ...existing code...
            ce_df = oi_data[oi_data['option_type'] == 'CE']
            pe_df = oi_data[oi_data['option_type'] == 'PE']
            if ce_df.empty or pe_df.empty:
                logging.error("[OI_ANALYSIS][ERROR] No CE or PE data available.")
                logging.debug("[OI_ANALYSIS][DEBUG] ce_df empty? %s; pe_df empty? %s", ce_df.empty, pe_df.empty)
                return False
            ce_df = ce_df[ce_df['ltp'].notnull()]
            pe_df = pe_df[pe_df['ltp'].notnull()]
            if ce_df.empty or pe_df.empty:
                logging.error("[OI_ANALYSIS][ERROR] No CE or PE contracts with valid LTP.")
                logging.debug("[OI_ANALYSIS][DEBUG] ce_df.head():\n%s", ce_df.head().to_string())
                logging.debug("[OI_ANALYSIS][DEBUG] pe_df.head():\n%s", pe_df.head().to_string())
                return False
            # --- Enhanced Strike Selection Logic ---
            spot_price = self.live_prices.get('NSE:NIFTY', None)
            if spot_price is None:
                spot_price = ce_df['strike'].median()  # fallback
            atm_strike = round(spot_price / 100) * 100 if spot_price else None
            max_distance = self.max_strike_distance
            min_premium = self.min_premium_threshold
            ce_filtered = ce_df[(ce_df['strike'] >= atm_strike - max_distance) & (ce_df['strike'] <= atm_strike + max_distance)]
            pe_filtered = pe_df[(pe_df['strike'] >= atm_strike - max_distance) & (pe_df['strike'] <= atm_strike + max_distance)]
            ce_sorted = ce_filtered.sort_values('oi', ascending=False)
            pe_sorted = pe_filtered.sort_values('oi', ascending=False)
            highest_call_row = None
            for _, row in ce_sorted.iterrows():
                if row['ltp'] >= min_premium:
                    highest_call_row = row
                    break
            if highest_call_row is None and not ce_sorted.empty:
                highest_call_row = ce_sorted.iloc[0]
            highest_put_row = None
            for _, row in pe_sorted.iterrows():
                if row['ltp'] >= min_premium:
                    highest_put_row = row
                    break
            if highest_put_row is None and not pe_sorted.empty:
                highest_put_row = pe_sorted.iloc[0]
            if highest_call_row is None or highest_put_row is None:
                logging.error("[OI_ANALYSIS][ERROR] No suitable CE/PE strike found above premium threshold.")
                # Emit helpful debug info to diagnose why selection failed
                try:
                    logging.debug("[OI_ANALYSIS][DEBUG] atm_strike=%s, max_distance=%s, min_premium=%s", atm_strike, max_distance, min_premium)
                    logging.debug("[OI_ANALYSIS][DEBUG] ce_filtered.head():\n%s", ce_filtered.head().to_string())
                    logging.debug("[OI_ANALYSIS][DEBUG] pe_filtered.head():\n%s", pe_filtered.head().to_string())
                    logging.debug("[OI_ANALYSIS][DEBUG] ce_sorted.head():\n%s", ce_sorted.head().to_string())
                    logging.debug("[OI_ANALYSIS][DEBUG] pe_sorted.head():\n%s", pe_sorted.head().to_string())
                except Exception:
                    logging.exception("[OI_ANALYSIS][DEBUG] Failed to dump debug frames for OI analysis")
                return False
            self.highest_call_oi_strike = int(highest_call_row['strike'])
            self.highest_put_oi_strike = int(highest_put_row['strike'])
            self.highest_call_oi_symbol = highest_call_row['symbol']
            self.highest_put_oi_symbol = highest_put_row['symbol']
            breakout_pct = self.config.get('strategy', {}).get('breakout_pct', 10)
            self.call_breakout_level = float(highest_call_row['ltp']) * (1 + breakout_pct / 100)
            self.put_breakout_level = float(highest_put_row['ltp']) * (1 + breakout_pct / 100)
            logging.info(
                "OI_ANALYSIS: CE Strike=%s Symbol=%s Breakout=%.2f | PE Strike=%s Symbol=%s Breakout=%.2f",
                self.highest_call_oi_strike, self.highest_call_oi_symbol, self.call_breakout_level,
                self.highest_put_oi_strike, self.highest_put_oi_symbol, self.put_breakout_level
            )
            return True
        except Exception as e:
            logging.error(f"[OI_ANALYSIS][ERROR] Exception: {str(e)}")
            logging.error(traceback.format_exc())
            return False

    def subscribe_to_valid_symbols(self, symbols):
        """
        Subscribe only to valid symbols for monitoring.
        """
        # Do not allow subscriptions until breakout levels are fixed and it's >= 09:20
        try:
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            pid = os.getpid()
            analysis_time = datetime.strptime("09:20", "%H:%M").time()
            if not getattr(self, 'breakout_levels_fixed', False):
                logging.warning(f"[WS][BLOCKED] Attempted to subscribe to symbols before OI analysis completed. Subscription skipped.")
                return False
            if ist_now.time() < analysis_time:
                logging.warning(f"[WS][BLOCKED] Attempted to subscribe before 09:20. Current time: {ist_now.time().strftime('%H:%M:%S')} - subscription skipped.")
                return False
        except Exception:
            # If timezone/time check fails for any reason, be conservative and block subscription
            logging.warning("[WS][BLOCKED] Time check failed; subscription blocked as a safety measure.")
            return False

        valid_symbols = self.validate_fyers_symbols(symbols)
        if not valid_symbols:
            logging.error("No valid symbols to subscribe for monitoring.")
            return
        # Start websocket subscription for valid symbols
        self.data_socket = enhanced_start_market_data_websocket(valid_symbols, self.ws_price_update)
        logging.info(f"[WS][SUBSCRIBE] Subscribed to valid symbols: {valid_symbols}")
        # Start the tick consumer if available
        try:
            self.start_tick_consumer()
        except Exception:
            logging.debug("[WS][SUBSCRIBE] start_tick_consumer failed or not available")

    def ws_price_update(self, symbol, key, ticks, raw_ticks):
        """
        Callback function to handle WebSocket price updates.
        Accepts symbol, key, ticks, raw_ticks as per the callback handler's call signature.
        Uses canonical symbol as the key for self.live_prices and logging.
        Logs both incoming and canonical symbols for diagnostics.
        """
        try:
            canonical_symbol = self.get_canonical_symbol(symbol)
            canonical_symbol = canonical_symbol.strip()
            # ...existing code...
            # Try to extract expiry/strike/type from the canonical symbol.
            # Accept forms with or without the 'NSE:' prefix and CE/PE or single-letter C/P suffix.
            expiry = strike = option_type = None
            # Pattern 1: strike then option suffix (e.g. ...<strike><CE|PE>)
            pat1 = re.compile(r"(?:NSE:)?NIFTY(\d{2})([A-Z]{3})(\d{2})(\d{4,5})(C(?:E)?|P(?:E)?)$")
            # Pattern 2: option letter then strike (e.g. ...<C|P><strike>)
            pat2 = re.compile(r"(?:NSE:)?NIFTY(\d{2})([A-Z]{3})(\d{2})(C(?:E)?|P(?:E)?)(\d{4,5})$")

            m = pat1.match(canonical_symbol)
            if m:
                # groups: underlying, day, mon, year, strike, opt_type
                day = m.group(2)
                mon = m.group(3)
                yr = m.group(4)
                expiry = f"{day}{mon}{yr}"
                try:
                    strike = int(m.group(5))
                except Exception:
                    strike = None
                opt = m.group(6)
                option_type = 'CE' if opt.upper().startswith('C') else 'PE'
                underlying = m.group(1).upper()
                if strike is not None:
                    canonical_symbol = f"NSE:{underlying}{expiry}{str(strike).zfill(5)}{option_type}"
            else:
                m2 = pat2.match(canonical_symbol)
                if m2:
                    # groups: underlying, day, mon, year, opt_type, strike
                    day = m2.group(2)
                    mon = m2.group(3)
                    yr = m2.group(4)
                    expiry = f"{day}{mon}{yr}"
                    opt = m2.group(5)
                    option_type = 'CE' if opt.upper().startswith('C') else 'PE'
                    try:
                        strike = int(m2.group(6))
                    except Exception:
                        strike = None
                    underlying = m2.group(1).upper()
                    if strike is not None:
                        canonical_symbol = f"NSE:{underlying}{expiry}{str(strike).zfill(5)}{option_type}"
                else:
                    # As a last resort, try a looser parse (digits + month + digits) to salvage values
                    loose = re.search(r"(\d{2})([A-Z]{3})(\d{2})(\d{4,5})", canonical_symbol)
                    if loose:
                        day, mon, yr, st = loose.groups()
                        expiry = f"{day}{mon}{yr}"
                        try:
                            strike = int(st)
                        except Exception:
                            strike = None
                        # Option type still unknown in this fallback
                        option_type = None
                    # else leave expiry/strike/type as None
            ltp = ticks.get('ltp') if isinstance(ticks, dict) else None
            # ...existing code...
            if ltp is not None:
                self.live_prices[canonical_symbol] = ltp
                df_row = self.ltp_df[
                    (self.ltp_df.symbol == canonical_symbol) &
                    (self.ltp_df.expiry == expiry) &
                    (self.ltp_df.strike == strike) &
                    (self.ltp_df.option_type == option_type)
                ]
                if not df_row.empty:
                    self.ltp_df.loc[df_row.index, 'ltp'] = ltp
                else:
                    new_row = {
                        'symbol': canonical_symbol,
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': option_type,
                        'ltp': ltp
                    }
                    self.ltp_df = pd.concat([self.ltp_df, pd.DataFrame([new_row])], ignore_index=True)
                # Rate-limit WS_UPDATE logs: log only on meaningful LTP change or once every configured interval
                try:
                    now = time.time()
                    last_ltp = self._last_logged_ltp.get(canonical_symbol)
                    last_time = self._last_logged_time.get(canonical_symbol, 0)
                    should_log = False
                    if last_ltp is None:
                        should_log = True
                    else:
                        # Absolute change threshold or percent-based threshold can be used
                        if abs(ltp - float(last_ltp)) >= self._ws_update_min_change:
                            should_log = True
                        elif (now - last_time) >= self._ws_update_min_seconds:
                            should_log = True

                        if should_log:
                            # If monitoring is active or suppression flag set, keep these as DEBUG to avoid flooding
                            msg = f"[WS_UPDATE] Updated {canonical_symbol} | expiry: {expiry} | strike: {strike} | type: {option_type} | ltp: {ltp}"
                            if getattr(self, '_suppress_ws_update_info', False):
                                logging.debug(msg)
                            else:
                                if expiry is not None and strike is not None and option_type is not None:
                                    logging.info(msg)
                                else:
                                    logging.debug(msg)
                        self._last_logged_ltp[canonical_symbol] = ltp
                        self._last_logged_time[canonical_symbol] = now
                except Exception:
                    # Fallback to always log if anything goes wrong with rate-limiter
                    msg = f"[WS_UPDATE] Updated {canonical_symbol} | expiry: {expiry} | strike: {strike} | type: {option_type} | ltp: {ltp}"
                    if getattr(self, '_suppress_ws_update_info', False):
                        logging.debug(msg)
                    else:
                        if expiry is not None and strike is not None and option_type is not None:
                            logging.info(msg)
                        else:
                            logging.debug(msg)
            else:
                logging.warning(f"[WS_TICK][WARN] Tick update missing LTP for {canonical_symbol}: {ticks}")
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
        import datetime as _dt
        try:
            # Define required columns in order
            # Match Nifty-style header exactly
            columns = [
                'Entry DateTime','Index','Symbol','Direction','Entry Price','Exit DateTime','Exit Price',
                'Stop Loss','Target','Trailing SL','Quantity','Brokerage','P&L','Net P&L','Margin Required',
                '% Gain/Loss','Max Up (₹)','Max Down (₹)','Max Up (%)','Max Down (%)','VIX','Balance After Trade'
            ]

            # Build a normalized list of rows ensuring all required fields exist and derived fields are computed
            rows = []
            for t in list(self.trade_history):
                try:
                    entry_dt = t.get('Entry DateTime')
                    exit_dt = t.get('Exit DateTime')
                    symbol = t.get('Symbol')
                    idx = t.get('Index', '')
                    direction = t.get('Direction', '')
                    entry_price = float(t.get('Entry Price') or 0)
                    exit_price = float(t.get('Exit Price') or 0)
                    stop_loss = t.get('SL') if 'SL' in t else t.get('Stop Loss', t.get('stoploss', ''))
                    trailing_sl = t.get('trailing_stoploss', t.get('Trailing SL', ''))
                    target = t.get('Target', t.get('target', ''))
                    qty = t.get('Quantity', t.get('Quantity', t.get('quantity', 0)))
                    try:
                        qty = int(qty)
                    except Exception:
                        try:
                            qty = int(float(qty))
                        except Exception:
                            pass
                    pnl = float(t.get('P&L', t.get('P&L', 0) or 0))
                except Exception:
                    # Skip malformed entries but continue saving others
                    logging.debug("Skipping malformed trade history entry during save")
                    continue

                # Compute brokerage (approx) using available helper
                brokerage = 0.0
                try:
                    brokerage, _ = self.calculate_fyers_option_charges(entry_price, exit_price, qty)
                except Exception:
                    brokerage = 0.0

                net_pnl = pnl - float(brokerage)
                # Margin required: be careful not to double-count lot-size.
                # execute_trade() stores 'quantity' as actual contracts (lots * lot_size)
                # and also stores 'lots' as the configured number of lots. Use 'lots'
                # when available to compute margin as entry_price * lots * lot_size.
                try:
                    strategy_cfg = self.config.get('strategy', {}) if self.config else {}
                    quantity_is_lots = bool(strategy_cfg.get('quantity_is_lots', True))
                    lot_map = strategy_cfg.get('lot_size_map', {'NIFTY': 65})
                    # Determine lot_size for this symbol (default NIFTY=65)
                    lot_size = 1
                    sym_upper = (symbol or '').upper()
                    import re
                    for k, v in lot_map.items():
                        try:
                            pattern = r"\b" + re.escape(str(k).upper()) + r"\b"
                            if re.search(pattern, sym_upper):
                                lot_size = int(v)
                                break
                        except Exception:
                            continue
                    if lot_size == 1 and 'NIFTY' in sym_upper:
                        lot_size = 65

                    # If 'lots' is present in the saved trade record, compute margin from lots*lot_size.
                    lots_val = t.get('lots') if isinstance(t, dict) else None
                    try:
                        lots_val = int(lots_val) if lots_val is not None else None
                    except Exception:
                        lots_val = None

                    if lots_val is not None and quantity_is_lots:
                        margin_required = float(entry_price) * float(lots_val) * float(lot_size)
                    else:
                        # Otherwise assume 'quantity' already contains total contracts (lots * lot_size)
                        margin_required = float(entry_price) * float(qty)
                except Exception:
                    margin_required = 0.0
                pct_gain_loss = (pnl / margin_required * 100.0) if margin_required else 0.0

                max_up = float(t.get('Max Up', t.get('max up', t.get('max_up', 0)) or 0))
                max_down = float(t.get('Max Down', t.get('max down', t.get('max_down', 0)) or 0))
                max_up_pct = float(t.get('Max Up %', t.get('max up %', t.get('max_up_pct', 0)) or 0))
                max_down_pct = float(t.get('Max Down %', t.get('max down %', t.get('max_down_pct', 0)) or 0))

                vix = ''
                try:
                    vix = self.live_prices.get('NSE:VIX') or self.live_prices.get('VIX') or ''
                except Exception:
                    vix = ''

                balance_after = t.get('Balance After Trade', '')

                row = {
                    'Entry DateTime': entry_dt,
                    'Index': idx,
                    'Symbol': symbol,
                    'Direction': direction,
                    'Entry Price': entry_price,
                    'Exit DateTime': exit_dt,
                    'Exit Price': exit_price,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'Trailing SL': trailing_sl,
                    'Quantity': qty,
                    'Brokerage': brokerage,
                    'P&L': pnl,
                    'Net P&L': net_pnl,
                    'Margin Required': margin_required,
                    '% Gain/Loss': pct_gain_loss,
                    'Max Up (₹)': max_up,
                    'Max Down (₹)': max_down,
                    'Max Up (%)': max_up_pct,
                    'Max Down (%)': max_down_pct,
                    'VIX': vix,
                    'Balance After Trade': balance_after
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            # Ensure all columns present
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            df = df[columns]
            # Save to CSV
            df.to_csv('logs/trade_history.csv', index=False)
            # Save to Excel with today's date
            today = date.today().strftime('%Y%m%d')
            excel_path = f'logs/trade_history_{today}.xlsx'
            # Excel does not support timezone-aware datetimes. Convert any timezone-aware
            # datetime columns to naive datetimes by removing tzinfo to avoid write errors.
            def _drop_tz(v):
                try:
                    if isinstance(v, _dt.datetime) and v.tzinfo is not None:
                        return v.replace(tzinfo=None)
                except Exception:
                    pass
                return v

            for col in ['Entry DateTime', 'Exit DateTime']:
                if col in df.columns:
                    try:
                        df[col] = df[col].apply(_drop_tz)
                    except Exception:
                        # Best-effort: if conversion fails, coerce via pandas to_datetime then drop tz
                        try:
                            df[col] = pd.to_datetime(df[col]).apply(lambda x: x.replace(tzinfo=None) if getattr(x, 'tzinfo', None) is not None else x)
                        except Exception:
                            logging.debug(f"Could not sanitize timezone for column {col}")

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            logging.info(f"Trade history saved to CSV and Excel: {excel_path}")
        except Exception as e:
            logging.exception(f"Error saving trade history: {e}")

    # --- Balance persistence and performance helpers ---
    def load_balance(self):
        """Load persisted balance from disk or initialize from config/defaults"""
        try:
            path = 'logs/balance.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.current_balance = float(data.get('current_balance', 0))
                    self.initial_balance = float(data.get('initial_balance', self.current_balance))
                    logging.info(f"Loaded persisted balance: {self.current_balance}")
            else:
                # Initialize from config or default
                cfg = self.config.get('strategy', {}) if self.config else {}
                start_bal = float(cfg.get('starting_balance', 100000.0))
                # Default: reset to 100000 as requested by user
                start_bal = 100000.0
                self.initial_balance = start_bal
                self.current_balance = start_bal
                # Persist initial balance
                self.save_balance()
                logging.info(f"Initialized balance to {self.current_balance}")
        except Exception:
            logging.exception("Failed to load or initialize balance; using defaults")
            self.initial_balance = self.initial_balance or 100000.0
            self.current_balance = self.current_balance or self.initial_balance

    def save_balance(self):
        """Persist current balance to disk"""
        try:
            path = 'logs'
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            data = {
                'initial_balance': float(self.initial_balance or 0),
                'current_balance': float(self.current_balance or 0),
            }
            with open(os.path.join(path, 'balance.json'), 'w') as f:
                json.dump(data, f)
            logging.info(f"Persisted balance to {os.path.join(path, 'balance.json')}")
        except Exception:
            logging.exception("Failed to persist balance to disk")
        except Exception as e:
            logging.error(f"Error saving trade history: {str(e)}")

    def reset_balance(self, amount=100000.0):
        """Force-reset the in-memory and persisted balance to the given amount."""
        try:
            self.initial_balance = float(amount)
            self.current_balance = float(amount)
            try:
                self.save_balance()
            except Exception:
                logging.debug("Failed to persist balance during reset")
            logging.info(f"[BALANCE] Reset balance to {self.current_balance:,.2f}")
            return True
        except Exception:
            logging.exception("Failed to reset balance")
            return False

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
                logging.info(f"Waiting for market to open (09:15 IST)... Current time: {current_time.strftime('%H:%M:%S')}")
                time.sleep(30)
                ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                current_time = ist_now.time()
            # At market open, compute time until analysis (9:20) and log a single wait message similar to the sample logs
            secs_to_analysis = int((datetime.combine(ist_now.date(), analysis_time) - datetime.combine(ist_now.date(), current_time)).total_seconds())
            if secs_to_analysis > 0:
                logging.info(f"Waiting {secs_to_analysis} seconds until 9:20 IST for first 5-min candle to form...")
                # Sleep until analysis_time (keep simple and blocking to match the single-message behaviour)
                time.sleep(secs_to_analysis)
            logging.info("[MARKET][INFO] It's 9:20 or later. Ready to run strategy and OI analysis.")
            # Do NOT call run_strategy here. Just return control to caller.
            return {"success": True, "message": "Market open and 9:20 reached. Ready for OI analysis."}
        except Exception as e:
            logging.error(f"[MARKET][ERROR] Error in wait_for_market_open: {str(e)}")
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
                    logging.info(f"[LOG][BACKUP] Log file backed up to {backup_file}")
                    
                # Clear the current log file
                with open(log_file, 'w') as f:
                    f.write(f"Log file cleared on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                logging.info("[LOG][CLEAR] Log file has been cleared for new trading day")
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
            
            logging.info("STRATEGY INIT: Initializing strategy for the day")
            # Reset daily state variables
            self.trade_taken_today = False
            self.market_closed = False
            self.put_breakout_level = 0
            self.call_breakout_level = 0
            self.highest_put_oi_strike = 0
            self.highest_call_oi_strike = 0
            
            # Clear any active trades from previous day
            self.active_trade = {}
            # Do NOT subscribe to symbols here; defer until after market open and OI analysis
            return True
        except Exception as e:
            logging.error(f"[INIT][ERROR] Error initializing strategy for the day: {str(e)}")
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
            logging.info("STRATEGY START: Running Open Interest Option Buying Strategy")
            # Get current time in IST
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            current_time = ist_now.time()
            market_open_time = datetime.strptime("09:15", "%H:%M").time()
            market_close_time = datetime.strptime("15:30", "%H:%M").time()
            # Check if market is closed
            if self.market_closed or current_time >= market_close_time:
                logging.info("[STRATEGY][INFO] Market is closed. Skipping strategy execution.")
                return {"success": False, "message": "Market closed"}
            # Check if today is a weekday
            if ist_now.weekday() > 4:
                logging.info("[STRATEGY][INFO] Today is weekend. Market closed.")
                return {"success": False, "message": "Weekend"}
            # Check if trade already taken today
            if self.trade_taken_today and not force_analysis:
                logging.info("[STRATEGY][INFO] Trade already taken today. Skipping strategy execution.")
                return {"success": True, "message": "Trade already taken today"}
            # Wait for market open if needed
            if current_time < market_open_time:
                logging.info("[STRATEGY][INFO] Market not open yet. Waiting for market open...")
                return self.wait_for_market_open()
            # Wait for 9:20 before running OI analysis and breakout monitoring
            analysis_time = datetime.strptime("09:20", "%H:%M").time()
            # Add a small tolerance to avoid race conditions around the exact second boundary
            remaining_secs = (datetime.combine(ist_now.date(), analysis_time) - datetime.combine(ist_now.date(), current_time)).total_seconds()
            if remaining_secs > 1 and not force_analysis:
                mins, secs = divmod(remaining_secs, 60)
                logging.info(f"[STRATEGY][INFO] Waiting for 9:20... Current time: {current_time.strftime('%H:%M:%S')}, OI analysis in: {int(mins)}m {int(secs)}s")
                return {"success": False, "message": "Waiting for 9:20"}
            # Step 1: OI analysis at/after 9:20
            if force_analysis or (current_time >= analysis_time):
                if not getattr(self, 'breakout_levels_fixed', False):
                    oi_result = self.identify_high_oi_strikes()
                    if not oi_result:
                        logging.error("[STRATEGY][ERROR] OI analysis failed. Exiting strategy run.")
                        return {"success": False, "message": "OI analysis failed"}
                    self.breakout_levels_fixed = True
                # --- Original Trade Entry Logic ---
                if (self.highest_call_oi_symbol and self.call_breakout_level) or (self.highest_put_oi_symbol and self.put_breakout_level):
                    # Subscribe to valid symbols for monitoring only after OI analysis and after market open
                    symbols = []
                    if hasattr(self, 'highest_put_oi_symbol') and self.highest_put_oi_symbol:
                        symbols.append(self.highest_put_oi_symbol)
                    if hasattr(self, 'highest_call_oi_symbol') and self.highest_call_oi_symbol:
                        symbols.append(self.highest_call_oi_symbol)
                    if self.active_trade and 'symbol' in self.active_trade:
                        trade_symbol = self.active_trade['symbol']
                        if trade_symbol and trade_symbol not in symbols:
                            symbols.append(trade_symbol)
                    # Place OCO GTT orders for both legs (CE and PE) so that the first to trigger cancels the other
                    try:
                        qty = int(self.config.get('strategy', {}).get('quantity', 25))
                    except Exception:
                        qty = 25
                    try:
                        ce_sym = getattr(self, 'highest_call_oi_symbol', None)
                        pe_sym = getattr(self, 'highest_put_oi_symbol', None)
                        # Only place OCO if both symbols are available
                        if ce_sym and pe_sym:
                            logging.info(f"[OCO][PLACE] Placing OCO BRACKET orders for CE:{ce_sym} @ {self.call_breakout_level} and PE:{pe_sym} @ {self.put_breakout_level}")
                            # Place simulated bracket OCO orders (entry limit = breakout level)
                            self.place_oco_bracket_orders(ce_symbol=ce_sym, ce_entry=self.call_breakout_level,
                                                          pe_symbol=pe_sym, pe_entry=self.put_breakout_level,
                                                          qty=qty)
                        else:
                            logging.info("[OCO][SKIP] Not enough symbols to place OCO orders (need both CE and PE).")
                    except Exception:
                        logging.exception("[OCO][ERROR] Failed to place OCO GTT orders")
                    self.subscribe_to_valid_symbols(symbols)
                    breakout_detected = self.monitor_for_breakout()
                    # Only log trade entry in monitor_for_breakout/execute_trade
                else:
                    logging.warning("[STRATEGY][WARN] Trade entry skipped: missing symbol or breakout level.")
            # Position management is handled by continuous_position_monitor thread
            return {"success": True, "message": "Strategy executed successfully"}
        except Exception as e:
            logging.error(f"[STRATEGY][ERROR] Exception: {str(e)}")
            logging.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
            
    def unsubscribe_non_triggered_symbol(self, triggered_symbol, all_symbols):
        """Unsubscribe from the symbol(s) where trade was not triggered."""
        non_triggered = [s for s in all_symbols if s != triggered_symbol]
        # Assuming your data_socket has an unsubscribe method
        if hasattr(self.data_socket, 'unsubscribe'):
            for s in non_triggered:
                self.data_socket.unsubscribe(s)
                logging.info(f"[WS][UNSUBSCRIBE] Unsubscribed from {s} after trade triggered for {triggered_symbol}")
        else:
            logging.warning("[WS][UNSUBSCRIBE][WARN] WebSocket unsubscribe method not available. Manual unsubscribe required.")

    def retry_websocket_connection(self, symbols, max_retries=3, delay=5):
        """Retry websocket connection if it fails."""
        for attempt in range(1, max_retries + 1):
            try:
                self.data_socket = enhanced_start_market_data_websocket(
                    symbols=symbols,
                    callback_handler=self.ws_price_update
                )
                logging.info(f"[WS][RETRY] WebSocket connection established on attempt {attempt} for symbols: {symbols}")
                return True
            except Exception as e:
                logging.error(f"[WS][RETRY][ERROR] WebSocket connection attempt {attempt} failed: {str(e)}")
                time.sleep(delay)
        logging.error(f"[WS][RETRY][ERROR] All {max_retries} websocket connection attempts failed for symbols: {symbols}")
        return False

    def monitor_for_breakout(self):
        """Continuously monitor both CE and PE option premiums for breakout using websocket for real-time data"""
        try:
            # Enforce: Do not monitor before market open (09:15) or before 9:20
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            current_time = ist_now.time()
            market_open_time = datetime.strptime("09:15", "%H:%M").time()
            analysis_time = datetime.strptime("09:20", "%H:%M").time()
            if current_time < market_open_time:
                logging.warning(f"[MONITOR][BLOCKED] Attempted to start monitoring before market open (09:15). Current time: {current_time.strftime('%H:%M:%S')}")
                return False
            if current_time < analysis_time:
                logging.warning(f"[MONITOR][BLOCKED] Attempted to start monitoring before 9:20. Current time: {current_time.strftime('%H:%M:%S')}")
                return False
            # Ensure breakout levels have been fixed by OI analysis before monitoring
            if not getattr(self, 'breakout_levels_fixed', False):
                logging.warning(f"[MONITOR][BLOCKED] Attempted to start monitoring before breakout levels were fixed by OI analysis. Monitoring skipped.")
                return False
            logging.info(f"BREAKOUT MONITORING: Monitoring for breakout on CE and PE")
            # Only monitor the two highest OI strikes (one CE, one PE)
            symbols_to_monitor = []
            breakout_levels = {}
            if self.put_breakout_level and self.highest_put_oi_symbol:
                symbols_to_monitor.append(self.highest_put_oi_symbol)
                breakout_levels[self.get_canonical_symbol(self.highest_put_oi_symbol)] = self.put_breakout_level
            if self.call_breakout_level and self.highest_call_oi_symbol:
                # Avoid duplicate if both symbols are the same (shouldn't happen, but safe)
                if self.highest_call_oi_symbol != self.highest_put_oi_symbol:
                    symbols_to_monitor.append(self.highest_call_oi_symbol)
                breakout_levels[self.get_canonical_symbol(self.highest_call_oi_symbol)] = self.call_breakout_level
            # Remove duplicates just in case
            symbols_to_monitor = list(dict.fromkeys(symbols_to_monitor))
            # Enforce only one CE and one PE symbol
            ce = next((s for s in symbols_to_monitor if "CE" in s), None)
            pe = next((s for s in symbols_to_monitor if "PE" in s), None)
            symbols_to_monitor = [s for s in (ce, pe) if s]
            logging.info(f"[BREAKOUT][MONITOR] Final symbols to monitor: {symbols_to_monitor}")
            if not symbols_to_monitor:
                logging.info("No valid option symbols to monitor for breakout.")
                return
            logging.info(f"[BREAKOUT][WS] Subscribing to only the two highest OI option symbols for breakout monitoring: {symbols_to_monitor}")
            if not self.retry_websocket_connection(symbols_to_monitor):
                logging.error("[BREAKOUT][WS][ERROR] Could not establish websocket connection after retries. Aborting breakout monitoring.")
                return            
            logging.info(f"[BREAKOUT][WS] WebSocket subscription started for symbols: {symbols_to_monitor}")
            canonical_symbols = [self.get_canonical_symbol(s) for s in symbols_to_monitor]
            # Emit concise per-leg monitoring lines (match example output)
            try:
                for s in symbols_to_monitor:
                    cb = breakout_levels.get(self.get_canonical_symbol(s))
                    if 'CE' in s:
                        logging.info(f"Monitoring CE {s} for breakout above {cb} (buffer: 0.1)")
                    elif 'PE' in s:
                        logging.info(f"Monitoring PE {s} for breakout above {cb} (buffer: 0.1)")
            except Exception:
                logging.debug("[BREAKOUT][MONITOR] Could not emit per-leg monitoring lines")
            
            while True:
                # Forcibly exit if time is before market open or 9:20
                ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                current_time = ist_now.time()
                market_open_time = datetime.strptime("09:15", "%H:%M").time()
                analysis_time = datetime.strptime("09:20", "%H:%M").time()
                if current_time < market_open_time or current_time < analysis_time:
                    logging.error(f"[MONITOR][FORCE-EXIT] Monitoring loop forcibly stopped: current time {current_time.strftime('%H:%M:%S')} is before allowed window.")
                    break
                for symbol, canonical_symbol in zip(symbols_to_monitor, canonical_symbols):
                    df_row = self.ltp_df[self.ltp_df.symbol == canonical_symbol]
                    price = float(df_row.iloc[0]['ltp']) if not df_row.empty else None
                    breakout_level = breakout_levels[canonical_symbol]
                    option_type = "unknown"
                    if "CE" in canonical_symbol:
                        option_type = "CE"
                    elif "PE" in canonical_symbol:
                        option_type = "PE"
                    # ...existing code...
                    if price is not None:
                        if price >= breakout_level:
                            # ...existing code...
                            trade_result = self.execute_trade(symbol=canonical_symbol, side='BUY', entry_price=price)
                            if trade_result:
                                logging.info(f"[TRADE][ENTRY] Trade entry successful for {canonical_symbol} at price {price}.")
                            else:
                                logging.error(f"[TRADE][ENTRY][ERROR] Trade entry failed for {canonical_symbol} at price {price}.")
                            self.unsubscribe_non_triggered_symbol(triggered_symbol=canonical_symbol, all_symbols=symbols_to_monitor)
                            return True
                time.sleep(2)
            return False
        except Exception as e:
            logging.error(f"[BREAKOUT][ERROR] Error monitoring for breakout: {str(e)}")
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
            logging.error(f"[TRADE][UPDATE][ERROR] No tick DataFrame LTP available for active trade contract. Skipping trade update.")
            return
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
        max_up = self.active_trade.get('max_up', None)
        max_up_pct = self.active_trade.get('max_up_pct', None)
        max_down = self.active_trade.get('max_down', None)
        max_down_pct = self.active_trade.get('max_down_pct', None)
        trailing_sl = stoploss
        if pnl > 0 and (max_up is None or pnl > max_up):
            self.active_trade['max_up'] = pnl
            self.active_trade['max_up_pct'] = pnl_pct
        if pnl < 0 and (max_down is None or pnl < max_down):
            self.active_trade['max_down'] = pnl
            self.active_trade['max_down_pct'] = pnl_pct
        # Apply configurable trailing stop policy (separate helper)
        try:
            self._apply_trailing_stop(current_price, entry_price)
        except Exception:
            logging.debug("[TRADE][TRAILING] Failed to apply trailing stop policy")
        max_down_val = float(self.active_trade.get('max_down', 0) or 0)
        max_down_pct_val = float(self.active_trade.get('max_down_pct', 0) or 0)
        max_up_val = float(self.active_trade.get('max_up', 0) or 0)
        max_up_pct_val = float(self.active_trade.get('max_up_pct', 0) or 0)
        # Rate-limit PAPER STATUS logs to avoid flooding; log at most once per configured interval
        try:
            now = time.time()
            if (now - getattr(self, '_last_paper_status_time', 0)) >= getattr(self, '_paper_status_min_seconds', 5):
                # Format numbers to two decimals and percentages with sign
                ltp_str = f"{float(current_price):.2f}"
                entry_str = f"{float(entry_price):.2f}"
                sl_str = f"{float(self.active_trade.get('stoploss', 0)):.2f}"
                tgt_str = f"{float(target):.2f}"
                # Prepare PnL and MaxUp/MaxDn formatting
                pnl_str = f"{pnl:.2f}"
                max_up_val_f = float(max_up or 0.0)
                max_down_val_f = float(max_down or 0.0)
                max_up_pct_f = float(max_up_pct or 0.0)
                max_down_pct_f = float(max_down_pct or 0.0)
                max_up_str = f"{max_up_val_f:.2f} ({max_up_pct_f:.2f}%)"
                max_down_str = f"{max_down_val_f:.2f} ({max_down_pct_f:.2f}%)"
                try:
                    # Emit full PAPER STATUS including MaxUp/MaxDn with percentages
                    logging.info(
                        f"[PAPER STATUS] {symbol} | LTP: {float(current_price):.2f} | Entry: {float(entry_price):.2f} | SL: {float(self.active_trade.get('stoploss', 0)):.2f} | Target: {float(target):.2f} | PnL: {pnl:.2f} | MaxUp: {max_up_val_f:.2f} ({max_up_pct_f:.2f}%) | MaxDn: {max_down_val_f:.2f} ({max_down_pct_f:.2f}%)"
                    )
                    # Update last paper status time to enforce rate-limiting
                    try:
                        self._last_paper_status_time = now
                    except Exception:
                        pass
                except Exception:
                    # Fallback shorter form if formatting fails
                    logging.info(f"[PAPER STATUS] {symbol} | LTP: {current_price} | Entry: {entry_price} | PnL: {pnl:.2f}")
        except Exception:
            logging.exception("[PAPER STATUS][ERROR] Exception while preparing PAPER STATUS")

    def cleanup(self):
        """Cleanup resources before exiting"""
        try:
            logging.info("[CLEANUP][START] Cleaning up strategy resources")
            # Force exit any open trade before shutdown
            if self.active_trade and not self.active_trade.get('exit_reason'):
                logging.info("[CLEANUP][FORCE_EXIT] Forcing exit of open trade during cleanup to ensure exit is logged.")
                self.process_exit(exit_reason="FORCED_CLEANUP")
            # Save any pending data
            self.save_trade_history()
            # Close any connections
            if self.fyers:
                # Close any active websocket connections, etc.
                pass
            logging.info("[CLEANUP][COMPLETE] Cleanup completed")
        except Exception as e:
            logging.error(f"[CLEANUP][ERROR] Error during cleanup: {str(e)}")
            return False
        return True

    def _final_shutdown(self):
        """Perform aggressive cleanup and exit the process to ensure no background threads or websockets
        remain active after a trade exit. This does best-effort attempts to disable reconnects,
        stop background threads, close websockets, and then terminate the process.
        """
        try:
            logging.info("[SHUTDOWN] Finalizing and exiting process after trade exit")
            # Prevent further monitoring loops from running
            try:
                self.market_closed = True
            except Exception:
                pass

            # Disable websocket auto-reconnect if available
            try:
                if hasattr(self, 'data_socket') and self.data_socket:
                    try:
                        if hasattr(self.data_socket, 'reconnect'):
                            try:
                                setattr(self.data_socket, 'reconnect', False)
                            except Exception:
                                pass
                        if hasattr(self.data_socket, 'auto_reconnect'):
                            try:
                                setattr(self.data_socket, 'auto_reconnect', False)
                            except Exception:
                                pass
                        # If connection_status dict exists, mark as disconnected
                        if hasattr(self.data_socket, 'connection_status') and isinstance(self.data_socket.connection_status, dict):
                            try:
                                self.data_socket.connection_status['connected'] = False
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            # Stop tick consumer and price monitoring
            try:
                self.stop_tick_consumer()
            except Exception:
                pass
            try:
                self.stop_price_monitoring()
            except Exception:
                pass

            # Attempt to join common background threads with short timeouts
            try:
                if hasattr(self, '_paper_status_thread') and self._paper_status_thread:
                    try:
                        self._paper_status_thread.join(timeout=2)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if hasattr(self, '_exit_monitor_thread') and self._exit_monitor_thread:
                    try:
                        self._exit_monitor_thread.join(timeout=2)
                    except Exception:
                        pass
            except Exception:
                pass

            # Try to join bracket/gtt monitor threads
            try:
                if hasattr(self, '_bracket_monitor_threads') and isinstance(self._bracket_monitor_threads, dict):
                    for k, t in list(self._bracket_monitor_threads.items()):
                        try:
                            if t and t.is_alive():
                                t.join(timeout=1)
                        except Exception:
                            pass
                    self._bracket_monitor_threads = {}
            except Exception:
                pass
            try:
                if hasattr(self, '_gtt_monitor_threads') and isinstance(self._gtt_monitor_threads, dict):
                    for k, t in list(self._gtt_monitor_threads.items()):
                        try:
                            if t and t.is_alive():
                                t.join(timeout=1)
                        except Exception:
                            pass
                    self._gtt_monitor_threads = {}
            except Exception:
                pass

            # Ask order_manager to shutdown if it exposes a method
            try:
                if hasattr(self, 'order_manager') and self.order_manager:
                    if hasattr(self.order_manager, 'shutdown'):
                        try:
                            self.order_manager.shutdown()
                        except Exception:
                            pass
            except Exception:
                pass

            # Give logging a brief moment to flush
            try:
                time.sleep(0.2)
            except Exception:
                pass

            logging.info("[SHUTDOWN] Exiting process now (exit_after_trade configured)")
            try:
                os._exit(0)
            except Exception:
                try:
                    sys.exit(0)
                except Exception:
                    pass
        except Exception:
            logging.exception("[SHUTDOWN] Exception during final shutdown")
            try:
                os._exit(0)
            except Exception:
                pass

    def get_ist_datetime(self):
        """Return current datetime in IST timezone"""
        return datetime.now(pytz.timezone('Asia/Kolkata'))

    def get_active_trade_ltp(self):
        """
        Return the current LTP for the active trade's contract, preferring the live_prices map
        then falling back to the internal ltp_df DataFrame. Returns a float or None.
        """
        try:
            if not self.active_trade:
                return None
            symbol = self.active_trade.get('symbol')
            if not symbol:
                return None
            # Prefer live_prices mapping
            price = self.live_prices.get(symbol)
            if price is not None:
                try:
                    return float(price)
                except Exception:
                    return None
            # Fallback to ltp_df lookup
            df_row = self.ltp_df[self.ltp_df.symbol == symbol]
            if not df_row.empty:
                try:
                    return float(df_row.iloc[-1]['ltp'])
                except Exception:
                    return None
            return None
        except Exception:
            logging.exception("[GET_ACTIVE_LTP] Exception while getting active trade LTP")
            return None
    def _apply_trailing_stop(self, current_price, entry_price):
        """
        Apply trailing stop policy based on configured percentage bands.
        The rules are a list of (trigger_profit_pct, lock_sl_pct) pairs. We select the
        highest trigger satisfied and lock SL to entry_price * (1 + lock_sl_pct/100).
        SL is only moved up (never reduced).
        """
        try:
            if not self.active_trade or not entry_price or current_price is None:
                return
            cfg = self.config.get('strategy', {}) if self.config else {}
            use_trailing = bool(cfg.get('use_trailing_stop', True))
            if not use_trailing:
                return
            # Default trailing rules matching user's requested scheme (percent values):
            # trigger_pct -> lock SL to this pct above entry
            rules = [
                (13.0, 3.0),
                (20.0, 10.0),
                (25.0, 15.0),
                (30.0, 25.0),
                (35.0, 30.0),
            ]
            profit_pct = ((current_price - entry_price) / entry_price) * 100.0
            # find highest rule satisfied
            applicable = None
            for trig, lock in rules:
                if profit_pct >= trig:
                    applicable = (trig, lock)
            if not applicable:
                return
            lock_pct = applicable[1]
            new_sl = round(entry_price * (1 + lock_pct / 100.0), 2)
            cur_sl = float(self.active_trade.get('stoploss', 0) or 0)
            if new_sl > cur_sl:
                # Safety/config guards
                trail_cfg = cfg.get('trailing', {}) if isinstance(cfg, dict) else {}
                prefer_amend = bool(trail_cfg.get('prefer_amend', True))
                min_move_pct = float(trail_cfg.get('min_move_pct', 0.5))
                cooldown_secs = int(trail_cfg.get('cooldown_secs', 30))
                max_mods = int(trail_cfg.get('max_modifications', 20))

                nowt = time.time()
                last_mod = float(self.active_trade.get('last_trailing_mod_time', 0) or 0)
                mod_count = int(self.active_trade.get('trailing_mod_count', 0) or 0)
                move_pct = ((new_sl - cur_sl) / entry_price) * 100.0 if entry_price else 0.0

                if move_pct < min_move_pct:
                    logging.debug(f"[TRAILING][SKIP] Proposed SL move {move_pct:.3f}% < min_move_pct {min_move_pct}%; skipping")
                elif (nowt - last_mod) < cooldown_secs:
                    logging.debug(f"[TRAILING][SKIP] Cooldown active: {(nowt-last_mod):.1f}s elapsed < {cooldown_secs}s; skipping")
                elif mod_count >= max_mods:
                    logging.warning(f"[TRAILING][SKIP] Reached max modifications ({mod_count}) for this trade; skipping further trailing")
                else:
                    bracket_id = self.active_trade.get('bracket_order_id') or self.active_trade.get('stop_loss_order_id')
                    method_used = 'IN-MEMORY'
                    success = False
                    new_bracket_id = None
                    try:
                        if bracket_id:
                            # Try amend first if preferred
                            if prefer_amend:
                                try:
                                    resp = self.order_manager.modify_order_stoploss(bracket_id, new_sl)
                                    if resp and resp.get('s') == 'ok':
                                        method_used = 'AMEND'
                                        success = True
                                    else:
                                        logging.debug(f"[TRAILING][AMEND] Amend response: {resp}")
                                except Exception:
                                    logging.exception("[TRAILING][AMEND] Exception during amend attempt")
                            # Fallback to cancel+replace if amend not successful
                            if not success:
                                try:
                                    resp2 = self.order_manager.cancel_and_replace_stop(bracket_id, new_sl)
                                    # resp2 format mimics place_bracket_order
                                    if resp2 and (resp2.get('order') or resp2.get('order_id') or resp2.get('status') == 'pending'):
                                        method_used = 'CANCEL+REPLACE'
                                        success = True
                                        new_bracket_id = resp2.get('order', {}).get('order_id') or resp2.get('order_id')
                                except Exception:
                                    logging.exception("[TRAILING][REPLACE] Exception during cancel+replace attempt")
                        else:
                            # No associated bracket id — fall back to in-memory update
                            logging.warning("[TRAILING][WARN] No bracket/stop order id available, updating in-memory only")
                            success = True
                            method_used = 'IN-MEMORY'
                    except Exception:
                        logging.exception("[TRAILING][ERROR] Error while attempting to modify stop order")

                    if success:
                        old_sl = cur_sl
                        # If cancel+replace returned a new id, update active_trade
                        if new_bracket_id:
                            self.active_trade['bracket_order_id'] = new_bracket_id
                        self.active_trade['stoploss'] = new_sl
                        self.active_trade['trailing_locked_to_pct'] = lock_pct
                        self.active_trade['last_trailing_mod_time'] = nowt
                        self.active_trade['trailing_mod_count'] = mod_count + 1
                        logging.info(f"[TRAILING][UPDATE] ({method_used}) Profit {profit_pct:.2f}% >= {applicable[0]}% -> moving SL from {old_sl} to {new_sl} (locked +{lock_pct}%)")
                    else:
                        logging.error("[TRAILING][ERROR] Failed to update stoploss via amend or cancel+replace; in-memory SL not changed")
        except Exception:
            logging.exception("[TRAILING][ERROR] Exception in _apply_trailing_stop")
    def get_canonical_symbol(self, symbol):
        """
        Convert any incoming symbol (raw or exchange-formatted) to the canonical format used for logging and processing.
        Ensures every unique contract (expiry, strike, type) gets a unique symbol.
        Logs original and converted symbol for diagnostics.
        """
        import re
        import logging
        orig_symbol = symbol
        # Initialize rate-limiting state for symbol map logging
        if not hasattr(self, '_symbol_map_last'):
            # map of message -> last log time
            self._symbol_map_last = {}
            # map canonical target -> last original seen (to only log when mapping changes)
            self._symbol_map_target_last = {}
            # seconds to suppress repeated identical symbol map messages (increase cooldown to reduce noise)
            self._symbol_map_cooldown = 60

        # If already in Fyers format and strike is 5 digits, return as is
        if symbol.startswith('NSE:') and (symbol.endswith('CE') or symbol.endswith('PE')):
            match = re.match(r'NSE:[A-Z]+\d{6}(\d{4,5})(CE|PE)$', symbol)
            if match:
                # Rate-limit repeated identical SYMBOL MAP logs
                now = time.time()
                key = f"{orig_symbol} -> {symbol}"
                last = self._symbol_map_last.get(key, 0)
                if now - last >= self._symbol_map_cooldown:
                    # If an active trade exists, suppress further SYMBOL MAP INFO logs to
                    # avoid masking trade monitoring output; still update rate-limit state.
                    suppress = False
                    try:
                        # Also suppress if there is a pending GTT for this symbol
                        if hasattr(self, 'order_manager') and self.order_manager:
                            pending = [o for o in self.order_manager.get_orders_by_symbol(symbol) if o.get('status_code') == 3]
                            if pending:
                                suppress = True
                    except Exception:
                        suppress = False
                    if getattr(self, 'active_trade', None) or suppress:
                        self._symbol_map_last[key] = now
                    else:
                        logging.info(f"[SYMBOL MAP] {orig_symbol} -> {symbol}")
                        self._symbol_map_last[key] = now
                return symbol
            else:
                return symbol
        # Try to use the formatter utility for all other cases.
        # Perform a lazy, cached import so that we avoid logging the same ImportError repeatedly.
        if not hasattr(self, '_convert_option_symbol_format'):
            try:
                from src.symbol_formatter import convert_option_symbol_format as _conv
                self._convert_option_symbol_format = _conv
                self._symbol_formatter_import_failed = False
            except Exception as e:
                # Fallback to identity converter and warn once so logs are not flooded.
                self._convert_option_symbol_format = lambda s: s
                self._symbol_formatter_import_failed = True
                logging.warning(f"[SYMBOL MAP] symbol_formatter import failed; falling back to identity converter. Error: {e}")

        try:
            converted = self._convert_option_symbol_format(symbol)
            # Normalize and ensure canonical suffix/prefix (handle feeds that use 'C'/'P')
            conv = converted.strip() if isinstance(converted, str) else str(converted)
            if not conv.startswith('NSE:'):
                conv = 'NSE:' + conv
            # Normalize single-letter option type suffixes
            if conv.endswith('C') and not conv.endswith('CE'):
                conv = conv + 'E'
            if conv.endswith('P') and not conv.endswith('PE'):
                conv = conv + 'E'
            # Rate-limit SYMBOL MAP logs: only log when the canonical mapping for this target
            # changes (orig symbol differs) or when the cooldown has elapsed. This avoids
            # flooding when many raw variants map to the same canonical symbol repeatedly.
            now = time.time()
            key = conv
            last_orig = self._symbol_map_target_last.get(key)
            last_time = self._symbol_map_last.get(key, 0)
            # Only emit INFO when the mapping actually changed (orig -> conv differs). Otherwise, keep it DEBUG
            if last_orig != orig_symbol:
                # Suppress SYMBOL MAP INFO lines once a trade is active or while a pending
                # GTT exists for this symbol so PAPER STATUS and bracket logs remain visible.
                suppress = False
                try:
                    if hasattr(self, 'order_manager') and self.order_manager:
                        pending = [o for o in self.order_manager.get_orders_by_symbol(conv) if o.get('status_code') == 3]
                        if pending:
                            suppress = True
                except Exception:
                    suppress = False
                if getattr(self, 'active_trade', None) or suppress:
                    # Update internal state but do not emit INFO-level symbol mapping logs
                    self._symbol_map_target_last[key] = orig_symbol
                    self._symbol_map_last[key] = now
                else:
                    logging.info(f"[SYMBOL MAP] {orig_symbol} -> {conv}")
                    self._symbol_map_target_last[key] = orig_symbol
                    self._symbol_map_last[key] = now
            elif (now - last_time) >= self._symbol_map_cooldown:
                logging.debug(f"[SYMBOL MAP - REFRESH] {orig_symbol} -> {conv} (refreshed after cooldown)")
                self._symbol_map_last[key] = now
            return conv
        except Exception as e:
            logging.error(f"[SYMBOL MAP] Error converting {orig_symbol}: {e}")
            logging.error(traceback.format_exc())
            return symbol

    def stop_price_monitoring(self, symbol=None):
        """Stop all price monitoring and unsubscribe from all symbols after trade exit."""
        if hasattr(self, 'data_socket') and self.data_socket:
            # First try a bulk unsubscribe if available
            try:
                if hasattr(self.data_socket, 'unsubscribe_all') and symbol is None:
                    try:
                        self.data_socket.unsubscribe_all()
                        logging.info("Unsubscribed from all symbols after trade exit.")
                    except Exception:
                        logging.debug("unsubscribe_all() call failed")
                else:
                    # If unsubscribe_all not available, try per-symbol unsubscribe
                    if symbol and hasattr(self.data_socket, 'unsubscribe'):
                        try:
                            # improved websocket expects unsubscribe(symbols=[...]) or our wrapper handles single symbol
                            self.data_socket.unsubscribe(symbol)
                            logging.info(f"Unsubscribed from symbol: {symbol}")
                        except Exception:
                            logging.debug(f"Failed to unsubscribe single symbol: {symbol}")
                    else:
                        # Attempt to unsubscribe everything we know the client subscribed to (market_data index or stored list)
                        try:
                            if hasattr(self.data_socket, 'market_data'):
                                symbols = list(self.data_socket.market_data.index)
                                # Try native unsubscribe first
                                for s in symbols:
                                    try:
                                        if hasattr(self.data_socket, 'unsubscribe'):
                                            self.data_socket.unsubscribe(s)
                                    except Exception:
                                        continue
                                # If client expects a JSON unsubscribe message (some Fyers WS clients), send SUB_MDATA
                                try:
                                    import json
                                    if hasattr(self.data_socket, 'send') and symbols:
                                        msg = json.dumps({"T": "SUB_MDATA", "S": symbols, "SUB_T": -1})
                                        try:
                                            self.data_socket.send(msg)
                                            logging.info("Sent manual SUB_MDATA unsubscribe message for market symbols")
                                        except Exception:
                                            logging.debug("Failed to send SUB_MDATA unsubscribe message")
                                except Exception:
                                    logging.debug("Could not send SUB_MDATA unsubscribe message")
                                logging.info("Attempted per-symbol unsubscribe for all market_data symbols")
                        except Exception:
                            logging.debug("Could not iterate market_data for unsubscribe")
            except Exception:
                logging.debug("Error during unsubscribe attempts")

            # Try to close the websocket/data socket using common hooks on the client
            try:
                # Before closing, some clients require an explicit unsubscribe for order/action feeds
                try:
                    import json
                    # Common action feeds to unsubscribe from (orders/trades/positions etc.)
                    action_list = ['orders', 'trades', 'positions', 'edis', 'pricealerts', 'login']
                    if hasattr(self.data_socket, 'send'):
                        try:
                            msg_actions = json.dumps({"T": "SUB_ORD", "SLIST": action_list, "SUB_T": -1})
                            self.data_socket.send(msg_actions)
                            logging.info("Sent SUB_ORD unsubscribe for action feeds")
                        except Exception:
                            logging.debug("Failed to send SUB_ORD unsubscribe message")
                except Exception:
                    logging.debug("Could not prepare SUB_ORD unsubscribe message")

                if hasattr(self.data_socket, 'close_connection'):
                    try:
                        self.data_socket.close_connection()
                        logging.info("Closed data socket using close_connection() after trade exit.")
                    except Exception:
                        logging.debug("close_connection() failed, will try other close methods")
                if hasattr(self.data_socket, 'terminate'):
                    try:
                        self.data_socket.terminate()
                        logging.info("Terminated data socket after trade exit.")
                    except Exception:
                        logging.debug("terminate() failed on data_socket")
                elif hasattr(self.data_socket, 'close'):
                    try:
                        self.data_socket.close()
                        logging.info("Closed data socket after trade exit.")
                    except Exception as e:
                        logging.error(f"Error closing data socket: {e}")
            except Exception:
                logging.debug("Error closing data socket (general) after trade exit")
        # Clear client reference and any cached live prices to avoid using stale data
        try:
            if hasattr(self, 'live_prices') and isinstance(self.live_prices, dict):
                self.live_prices.clear()
        except Exception:
            logging.debug("Could not clear live_prices cache")
        self.data_socket = None
    def _close_active_trade(self, exit_reason="manual", exit_price=None):
        """
        Close the currently active trade: cancel bracket order, record exit, save trade history and log PAPER EXIT.
        """
        try:
            if not self.active_trade:
                logging.warning("[EXIT] No active trade to close")
                return False
            symbol = self.active_trade.get('symbol')
            qty = self.active_trade.get('quantity')
            entry = self.active_trade.get('entry_price')
            exit_p = exit_price if exit_price is not None else self.get_active_trade_ltp()
            # Cancel bracket order if present
            bracket_id = self.active_trade.get('bracket_order_id')
            if bracket_id:
                try:
                    self.order_manager.cancel_order(bracket_id, reason=f"Exit by {exit_reason}")
                except Exception:
                    logging.exception("[EXIT] Failed to cancel bracket order")
            # Mark trade history
            pnl = (exit_p - entry) * qty if entry and exit_p else 0
            # Record extended trade metadata to match Nifty-style logs
            try:
                filled_dt = datetime.now(pytz.timezone('Asia/Kolkata'))
            except Exception:
                filled_dt = datetime.now()

            # Compose a simulated order id for paper trades if none provided
            order_id = bracket_id or self.active_trade.get('order_id') or f"SIM-{symbol}-{entry}-{qty}"

            self.trade_history.append({
                'Entry DateTime': self.active_trade.get('entry_time'),
                'Symbol': symbol,
                'Direction': self.active_trade.get('side'),
                'Entry Price': entry,
                'Exit DateTime': filled_dt,
                'Exit Price': exit_p,
                'P&L': pnl,
                'Quantity': qty,
                'Exit Reason': exit_reason,
                # Additional metadata
                'Order ID': order_id,
                'Group ID': self.active_trade.get('group_id'),
                'Bracket Order ID': bracket_id,
                'SL': self.active_trade.get('stoploss'),
                # Trailing SL (price) snapshot and legacy header name for compatibility
                'trailing_stoploss': self.active_trade.get('trailing_stoploss', ''),
                'Trailing SL': self.active_trade.get('trailing_stoploss', ''),
                'Target': self.active_trade.get('target'),
                'Max Up': self.active_trade.get('max_up', 0),
                'Max Down': self.active_trade.get('max_down', 0),
                'Max Up %': self.active_trade.get('max_up_pct', 0),
                'Max Down %': self.active_trade.get('max_down_pct', 0),
                'Filled Time': filled_dt.strftime('%H:%M:%S') if hasattr(filled_dt, 'strftime') else str(filled_dt),
                # VIX snapshot at exit time (if available)
                'VIX': self.live_prices.get('NSE:VIX') or self.live_prices.get('VIX') or ''
            })
            # Format numbers with thousands separators for clearer operator logs
            try:
                exit_str = f"{float(exit_p):,.2f}"
            except Exception:
                exit_str = str(exit_p)
            try:
                pnl_str = f"{float(pnl):,.2f}"
            except Exception:
                pnl_str = str(pnl)

            # Normalize exit headline to match operator-friendly phrasing
            er = str(exit_reason or '').upper()
            if 'TARGET' in er or 'TARGET HIT' in er:
                logging.info(f"[PAPER EXIT] Target hit at {exit_str}")
            elif 'STOP' in er or 'STOPLOSS' in er:
                logging.info(f"[PAPER EXIT] Stoploss hit at {exit_str}")
            else:
                logging.info(f"[PAPER EXIT] {exit_reason} at {exit_str}")

            # Emit a concise PAPER TRADING SUMMARY block for operator visibility
            try:
                max_up = float(self.active_trade.get('max_up', 0) or 0)
                max_down = float(self.active_trade.get('max_down', 0) or 0)
                max_up_pct = float(self.active_trade.get('max_up_pct', 0) or 0)
                max_down_pct = float(self.active_trade.get('max_down_pct', 0) or 0)
            except Exception:
                max_up = max_down = max_up_pct = max_down_pct = 0

            # Compose a simulated order id for paper trades if none provided
            order_id = bracket_id or self.active_trade.get('order_id') or f"SIM-{symbol}-{entry}-{qty}"

            logging.info("")
            logging.info("==============================================================================")
            logging.info("PAPER TRADING SUMMARY - Bracket Order OCO Test")
            logging.info("==============================================================================")
            logging.info("")
            logging.info(f"Order ID: {order_id}")
            logging.info(f"  Symbol: {symbol}")
            logging.info(f"  Status: FILLED")
            logging.info(f"  Entry Limit: {format(entry, ',.2f')}")
            # Display SL/Target with two decimals for readability
            try:
                sl_display = format(float(self.active_trade.get('stoploss', 0)), ',.2f') if self.active_trade.get('stoploss', '') != '' else ''
            except Exception:
                sl_display = str(self.active_trade.get('stoploss', ''))
            try:
                tgt_display = format(float(self.active_trade.get('target', 0)), ',.2f') if self.active_trade.get('target', '') != '' else ''
            except Exception:
                tgt_display = str(self.active_trade.get('target', ''))
            logging.info(f"  SL: {sl_display} | Target: {tgt_display}")
            # Ensure qty printed as integer when possible
            try:
                qty_display = int(qty)
            except Exception:
                qty_display = qty
            logging.info(f"  Qty: {qty_display}")
            try:
                filled_time = self.active_trade.get('entry_time').strftime('%H:%M:%S') if self.active_trade.get('entry_time') else ''
            except Exception:
                filled_time = ''
            logging.info(f"  Filled at: {filled_time} @ {format(entry, ',.2f')}")
            logging.info(f"  Max Up: {format(max_up, ',.2f')} ({max_up_pct:.2f}%) | Max Down: {format(max_down, ',.2f')} ({max_down_pct:.2f}%)")
            logging.info("")
            # Log OCO companion info if available from order manager
            try:
                # Attempt to find other leg orders in order_manager if API present
                other_orders = []
                if hasattr(self, 'order_manager') and self.order_manager:
                    try:
                        # get_orders_by_group may not exist; gracefully ignore
                        if hasattr(self.order_manager, 'get_orders_by_group') and self.active_trade.get('group_id'):
                            other_orders = self.order_manager.get_orders_by_group(self.active_trade.get('group_id'))
                    except Exception:
                        other_orders = []
                companion_filled = None
                companion_cancelled = []
                if other_orders:
                    for o in other_orders:
                        s = str(o.get('status', '') or o.get('status_text', '') or o.get('status_code', ''))
                        sym = o.get('symbol')
                        q = o.get('quantity')
                        try:
                            q = int(q)
                        except Exception:
                            pass
                        # Consider typical simulated statuses
                        s_up = s.upper()
                        if 'FILL' in s_up or 'FILLED' in s_up:
                            companion_filled = sym
                            logging.info(f"Order ID: {o.get('order_id', '')}")
                            logging.info(f"  Symbol: {sym}")
                            logging.info(f"  Status: FILLED")
                            logging.info(f"  Entry Limit: {format(o.get('entry_price', o.get('price', '')), ',.2f') if o.get('entry_price', None) else ''}")
                            logging.info(f"  SL: {format(o.get('stoploss', ''), ',.2f')} | Target: {format(o.get('target', ''), ',.2f')}")
                            logging.info(f"  Qty: {q}")
                        else:
                            # Treat it as cancelled if status contains CANCEL
                            if 'CANCEL' in s_up or 'CANCELLED' in s_up:
                                companion_cancelled.append(sym)
                                logging.info(f"Order ID: {o.get('order_id', '')}")
                                logging.info(f"  Symbol: {sym}")
                                logging.info(f"  Status: CANCELLED")
                                logging.info(f"  Entry Limit: {format(o.get('entry_price', o.get('price', '')), ',.2f') if o.get('entry_price', None) else ''}")
                                logging.info(f"  SL: {format(o.get('stoploss', ''), ',.2f')} | Target: {format(o.get('target', ''), ',.2f')}")
                                logging.info(f"  Qty: {q}")
                            else:
                                logging.info(f"  Companion order: {sym} | Status: {s} | Qty: {q}")
                else:
                    logging.info(f"  (OCO companion order details not available)")
            except Exception:
                logging.debug("Could not retrieve OCO companion order details")

            logging.info("")
            logging.info("==============================================================================")
            logging.info("OCO Test Result:")
            logging.info(f"[OK] OCO LOGIC WORKING CORRECTLY!")
            # Report specifically which leg filled and which cancelled when possible
            try:
                filled_leg = symbol
                cancelled_leg_list = companion_cancelled if companion_cancelled else ['(companion leg)']
                logging.info(f"   One order filled: {filled_leg}")
                logging.info(f"   Other order cancelled: {', '.join(cancelled_leg_list)}")
            except Exception:
                logging.info(f"   One order filled: {symbol}")
                logging.info(f"   Other order cancelled: (companion leg)")
            logging.info("==============================================================================")

            # Update persistent balance and performance summary
            try:
                previous = float(self.current_balance or 0)
                new_bal = previous + float(pnl)
                self.current_balance = new_bal
                # Record balance after trade on the most recent trade entry if present
                try:
                    if len(self.trade_history) > 0:
                        self.trade_history[-1]['Balance After Trade'] = float(new_bal)
                except Exception:
                    logging.debug("Could not set Balance After Trade on trade history entry")
                # Persist balance
                try:
                    self.save_balance()
                except Exception:
                    logging.debug("Failed to persist balance after trade exit")
                # Log balance line
                change = new_bal - previous
                sign = '+' if change >= 0 else ''
                logging.info(f"[BALANCE] Updated Balance: {new_bal:,.2f} (Change: {sign}{change:,.2f})")
                # Compute simple performance metrics from trade_history
                try:
                    wins = sum(1 for t in self.trade_history if float(t.get('P&L', 0)) > 0)
                    losses = sum(1 for t in self.trade_history if float(t.get('P&L', 0)) <= 0)
                    total = wins + losses
                    win_rate = (wins / total * 100.0) if total > 0 else 0.0
                    net_pnl = sum(float(t.get('P&L', 0)) for t in self.trade_history)
                    sign_net = '+' if net_pnl >= 0 else ''
                    logging.info(f"[PERF] Performance: {wins}W/{losses}L | Win Rate: {win_rate:.1f}% | Net P&L: {sign_net}{net_pnl:,.2f}")
                except Exception:
                    logging.debug("Could not compute performance summary")
            except Exception:
                logging.exception("Error updating balance/performance after exit")

            # Persist trade history
            try:
                self.save_trade_history()
            except Exception:
                logging.exception("[EXIT] Failed to save trade history")

            # Stop any background tick consumer and unsubscribe/close data socket so we stop receiving ticks
            try:
                try:
                    self.stop_tick_consumer()
                except Exception:
                    logging.debug("Could not cleanly stop tick consumer thread")
                try:
                    # Unsubscribe from all price feeds and close websocket
                    self.stop_price_monitoring()
                except Exception:
                    logging.debug("Could not stop price monitoring / close data socket")
            except Exception:
                logging.debug("Error while attempting to stop background price threads/sockets")

            # Emit target/exit completion line for operator visibility
            try:
                # Trade number is the length of trade_history
                trade_no = len(self.trade_history)
                if 'TARGET' in str(exit_reason).upper() or 'TARGET' in str(exit_reason).upper():
                    result_tag = 'PROFIT [WIN]' if pnl > 0 else 'LOSS'
                    logging.info(f"[TARGET] Trade #{trade_no} completed: {symbol} | Net P&L: {pnl_str} ({'PROFIT [WIN]' if pnl>0 else 'LOSS'})")
                else:
                    # generic exit message
                    logging.info(f"[EXIT] Trade #{trade_no} closed: {symbol} | Net P&L: {pnl_str} | Reason: {exit_reason}")
            except Exception:
                logging.debug("Could not emit TARGET completion line")

            # Log saved filenames and compact trade-line for quick scanning
            try:
                # CSV path is constant in save_trade_history
                csv_path = 'logs/trade_history.csv'
                today = datetime.now().strftime('%Y%m%d')
                excel_path = f'logs/trade_history_{today}.xlsx'
                logging.info(f"Trade data saved to Excel: {excel_path}")
                logging.info(f"Trade data saved to CSV: {csv_path}")
                # One-line trade log
                ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                side = self.trade_history[-1].get('Direction', '')
                qty = self.trade_history[-1].get('Quantity', '')
                logging.info(f"Trade logged: {ts},{symbol},{side},{exit_p},{qty},{exit_reason}")
                logging.info(f"[EXCEL] Trade logged to Excel: {symbol} | Entry: {entry} | Exit: {exit_p} | P&L: {pnl_str}")
            except Exception:
                logging.debug("Could not emit post-save trade lines")

            # Clear active trade state
            self.active_trade = {}
            # Optionally perform a final shutdown so the process fully stops after a trade exit
            try:
                cfg = self.config.get('strategy', {}) if self.config else {}
                stop_after = bool(cfg.get('stop_after_trade', True))
            except Exception:
                stop_after = True

            if stop_after:
                try:
                    # Run final shutdown in a separate daemon thread to avoid blocking the caller
                    t = threading.Thread(target=self._final_shutdown, name="FinalShutdownThread", daemon=True)
                    t.start()
                except Exception:
                    try:
                        self._final_shutdown()
                    except Exception:
                        pass
            return True
        except Exception as e:
            logging.error(f"[EXIT][ERROR] Exception closing trade: {e}")
            logging.error(traceback.format_exc())
            return False
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

    # --- OCO / GTT helpers and execution flow ---
    def place_oco_gtt_orders(self, ce_symbol, ce_trigger, pe_symbol, pe_trigger, qty=25):
        """
        Place two GTT orders (one for CE, one for PE) in the same group so that when one triggers
        the other is cancelled (mutual exclusivity / OCO semantics for paper-mode simulation).
        """
        try:
            # Create a concise group id (timestamp only) to match requested log style
            group_id = f"OCO-{int(time.time())}"
            # Normalize symbols to canonical form for storage
            ce_canon = self.get_canonical_symbol(ce_symbol)
            pe_canon = self.get_canonical_symbol(pe_symbol)
            logging.info(f"[OCO][PLACING] Group={group_id} CE={ce_canon}@{ce_trigger} | PE={pe_canon}@{pe_trigger} | QTY={qty}")
            # Side: 1 for BUY trigger, -1 for SELL trigger (we are buying options)
            r1 = self.order_manager.place_gtt_order(symbol=ce_canon, side=1, qty=qty, trigger_price=ce_trigger, tag='OCO', group_id=group_id)
            r2 = self.order_manager.place_gtt_order(symbol=pe_canon, side=1, qty=qty, trigger_price=pe_trigger, tag='OCO', group_id=group_id)
            logging.info(f"[OCO][SIM] Placed CE GTT: {r1.get('order_id')} | PE GTT: {r2.get('order_id')}")
            # Start monitor thread for this group to detect triggers and convert to an active trade
            t = threading.Thread(target=self._gtt_monitor_thread_fn, args=(group_id,), name=f"GTTMonitor-{group_id}", daemon=True)
            t.start()
            # Keep a reference so we can join/stop if needed
            if not hasattr(self, '_gtt_monitor_threads'):
                self._gtt_monitor_threads = {}
            self._gtt_monitor_threads[group_id] = t
            return True
        except Exception as e:
            logging.error(f"[OCO][ERROR] Failed to place OCO GTT orders: {e}")
            logging.error(traceback.format_exc())
            return False

    def place_oco_bracket_orders(self, ce_symbol, ce_entry, pe_symbol, pe_entry, qty=25):
        """
        Place two simulated bracket orders (one CE, one PE) with entry limits equal to breakout levels.
        Monitor them and when one fills, cancel the other (OCO semantics) and convert the filled
        bracket into an active PAPER trade.
        """
        try:
            group_id = f"OCO-{int(time.time())}"
            ce_canon = self.get_canonical_symbol(ce_symbol)
            pe_canon = self.get_canonical_symbol(pe_symbol)
            # Suppress frequent websocket INFO ticks while placing and monitoring OCO orders
            try:
                self._suppress_ws_update_info = True
            except Exception:
                pass
            logging.info("==============================================================================")
            logging.info("PLACING OCO BRACKET ORDERS AT BREAKOUT LEVELS")
            logging.info("==============================================================================")
            logging.info(f"Placing CE BO: {ce_canon} @ {ce_entry:.2f} (trigger when price >= {ce_entry:.2f})")
            # Place bracket orders via OrderManager (use configured SL/TG percentages when available)
            cfg = self.config.get('strategy', {}) if self.config else {}
            sl_pct = float(cfg.get('stoploss_pct', 20))
            tgt_pct = float(cfg.get('target_pct', 40))
            ce_sl = round(ce_entry * (1 - sl_pct / 100.0), 2)
            ce_tg = round(ce_entry * (1 + tgt_pct / 100.0), 2)
            r1 = self.order_manager.place_bracket_order(symbol=ce_canon, side=1, qty=qty, entry_price=ce_entry, stoploss=ce_sl, target=ce_tg, tag='OCO')
            logging.info(f"[SIMULATION] Placed bracket order for {ce_canon} @ {ce_entry} qty={qty} (breakout={ce_entry})")
            logging.info(f"Placing PE BO: {pe_canon} @ {pe_entry:.2f} (trigger when price >= {pe_entry:.2f})")
            pe_sl = round(pe_entry * (1 - sl_pct / 100.0), 2)
            pe_tg = round(pe_entry * (1 + tgt_pct / 100.0), 2)
            r2 = self.order_manager.place_bracket_order(symbol=pe_canon, side=1, qty=qty, entry_price=pe_entry, stoploss=pe_sl, target=pe_tg, tag='OCO')
            logging.info(f"[SIMULATION] Placed bracket order for {pe_canon} @ {pe_entry} qty={qty} (breakout={pe_entry})")
            logging.info("==============================================================================")
            logging.info("Both OCO bracket orders placed successfully!")
            # Report order ids if available
            try:
                ce_id = r1.get('order_id') if isinstance(r1, dict) else None
                pe_id = r2.get('order_id') if isinstance(r2, dict) else None
                logging.info(f"CE Order ID: {ce_id}")
                logging.info(f"PE Order ID: {pe_id}")
            except Exception:
                logging.debug("Could not read order ids from order manager response")
            logging.info("Orders are now at broker with SL/TP configured.")
            logging.info("Monitoring order status... whichever triggers first will cancel the other.")
            # Track group -> order ids for cancellation when one fills
            if not hasattr(self, 'bracket_groups'):
                self.bracket_groups = {}
            ids = []
            if isinstance(r1, dict) and r1.get('order_id'):
                ids.append(r1.get('order_id'))
            if isinstance(r2, dict) and r2.get('order_id'):
                ids.append(r2.get('order_id'))
            self.bracket_groups[group_id] = ids
            # Start monitor thread for bracket fills
            t = threading.Thread(target=self._bracket_monitor_thread_fn, args=(group_id, ce_canon, pe_canon), name=f"BracketMonitor-{group_id}", daemon=True)
            t.start()
            if not hasattr(self, '_bracket_monitor_threads'):
                self._bracket_monitor_threads = {}
            self._bracket_monitor_threads[group_id] = t
            return True
        except Exception as e:
            logging.error(f"[OCO][ERROR] Failed to place OCO bracket orders: {e}")
            logging.error(traceback.format_exc())
            return False

    def _bracket_monitor_thread_fn(self, group_id, ce_symbol, pe_symbol, poll_interval=1.0):
        """
        Background thread to monitor simulated bracket orders. When one fills, cancel the other
        and convert the filled bracket into an active PAPER trade.
        """
        logging.info(f"[OCO][MONITOR] Starting BRACKET monitor for group {group_id}")
        try:
            last_status_log = 0
            status_log_interval = 7  # seconds between concise summary logs
            while True:
                # Ask OrderManager to check bracket orders based on current live prices
                filled = self.order_manager.monitor_bracket_orders(get_price_func=lambda s: self.live_prices.get(s))
                # Periodically emit a concise status summary (breakout levels, LTPs, order status)
                try:
                    now = time.time()
                    if (now - last_status_log) >= status_log_interval:
                        # Read LTPs for CE and PE
                        ce_ltp = self.live_prices.get(ce_symbol) or 0.0
                        pe_ltp = self.live_prices.get(pe_symbol) or 0.0
                        ce_ltp_f = float(ce_ltp) if ce_ltp is not None else 0.0
                        pe_ltp_f = float(pe_ltp) if pe_ltp is not None else 0.0
                        # Breakout levels
                        ce_break = float(self.call_breakout_level or 0.0)
                        pe_break = float(self.put_breakout_level or 0.0)
                        logging.info(f"[DEBUG] Breakout levels: CE={ce_break:.2f}, PE={pe_break:.2f} | LTPs: CE={ce_ltp_f:.2f}, PE={pe_ltp_f:.2f}")
                        # Try to get order status strings from OrderManager
                        try:
                            ce_orders = self.order_manager.get_orders_by_symbol(ce_symbol) if hasattr(self.order_manager, 'get_orders_by_symbol') else []
                            pe_orders = self.order_manager.get_orders_by_symbol(pe_symbol) if hasattr(self.order_manager, 'get_orders_by_symbol') else []
                            ce_status = 'PENDING'
                            pe_status = 'PENDING'
                            if ce_orders and isinstance(ce_orders, list) and len(ce_orders) > 0:
                                s = ce_orders[0].get('status') or ce_orders[0].get('status_text') or ce_orders[0].get('status_code')
                                ce_status = str(s)
                            if pe_orders and isinstance(pe_orders, list) and len(pe_orders) > 0:
                                s = pe_orders[0].get('status') or pe_orders[0].get('status_text') or pe_orders[0].get('status_code')
                                pe_status = str(s)
                            logging.info(f"Order Status: CE={ce_status} | PE={pe_status}")
                            logging.info(f"Current Prices: CE LTP: {ce_ltp_f:.2f} | PE LTP: {pe_ltp_f:.2f}")
                        except Exception:
                            logging.debug("[OCO][MONITOR] Could not fetch order statuses for concise log")
                        last_status_log = now
                except Exception:
                    logging.debug("[OCO][MONITOR] Status summary failed (non-fatal)")

                if filled:
                    for order in filled:
                        oid = order.get('order_id')
                        symbol = order.get('symbol')
                        price = order.get('filled_price') or order.get('entry_price') or order.get('price')
                        qty = order.get('qty')
                        # Only process orders that belong to our group
                        group_ids = self.bracket_groups.get(group_id, [])
                        if oid not in group_ids:
                            continue
                        # Cancel sibling(s)
                        for other_oid in group_ids:
                            if other_oid != oid:
                                try:
                                    self.order_manager.cancel_order(other_oid, reason='OCO - sibling filled')
                                except Exception:
                                    logging.exception(f"[OCO][MONITOR] Failed to cancel sibling order {other_oid}")
                        # Emit compact simulation-style messages matching sample
                        leg = 'CE' if 'CE' in symbol else 'PE'
                        other_leg = 'PE' if leg == 'CE' else 'CE'
                        try:
                            logging.info(f"[SIMULATION] {leg} breakout triggered! {leg} order FILLED at {price}. {other_leg} order CANCELLED (OCO logic).")
                            # Update shown order status line
                            if leg == 'CE':
                                logging.info("Order Status: CE=FILLED | PE=CANCELLED")
                            else:
                                logging.info("Order Status: CE=CANCELLED | PE=FILLED")
                            logging.info("==============================================================================")
                            logging.info(f"{leg} ORDER TRIGGERED/FILLED! Cancelling {other_leg} order...")
                            logging.info("==============================================================================")
                            logging.info(f"{leg} position active with automatic SL and Target management by broker")
                        except Exception:
                            logging.debug("[OCO][MONITOR] Filled order logged")
                        # Stop suppressing websocket INFO ticks now that a trade was entered
                        try:
                            self._suppress_ws_update_info = False
                        except Exception:
                            pass
                        # Convert filled bracket into an active trade
                        try:
                            self.execute_trade(symbol=symbol, side='BUY', entry_price=price, quantity=qty)
                        except Exception:
                            logging.exception("[OCO][MONITOR] Failed to execute trade from filled bracket order")
                        # Done with this group
                        return
                time.sleep(poll_interval)
        except Exception as e:
            logging.error(f"[OCO][MONITOR][ERROR] Error in BRACKET monitor for group {group_id}: {e}")
            logging.error(traceback.format_exc())

    def _gtt_monitor_thread_fn(self, group_id, poll_interval=1.5):
        """
        Background thread which polls OrderManager.monitor_active_gtt_orders to detect when a GTT
        in the group triggers. When trigger detected, convert trigger into an active_trade, cancel others
        (OrderManager already cancels the group members), and start trade monitoring.
        """
        logging.info(f"[OCO][MONITOR] Starting GTT monitor for group {group_id}")
        try:
            # Keep track of triggers we've already processed to avoid duplicates
            processed_triggers = set()
            while True:
                # Poll and check for triggered orders
                triggered = self.order_manager.monitor_active_gtt_orders(get_price_func=lambda s: self.live_prices.get(s))
                # monitor_active_gtt_orders returns list of triggered order dicts
                for order in triggered:
                    if order.get('group_id') != group_id:
                        continue
                    # We have a triggered GTT -> treat as filled entry for paper trading
                    symbol = order.get('symbol')
                    # Prefer order-reported price, otherwise try live websocket price
                    price = order.get('price') or order.get('trigger_price')
                    if price is None:
                        # Try to read from live prices map first; wait briefly to allow websocket to populate
                        price = self.live_prices.get(symbol)
                        if price is None:
                            # Give websocket a short window to deliver ticks
                            logging.debug(f"[OCO][MONITOR] No live price for {symbol} yet; waiting up to 2s for websocket ticks before REST fallback")
                            waited = 0.0
                            while waited < 2.0 and price is None:
                                time.sleep(0.25)
                                waited += 0.25
                                price = self.live_prices.get(symbol)
                    if price is None:
                        # As a last resort, attempt REST quotes fallback via fyers API
                        try:
                            from src.fyers_api_utils import get_ltp as fyers_get_ltp
                            logging.info(f"[OCO][MONITOR] Falling back to REST LTP for {symbol}")
                            rest_price = fyers_get_ltp(self.fyers, symbol, websocket_client=getattr(self, 'data_socket', None))
                            if rest_price:
                                price = rest_price
                                logging.info(f"[OCO][MONITOR] REST LTP for {symbol}: {price}")
                            else:
                                logging.warning(f"[OCO][MONITOR] REST LTP unavailable for {symbol}; proceeding with trigger price if present")
                        except Exception:
                            logging.exception(f"[OCO][MONITOR] Exception while fetching REST LTP for {symbol}")
                    qty = order.get('qty')
                    logging.info(f"[OCO][TRIGGERED] Group={group_id} Order={order.get('order_id')} Symbol={symbol} Price={price} Qty={qty}")
                    # Cancel other orders (OrderManager.monitor_active_gtt_orders already called cancel_group_gtt_orders)
                    # Convert trigger into an active trade record
                    self.execute_trade(symbol=symbol, side='BUY', entry_price=price, quantity=qty)
                    # Stop monitoring this group after first trigger
                    logging.info(f"[OCO][MONITOR] Stopping monitor for group {group_id} after trigger.")
                    return
                # Additionally, check for any orders that may have been triggered by other parts of
                # the system (or by order_manager from another thread). This ensures we don't miss
                # externally-updated triggers that occurred between polls.
                try:
                    # Use get_orders_by_status helper to find already-triggered orders
                    triggered_orders = self.order_manager.get_orders_by_status(2)  # 2 == TRIGGERED
                    for order in triggered_orders:
                        oid = order.get('order_id')
                        if oid in processed_triggers:
                            continue
                        if order.get('group_id') != group_id:
                            continue
                        # Process this triggered order
                        symbol = order.get('symbol')
                        qty = order.get('qty')
                        price = order.get('price') or order.get('trigger_price') or self.live_prices.get(symbol)
                        logging.info(f"[OCO][MONITOR][DETECT] Found externally-triggered GTT in group {group_id}: {oid} {symbol} price={price}")
                        # Ensure we have a price (try REST fallback)
                        if price is None:
                            try:
                                from src.fyers_api_utils import get_ltp as fyers_get_ltp
                                rest_price = fyers_get_ltp(self.fyers, symbol, websocket_client=getattr(self, 'data_socket', None))
                                if rest_price:
                                    price = rest_price
                                    logging.info(f"[OCO][MONITOR] REST price for {symbol}: {price}")
                            except Exception:
                                logging.exception(f"[OCO][MONITOR] Exception while fetching REST LTP for {symbol}")
                        # Execute trade using detected trigger
                        processed_triggers.add(oid)
                        self.execute_trade(symbol=symbol, side='BUY', entry_price=price, quantity=qty)
                        logging.info(f"[OCO][MONITOR] Processed externally-triggered GTT {oid} for group {group_id}")
                        return
                except Exception:
                    logging.exception(f"[OCO][MONITOR] Error checking externally-triggered orders for group {group_id}")
                time.sleep(poll_interval)
        except Exception as e:
            logging.error(f"[OCO][MONITOR][ERROR] Error in GTT monitor thread for group {group_id}: {e}")
            logging.error(traceback.format_exc())

    def _on_order_manager_gtt_callback(self, order):
        """
        Callback invoked (in a separate thread) when OrderManager notifies of a triggered GTT.
        This ensures we immediately convert the trigger into an active trade without waiting
        for the next poll cycle.
        """
        try:
            group_id = order.get('group_id')
            symbol = order.get('symbol')
            qty = order.get('qty')
            price = order.get('price') or order.get('trigger_price') or self.live_prices.get(symbol)
            logging.info(f"[OCO][CALLBACK] Received GTT trigger callback: group={group_id} order={order.get('order_id')} symbol={symbol} price={price}")
            # Try REST fallback if price missing
            if price is None:
                try:
                    from src.fyers_api_utils import get_ltp as fyers_get_ltp
                    rest_price = fyers_get_ltp(self.fyers, symbol, websocket_client=getattr(self, 'data_socket', None))
                    if rest_price:
                        price = rest_price
                        logging.info(f"[OCO][CALLBACK] REST price for {symbol}: {price}")
                except Exception:
                    logging.exception(f"[OCO][CALLBACK] Exception fetching REST LTP for {symbol}")
            # Execute trade if we have a price (or even if price None, execute_trade will validate)
            self.execute_trade(symbol=symbol, side='BUY', entry_price=price, quantity=qty)
        except Exception:
            logging.exception("[OCO][CALLBACK] Exception while handling GTT callback")

    def execute_trade(self, symbol, side='BUY', entry_price=None, quantity=None):
        """
        Convert an entry (market/GTT trigger) into the internal active_trade representation and
        start the periodic PAPER STATUS monitoring. Returns True on success.
        """
        try:
            if not symbol:
                logging.error("[EXECUTE][ERROR] No symbol provided for execute_trade")
                return False
            # Accept either canonical or raw; normalize for internal storage
            canonical = self.get_canonical_symbol(symbol)
            if quantity is None:
                try:
                    quantity = int(self.config.get('strategy', {}).get('quantity', 25))
                except Exception:
                    quantity = 25
            # Interpret configured 'quantity' as number of lots when a lot_size_map is provided
            try:
                cfg = self.config.get('strategy', {}) if self.config else {}
                lot_map = cfg.get('lot_size_map', {}) if isinstance(cfg, dict) else {}
                quantity_is_lots = bool(cfg.get('quantity_is_lots', True))
                actual_quantity = int(quantity)
                if quantity_is_lots and lot_map:
                    # find a matching key in lot_map that appears in the symbol name
                    for k, v in lot_map.items():
                        try:
                            if k.upper() in canonical.upper():
                                actual_quantity = int(quantity) * int(v)
                                logging.info(f"[EXECUTE] Interpreting quantity {quantity} lots for {k} with lot size {v} -> actual qty {actual_quantity}")
                                break
                        except Exception:
                            continue
                else:
                    # Fallback: if no lot_map but symbol looks like NIFTY, apply known default of 65 per lot
                    if quantity_is_lots and 'NIFTY' in canonical.upper() and not lot_map:
                        actual_quantity = int(quantity) * 65
                        logging.info(f"[EXECUTE] No lot_map in config; assuming NIFTY lot size=65 -> actual qty {actual_quantity}")
                quantity = actual_quantity
            except Exception:
                try:
                    quantity = int(quantity)
                except Exception:
                    quantity = 25
            if entry_price is None:
                # Try to read from live prices / ltp_df
                entry_price = float(self.live_prices.get(canonical) or 0)
            if not entry_price or entry_price <= 0:
                logging.error(f"[EXECUTE][ERROR] Invalid entry price for {canonical}: {entry_price}")
                return False
            # Build active trade structure
            self.active_trade = {
                'symbol': canonical,
                'entry_price': float(entry_price),
                'entry_time': datetime.now(pytz.timezone('Asia/Kolkata')), 
                # quantity stored as actual contracts (lots * lot_size when configured)
                'quantity': int(quantity),
                'lots': int(self.config.get('strategy', {}).get('quantity', 25)) if isinstance(self.config.get('strategy', {}).get('quantity', None), int) else None,
                'side': side,
                'stoploss': round(entry_price * 0.7, 2),  # placeholder: initial stoploss at 30% of entry
                'target': round(entry_price * 1.5, 2),    # placeholder: target at +50%
            }
            logging.info(f"[TRADE][ENTER] Entered PAPER trade: {self.active_trade['symbol']} | Entry: {self.active_trade['entry_price']} | Qty: {self.active_trade['quantity']}")
            # After entering a trade, prefer trade-monitoring logs over symbol-mapping and
            # websocket heartbeat logs. Signal the websocket/data socket (if present)
            # to reduce heartbeat INFO spam, and avoid emitting SYMBOL MAP INFO lines.
            try:
                if hasattr(self, 'data_socket') and getattr(self, 'data_socket') is not None:
                    md = getattr(self.data_socket, 'market_data', None)
                    if isinstance(md, dict) or hasattr(md, 'market_status'):
                        try:
                            # set a flag so websocket implementation can suppress heartbeat INFO
                            md.market_status['suppress_heartbeat'] = True
                        except Exception:
                            # If market_data is a plain dict, ensure key exists
                            try:
                                md['market_status']['suppress_heartbeat'] = True
                            except Exception:
                                pass
            except Exception:
                logging.debug("[EXECUTE] Could not set websocket suppress_heartbeat flag")
            # Place a simulated bracket order (entry + SL + Target) in OrderManager for paper mode
            try:
                cfg = self.config.get('strategy', {}) if self.config else {}
                sl_pct = float(cfg.get('stoploss_pct', 20))
                tgt_pct = float(cfg.get('target_pct', 40))
                stoploss = round(self.active_trade['entry_price'] * (1 - sl_pct / 100.0), 2)
                target = round(self.active_trade['entry_price'] * (1 + tgt_pct / 100.0), 2)
                bracket = self.order_manager.place_bracket_order(symbol=canonical, side=1, qty=self.active_trade['quantity'], entry_price=self.active_trade['entry_price'], stoploss=stoploss, target=target, tag='BRACKET')
                # Debug/log the raw response from OrderManager to diagnose missing bracket placements
                # Log raw bracket response at INFO so it's visible in normal runs/simulations
                logging.info(f"[EXECUTE][DEBUG] Raw bracket response: {bracket}")
                # Handle both forms: {'order_id': id, ...} or {'order_id': id, 'order': {...}}
                try:
                    bracket_order_id = None
                    if isinstance(bracket, dict):
                        bracket_order_id = bracket.get('order_id') or (bracket.get('order') or {}).get('order_id')
                    if bracket_order_id:
                        self.active_trade['bracket_order_id'] = bracket_order_id
                        logging.info(f"[ORDER][BRACKET] Placed bracket order id: {bracket_order_id} SL:{stoploss} TG:{target}")
                        logging.info(f"[ORDER][BRACKET][DETAILS] {bracket}")
                        # Ensure the active trade's SL/Target reflect the bracket order placed
                        try:
                            self.active_trade['stoploss'] = float(stoploss)
                            self.active_trade['target'] = float(target)
                        except Exception:
                            logging.debug("[EXECUTE] Could not sync active_trade SL/TG from bracket values")
                        # Record trailing stop percentage from config for later updates
                        try:
                            trailing_pct = float(cfg.get('trailing_stop_pct', 8)) if cfg else 8.0
                            self.active_trade['trailing_stop_pct'] = trailing_pct
                        except Exception:
                            self.active_trade['trailing_stop_pct'] = 8.0
                        # Start exit monitor to watch for SL / Target hits
                        try:
                            self.start_exit_monitor()
                        except Exception:
                            logging.exception("[EXECUTE] Failed to start exit monitor thread")
                        # Also log current live LTP for the contract so simulation fill checks can be correlated
                        try:
                            cur_ltp = self.live_prices.get(self.active_trade['symbol'])
                            logging.info(f"[EXECUTE][INFO] Current live LTP for {self.active_trade['symbol']}: {cur_ltp}")
                        except Exception:
                            logging.debug("[EXECUTE][INFO] Could not read live_prices for bracket symbol")
                    else:
                        logging.error(f"[ORDER][BRACKET][ERROR] Bracket placement failed for {canonical}. Response: {bracket}")
                except Exception:
                    logging.exception("[ORDER][BRACKET][ERROR] Exception while processing bracket placement response")
            except Exception:
                logging.exception("[ORDER][BRACKET] Failed to place simulated bracket order")
            # Start periodic PAPER STATUS monitor thread
            if not hasattr(self, '_paper_status_thread') or not getattr(self, '_paper_status_thread'):
                def paper_status_runner():
                    logging.info("[PAPER STATUS THREAD] Started")
                    while self.active_trade and not self.market_closed:
                        try:
                            self.log_trade_update()
                        except Exception:
                            logging.exception("[PAPER STATUS THREAD] Exception in log_trade_update")
                        time.sleep(max(1, getattr(self, '_paper_status_min_seconds', 5)))
                    logging.info("[PAPER STATUS THREAD] Exiting")
                # Guard thread creation against interpreter shutdown (prevents RuntimeError seen when process is exiting)
                try:
                    import sys as _sys
                    if hasattr(_sys, 'is_finalizing') and _sys.is_finalizing():
                        logging.error("[EXECUTE][WARN] Interpreter is finalizing; skipping background PAPER STATUS thread")
                    else:
                        self._paper_status_thread = threading.Thread(target=paper_status_runner, name="PaperStatusThread", daemon=True)
                        try:
                            self._paper_status_thread.start()
                        except RuntimeError as e:
                            # This can happen during interpreter shutdown; fall back to immediate status log
                            logging.error(f"[EXECUTE][ERROR] Could not start PAPER STATUS thread: {e}")
                            try:
                                self.log_trade_update()
                            except Exception:
                                logging.exception("[EXECUTE] Failed to emit immediate PAPER STATUS after thread start failure")
                            self._paper_status_thread = None
                except Exception:
                    # Any unexpected error here should not stop execution of the strategy
                    logging.exception("[EXECUTE] Unexpected error when creating PAPER STATUS thread")
            # Emit an immediate PAPER STATUS snapshot so operator sees trade monitoring right after entry
            try:
                self.log_trade_update()
            except Exception:
                logging.exception("[EXECUTE] Failed to emit immediate PAPER STATUS after trade entry")
            return True
        except Exception as e:
            logging.error(f"[EXECUTE][ERROR] Failed to execute trade for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return False
