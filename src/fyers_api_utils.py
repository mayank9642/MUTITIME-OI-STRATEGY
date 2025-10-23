from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws, order_ws
import logging
import datetime
import time
import pandas as pd
import threading
import queue
import random
import functools
import re
from src.config import load_config
from src.token_helper import ensure_valid_token

# Monkey patch the logging system to filter sensitive information
orig_info = logging.Logger.info
@functools.wraps(orig_info)
def filtered_info(self, msg, *args, **kwargs):
    # Filter out any debug messages containing token information
    if isinstance(msg, str) and "[DEBUG] get_fyers_client" in msg and ("token" in msg or "access_token" in msg):
        # Replace token information with placeholder
        safe_msg = msg
        if "client_id=" in safe_msg:
            safe_msg = re.sub(r'client_id=[^,\s]+', 'client_id=***FILTERED***', safe_msg)
        if "token" in safe_msg:
            safe_msg = re.sub(r'access_token_head=[^,\s]+', 'access_token=***FILTERED***', safe_msg)
            safe_msg = re.sub(r'token_combo=[^,\s]+', 'token_combo=***FILTERED***', safe_msg)
        # Call original with filtered message
        return orig_info(self, safe_msg, *args, **kwargs)
    # Pass through other messages
    return orig_info(self, msg, *args, **kwargs)

# Apply the monkey patch
logging.Logger.info = filtered_info

def get_fyers_client(check_token=True):
    """
    Create and return authenticated Fyers API client
    
    Args:
        check_token (bool): If True, verify and refresh token if needed
        
    Returns:
        FyersModel: Authenticated Fyers client
    """
    try:
        config = load_config()
        client_id = config['fyers']['client_id']
        
        if check_token:
            access_token = ensure_valid_token()
        else:
            access_token = config['fyers']['access_token']

        # IMPORTANT: For REST API, use only the access_token (not client_id:access_token)
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path="logs/")# Test connection with profile API
        profile_response = fyers.get_profile()
        if profile_response.get('s') == 'ok':
            logging.debug(f"Successfully authenticated with Fyers API")
            return fyers
        else:
            logging.error(f"Fyers authentication failed: {profile_response}")
            if 'message' in profile_response and 'token' in profile_response['message'].lower():
                logging.error("[FATAL] Fyers token rejected. Please re-authenticate using python -m src.auth and ensure your app permissions are correct.")
            return None
    except Exception as e:
        logging.error(f"Error creating Fyers client: {str(e)}")
        return None

def place_market_order(fyers, symbol, qty, side):
    """
    Place a market order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 2,  # 2 = Market order
            "side": 1 if side == "BUY" else -1,  # 1 = Buy, -1 = Sell
            "productType": "INTRADAY",
            "validity": "DAY",
            "offlineOrder": False,
            "stopPrice": 0,
            "limitPrice": 0
        }
        
        response = fyers.place_order(data=order_data)
        logging.info(f"Order placed: {symbol} {side} {qty} - Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error placing order: {str(e)}")
        return None

def modify_order(fyers, order_id, price=None, stop_price=None):
    """Modify an existing order (for SL/target modifications)"""
    try:
        modify_data = {
            "id": order_id
        }
        
        if price is not None:
            modify_data["limitPrice"] = price
            
        if stop_price is not None:
            modify_data["stopPrice"] = stop_price
            
        response = fyers.modify_order(data=modify_data)
        logging.info(f"Order modified: {order_id} - Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error modifying order: {str(e)}")
        return None

def exit_position(fyers, symbol, qty, side):
    """Exit an existing position"""
    try:
        return place_market_order(fyers, symbol, qty, side)
    except Exception as e:
        logging.error(f"Error exiting position: {str(e)}")
        return None

def get_current_positions(fyers):
    """Get current positions"""
    try:
        positions = fyers.positions()
        return positions
    except Exception as e:
        logging.error(f"Error getting positions: {str(e)}")
        return None

def place_limit_order(fyers, symbol, qty, side, limit_price):
    """
    Place a limit order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        limit_price: Limit price for the order
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 1,  # 1 = Limit order
            "side": 1 if side == "BUY" else -1,  # 1 = Buy, -1 = Sell
            "productType": "INTRADAY",
            "validity": "DAY",
            "offlineOrder": False,
            "stopPrice": 0,
            "limitPrice": limit_price
        }
        
        response = fyers.place_order(data=order_data)
        logging.info(f"Limit order placed: {symbol} {side} {qty} @ {limit_price} - Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error placing limit order: {str(e)}")
        return None

def place_sl_order(fyers, symbol, qty, side, trigger_price):
    """
    Place a stop-loss order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        trigger_price: Stop-loss trigger price
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 3,  # 3 = Stop order (SL-M)
            "side": 1 if side == "BUY" else -1,
            "productType": "INTRADAY",
            "validity": "DAY",
            "offlineOrder": False, 
            "stopPrice": trigger_price,
            "limitPrice": 0
        }
        
        response = fyers.place_order(data=order_data)
        logging.info(f"SL order placed: {symbol} {side} {qty} @ trigger {trigger_price} - Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error placing SL order: {str(e)}")
        return None

def place_sl_limit_order(fyers, symbol, qty, side, trigger_price, limit_price):
    """
    Place a stop-loss limit order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        trigger_price: Stop-loss trigger price
        limit_price: Limit price for order execution
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 4,  # 4 = Stop limit order (SL-L)
            "side": 1 if side == "BUY" else -1,
            "productType": "INTRADAY",
            "validity": "DAY",
            "offlineOrder": False,
            "stopPrice": trigger_price,
            "limitPrice": limit_price
        }
        
        response = fyers.place_order(data=order_data)
        logging.info(f"SL-L order placed: {symbol} {side} {qty} @ trigger {trigger_price}, limit {limit_price} - Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error placing SL-L order: {str(e)}")
        return None

def get_order_status(fyers, order_id):
    """Get status of an existing order"""
    try:
        data = {
            "id": order_id
        }
        response = fyers.get_orders(data=data)
        logging.info(f"Order status for {order_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Error getting order status: {str(e)}")
        return None

def get_historical_data(fyers, symbol, resolution, date_format, range_from, range_to):
    """
    Get historical data for a symbol
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY-INDEX")
        resolution: Timeframe resolution (1, 5, 15, 60, 1D, etc.)
        date_format: Date format (1 for epoch)
        range_from: Start date (epoch or datetime format)
        range_to: End date (epoch or datetime format)
        
    Returns:
        DataFrame: Historical data in pandas DataFrame
    """
    try:
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": date_format,
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        
        response = fyers.get_historical_data(data)
        
        if isinstance(response, dict) and 'candles' in response:
            df = pd.DataFrame(response['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            if date_format == 1:  # If date is in epoch format
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            return df
        else:
            logging.error(f"Invalid response format: {response}")
            return None
    except Exception as e:
        logging.error(f"Error getting historical data: {str(e)}")
        return None

def get_option_chain(fyers, underlying):
    """
    Get option chain data for an underlying
    
    Args:
        fyers: Authenticated Fyers client
        underlying: Underlying symbol (e.g., "NSE:NIFTY-INDEX")
        
    Returns:
        dict: Option chain data
    """
    try:
        response = fyers.get_option_chain({"symbol": underlying})
        logging.info(f"Option chain fetched for {underlying}")
        return response
    except Exception as e:
        logging.error(f"Error getting option chain: {str(e)}")
        return None

def robust_market_data_websocket(symbols, callback_handler=None, data_type="SymbolUpdate", on_success=None, on_failure=None):
    """
    Robust WebSocket connection with auto-reconnect, heartbeat, thread-safe queue, and throttling.
    Returns the ws_client with .tick_queue for tick data.
    """
    import queue, threading, random, time
    config = load_config()
    client_id = config['fyers']['client_id']
    access_token = ensure_valid_token()
    tick_queue = queue.Queue()
    throttle_interval = 1  # seconds
    last_tick_time = [0]
    min_delay = 1
    max_delay = 60
    backoff = min_delay
    heartbeat_interval = 30
    last_ping_time = [time.time()]
    # Track symbols we've already logged to reduce duplicate logs
    logged_symbols = set()
    
    expected_columns = [
        'ltp', 'vol_traded_today', 'last_traded_time', 'exch_feed_time',
        'bid_size', 'ask_size', 'bid_price', 'ask_price', 
        'last_traded_qty', 'tot_buy_qty', 'tot_sell_qty', 'avg_trade_price',
        'low_price', 'high_price', 'open_price', 'prev_close_price', 'symbol'
    ]
    
    market_data_df = pd.DataFrame(columns=expected_columns)
    market_data_df.set_index('symbol', inplace=True)
    for symbol in symbols:
        market_data_df.loc[symbol] = [None] * (len(expected_columns) - 1)
        
    def on_message(ticks):
        now = time.time()
        if now - last_tick_time[0] < throttle_interval:
            return
        last_tick_time[0] = now
        if ticks.get('symbol'):
            symbol = ticks.get('symbol')
            is_option = "CE" in symbol or "PE" in symbol
            
            # Track market status data
            if not hasattr(market_data_df, 'market_status'):
                market_data_df.market_status = {
                    'last_check': now,
                    'is_open': False,
                    'status_logged': False,
                    'has_ltp_data': False
                }
            
            # Store all tick data in the DataFrame
            for key, value in ticks.items():
                if key != 'symbol':
                    try:
                        market_data_df.loc[symbol, key] = value
                    except Exception as e:
                        logging.warning(f"Failed to update DataFrame for {symbol} - {key}: {e}")
            
            # Put the tick in the queue for consumption
            tick_queue.put(ticks)
            
            # Call the handler with the expected format
            if callback_handler:
                callback_handler(symbol, 'tick', ticks, ticks)
            
            # Process and log LTP data
            if 'ltp' in ticks and ticks['ltp'] is not None:
                market_data_df.market_status['has_ltp_data'] = True
                market_data_df.market_status['is_open'] = True
                
                if symbol not in logged_symbols:
                    if is_option:
                        logging.info(f"WebSocket option tick: {symbol} LTP: {ticks['ltp']}")
                    else:
                        logging.info(f"WebSocket tick: {symbol} LTP: {ticks['ltp']}")
                    logged_symbols.add(symbol)
                    
                # Always update a running ticker count to verify WS is active
                if not hasattr(market_data_df, 'tick_count'):
                    market_data_df.tick_count = 0
                market_data_df.tick_count += 1
                
                # Log an active heartbeat every 50 ticks
                if market_data_df.tick_count % 50 == 0:
                    logging.info(f"WebSocket active: {market_data_df.tick_count} ticks received")
            
            # Check for market closed condition (receiving data but no LTP)
            elif now - market_data_df.market_status['last_check'] > 30:  # Check every 30 seconds
                market_data_df.market_status['last_check'] = now
                
                # If we're receiving ticks but they don't have LTP, market is likely closed
                if not market_data_df.market_status['has_ltp_data'] and not market_data_df.market_status['status_logged']:
                    if 'prev_close_price' in ticks and ticks['prev_close_price'] is not None:
                        logging.warning(f"MARKET IS CLOSED - WebSocket receiving prev_close_price but no LTP for {symbol}")
                    else:                        
                        available_fields = [k for k, v in ticks.items() if v is not None]
                        logging.warning(f"MARKET IS CLOSED - WebSocket receiving data without LTP. Available fields: {available_fields}")
                    
                    market_data_df.market_status['status_logged'] = True
                    market_data_df.market_status['is_open'] = False
                    
            # Clear the logged symbols set every 2 minutes to allow fresh logs
            if not hasattr(market_data_df, 'last_symbol_clear'):
                market_data_df.last_symbol_clear = now
            if now - market_data_df.last_symbol_clear > 120:  # 2 minutes
                logged_symbols.clear()
                market_data_df.last_symbol_clear = now
                
    def on_error(error):
        logging.error(f"WebSocket error: {error}")

    def on_close(message):
        logging.info(f"WebSocket connection closed: {message}")
        
    def on_subscribe_success(response):
        logging.info(f"WebSocket subscription success: {response}")
        if callable(on_success):
            on_success(response)
            
    def on_subscribe_failure(error_code, message):
        logging.error(f"WebSocket subscription failed: {error_code} - {message}")
        if callable(on_failure):
            on_failure(error_code, message)
            
    def on_open(client):
        logging.info(f"WebSocket connection opened for {len(symbols)} symbols")
        try:
            # Add a short delay before subscribing to ensure socket is ready
            time.sleep(1)
            
            # Validate symbols format before subscribing
            valid_symbols = []
            for symbol in symbols:
                if ":" in symbol:  # Ensure symbol has proper format
                    valid_symbols.append(symbol)
                else:
                    logging.warning(f"Invalid symbol format: {symbol} - must contain a colon (e.g. NSE:SYMBOL)")
            
            if not valid_symbols:
                logging.error("No valid symbols to subscribe!")
                if callable(on_failure):
                    on_failure("ERROR", "No valid symbols to subscribe")
                return
                
            # Subscribe to symbols and verify subscription
            logging.info(f"Requesting subscription for {len(valid_symbols)} symbols...")
            result = client.subscribe(symbols=valid_symbols, data_type=data_type)
            
            # Verify subscription result
            if result:
                logging.info(f"Subscription successful: {result}")
                if callable(on_success):
                    on_success({"status": "Subscription confirmed", "symbols": valid_symbols})
            else:
                logging.warning(f"Subscription request sent but returned: {result}")
                logging.warning("Will continue but subscription may not be active")
                if callable(on_success):
                    on_success({"status": "Subscription requested but unconfirmed", "symbols": valid_symbols})
        except Exception as e:
            logging.error(f"Error subscribing to symbols: {str(e)}")
            import traceback
            logging.error(f"Subscription error traceback: {traceback.format_exc()}")
            if callable(on_failure):
                on_failure("ERROR", str(e))

    def heartbeat_thread(ws_client):
        while True:
            now = time.time()
            if now - last_ping_time[0] > heartbeat_interval:
                try:
                    if hasattr(ws_client, 'ping'):
                        ws_client.ping()
                        logging.debug("WebSocket ping sent.")  # Reduced to debug level
                    last_ping_time[0] = now
                except Exception as e:
                    logging.warning(f"WebSocket ping failed: {e}")
            time.sleep(heartbeat_interval)

    def connect_with_retries():
        nonlocal backoff
        while True:
            try:
                # Create WebSocket client
                local_client = data_ws.FyersDataSocket(
                    access_token=f"{client_id}:{access_token}",
                    log_path="logs/",
                    litemode=False,
                    write_to_file=False,
                    reconnect=False,  # We'll handle reconnection
                    on_close=on_close,
                    on_error=on_error,
                    on_message=on_message
                )
                
                # Set the on_connect callback with a closure that has access to the client
                local_client.on_connect = lambda: on_open(local_client)                # Attempt to connect with more robust error handling
                connect_result = local_client.connect()
                if connect_result:
                    logging.info("Websocket connected")
                    # Explicitly wait a bit to ensure connection stabilizes
                    time.sleep(1)
                else:
                    logging.error("Websocket connection failed - returned False")
                    logging.debug("Connection details: " + str(dir(local_client)))
                    
                    # Try to reconnect with a retry
                    logging.info("Attempting reconnection...")
                    time.sleep(2)  # Wait before retry
                    connect_result = local_client.connect()
                    if connect_result:
                        logging.info("Websocket reconnected successfully")
                    else:
                        logging.error("Websocket reconnection failed")
                
                # Store data and queue references on the client
                local_client.market_data = market_data_df
                local_client.tick_queue = tick_queue
                local_client.connection_time = time.time()
                
                # Ensure we have proper close methods on the client
                if not hasattr(local_client, 'close_connection'):
                    def close_connection():
                        try:
                            logging.info("Closing websocket connection...")
                            if hasattr(local_client, 'terminate'):
                                local_client.terminate()
                            elif hasattr(local_client, 'close'):
                                local_client.close()
                            else:
                                logging.error("No valid close method found on websocket client")
                            logging.info("Websocket connection terminated")
                        except Exception as e:
                            logging.error(f"Error terminating websocket: {str(e)}")
                    local_client.close_connection = close_connection
                threading.Thread(target=heartbeat_thread, args=(local_client,), daemon=True).start()
                return local_client
            except Exception as e:
                logging.error(f"WebSocket connection failed: {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff = min(max_delay, backoff * 2 + random.uniform(0, 1))

    ws_client = connect_with_retries()
    return ws_client

def start_market_data_websocket(symbols, callback_handler=None, data_type="SymbolUpdate", 
                        on_success=None, on_failure=None, debug_mode=False):
    """
    Start a websocket connection for market data with tick-by-tick updates
    
    Args:
        symbols: List of symbols to subscribe to
        callback_handler: Function to handle data updates (receives symbol, key, value)
        data_type: Type of data to receive (SymbolUpdate, DepthUpdate)
        on_success: Callback function when subscription succeeds
        on_failure: Callback function when subscription fails
        debug_mode: Enable extra debug logging
        
    Returns:
        WebSocket connection object
    """
    try:
        # Check market status
        from src.market_utils import check_and_log_market_status
        is_market_open = check_and_log_market_status()
        
        # Log details about the request
        logging.info(f"Setting up websocket connection for {len(symbols)} symbols...")
        if debug_mode:
            logging.debug(f"Symbols: {symbols}")
            logging.debug(f"Data type: {data_type}")
            logging.debug(f"Market status: {'Open' if is_market_open else 'Closed'}")
        
        # Verify token before starting
        token = ensure_valid_token()
        if not token:
            logging.error("Could not obtain a valid token for WebSocket connection")
            return None
            
        if debug_mode:
            logging.debug("Token validation successful")
        
        # Use the robust websocket implementation
        ws_client = robust_market_data_websocket(
            symbols,
            callback_handler=callback_handler,
            data_type=data_type,
            on_success=on_success,
            on_failure=on_failure
        )
        
        if ws_client:
            # Store market status on the client
            ws_client.market_open = is_market_open
            
            # Check if the connection is actually working
            if hasattr(ws_client, 'connection_time'):
                logging.info(f"WebSocket connection established and active for {len(symbols)} symbols")
            else:
                logging.warning("WebSocket client created but may not be properly initialized")
            
            # Log what to expect
            if not is_market_open:
                logging.warning("MARKET CLOSED: Websocket will not provide real-time price updates. "
                               "You may receive prev_close_price instead of ltp.")
        else:
            logging.error("Failed to create WebSocket client")
        
        return ws_client
    except Exception as e:
        logging.error(f"Error starting market data websocket: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_nifty_spot_price():
    """
    Fetch the current Nifty spot price using the Fyers quotes API.
    
    Returns:
        float: The current Nifty spot price, or 0 if unavailable.
    """
    try:
        fyers = get_fyers_client()
        if not fyers:
            logging.error("Fyers client not available for spot price fetch.")
            return 0
        data = {"symbols": "NSE:NIFTY50-INDEX"}
        response = fyers.quotes(data=data)
        if response.get('s') == 'ok' and 'd' in response and len(response['d']) > 0:
            spot = response['d'][0].get('v', {}).get('lp', 0)
            logging.info(f"Fetched Nifty spot price from quotes API: {spot}")
            return spot
        else:
            logging.error(f"Failed to fetch Nifty spot price from quotes API: {response}")
            return 0
    except Exception as e:
        logging.error(f"Error fetching Nifty spot price: {str(e)}")
        return 0

def get_nifty_spot_price_direct(fyers):
    """
    Get the current Nifty spot price using direct quotes API
    
    Args:
        fyers: Authenticated Fyers client
        
    Returns:
        float: Nifty spot price or 0 if unavailable
    """
    try:
        if not fyers:
            logging.error("Fyers client is not initialized")
            return 0
            
        # Use Fyers quotes API to get Nifty spot price
        nifty_symbol = "NSE:NIFTY50-INDEX"  # Nifty index symbol
        quotes_response = fyers.quotes({"symbols": nifty_symbol})
        
        if quotes_response.get('s') == 'ok' and 'd' in quotes_response:
            data = quotes_response['d']
            if data and len(data) > 0 and 'v' in data[0]:
                # Extract the last price from the response
                ltp = data[0]['v'].get('lp', 0)
                return float(ltp)
                
        logging.warning(f"Failed to get Nifty spot price: {quotes_response}")
        return 0
    except Exception as e:
        logging.error(f"Error getting Nifty spot price: {str(e)}")
        return 0
    except Exception as e:
        logging.error(f"Error getting Nifty spot price: {str(e)}")
        return 0

def get_ltp(fyers, symbol, websocket_client=None):
    """
    Get the Last Traded Price (LTP) for a symbol
    First tries from websocket data if available, otherwise uses quotes API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        websocket_client: Optional websocket client with market data
        
    Returns:
        float: LTP if available, None otherwise
    """
    try:
        # Try to get price from WebSocket first if available
        if websocket_client and hasattr(websocket_client, 'market_data'):
            if symbol in websocket_client.market_data.index:
                ltp = websocket_client.market_data.loc[symbol, 'ltp']
                if pd.notna(ltp):
                    logging.info(f"LTP from WebSocket for {symbol}: {ltp}")
                    return float(ltp)
        
        # If websocket data is not available or the symbol isn't there, fall back to API
        if not fyers:
            logging.error("Fyers client is not initialized")
            return None
            
        # Use Fyers quotes API to get LTP
        quotes_response = fyers.quotes({"symbols": symbol})
        
        if quotes_response.get('s') == 'ok' and 'd' in quotes_response:
            data = quotes_response['d']
            if data and len(data) > 0 and 'v' in data[0]:
                # Extract the last price from the response
                ltp = data[0]['v'].get('lp', 0)
                logging.debug(f"LTP from API for {symbol}: {ltp}")
                return float(ltp)
                
        logging.warning(f"Failed to get LTP for {symbol}: {quotes_response}")
        return None
    except Exception as e:
        logging.error(f"Error getting LTP for {symbol}: {str(e)}")
        return None