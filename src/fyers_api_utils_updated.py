from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws, order_ws
import logging
import datetime
import time
import pandas as pd
import threading
import queue
import random
from src.config import load_config
from src.token_helper import ensure_valid_token

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
            for key, value in ticks.items():
                if key != 'symbol':
                    try:
                        market_data_df.loc[symbol, key] = value
                    except Exception as e:
                        logging.warning(f"Failed to update DataFrame for {symbol} - {key}: {e}")
            tick_queue.put(ticks)
            if callback_handler:
                callback_handler(symbol, 'tick', ticks, ticks)
            if 'ltp' in ticks:
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
                    
            # Clear the logged symbols set every 2 minutes to allow fresh logs
            now = time.time()
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
            client.subscribe(symbols=symbols, data_type=data_type)
            logging.info(f"Subscription requested for symbols: {symbols}")
            if callable(on_success):
                on_success({"status": "Subscription requested", "symbols": symbols})
        except Exception as e:
            logging.error(f"Error subscribing to symbols: {str(e)}")
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
                local_client.on_connect = lambda: on_open(local_client)
                local_client.connect()
                local_client.market_data = market_data_df
                local_client.tick_queue = tick_queue
                if not hasattr(local_client, 'close_connection'):
                    def close_connection():
                        try:
                            logging.info("Closing websocket connection...")
                            local_client.terminate()
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

def get_fyers_client(check_token=True):
    """
    Create and return authenticated Fyers API client
    
    Args:
        check_token (bool): If True, verify and refresh token if needed
        
    Returns:
        FyersModel: Authenticated Fyers client
    """
    try:
        if check_token:
            access_token = ensure_valid_token()
        else:            
            config = load_config()
            access_token = config['fyers']['access_token']
            client_id = config['fyers']['client_id']

        # IMPORTANT: For REST API, use only the access_token (not client_id:access_token)
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path="logs/")
        # Test connection with profile API
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
    # Exiting a position is just placing a market order in the opposite direction
    exit_side = "SELL" if side == "BUY" else "BUY"
    return place_market_order(fyers, symbol, qty, exit_side)

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
    Place a stop loss order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        trigger_price: Price at which the order should trigger
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 4,  # 4 = Stop order (SL-M)
            "side": 1 if side == "BUY" else -1,  # 1 = Buy, -1 = Sell
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
    Place a stop loss limit order using Fyers API
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY2560619500CE")
        qty: Quantity to trade
        side: "BUY" or "SELL"
        trigger_price: Price at which the order should trigger
        limit_price: Limit price for the order
        
    Returns:
        Order response from Fyers API
    """
    try:
        order_data = {
            "symbol": symbol,
            "qty": qty,
            "type": 3,  # 3 = SL-L order
            "side": 1 if side == "BUY" else -1,  # 1 = Buy, -1 = Sell
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
        order_status = fyers.orderbook()
        logging.info(f"Order status for {order_id}: {order_status}")
        
        if isinstance(order_status, dict) and order_status.get('s') == 'ok' and 'orderBook' in order_status:
            for order in order_status['orderBook']:
                if order.get('id') == order_id:
                    return order
        
        return None
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

def start_market_data_websocket(symbols, callback_handler=None, data_type="SymbolUpdate", 
                        on_success=None, on_failure=None):
    """
    Start a market data WebSocket connection
    
    Args:
        symbols: List of symbols to subscribe to
        callback_handler: Callback function for handling data updates
        
    Returns:
        WebSocket client
    """
    return robust_market_data_websocket(symbols, callback_handler, data_type, on_success, on_failure)

def get_nifty_spot_price():
    """
    Get the current Nifty spot price
    
    Returns:
        float: Nifty spot price or 0 if unavailable
    """
    try:
        fyers = get_fyers_client()
        return get_nifty_spot_price_direct(fyers)
    except Exception as e:
        logging.error(f"Error getting Nifty spot price: {str(e)}")
        return 0
    except Exception as e:
        logging.error(f"Error getting Nifty spot price: {str(e)}")
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
                spot = float(ltp)
                logging.info(f"Fetched Nifty spot price from quotes API: {spot}")
                return spot
                
        logging.warning(f"Failed to get Nifty spot price: {quotes_response}")
        return 0
    except Exception as e:
        logging.error(f"Error getting Nifty spot price: {str(e)}")
        return 0

def get_ltp(fyers, symbol, websocket_client=None):
    """
    Get the last traded price for a symbol
    
    Args:
        fyers: Authenticated Fyers client
        symbol: Trading symbol (e.g., "NSE:NIFTY50-INDEX")
        websocket_client: Optional websocket client for faster price updates
        
    Returns:
        float: Last traded price or None if unavailable
    """
    try:
        # Try to get price from websocket data first (faster)
        if websocket_client and hasattr(websocket_client, 'market_data'):
            try:
                market_data = websocket_client.market_data
                if symbol in market_data.index:
                    ltp = market_data.loc[symbol, 'ltp']
                    if ltp is not None:
                        logging.info(f"LTP from WebSocket for {symbol}: {ltp}")
                        return float(ltp)
            except Exception as ws_error:
                logging.warning(f"Failed to get LTP from WebSocket: {ws_error}")
        
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
