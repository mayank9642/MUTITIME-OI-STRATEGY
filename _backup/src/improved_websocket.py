"""
Improved WebSocket helper functions with better error handling and diagnostic capabilities
"""
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws, order_ws
import logging
import datetime
import time
import pandas as pd
import threading
import queue
import random
import traceback
from src.config import load_config
from src.token_helper import ensure_valid_token

def improved_market_data_websocket(symbols, callback_handler=None, data_type="SymbolUpdate", debug=False):
    """
    Improved WebSocket connection with better error handling and diagnostic info
    
    Args:
        symbols: List of symbols to subscribe to
        callback_handler: Function to handle data updates
        data_type: Type of data to receive (SymbolUpdate, DepthUpdate)
        debug: Whether to enable verbose debug logging
        
    Returns:
        WebSocket client or None if connection fails
    """
    # Load configuration
    config = load_config()
    client_id = config['fyers']['client_id']
    
    # Get a fresh token
    access_token = ensure_valid_token()
    if not access_token:
        logging.error("Could not obtain valid token for websocket connection")
        return None
    
    # Full token format needed for websocket
    token = f"{client_id}:{access_token}"
    
    # Track received data
    tick_queue = queue.Queue()
    last_tick_time = time.time()
    logged_symbols = set()
    
    # Create data structure for storing market data
    columns = [
        'ltp', 'vol_traded_today', 'timestamp', 'exchange_code',
        'bid_size', 'ask_size', 'bid_price', 'ask_price', 
        'open_price', 'high_price', 'low_price', 'prev_close_price'
    ]
    
    market_data = pd.DataFrame(columns=columns + ['symbol'])
    market_data.set_index('symbol', inplace=True)
    for symbol in symbols:
        market_data.loc[symbol] = [None] * len(columns)
    
    # Track connection status
    connection_status = {
        'connected': False,
        'subscribed': False,
        'last_message_time': 0,
        'connection_time': 0,
        'tick_count': 0,
        'status_logged': False
    }
    
    # Handle incoming messages
    def on_message(ws_ticks):
        now = time.time()
        connection_status['last_message_time'] = now
        
        if debug:
            logging.debug(f"Received tick data: {ws_ticks}")
        
        # Extract symbol from the data
        symbol = ws_ticks.get('symbol')
        if not symbol:
            logging.debug("Received tick data without symbol field")
            return
        
        # Track that we received data
        connection_status['tick_count'] += 1
        
        # Store all tick data in the dataframe
        for key, value in ws_ticks.items():
            if key != 'symbol' and key in columns:
                try:
                    market_data.loc[symbol, key] = value
                except Exception as e:
                    logging.warning(f"Error updating market data for {symbol}.{key}: {e}")
        
        # Put in queue for later processing
        tick_queue.put(ws_ticks)
          # Call the handler
        if callback_handler:
            try:
                callback_handler(symbol, 'tick', ws_ticks, ws_ticks)
            except Exception as e:
                logging.error(f"Error in callback handler: {e}")
                logging.debug(f"Callback error details: {traceback.format_exc()}")
        
        # Log diagnostic information
        if connection_status['tick_count'] % 50 == 0 and not connection_status['status_logged']:
            logging.info(f"WebSocket active: received {connection_status['tick_count']} ticks")
            connection_status['status_logged'] = True
        
        if connection_status['tick_count'] % 100 == 0:
            connection_status['status_logged'] = False
    
    # Handle connection errors
    def on_error(error):
        logging.error(f"WebSocket connection error: {error}")
        connection_status['connected'] = False
    
    # Handle connection close
    def on_close():
        logging.info("WebSocket connection closed")
        connection_status['connected'] = False
    
    # Handle successful connection
    def on_connect():
        logging.info("WebSocket connected successfully!")
        connection_status['connected'] = True
        connection_status['connection_time'] = time.time()
        
        # Subscribe to symbols
        try:
            logging.info(f"Subscribing to {len(symbols)} symbols...")
            client.subscribe(symbols=symbols, data_type=data_type)
            logging.info(f"Subscription requested for symbols: {symbols}")
            connection_status['subscribed'] = True
        except Exception as e:
            logging.error(f"Failed to subscribe to symbols: {e}")
            connection_status['subscribed'] = False
    
    # Create and configure the client
    try:
        # Create websocket client
        client = data_ws.FyersDataSocket(
            access_token=token,
            log_path="logs/",
            litemode=False,
            write_to_file=False,
            reconnect=True,  # Enable auto-reconnect
            on_connect=on_connect,
            on_close=on_close,
            on_error=on_error,
            on_message=on_message
        )
        
        # Connect to websocket
        if debug:
            logging.debug(f"Connecting with token prefix: {token[:15]}...")
        
        connect_result = client.connect()
        
        if connect_result:
            logging.info("WebSocket connection initiated successfully")
            
            # Wait briefly to allow connection to establish
            time.sleep(1)
            
            # Add important attributes to the client
            client.market_data = market_data
            client.tick_queue = tick_queue
            client.connection_status = connection_status
            
            # Define a clean close method
            def close_connection():
                try:
                    logging.info("Closing websocket connection...")
                    if hasattr(client, 'terminate'):
                        client.terminate()
                    elif hasattr(client, 'close'):
                        client.close()
                    else:
                        logging.error("No valid close method found")
                    logging.info("WebSocket connection terminated")
                except Exception as e:
                    logging.error(f"Error closing websocket: {e}")
            
            # Attach the close method
            client.close_connection = close_connection
            
            # Start heartbeat thread to monitor connection
            def heartbeat_monitor():
                last_ping = time.time()
                
                while connection_status['connected']:
                    now = time.time()
                    
                    # Send ping every 30 seconds
                    if now - last_ping > 30:
                        try:
                            if hasattr(client, 'ping'):
                                client.ping()
                                if debug:
                                    logging.debug("Sent ping to websocket server")
                            last_ping = now
                        except Exception as e:
                            logging.warning(f"Failed to send ping: {e}")
                    
                    # Check if we're receiving data
                    if now - connection_status['last_message_time'] > 60 and connection_status['last_message_time'] > 0:
                        logging.warning("No data received for 60 seconds - connection may be stale")
                    
                    # Sleep to avoid high CPU usage
                    time.sleep(5)
            
            # Start the heartbeat thread
            heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
            heartbeat_thread.start()
            
            return client
        else:
            logging.error("Failed to establish websocket connection")
            return None
            
    except Exception as e:
        logging.error(f"Error creating websocket client: {e}")
        logging.error(traceback.format_exc())
        return None

def enhanced_start_market_data_websocket(symbols, callback_handler=None):
    """
    Start a websocket connection with enhanced error handling
    
    Args:
        symbols: List of symbols to monitor
        callback_handler: Function to handle data updates
        
    Returns:
        WebSocket client or None if connection fails
    """
    # Check market status
    from src.market_utils import check_and_log_market_status
    market_open = check_and_log_market_status()
    
    # Log market status
    if market_open:
        logging.info("Market is OPEN - expecting real-time price updates")
    else:
        logging.warning("Market is CLOSED - price updates may be limited or unavailable")
    
    # Try to establish websocket connection
    return improved_market_data_websocket(symbols, callback_handler, debug=True)
