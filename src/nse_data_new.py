import logging
import pandas as pd
import time
from src.fyers_api_utils import get_fyers_client

def get_nifty_option_chain(expiry_index=0):
    """
    Fetch the Nifty 50 option chain using Fyers API with fallback to NSE if needed.
    Args:
        expiry_index (int): Index of expiry to use (0=current, 1=next, etc.)
    Returns:
        DataFrame: Option chain data in pandas DataFrame format
    """
    try:
        logging.info(f"Fetching Nifty option chain data using Fyers API for expiry index: {expiry_index}")
        fyers = get_fyers_client()
        if not fyers:
            logging.error("Fyers API client not available, using fallback method")
            return _get_nifty_option_chain_fallback()
        symbol = "NSE:NIFTY50-INDEX"
        strike_count = 20
        # Step 1: Get expiry dates
        data = {"symbol": symbol, "strikecount": strike_count, "timestamp": ""}
        response = fyers.optionchain(data=data)
        if not response or response.get('s') != 'ok' or 'data' not in response:
            logging.error(f"Failed to fetch expiry dates: {response}")
            return _get_nifty_option_chain_fallback()
        expiry_data = response['data'].get('expiryData', [])
        if not expiry_data:
            logging.error("No expiry dates found in response")
            return _get_nifty_option_chain_fallback()
        if expiry_index >= len(expiry_data):
            logging.error(f"Requested expiry index {expiry_index} exceeds available expiries (total: {len(expiry_data)})")
            expiry_index = len(expiry_data) - 1
            logging.info(f"Using last available expiry at index {expiry_index} instead")
        expiry_timestamp = expiry_data[expiry_index]['expiry']
        expiry_str = expiry_data[expiry_index].get('date', str(expiry_timestamp))
        logging.info(f"Using option chain expiry: {expiry_str} (index {expiry_index})")
        # Step 2: Now fetch the full option chain with the expiry timestamp
        data = {"symbol": symbol, "strikecount": strike_count, "timestamp": expiry_timestamp}
        response = fyers.optionchain(data=data)
        if not response or response.get('s') != 'ok' or 'data' not in response:
            logging.error(f"Failed to fetch option chain: {response}")
            return _get_nifty_option_chain_fallback()
        options_chain_data = response['data'].get('optionsChain', [])
        if not options_chain_data:
            logging.error("Empty options chain returned")
            return _get_nifty_option_chain_fallback()
        return pd.DataFrame(options_chain_data)
    except Exception as e:
        logging.error(f"Exception in get_nifty_option_chain: {str(e)}")
        return _get_nifty_option_chain_fallback()

def _get_nifty_option_chain_fallback():
    """
    Fallback method to fetch Nifty 50 option chain from NSE website
    This is used when Fyers API is not available or fails
    """
    try:
        logging.info("Using fallback method to fetch option chain from NSE")
        # Placeholder: implement NSE scraping if needed
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching Nifty option chain: {str(e)}")
        return pd.DataFrame()