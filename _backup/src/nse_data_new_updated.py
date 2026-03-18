import requests
import pandas as pd
import logging
import datetime
import time
from io import StringIO
from src.fyers_api_utils import get_fyers_client, get_nifty_spot_price


def get_nifty_option_chain(expiry_index=0):
    """
    Fetch the Nifty 50 option chain using Fyers API with the correct symbol format
    
    Args:
        expiry_index (int): Index of expiry to use (0=current, 1=next, etc.)
        
    Returns:
        DataFrame: Option chain data in pandas DataFrame format with proper symbol names
    """
    try:
        logging.info(f"Fetching Nifty option chain data using Fyers API for expiry index: {expiry_index}")
        fyers = get_fyers_client()

        # If Fyers API client is not available, fall back to alternative method
        if not fyers:
            logging.error("Fyers API client not available, using fallback method")
            return _get_nifty_option_chain_fallback()

        symbol = "NSE:NIFTY50-INDEX"
        strike_count = 20        
        # Step 1: Get expiry dates
        data = {"symbol": symbol, "strikecount": strike_count, "timestamp": ""}
        try:
            response = fyers.optionchain(data=data)
            # Enhanced error handling: Check for both authentication errors and other API failures
            if isinstance(response, dict):
                # Check for authentication errors specifically
                if response.get('code') == -15:
                    logging.error("Authentication token error detected. Attempting to refresh token...")
                    from src.token_helper import ensure_valid_token
                    new_token = ensure_valid_token(max_retries=2)
                    if new_token:
                        logging.info("Token refreshed successfully, retrying API call")
                        fyers = get_fyers_client(check_token=False)  # Get client with new token
                        response = fyers.optionchain(data=data)
                    else:
                        logging.error("Failed to refresh authentication token")
                        return _get_nifty_option_chain_fallback()
                # Check for rate limiting or temporary server issues
                elif response.get('code') in [-5, -11, -30, -31, -32]:
                    # These codes often indicate temporary issues
                    logging.warning(f"API temporary error (code {response.get('code')}): {response.get('message')}. Retrying in 3 seconds...")
                    time.sleep(3)  # Add delay before retry
                    try:
                        response = fyers.optionchain(data=data)
                    except Exception as retry_e:
                        logging.error(f"Retry failed: {str(retry_e)}")
                        return _get_nifty_option_chain_fallback()
            
            if not response or response.get('s') != 'ok' or 'data' not in response:
                logging.error(f"Failed to fetch expiry dates: {response}")
                return _get_nifty_option_chain_fallback()
        except Exception as e:
            logging.error(f"Exception while fetching expiry dates: {str(e)}")
            return _get_nifty_option_chain_fallback()

        expiry_data = response['data'].get('expiryData', [])

        if not expiry_data:
            logging.error("No expiry dates found in response")
            return _get_nifty_option_chain_fallback()

        # Check if requested expiry index is valid
        if expiry_index >= len(expiry_data):
            logging.error(f"Requested expiry index {expiry_index} exceeds available expiries (total: {len(expiry_data)})")
            # Fall back to using the last available expiry
            expiry_index = len(expiry_data) - 1
            logging.info(f"Using last available expiry at index {expiry_index} instead")

        # Get the specified expiry date
        expiry_timestamp = expiry_data[expiry_index]['expiry']
        expiry_str = expiry_data[expiry_index].get('date', str(expiry_timestamp))
        logging.info(f"Using option chain expiry: {expiry_str} (index {expiry_index})")        
        
        # Step 2: Now fetch the full option chain with the expiry timestamp
        data = {"symbol": symbol, "strikecount": strike_count, "timestamp": expiry_timestamp}
        try:
            response = fyers.optionchain(data=data)
            # Enhanced error handling: Check for both authentication errors and other API failures
            if isinstance(response, dict):
                # Check for authentication errors specifically
                if response.get('code') == -15:
                    logging.error("Authentication token error detected. Attempting to refresh token...")
                    from src.token_helper import ensure_valid_token
                    new_token = ensure_valid_token(max_retries=2)
                    if new_token:
                        logging.info("Token refreshed successfully, retrying API call")
                        fyers = get_fyers_client(check_token=False)  # Get client with new token
                        response = fyers.optionchain(data=data)
                    else:
                        logging.error("Failed to refresh authentication token")
                        return _get_nifty_option_chain_fallback()
                # Check for rate limiting or temporary server issues
                elif response.get('code') in [-5, -11, -30, -31, -32]:
                    # These codes often indicate temporary issues
                    logging.warning(f"API temporary error (code {response.get('code')}): {response.get('message')}. Retrying in 3 seconds...")
                    time.sleep(3)  # Add delay before retry
                    try:
                        response = fyers.optionchain(data=data)
                    except Exception as retry_e:
                        logging.error(f"Retry failed: {str(retry_e)}")
                        return _get_nifty_option_chain_fallback()
            
            if not response or response.get('s') != 'ok' or 'data' not in response:
                logging.error(f"Failed to fetch option chain: {response}")
                return _get_nifty_option_chain_fallback()
        except Exception as e:
            logging.error(f"Exception while fetching option chain: {str(e)}")
            return _get_nifty_option_chain_fallback()
            
        # Get the options chain data
        options_chain_data = response['data'].get('optionsChain', [])

        if not options_chain_data:
            logging.error("Empty options chain returned")
            return _get_nifty_option_chain_fallback()

        # Get the underlying spot price using the quotes API (fixes spot price = 0)
        spot_price = get_nifty_spot_price()
        logging.info(f"Current Nifty spot price: {spot_price}")
        
        # Process option chain data into a consistent format
        processed_options = []
        
        # REMOVED: No longer logging option data structure on every fetch to reduce logs
        
        for option_data in options_chain_data:
            strike_price = option_data.get('strike_price', 0)
            option_type = option_data.get('option_type', '')
            symbol = option_data.get('symbol', '')
            expiry = option_data.get('expiry', expiry_str)

            # Extract data for CE or PE options
            if option_type in ['CE', 'PE']:
                processed_options.append({
                    'symbol': symbol,
                    'strikePrice': strike_price,
                    'option_type': option_type,
                    'lastPrice': option_data.get('ltp', 0),  # Last traded price
                    'openInterest': option_data.get('oi', 0),  # Open interest
                    'change': option_data.get('ltpch', 0),  # Change in LTP
                    'changePercent': option_data.get('ltpchp', 0),  # Change % in LTP
                    'volume': option_data.get('volume', 0),  # Volume
                    'bidPrice': option_data.get('bid', 0),  # Bid price
                    'askPrice': option_data.get('ask', 0),  # Ask price
                    'underlyingValue': spot_price,
                    'expiry': expiry
                })

        options_df = pd.DataFrame(processed_options)
        
        if not options_df.empty:
            # Log top 5 OI for CE and PE
            ce_oi = options_df[options_df['option_type'] == 'CE'].sort_values('openInterest', ascending=False).head(5)
            pe_oi = options_df[options_df['option_type'] == 'PE'].sort_values('openInterest', ascending=False).head(5)
            
            # Reduced logging - only log summary of top OI strikes
            if hasattr(ce_oi, 'shape') and ce_oi.shape[0] > 0:
                logging.info(f"Top CE strike by OI: {ce_oi.iloc[0]['strikePrice']} (OI: {ce_oi.iloc[0]['openInterest']})")
            if hasattr(pe_oi, 'shape') and pe_oi.shape[0] > 0:
                logging.info(f"Top PE strike by OI: {pe_oi.iloc[0]['strikePrice']} (OI: {pe_oi.iloc[0]['openInterest']})")
            
            # Only log minimal summary of highest OI strikes
            logging.info(f"Successfully fetched option chain with {len(options_df)} options")
        else:
            logging.warning("Option chain DataFrame is empty after processing.")

        return options_df

    except Exception as e:
        logging.error(f"Error in option chain retrieval: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def _get_nifty_option_chain_fallback():
    """
    Fallback method to fetch Nifty 50 option chain from NSE website
    This is used when Fyers API is not available or fails
    """
    try:
        logging.info("Using fallback method to fetch option chain from NSE")

        # First get the cookies from the main NSE page
        main_url = "https://www.nseindia.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.nseindia.com/",
            "Accept": "*/*"
        }

        # Create a session to maintain cookies
        session = requests.Session()

        # Get main page to set cookies
        logging.info("Fetching NSE main page to set cookies")
        main_response = session.get(main_url, headers=headers, timeout=15)
        if main_response.status_code != 200:
            logging.error(f"Failed to access NSE main page: {main_response.status_code}")
            return pd.DataFrame()

        # Add a small delay to avoid being blocked as a bot
        time.sleep(2)

        # Now get the option chain data
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        logging.info("Fetching option chain with established session")
        try:
            response = session.get(url, headers=headers, timeout=15)
            logging.info(f"NSE response status: {response.status_code}")
            if response.status_code != 200:
                logging.error(f"Failed to fetch option chain: Status code {response.status_code}")
                return pd.DataFrame()
            # --- DEBUG: Log first 200 chars of response ---
            logging.debug(f"NSE response content (first 200): {response.text[:200]}")
        except Exception as e:
            logging.error(f"Error fetching from NSE: {str(e)}")
            return pd.DataFrame()
        if response.status_code == 200:
            try:
                data = response.json()
                if 'records' not in data or 'data' not in data['records']:
                    logging.error("Invalid JSON structure from NSE API")
                    return pd.DataFrame()

                records = data['records']['data']

                # Get current date for expiry calculation
                today = datetime.datetime.now()

                # Find next Thursday for weekly expiry
                days_to_thursday = (3 - today.weekday()) % 7
                if days_to_thursday == 0:
                    # If today is Thursday, check if market closed
                    if today.hour > 15 or (today.hour == 15 and today.minute >= 30):
                        days_to_thursday = 7

                expiry_date = today + datetime.timedelta(days=days_to_thursday)
                expiry_str = f"{expiry_date.strftime('%y%b').upper()}{expiry_date.day}"

                option_chain = []
                for record in records:
                    strike_price = record['strikePrice']
                    underlying_value = record.get('underlyingValue', 0)

                    # Process CE (Call) data
                    if 'CE' in record:
                        ce = record['CE']
                        call_symbol = f"NSE:NIFTY{expiry_str}{strike_price}CE"
                        ce.update({
                            'symbol': call_symbol,
                            'strikePrice': strike_price,
                            'option_type': 'CE',
                            'underlyingValue': underlying_value
                        })
                        option_chain.append(ce)

                    # Process PE (Put) data
                    if 'PE' in record:
                        pe = record['PE']
                        put_symbol = f"NSE:NIFTY{expiry_str}{strike_price}PE"
                        pe.update({
                            'symbol': put_symbol,
                            'strikePrice': strike_price,
                            'option_type': 'PE',
                            'underlyingValue': underlying_value
                        })
                        option_chain.append(pe)

                result_df = pd.DataFrame(option_chain)
                logging.info(f"Fallback: Successfully fetched {len(result_df)} options")
                return result_df
            except Exception as e:
                logging.error(f"Error processing NSE response: {str(e)} | Raw content: {response.text[:500]}")
                return pd.DataFrame()
        else:
            logging.error(f"Failed to fetch option chain: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error fetching Nifty option chain: {str(e)}")
        # Return empty dataframe in case of error
        return pd.DataFrame()
