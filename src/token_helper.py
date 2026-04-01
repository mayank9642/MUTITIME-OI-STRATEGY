import datetime
import sys
import os
import logging

# Add the project root directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try using package import first (when running as module)
    from src.config import load_config
    from src.auth import generate_access_token
except ModuleNotFoundError:
    # Fall back to relative import (when running as script)
    from config import load_config
    from auth import generate_access_token

def is_token_valid():
    """
    Check if the access token is still valid or needs to be refreshed.
    
    Returns:
        bool: True if token is valid, False otherwise
    """
    try:
        config = load_config()
        token_expiry_str = config.get('fyers', {}).get('token_expiry', '')
        
        if not token_expiry_str:
            logging.warning("No token expiry found in config.")
            return False
        
        expiry_time = datetime.datetime.strptime(token_expiry_str, '%Y-%m-%d %H:%M:%S')
        current_time = datetime.datetime.now()
        
        # Add a buffer of 5 minutes to ensure we don't use a token that's about to expire
        buffer_time = datetime.timedelta(minutes=5)
        
        if current_time + buffer_time < expiry_time:
            return True
        else:
            logging.info("Token expired or about to expire.")
            return False
    
    except Exception as e:
        logging.error(f"Error checking token validity: {str(e)}")
        return False

def ensure_valid_token(use_totp=False, max_retries=3):
    """
    Check if token is valid, and if not, generate a new one with exponential backoff retry.
    
    Args:
        use_totp (bool): Whether to use TOTP for authentication
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Valid access token or None if all attempts fail
    """
    import time
    retry_count = 0
    retry_delay = 2  # Initial delay in seconds
    
    # Only attempt automatic generation of a token if a TOTP key is configured.
    # FYERS regulatory changes may require interactive 2FA daily; if no TOTP
    # key is available in config, do not attempt to auto-open an interactive
    # auth flow here. Instead, return None and let the operator perform the
    # interactive login (see README or src/auth.py).
    try:
        config = load_config()
    except Exception:
        config = {}

    totp_key = config.get('fyers', {}).get('totp_key')

    # If token still valid, return it
    if is_token_valid():
        try:
            config = load_config()
            token = config['fyers']['access_token']
            logging.info("Using existing valid access token")
            return token
        except Exception:
            logging.error("Token validity reported true but could not read token from config")
            return None

    # Token is invalid/expired at this point
    if not totp_key:
        logging.critical(
            "Access token expired and no TOTP key is configured.\n"
            "Due to regulatory changes, automatic refresh may not be available.\n"
            "Please run the interactive auth flow now (e.g. `python -m src.auth`) to re-authorize the App ID, or configure a TOTP key in config/config.yaml if a non-interactive flow is supported."
        )
        return None

    # TOTP key is present: attempt non-interactive TOTP-based auth (may open browser or use API depending on implementation)
    while retry_count < max_retries:
        try:
            logging.info(f"Generating new access token using TOTP (attempt {retry_count + 1}/{max_retries})...")
            token = generate_access_token(use_totp=True)
            if token:
                logging.info("Generated new access token using TOTP")
                return token
        except Exception as e:
            logging.error(f"Token generation error (attempt {retry_count + 1}/{max_retries}): {str(e)}")
        except Exception as e:
            logging.error(f"Token error (attempt {retry_count + 1}/{max_retries}): {str(e)}")
        
        # Increment retry counter and delay before next attempt
        retry_count += 1
        if retry_count < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    logging.critical("Failed to obtain valid token after multiple attempts. Please check your credentials and network connection.")
    return None
