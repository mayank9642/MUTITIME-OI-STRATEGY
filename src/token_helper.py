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
    
    while retry_count < max_retries:
        try:
            # First check if we have a valid token
            if is_token_valid():
                config = load_config()
                token = config['fyers']['access_token']
                logging.info("Using existing valid access token")
                return token
            else:
                # Token is missing or expired, generate a new one
                logging.info(f"Generating new access token (attempt {retry_count + 1}/{max_retries})...")
                token = generate_access_token(use_totp)
                if token:
                    # Verify the token actually works by making a simple API call
                    from src.fyers_api_utils import get_fyers_client
                    fyers = get_fyers_client(check_token=False)  # Use the new token
                    if fyers:
                        try:
                            # Test with a simple API call
                            profile = fyers.get_profile()
                            if isinstance(profile, dict) and profile.get('s') == 'ok':
                                logging.info("Successfully generated and verified new access token")
                                return token
                            else:
                                logging.warning(f"Generated token verification failed: {profile}")
                        except Exception as verify_error:
                            logging.error(f"Error verifying new token: {str(verify_error)}")
                    else:
                        logging.info("Successfully generated new access token")
                        return token
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
