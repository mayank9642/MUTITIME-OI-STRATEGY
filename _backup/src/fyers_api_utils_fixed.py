import json
import os
import logging
import datetime
import time
import base64
import uuid
import re
import requests
import random
from fyers_api import fyersModel
from fyers_api import accessToken
from src.config import load_config
from src.token_helper import ensure_valid_token

def get_fyers_client(check_token=True):
    """
    Get an authenticated Fyers client instance
    
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
            client_id = load_config()['fyers']['client_id']

        # IMPORTANT: For REST API, use only the access_token (not client_id:access_token)
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path="logs/")
        # Test connection with profile API
        profile_response = fyers.get_profile()
        if profile_response.get('s') == 'ok':
            logging.debug(f"Successfully authenticated with Fyers API for user: {profile_response.get('data', {}).get('name')}")
            return fyers
        else:
            logging.error(f"Authentication failed: {profile_response}")
            return None
    except Exception as e:
        logging.error(f"Error in get_fyers_client: {str(e)}")
        return None
